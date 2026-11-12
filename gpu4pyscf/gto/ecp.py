# Copyright 2025 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict
import ctypes
import numpy as np
import cupy as cp
from pyscf import gto
from pyscf import __config__
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import load_library, contract, get_avail_mem
from gpu4pyscf.gto.mole import group_basis

libecp = load_library('libgecp')

ecp_cart_argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]

libecp.ECP_cart.argtypes = ecp_cart_argtypes
libecp.ECP_ip_cart.argtypes = ecp_cart_argtypes
libecp.ECP_ipipv_cart.argtypes = ecp_cart_argtypes
libecp.ECP_ipvip_cart.argtypes = ecp_cart_argtypes

# Spin-orbit ECP kernel (may be absent in an older libgecp build).
_HAS_SO = hasattr(libecp, 'ECP_so_cart')
if _HAS_SO:
    libecp.ECP_so_cart.argtypes = ecp_cart_argtypes

ECP_ATOM_ID = 7

# Enable/disable the shell-pair x ECP-center screening of the task list.
SCREEN_ECP = getattr(__config__, 'gto_ecp_screen', True)

# Exponential-argument cutoff for the 3-center overlap estimate used to skip
# negligible (shell i, shell j, ECP center C) triples.  exp(-EXPCUTOFF) ~ 1e-17,
# matching the value hard-coded in pyscf/lib/gto/nr_ecp.h.
EXPCUTOFF = 39.0


def _ecp_expcutoff(mol):
    '''Screening cutoff on the exponential argument.  Never looser than the
    pyscf default (EXPCUTOFF); tightened further when mol.precision is tighter.
    '''
    # Look only at the instance dict: Mole has no `precision`, and falling
    # through to Mole.__getattr__ triggers a lazy `import pyscf.__all__`
    # (seconds on a network filesystem) inside the first get_ecp call.
    # pbc.gto.Cell sets self.precision in __init__, so it is still honoured.
    prec = vars(mol).get('precision', 1e-13)
    if not prec or prec <= 0:
        prec = 1e-13
    return max(EXPCUTOFF, -float(np.log(prec)))


class _ScreenData(tuple):
    '''Tuple subclass `(bas_min_exp, bas_coords, ecp_min_exp, ecp_coords)`
    that also carries precomputed distance and Level-1 screening tables for
    vectorized 2-level screening in `_screen_block`.
    '''
    # NOTE: tuple subclasses cannot declare non-empty __slots__; the extra
    # attributes live in the instance __dict__.

    def __new__(cls, bas_min_exp, bas_coords, ecp_min_exp, ecp_coords,
                rrca=None, rrab=None, l1_lhs=None, l1_rhs_fac=None):
        obj = super().__new__(
            cls, (bas_min_exp, bas_coords, ecp_min_exp, ecp_coords))
        obj.rrca = rrca
        obj.rrab = rrab
        obj.l1_lhs = l1_lhs
        obj.l1_rhs_fac = l1_rhs_fac
        return obj


def _collapse_l_groups(uniq_l_ctr, l_ctr_counts):
    '''Collapse (l, nprim) groups from group_basis() by unique angular
    momentum l. Since group_basis() sorts shells by l ascending (and nprim
    descending within each l), shells sharing the same l already occupy a
    single contiguous shell index range.
    '''
    if len(uniq_l_ctr) == 0:
        return uniq_l_ctr, l_ctr_counts
    ls = uniq_l_ctr[:, 0]
    uniq_l, first_idx = np.unique(ls, return_index=True)
    if len(uniq_l) == len(uniq_l_ctr):
        return uniq_l_ctr, l_ctr_counts
    collapsed_counts = np.add.reduceat(l_ctr_counts, first_idx)
    collapsed_uniq = uniq_l_ctr[first_idx].copy()
    return collapsed_uniq, collapsed_counts


def _compute_bas_min_exp(bas, env):
    nbas = len(bas)
    if nbas == 0:
        return np.empty(0, dtype=np.float64)
    ptr = bas[:, gto.PTR_EXP]
    nprim = bas[:, gto.NPRIM_OF]
    if np.all(nprim == 1):
        return env[ptr].astype(np.float64, copy=True)
    bas_min_exp = np.empty(nbas, dtype=np.float64)
    one_mask = nprim == 1
    bas_min_exp[one_mask] = env[ptr[one_mask]]
    for sh in np.flatnonzero(~one_mask):
        n = int(nprim[sh])
        p = int(ptr[sh])
        bas_min_exp[sh] = env[p:p+n].min() if n > 0 else np.inf
    return bas_min_exp


def _compute_ecp_min_exp(ecpbas, ecp_loc, env):
    ngrp = len(ecp_loc) - 1
    if ngrp <= 0 or len(ecpbas) == 0:
        return np.zeros(max(ngrp, 0), dtype=np.float64)
    ptr = ecpbas[:, gto.PTR_EXP]
    nprim = ecpbas[:, gto.NPRIM_OF]
    if np.all(nprim == 1):
        row_min = env[ptr]
    else:
        row_min = np.array(
            [env[p:p+n].min() if n > 0 else np.inf
             for p, n in zip(ptr, nprim)],
            dtype=np.float64)
    ecp_min_exp = np.minimum.reduceat(row_min, ecp_loc[:-1])
    ecp_min_exp[~np.isfinite(ecp_min_exp)] = 0.0
    return ecp_min_exp


def _build_screen_data(_sorted_mol, ecpbas, ecp_loc,
                       bas_min_exp=None, ecp_min_exp=None):
    '''Per-shell and per-ECP-group data needed by _screen_grid and _screen_block.

    Returns
        _ScreenData((bas_min_exp[nbas], bas_coords[nbas,3],
                     ecp_min_exp[ngrp], ecp_coords[ngrp,3]))
    where a group is one (l, atom) block delimited by ecp_loc, matching the
    third column (ksh) of the task grids.  All exponents are the smallest
    primitive exponent (slowest decay = most conservative bound).
    '''
    bas = _sorted_mol._bas
    env = _sorted_mol._env
    atom_coords = _sorted_mol.atom_coords()  # Bohr

    if bas_min_exp is None:
        bas_min_exp = _compute_bas_min_exp(bas, env)
    bas_atom = bas[:, gto.ATOM_OF]
    bas_coords = atom_coords[bas_atom]

    ecpbas = np.asarray(ecpbas)
    ngrp = len(ecp_loc) - 1
    if ngrp > 0 and len(ecpbas) > 0:
        if ecp_min_exp is None:
            ecp_min_exp = _compute_ecp_min_exp(ecpbas, ecp_loc, env)
        ecp_atom = ecpbas[ecp_loc[:-1], gto.ATOM_OF]
        ecp_coords = atom_coords[ecp_atom]
    else:
        ecp_min_exp = np.zeros(max(ngrp, 0), dtype=np.float64)
        ecp_coords = np.empty((max(ngrp, 0), 3), dtype=np.float64)
        ecp_atom = np.empty(0, dtype=np.int32)

    diff_atm = atom_coords[:, None, :] - atom_coords[None, :, :]
    dist2_atm = np.einsum('ijk,ijk->ij', diff_atm, diff_atm)
    rrab = dist2_atm[bas_atom[:, None], bas_atom[None, :]]
    if ngrp > 0 and len(ecpbas) > 0:
        rrca = dist2_atm[bas_atom[:, None], ecp_atom[None, :]]
    else:
        rrca = np.empty((len(bas), 0), dtype=np.float64)

    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        l1_lhs = bas_min_exp[:, None] * ecp_min_exp[None, :] * rrca
        l1_rhs_fac = bas_min_exp[:, None] + ecp_min_exp[None, :]

    return _ScreenData(bas_min_exp, bas_coords, ecp_min_exp, ecp_coords,
                       rrca=rrca, rrab=rrab,
                       l1_lhs=l1_lhs, l1_rhs_fac=l1_rhs_fac)


def _screen_grid(grid, screen_data, expcutoff):
    '''Drop (ish, jsh, ksh) rows whose 3-center overlap estimate is negligible.

    Mirrors check_3c_overlap() in pyscf/lib/gto/nr_ecp.c:
        eijk = (ai*aj*|Ri-Rj|^2 + ai*ak*|C-Ri|^2 + aj*ak*|C-Rj|^2) / (ai+aj+ak)
    keep the triple when eijk < expcutoff.
    '''
    if grid.shape[0] == 0:
        return grid
    bas_min_exp, bas_coords, ecp_min_exp, ecp_coords = screen_data
    ish = grid[:, 0]
    jsh = grid[:, 1]
    ksh = grid[:, 2]
    ai = bas_min_exp[ish]
    aj = bas_min_exp[jsh]
    ak = ecp_min_exp[ksh]
    if getattr(screen_data, 'rrca', None) is not None:
        rrab = screen_data.rrab[ish, jsh]
        rrca = screen_data.rrca[ish, ksh]
        rrcb = screen_data.rrca[jsh, ksh]
    else:
        rab = bas_coords[ish] - bas_coords[jsh]
        rca = ecp_coords[ksh] - bas_coords[ish]
        rcb = ecp_coords[ksh] - bas_coords[jsh]
        rrab = np.sum(rab*rab, axis=1)
        rrca = np.sum(rca*rca, axis=1)
        rrcb = np.sum(rcb*rcb, axis=1)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        eijk = (ai*aj*rrab + ai*ak*rrca + aj*ak*rrcb) / (ai + aj + ak)
    return grid[eijk < expcutoff]


def sort_ecp_basis(_ecpbas, cart=True, log=None):
    '''
    # Sort ECP basis based on angular momentum
    # Remove SO Type basis functions
    '''
    not_so_type = _ecpbas[:, gto.SO_TYPE_OF] == 0
    _ecpbas = _ecpbas[not_so_type]

    # Sort ECP basis based on angular momentum and atom_id
    l_atm = _ecpbas[:,[gto.ANG_OF, gto.ATOM_OF]]

    uniq_l_atm, inv_idx, l_atm_counts = np.unique(
        l_atm, return_inverse=True, return_counts=True, axis=0)
    sorted_idx = np.argsort(inv_idx.ravel(), kind='stable').astype(np.int32)

    # Sort basis inplace
    _ecpbas = _ecpbas[sorted_idx]

    # Group ECP basis based on angular momentum and atom id
    # Each group contains basis with multiple power order
    ecp_loc = np.append(0, np.cumsum(l_atm_counts))

    # Further group based on angular momentum for counting
    uniq_l, l_counts = np.unique(uniq_l_atm[:,0], return_counts=True, axis=0)

    return _ecpbas, uniq_l, l_counts, ecp_loc


def _screen_block(ish_range, jsh_range, ksh_range, screen_data, expcutoff,
                  triangular, return_columns=False):
    '''(ish, jsh, ksh) task rows surviving the 3-center overlap screen, built
    without ever materializing the full ish x jsh x ksh grid.

    Two levels, per ECP group k:
      1. keep shell i (resp. j) only if the best-case bound (partner
         exponent -> 0) is above threshold:
             ai*ak*|C-Ri|^2  <  expcutoff * (ai + ak)
      2. run the full check_3c_overlap estimate on the surviving (i, j) pairs.
    Tasks are ordered in (j, i, k) order (k fastest, i next, j slowest) to
    maximize L1/L2 cache locality across CUDA threadblocks.
    '''
    bas_min_exp, bas_coords, ecp_min_exp, ecp_coords = screen_data
    ish_range = np.asarray(ish_range, dtype=np.int32)
    jsh_range = np.asarray(jsh_range, dtype=np.int32)
    ksh_range = np.asarray(ksh_range, dtype=np.int32)
    if ish_range.size == 0 or jsh_range.size == 0 or ksh_range.size == 0:
        return [] if return_columns else np.zeros((0, 3), dtype=np.int64)

    has_precomp = getattr(screen_data, 'rrca', None) is not None
    out_cols = []

    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        if has_precomp:
            rrca_mat = screen_data.rrca
            rrab_mat = screen_data.rrab
            l1_lhs = screen_data.l1_lhs
            l1_rhs_fac = screen_data.l1_rhs_fac
            keep_ik = (l1_lhs[np.ix_(ish_range, ksh_range)]
                       < expcutoff * l1_rhs_fac[np.ix_(ish_range, ksh_range)])
            if triangular and ish_range.shape == jsh_range.shape and (
                    ish_range[0] == jsh_range[0] and ish_range[-1] == jsh_range[-1]):
                keep_jk = keep_ik
            else:
                keep_jk = (l1_lhs[np.ix_(jsh_range, ksh_range)]
                           < expcutoff * l1_rhs_fac[np.ix_(jsh_range, ksh_range)])

            for idx_k, k in enumerate(ksh_range):
                i_keep = ish_range[keep_ik[:, idx_k]]
                j_keep = jsh_range[keep_jk[:, idx_k]]
                if i_keep.size == 0 or j_keep.size == 0:
                    continue
                # indexing='xy' orders jj outer (slowest), ii inner (faster)
                ii, jj = np.meshgrid(i_keep, j_keep, indexing='xy')
                ii = ii.ravel()
                jj = jj.ravel()
                if triangular:
                    m = ii <= jj
                    ii = ii[m]
                    jj = jj[m]
                    if ii.size == 0:
                        continue
                aii = bas_min_exp[ii]
                ajj = bas_min_exp[jj]
                ak = ecp_min_exp[k]
                eijk = (aii * ajj * rrab_mat[ii, jj]
                        + aii * ak * rrca_mat[ii, k]
                        + ajj * ak * rrca_mat[jj, k]) / (aii + ajj + ak)
                m = eijk < expcutoff
                if np.any(m):
                    ii_m = ii[m]
                    jj_m = jj[m]
                    kk_m = np.full(ii_m.size, k, dtype=np.int32)
                    out_cols.append((ii_m, jj_m, kk_m))
        else:
            ai_all = bas_min_exp[ish_range]
            aj_all = bas_min_exp[jsh_range]
            ri_all = bas_coords[ish_range]
            rj_all = bas_coords[jsh_range]
            for k in ksh_range:
                ak = ecp_min_exp[k]
                ck = ecp_coords[k]
                rrca = np.sum((ri_all - ck)**2, axis=1)
                rrcb = np.sum((rj_all - ck)**2, axis=1)
                i_keep = ish_range[ai_all*ak*rrca < expcutoff * (ai_all + ak)]
                j_keep = jsh_range[aj_all*ak*rrcb < expcutoff * (aj_all + ak)]
                if i_keep.size == 0 or j_keep.size == 0:
                    continue
                ii, jj = np.meshgrid(i_keep, j_keep, indexing='xy')
                ii = ii.ravel()
                jj = jj.ravel()
                if triangular:
                    m = ii <= jj
                    ii = ii[m]
                    jj = jj[m]
                    if ii.size == 0:
                        continue
                aii = bas_min_exp[ii]
                ajj = bas_min_exp[jj]
                rab = bas_coords[ii] - bas_coords[jj]
                rca = ck - bas_coords[ii]
                rcb = ck - bas_coords[jj]
                eijk = (aii*ajj*np.sum(rab*rab, axis=1)
                        + aii*ak*np.sum(rca*rca, axis=1)
                        + ajj*ak*np.sum(rcb*rcb, axis=1)) / (aii + ajj + ak)
                m = eijk < expcutoff
                if np.any(m):
                    ii_m = ii[m]
                    jj_m = jj[m]
                    kk_m = np.full(ii_m.size, k, dtype=np.int32)
                    out_cols.append((ii_m, jj_m, kk_m))

    if return_columns:
        return out_cols
    if not out_cols:
        return np.zeros((0, 3), dtype=np.int64)
    if len(out_cols) == 1:
        ii_m, jj_m, kk_m = out_cols[0]
    else:
        ii_m = np.concatenate([c[0] for c in out_cols])
        jj_m = np.concatenate([c[1] for c in out_cols])
        kk_m = np.concatenate([c[2] for c in out_cols])
    return np.column_stack((ii_m, jj_m, kk_m)).astype(np.int64)


def _unscreened_block_columns(ish_range, jsh_range, ksh_range, triangular):
    '''Build unscreened (ii, jj, kk) int32 column arrays in (j, i, k) order
    (k fastest, i next, j slowest), matching original GPU4PySCF ordering.
    '''
    ish_range = np.asarray(ish_range, dtype=np.int32)
    jsh_range = np.asarray(jsh_range, dtype=np.int32)
    ksh_range = np.asarray(ksh_range, dtype=np.int32)
    if ish_range.size == 0 or jsh_range.size == 0 or ksh_range.size == 0:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty, empty
    # np.meshgrid(ish_range, jsh_range, ksh_range) uses indexing='xy', which
    # orders jsh slowest, ish middle, ksh fastest.
    ii, jj, kk = np.meshgrid(ish_range, jsh_range, ksh_range)
    ii = ii.ravel()
    jj = jj.ravel()
    kk = kk.ravel()
    if triangular:
        m = ii <= jj
        ii = ii[m]
        jj = jj[m]
        kk = kk[m]
    return ii, jj, kk


def _build_tasks(l_ctr_offsets, lecp_ctr_offsets, screen_data, expcutoff,
                 triangular):
    tasks = {}
    n_groups = len(l_ctr_offsets) - 1
    n_ecp_groups = len(lecp_ctr_offsets) - 1
    do_screen = screen_data is not None and SCREEN_ECP
    for i in range(n_groups):
        j_start = i if triangular else 0
        for j in range(j_start, n_groups):
            for k in range(n_ecp_groups):
                ish_range = np.arange(
                    l_ctr_offsets[i], l_ctr_offsets[i+1], dtype=np.int32)
                jsh_range = np.arange(
                    l_ctr_offsets[j], l_ctr_offsets[j+1], dtype=np.int32)
                ksh_range = np.arange(
                    lecp_ctr_offsets[k], lecp_ctr_offsets[k+1], dtype=np.int32)
                if do_screen:
                    grid = _screen_block(ish_range, jsh_range, ksh_range,
                                         screen_data, expcutoff, triangular)
                else:
                    ii, jj, kk = _unscreened_block_columns(
                        ish_range, jsh_range, ksh_range, triangular)
                    grid = np.column_stack((ii, jj, kk)).astype(np.int64)
                tasks[i, j, k] = grid
    return tasks


def make_tasks(l_ctr_offsets, lecp_ctr_offsets, screen_data=None,
               expcutoff=EXPCUTOFF):
    return _build_tasks(l_ctr_offsets, lecp_ctr_offsets, screen_data,
                        expcutoff, triangular=True)


def make_full_tasks(l_ctr_offsets, lecp_ctr_offsets, screen_data=None,
                    expcutoff=EXPCUTOFF):
    return _build_tasks(l_ctr_offsets, lecp_ctr_offsets, screen_data,
                        expcutoff, triangular=False)


def _build_packed_tasks(l_ctr_offsets, lecp_ctr_offsets, uniq_l_ctr, uniq_lecp,
                        screen_data, expcutoff, triangular):
    '''Build and pack all non-empty (i, j, k) task grids into a single 1D int32
    GPU buffer.

    Each CUDA kernel expects `tasks` of length `3 * ntasks` laid out as
    `[ish_0..ish_{n-1}, jsh_0..jsh_{n-1}, ksh_0..ksh_{n-1}]` (Fortran order of
    an `(ntasks, 3)` array). Packing all `(i, j, k)` buckets end-to-end into one
    contiguous 1D buffer replaces hundreds of small `cp.asarray(..., order='F')`
    allocations and host-to-device transfers with a single transfer.

    Returns
        (tasks_flat_gpu, launch_specs)
        where `launch_specs` is a list of `(off_ints, ntasks, li, lj, lk)`.
    '''
    n_groups = len(l_ctr_offsets) - 1
    n_ecp_groups = len(lecp_ctr_offsets) - 1
    do_screen = screen_data is not None and SCREEN_ECP

    ish_ranges = [np.arange(l_ctr_offsets[i], l_ctr_offsets[i+1], dtype=np.int32)
                  for i in range(n_groups)]
    ksh_ranges = [np.arange(lecp_ctr_offsets[k], lecp_ctr_offsets[k+1], dtype=np.int32)
                  for k in range(n_ecp_groups)]

    flat_segments = []
    launch_specs = []
    off_ints = 0

    for i in range(n_groups):
        ish_range = ish_ranges[i]
        li = int(uniq_l_ctr[i, 0])
        j_start = i if triangular else 0
        for j in range(j_start, n_groups):
            jsh_range = ish_ranges[j]
            lj = int(uniq_l_ctr[j, 0])
            for k in range(n_ecp_groups):
                ksh_range = ksh_ranges[k]
                lk = int(uniq_lecp[k])
                if do_screen:
                    cols = _screen_block(
                        ish_range, jsh_range, ksh_range,
                        screen_data, expcutoff, triangular,
                        return_columns=True)
                    if not cols:
                        continue
                    if len(cols) == 1:
                        ii, jj, kk = cols[0]
                    else:
                        ii = np.concatenate([c[0] for c in cols])
                        jj = np.concatenate([c[1] for c in cols])
                        kk = np.concatenate([c[2] for c in cols])
                else:
                    ii, jj, kk = _unscreened_block_columns(
                        ish_range, jsh_range, ksh_range, triangular)
                    if ii.size == 0:
                        continue
                ntasks = int(ii.size)
                flat_segments.extend((ii, jj, kk))
                launch_specs.append((off_ints, ntasks, li, lj, lk))
                off_ints += 3 * ntasks

    if not launch_specs:
        return cp.empty(0, dtype=np.int32), []
    tasks_flat_host = np.concatenate(flat_segments)
    tasks_flat_gpu = cp.asarray(tasks_flat_host, dtype=np.int32)
    return tasks_flat_gpu, launch_specs


def select_basis(ecpbas, ecp_atoms):
    """
    Select ECP basis for given ECP atoms, and reindexing atoms
    """
    ecp_atoms_list = list(ecp_atoms)
    if len(ecpbas) == 0 or len(ecp_atoms_list) == 0:
        return np.empty((0, ecpbas.shape[1] if ecpbas.ndim == 2 else 8),
                        dtype=ecpbas.dtype)
    ecpbas = np.asarray(ecpbas)
    atm_ids = ecpbas[:, gto.ATOM_OF]
    mask = np.isin(atm_ids, ecp_atoms_list)
    selected = ecpbas[mask].copy()
    if len(selected) == 0:
        return selected
    atom_map = {int(a): idx for idx, a in enumerate(ecp_atoms_list)}
    selected[:, ECP_ATOM_ID] = [atom_map[int(a)] for a in selected[:, gto.ATOM_OF]]
    return selected


_TOPO_CACHE = OrderedDict()
_TOPO_CACHE_MAXSIZE = 8
_GEOM_CACHE = OrderedDict()
_GEOM_CACHE_MAXSIZE = 16


def _make_topo_key(mol):
    device_id = cp.cuda.Device().id
    env_nocoord = mol._env.copy()
    if len(mol._atm) > 0:
        ptrs = mol._atm[:, gto.PTR_COORD][:, None] + np.arange(3)
        valid = (ptrs >= 0) & (ptrs < env_nocoord.size)
        env_nocoord[ptrs[valid]] = 0.0
    atm_topo = np.delete(mol._atm, gto.PTR_COORD, axis=1) if len(mol._atm) else mol._atm
    return (
        device_id,
        bool(mol.cart),
        mol._bas.shape,
        hash(mol._bas.tobytes()),
        mol._ecpbas.shape,
        hash(mol._ecpbas.tobytes()),
        atm_topo.shape,
        hash(atm_topo.tobytes()),
        env_nocoord.shape,
        hash(env_nocoord.tobytes()),
    )


class _EcpTopologyCache:
    '''Tier-1 topology cache: stores geometry-invariant sorted basis, angular
    momentum group offsets, AO transformation matrix, and sorted ECP tables.
    '''
    def __init__(self, mol, key):
        self.key = key
        _sorted_mol, coeff, uniq_l_ctr, l_ctr_counts = group_basis(mol)
        uniq_l_ctr, l_ctr_counts = _collapse_l_groups(uniq_l_ctr, l_ctr_counts)
        self._sorted_mol = _sorted_mol
        self.base_env = _sorted_mol._env.copy()
        self.uniq_l_ctr = uniq_l_ctr
        self.l_ctr_counts = l_ctr_counts
        self.l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
        self.n_groups = len(uniq_l_ctr)

        self.bas_min_exp = _compute_bas_min_exp(
            _sorted_mol._bas, _sorted_mol._env)

        ao_loc = _sorted_mol.ao_loc_nr(cart=True)
        self.nao = int(ao_loc[-1])
        self.ao_loc_gpu = cp.asarray(ao_loc, dtype=np.int32)
        self.coeff_gpu = cp.asarray(coeff)
        self.atm_gpu = cp.asarray(_sorted_mol._atm, dtype=np.int32)
        self.bas_gpu = cp.asarray(_sorted_mol._bas, dtype=np.int32)

        # Scalar ECP basis (full molecule)
        ecpbas_s, uniq_lecp_s, lecp_counts_s, ecp_loc_s = sort_ecp_basis(
            _sorted_mol._ecpbas)
        self.ecpbas_scalar = ecpbas_s
        self.uniq_lecp_scalar = uniq_lecp_s
        self.lecp_offsets_scalar = np.append(0, np.cumsum(lecp_counts_s))
        self.ecp_loc_scalar = ecp_loc_s
        self.ecp_min_exp_scalar = _compute_ecp_min_exp(
            ecpbas_s, ecp_loc_s, _sorted_mol._env)
        self.ecpbas_scalar_gpu = cp.asarray(ecpbas_s, dtype=np.int32)
        self.ecploc_scalar_gpu = cp.asarray(ecp_loc_s, dtype=np.int32)

        # Default derivative ECP basis (with ECP_ATOM_ID reindexed 0..n_ecp_atm-1)
        self.default_ecp_atoms = sorted(
            set(int(a) for a in mol._ecpbas[:, gto.ATOM_OF]))
        ecpbas_d = select_basis(mol._ecpbas, self.default_ecp_atoms)
        ecpbas_d, uniq_lecp_d, lecp_counts_d, ecp_loc_d = sort_ecp_basis(
            ecpbas_d)
        self.ecpbas_deriv_default = ecpbas_d
        self.uniq_lecp_deriv_default = uniq_lecp_d
        self.lecp_offsets_deriv_default = np.append(
            0, np.cumsum(lecp_counts_d))
        self.ecp_loc_deriv_default = ecp_loc_d
        self.ecp_min_exp_deriv_default = _compute_ecp_min_exp(
            ecpbas_d, ecp_loc_d, _sorted_mol._env)
        self.ecpbas_deriv_default_gpu = cp.asarray(ecpbas_d, dtype=np.int32)
        self.ecploc_deriv_default_gpu = cp.asarray(ecp_loc_d, dtype=np.int32)

        # Spin-orbit ECP basis (lazy)
        self._so_initialized = False

    def ensure_so(self):
        if not self._so_initialized:
            ecpbas_so, uniq_lecp_so, lecp_counts_so, ecp_loc_so = (
                sort_ecp_basis_so(self._sorted_mol._ecpbas))
            self.ecpbas_so = ecpbas_so
            self.uniq_lecp_so = uniq_lecp_so
            self.lecp_offsets_so = np.append(0, np.cumsum(lecp_counts_so))
            self.ecp_loc_so = ecp_loc_so
            self.ecp_min_exp_so = _compute_ecp_min_exp(
                ecpbas_so, ecp_loc_so, self._sorted_mol._env)
            self.ecpbas_so_gpu = cp.asarray(ecpbas_so, dtype=np.int32)
            self.ecploc_so_gpu = cp.asarray(ecp_loc_so, dtype=np.int32)
            self._so_initialized = True


class _EcpGeometryCache:
    '''Tier-2 geometry cache: stores coordinate-dependent screening tables,
    device `_env` array, and packed GPU task buffers.
    '''
    def __init__(self, mol, topo, expcutoff):
        self.topo = topo
        self.expcutoff = expcutoff
        coords = np.asarray(mol.atom_coords(), dtype=np.float64)
        env = topo.base_env.copy()
        if len(topo._sorted_mol._atm) > 0:
            ptrs = topo._sorted_mol._atm[:, gto.PTR_COORD][:, None] + np.arange(3)
            env[ptrs] = coords
        self.env_host = env
        self.activate()
        self.env_gpu = cp.asarray(env, dtype=np.float64)

        self.screen_data_scalar = _build_screen_data(
            topo._sorted_mol, topo.ecpbas_scalar, topo.ecp_loc_scalar,
            bas_min_exp=topo.bas_min_exp,
            ecp_min_exp=topo.ecp_min_exp_scalar)

        self._packed_scalar_tri = None
        self._packed_deriv_default_full = None
        self._packed_so_tri = None

    def activate(self):
        '''Point the shared Tier-1 ``_sorted_mol`` at THIS geometry's host
        ``_env`` so that any screening table built lazily below (deriv / SO
        task lists, or per-batch subsets in ``_EcpDerivContext.run_batch``)
        sees the coordinates that match ``env_gpu``.  Several geometries can
        share one topology object (geometry optimisation, finite differences),
        and the last geometry constructed must not leak into the others.
        '''
        self.topo._sorted_mol._env = self.env_host

    def packed_scalar_tri(self):
        if self._packed_scalar_tri is None:
            t = self.topo
            self.activate()
            self._packed_scalar_tri = _build_packed_tasks(
                t.l_ctr_offsets, t.lecp_offsets_scalar,
                t.uniq_l_ctr, t.uniq_lecp_scalar,
                self.screen_data_scalar, self.expcutoff, triangular=True)
        return self._packed_scalar_tri

    def packed_deriv_default_full(self):
        if self._packed_deriv_default_full is None:
            t = self.topo
            self.activate()
            screen_data_d = _build_screen_data(
                t._sorted_mol, t.ecpbas_deriv_default, t.ecp_loc_deriv_default,
                bas_min_exp=t.bas_min_exp,
                ecp_min_exp=t.ecp_min_exp_deriv_default)
            self._packed_deriv_default_full = _build_packed_tasks(
                t.l_ctr_offsets, t.lecp_offsets_deriv_default,
                t.uniq_l_ctr, t.uniq_lecp_deriv_default,
                screen_data_d, self.expcutoff, triangular=False)
        return self._packed_deriv_default_full

    def packed_so_tri(self):
        if self._packed_so_tri is None:
            t = self.topo
            t.ensure_so()
            self.activate()
            screen_data_so = _build_screen_data(
                t._sorted_mol, t.ecpbas_so, t.ecp_loc_so,
                bas_min_exp=t.bas_min_exp,
                ecp_min_exp=t.ecp_min_exp_so)
            self._packed_so_tri = _build_packed_tasks(
                t.l_ctr_offsets, t.lecp_offsets_so,
                t.uniq_l_ctr, t.uniq_lecp_so,
                screen_data_so, self.expcutoff, triangular=True)
        return self._packed_so_tri


def _get_ecp_caches(mol):
    topo_key = _make_topo_key(mol)
    topo = _TOPO_CACHE.get(topo_key)
    if topo is None:
        topo = _EcpTopologyCache(mol, topo_key)
        _TOPO_CACHE[topo_key] = topo
        if len(_TOPO_CACHE) > _TOPO_CACHE_MAXSIZE:
            _TOPO_CACHE.popitem(last=False)
    else:
        _TOPO_CACHE.move_to_end(topo_key)

    expcutoff = _ecp_expcutoff(mol)
    coords_bytes = np.ascontiguousarray(mol.atom_coords(), dtype=np.float64).tobytes()
    geom_key = (topo_key, coords_bytes, expcutoff, bool(SCREEN_ECP))
    geom = _GEOM_CACHE.get(geom_key)
    if geom is None:
        geom = _EcpGeometryCache(mol, topo, expcutoff)
        _GEOM_CACHE[geom_key] = geom
        if len(_GEOM_CACHE) > _GEOM_CACHE_MAXSIZE:
            _GEOM_CACHE.popitem(last=False)
    else:
        _GEOM_CACHE.move_to_end(geom_key)
    return topo, geom


def get_ecp(mol):
    """
    Calculate sum of ECP integrals

    Returns:
        CuPy array: [nao, nao]
            sum of ECP integrals over all ecp atoms
    """
    assert len(mol._ecpbas) > 0

    topo, geom = _get_ecp_caches(mol)
    tasks_flat_gpu, launch_specs = geom.packed_scalar_tri()

    nao = topo.nao
    mat1 = cp.zeros([nao, nao])
    base_ptr = tasks_flat_gpu.data.ptr
    ao_loc_ptr = topo.ao_loc_gpu.data.ptr
    ecpbas_ptr = topo.ecpbas_scalar_gpu.data.ptr
    ecploc_ptr = topo.ecploc_scalar_gpu.data.ptr
    atm_ptr = topo.atm_gpu.data.ptr
    bas_ptr = topo.bas_gpu.data.ptr
    env_ptr = geom.env_gpu.data.ptr
    mat1_ptr = mat1.data.ptr

    for off_ints, ntasks, li, lj, lk in launch_specs:
        err = libecp.ECP_cart(
            mat1_ptr, ao_loc_ptr, nao,
            base_ptr + off_ints * 4, ntasks,
            ecpbas_ptr, ecploc_ptr,
            atm_ptr, bas_ptr, env_ptr,
            li, lj, lk)
        if err != 0:
            raise RuntimeError('ECP CUDA kernel failed.')
    return topo.coeff_gpu.T @ mat1 @ topo.coeff_gpu


def sort_ecp_basis_so(_ecpbas):
    '''Like :func:`sort_ecp_basis` but keeps only the spin-orbit projectors
    (``SO_TYPE_OF == 1``) and rewrites the ``ul`` term (``ANG_OF == -1``) to
    ``max_l(atom) + 1``, matching ECPtype_so_cart in pyscf/lib/gto/nr_ecp.c.
    '''
    _ecpbas = _ecpbas[_ecpbas[:, gto.SO_TYPE_OF] == 1].copy()
    if len(_ecpbas) == 0:
        return _ecpbas, np.zeros(0, dtype=int), np.zeros(0, dtype=int), \
            np.zeros(1, dtype=int)

    # ul term (lc == -1) is treated as L_max + 1 for that atom
    # CPU (ECPtype_so_cart): ecp_lmax[atom] starts at 0 and is raised by the
    # non-ul projectors; the ul term then becomes ecp_lmax[atom] + 1.
    ul = _ecpbas[:, gto.ANG_OF] == -1
    if np.any(ul):
        for atm_id in np.unique(_ecpbas[ul, gto.ATOM_OF]):
            rows = _ecpbas[:, gto.ATOM_OF] == atm_id
            lmax = _ecpbas[rows & ~ul, gto.ANG_OF].max(initial=0)
            _ecpbas[rows & ul, gto.ANG_OF] = lmax + 1

    l_atm = _ecpbas[:, [gto.ANG_OF, gto.ATOM_OF]]
    uniq_l_atm, inv_idx, l_atm_counts = np.unique(
        l_atm, return_inverse=True, return_counts=True, axis=0)
    sorted_idx = np.argsort(inv_idx.ravel(), kind='stable').astype(np.int32)
    _ecpbas = _ecpbas[sorted_idx]
    ecp_loc = np.append(0, np.cumsum(l_atm_counts))
    uniq_l, l_counts = np.unique(uniq_l_atm[:, 0], return_counts=True, axis=0)
    return _ecpbas, uniq_l, l_counts, ecp_loc


def get_ecp_so(mol):
    '''Spin-orbit ECP integrals, real cartesian-component form.

    Returns
        CuPy array ``[3, nao, nao]`` (spherical AO basis, real): the three
        components ``<i| l_a dU^SO |j>`` for ``a in {x, y, z}``.  Equivalent to
        ``mol.intor('ECPso')``.  Assemble the spinor operator with
        :func:`get_soc_1e`.
    '''
    assert len(mol._ecpbas) > 0
    if not _HAS_SO:
        raise NotImplementedError(
            'libgecp was built without ECP_so_cart; rebuild gpu4pyscf/lib.')
    if not np.any(mol._ecpbas[:, gto.SO_TYPE_OF] == 1):
        raise ValueError('mol has no spin-orbit ECP projectors')

    topo, geom = _get_ecp_caches(mol)
    tasks_flat_gpu, launch_specs = geom.packed_so_tri()

    nao = topo.nao
    mat1 = cp.zeros([3, nao, nao])
    base_ptr = tasks_flat_gpu.data.ptr
    ao_loc_ptr = topo.ao_loc_gpu.data.ptr
    ecpbas_ptr = topo.ecpbas_so_gpu.data.ptr
    ecploc_ptr = topo.ecploc_so_gpu.data.ptr
    atm_ptr = topo.atm_gpu.data.ptr
    bas_ptr = topo.bas_gpu.data.ptr
    env_ptr = geom.env_gpu.data.ptr
    mat1_ptr = mat1.data.ptr

    for off_ints, ntasks, li, lj, lk in launch_specs:
        err = libecp.ECP_so_cart(
            mat1_ptr, ao_loc_ptr, nao,
            base_ptr + off_ints * 4, ntasks,
            ecpbas_ptr, ecploc_ptr,
            atm_ptr, bas_ptr, env_ptr,
            li, lj, lk)
        if err != 0:
            raise RuntimeError('SO-ECP CUDA kernel failed.')

    coeff = topo.coeff_gpu
    mat1 = contract('aij,jq->aiq', mat1, coeff)
    mat1 = contract('aiq,ip->apq', mat1, coeff)
    return mat1


def get_soc_1e(mol):
    '''One-electron spin-orbit ECP operator in the 2-component (spinor) basis.

    Returns
        CuPy array ``[2*nao, 2*nao]`` complex, matching the SOC block that
        ``pyscf.scf.ghf.GHF.get_hcore`` adds:
        ``einsum('sxy,spq->xpyq', -1j * 0.5 * PauliMatrices, get_ecp_so(mol))``.
    '''
    from pyscf import lib as pyscf_lib
    so = get_ecp_so(mol)                       # [3, nao, nao] real
    nao = so.shape[-1]
    pauli = cp.asarray(pyscf_lib.PauliMatrices)   # [3, 2, 2] complex
    hso = cp.einsum('sxy,spq->xpyq', -1j * 0.5 * pauli,
                    so.astype(cp.complex128))
    return hso.reshape(2 * nao, 2 * nao)


_IP_FN = {
    'ip':    (libecp.ECP_ip_cart,    3),
    'ipipv': (libecp.ECP_ipipv_cart, 9),
    'ipvip': (libecp.ECP_ipvip_cart, 9),
}


class _EcpDerivContext:
    '''Batch-invariant data for the ECP-derivative kernels, built once so that
    slicing over ECP atoms only re-does the cheap per-batch work (basis
    selection, screening, task list, kernel launch, AO transform).
    '''
    def __init__(self, mol):
        self.topo, self.geom = _get_ecp_caches(mol)
        self._sorted_mol = self.topo._sorted_mol
        self.uniq_l_ctr = self.topo.uniq_l_ctr
        self.l_ctr_offsets = self.topo.l_ctr_offsets
        self.expcutoff = self.geom.expcutoff
        self.ecpbas_src = mol._ecpbas
        self.atm = self.topo.atm_gpu
        self.bas = self.topo.bas_gpu
        self.env = self.geom.env_gpu
        self.nao = self.topo.nao
        self.ao_loc = self.topo.ao_loc_gpu
        self.coeff = self.topo.coeff_gpu
        self.n_groups = self.topo.n_groups

    def run_batch(self, fn, comp, ecp_atoms_batch):
        '''[len(ecp_atoms_batch), comp, nao, nao] in the AO (transformed) basis.
        Row r corresponds to ecp_atoms_batch[r].'''
        batch_list = [int(a) for a in ecp_atoms_batch]
        if batch_list == self.topo.default_ecp_atoms:
            tasks_flat_gpu, launch_specs = self.geom.packed_deriv_default_full()
            ecpbas_gpu = self.topo.ecpbas_deriv_default_gpu
            ecploc_gpu = self.topo.ecploc_deriv_default_gpu
        else:
            ecpbas = select_basis(self.ecpbas_src, batch_list)
            ecpbas, uniq_lecp, lecp_counts, ecp_loc = sort_ecp_basis(ecpbas)
            lecp_offsets = np.append(0, np.cumsum(lecp_counts))
            self.geom.activate()
            screen_data = _build_screen_data(
                self._sorted_mol, ecpbas, ecp_loc,
                bas_min_exp=self.topo.bas_min_exp)
            tasks_flat_gpu, launch_specs = _build_packed_tasks(
                self.l_ctr_offsets, lecp_offsets,
                self.uniq_l_ctr, uniq_lecp,
                screen_data, self.expcutoff, triangular=False)
            ecpbas_gpu = cp.asarray(ecpbas, dtype=np.int32)
            ecploc_gpu = cp.asarray(ecp_loc, dtype=np.int32)

        nbatch = len(batch_list)
        mat1 = cp.zeros([nbatch, comp, self.nao, self.nao])
        base_ptr = tasks_flat_gpu.data.ptr
        ao_loc_ptr = self.ao_loc.data.ptr
        ecpbas_ptr = ecpbas_gpu.data.ptr
        ecploc_ptr = ecploc_gpu.data.ptr
        atm_ptr = self.atm.data.ptr
        bas_ptr = self.bas.data.ptr
        env_ptr = self.env.data.ptr
        mat1_ptr = mat1.data.ptr

        for off_ints, ntasks, li, lj, lk in launch_specs:
            err = fn(
                mat1_ptr, ao_loc_ptr, self.nao,
                base_ptr + off_ints * 4, ntasks,
                ecpbas_ptr, ecploc_ptr,
                atm_ptr, bas_ptr, env_ptr,
                li, lj, lk)
            if err != 0:
                raise RuntimeError('ECP CUDA kernel failed.')
        mat1 = contract('axij,jq->axiq', mat1, self.coeff)
        mat1 = contract('axiq,ip->axpq', mat1, self.coeff)
        return mat1


def _default_ecp_atoms(mol):
    return sorted(set(int(a) for a in mol._ecpbas[:, gto.ATOM_OF]))


def _ecp_atom_batch_size(comp, nao, n_ecp_atm, batch_size=None):
    '''How many ECP atoms to process at once so the transient
    [batch, comp, nao, nao] tensor (plus AO-transform scratch) stays well
    within free GPU memory.'''
    if batch_size is not None:
        return max(1, min(int(batch_size), n_ecp_atm))
    # mat1 + the two contract() temporaries ~ 3 buffers of this size
    per_atom = comp * nao * nao * 8 * 3
    avail = get_avail_mem()
    b = int(avail * 0.2 / max(per_atom, 1))
    return max(1, min(b, n_ecp_atm))


def loop_ecp_ip(mol, ip_type='ip', ecp_atoms=None, batch_size=None):
    '''Generator over ECP atoms in memory-bounded chunks.

    Yields ``(atom_ids, mat)`` where ``atom_ids`` is a list of atom indices and
    ``mat`` is a CuPy array ``[len(atom_ids), comp, nao, nao]`` (comp = 3 for
    ``ip``, 9 for ``ipipv``/``ipvip``); ``mat[r]`` is the derivative w.r.t.
    ``atom_ids[r]``.  Avoids ever allocating the full
    ``[n_ecp_atoms, comp, nao, nao]`` tensor.
    '''
    assert len(mol._ecpbas) > 0
    if ip_type not in _IP_FN:
        raise ValueError(f'Invalid IP type: {ip_type}')
    fn, comp = _IP_FN[ip_type]

    if ecp_atoms is None:
        ecp_atoms = _default_ecp_atoms(mol)
    else:
        ecp_atoms = [int(a) for a in ecp_atoms]
    if len(ecp_atoms) == 0:
        return

    ctx = _EcpDerivContext(mol)
    bs = _ecp_atom_batch_size(comp, ctx.nao, len(ecp_atoms), batch_size)
    for p0 in range(0, len(ecp_atoms), bs):
        batch = ecp_atoms[p0:p0+bs]
        yield batch, ctx.run_batch(fn, comp, batch)


def loop_ecp_ipip(mol, ip_type='ipipv', ecp_atoms=None, batch_size=None):
    '''Same as :func:`loop_ecp_ip` for the second derivatives
    (``ipipv``/``ipvip``, comp = 9).'''
    yield from loop_ecp_ip(mol, ip_type=ip_type, ecp_atoms=ecp_atoms,
                           batch_size=batch_size)


def get_ecp_ip(mol, ip_type='ip', ecp_atoms=None, batch_size=None):
    """
    First derivative of ECP integrals

    Returns:
        CuPy array: [n_ecp_atoms, 3, nao, nao],
            first dimension reindexed according to sorted(ecp_atoms)

    Note: materializes the full tensor.  For large systems prefer
    :func:`loop_ecp_ip` or :func:`get_ecp_ip_sum`.
    """
    mats = [mat for _batch, mat in
            loop_ecp_ip(mol, ip_type, ecp_atoms, batch_size)]
    if not mats:
        _, comp = _IP_FN[ip_type]
        return cp.zeros([0, comp, mol.nao, mol.nao])
    return cp.concatenate(mats, axis=0)


def get_ecp_ipip(mol, ip_type='ipipv', ecp_atoms=None, batch_size=None):
    """
    Second derivatives of ECP integrals
    Args:
        ip_type:
            ipipv -> (i''|ecp|j)
            ipvip -> (i'|ecp|j')

    Returns:
        CuPy array: [n_ecp_atoms, 9, nao, nao],
            first dimension reindexed according to sorted(ecp_atoms)

    Note: materializes the full tensor.  For large systems prefer
    :func:`loop_ecp_ipip` or :func:`get_ecp_ipip_sum`.
    """
    return get_ecp_ip(mol, ip_type, ecp_atoms, batch_size)


def get_ecp_ip_sum(mol, ip_type='ip', ecp_atoms=None, batch_size=None):
    '''sum over ECP atoms of the derivative integrals -> [comp, nao, nao].

    Equivalent to ``get_ecp_ip(mol, ...).sum(axis=0)`` but memory-bounded.
    '''
    out = None
    for _batch, mat in loop_ecp_ip(mol, ip_type, ecp_atoms, batch_size):
        s = mat.sum(axis=0)
        out = s if out is None else out + s
    if out is None:
        _, comp = _IP_FN[ip_type]
        return cp.zeros([comp, mol.nao, mol.nao])
    return out


def get_ecp_ipip_sum(mol, ip_type='ipipv', ecp_atoms=None, batch_size=None):
    '''sum over ECP atoms of the 2nd-derivative integrals -> [9, nao, nao].'''
    return get_ecp_ip_sum(mol, ip_type, ecp_atoms, batch_size)

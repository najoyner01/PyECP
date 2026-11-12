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

'''
Short-range ECP under periodic boundary conditions -- scalar and spin-orbit.

GPU port of ``pyscf.pbc.gto.ecp.ecp_int``.  Design + phasing:
``docs/ecp-pbc-design.md``.

A periodic ECP matrix is a lattice sum of *molecular* ECP integrals:

    H^ECP_k[i,j] = sum_L e^{i k.L} sum_{L_U} <phi_i(0) | U_ECP(L_U) | phi_j(L)>

so we build a super-molecule (reference-cell bra shells + image ket shells for
every lattice vector L + image ECP projectors for every L_U), reuse the
validated molecular kernels ``libgecp.ECP_cart`` / ``ECP_so_cart`` over the
cross-image task list, then fold the ket-image axis with per-k phase factors.
No new CUDA.

The ket-image axis is processed in memory-bounded batches (phase 4) so the
transient ``[comp, nao_sup, nao_sup]`` super-molecule matrix stays within GPU
memory for large cells; the ECP-projector image sum is always full (cheap).
'''

import math
import numpy as np
import cupy as cp
from pyscf import gto, lib
from gpu4pyscf.lib.cupy_helper import contract, get_avail_mem
from gpu4pyscf.gto.ecp import (
    libecp, sort_ecp_basis, sort_ecp_basis_so,
    _build_screen_data, _screen_block, _unscreened_block_columns,
    _collapse_l_groups, _ecp_expcutoff, ECP_ATOM_ID,
)
from gpu4pyscf.gto import ecp as _mol_ecp
from gpu4pyscf.gto.mole import group_basis

__all__ = ['ecp_int']

_INTORS = {
    # intor        : (sort function,      libgecp kernel,        comp)
    'ECPscalar': (sort_ecp_basis,    libecp.ECP_cart,    1),
}
if hasattr(libecp, 'ECP_so_cart'):          # libgecp built with SO-ECP
    _INTORS['ECPso'] = (sort_ecp_basis_so, libecp.ECP_so_cart, 3)


def _ecp_rcut(cell):
    '''Real-space cutoff for the ECP operator.

    Never shorter than ``cell.rcut`` (tuned for AO overlap); extended so the
    slowest-decaying ECP Gaussian is captured to ``cell.precision``.
    '''
    rcut = float(cell.rcut)
    if len(cell._ecpbas) == 0:
        return rcut
    env = cell._env
    a_min = np.inf
    for row in cell._ecpbas:
        p, n = row[gto.PTR_EXP], row[gto.NPRIM_OF]
        exps = env[p:p + n]
        exps = exps[exps > 0]
        if exps.size:
            a_min = min(a_min, float(exps.min()))
    if not np.isfinite(a_min) or a_min <= 0:
        return rcut
    prec = getattr(cell, 'precision', 1e-8) or 1e-8
    ecp_rcut = np.sqrt(max(-np.log(prec) / a_min, 0.0))
    return max(rcut, float(ecp_rcut))


def _image_batch_size(nao_ref_cart, comp, n_images, safety=0.25):
    '''How many ket lattice images to include in one super-molecule so that the
    transient ``[comp, nao_ref*(1+b), nao_ref*(1+b)]`` matrix (plus scratch)
    stays within ``safety`` of free GPU memory.'''
    avail = get_avail_mem()
    lim = safety * avail / (comp * 8.0 * max(nao_ref_cart, 1) ** 2)
    b = int(math.sqrt(max(lim, 1.0))) - 1
    return int(min(max(b, 1), n_images))


def _ecp_supmol(Ls, ket_img_ids, sorted_mol, sorted_ecpbas):
    '''Super-molecule for one batch of ket lattice images.

    ``_bas``    : [ref-cell bra shells] | [ket shells for m in ket_img_ids]
    ``_atm``    : [ref-cell atoms] | [image m atoms for every m in range(nL)]
    ``_ecpbas`` : ECP projectors for every image m in range(nL)   (always full)
    ``_env``    : original env + image atom coords for every m

    Image-major ket ordering: ``mat1[:nao_ref, nao_ref:]`` reshaped to
    ``[nao_ref, nket, nao_ref]`` maps column b onto lattice image
    ``ket_img_ids[b]``.
    '''
    natm = sorted_mol.natm
    nbas_ref = sorted_mol.nbas
    nL = len(Ls)
    nket = len(ket_img_ids)

    ref_ao_loc = sorted_mol.ao_loc_nr(cart=True)
    nao_ref_cart = int(ref_ao_loc[-1])
    ref_atom_coords = sorted_mol.atom_coords()          # [natm, 3] Bohr

    img_coords = (Ls[:, None, :] + ref_atom_coords[None, :, :]).reshape(-1, 3)
    n_orig_env = len(sorted_mol._env)
    _env = np.append(sorted_mol._env, img_coords.ravel())

    _atm_ref = sorted_mol._atm.copy()
    _atm_imgs = np.repeat(sorted_mol._atm[None, :, :], nL,
                          axis=0).reshape(-1, gto.ATM_SLOTS)
    _atm_imgs[:, gto.PTR_COORD] = n_orig_env + np.arange(nL * natm) * 3
    _atm = np.vstack([_atm_ref, _atm_imgs])

    _bas_ket = []
    for m in ket_img_ids:
        b = sorted_mol._bas.copy()
        b[:, gto.ATOM_OF] += natm + m * natm
        _bas_ket.append(b)
    _bas = np.vstack([sorted_mol._bas.copy()] + _bas_ket)

    _ecpbas_list = []
    for m in range(nL):
        e = sorted_ecpbas.copy()
        e[:, gto.ATOM_OF] = natm + m * natm + sorted_ecpbas[:, gto.ATOM_OF]
        e[:, ECP_ATOM_ID] = e[:, gto.ATOM_OF]
        _ecpbas_list.append(e)
    _ecpbas = np.concatenate(_ecpbas_list, axis=0)

    supmol = gto.Mole()
    supmol._atm = np.asarray(_atm, dtype=np.int32)
    supmol._bas = np.asarray(_bas, dtype=np.int32)
    supmol._ecpbas = _ecpbas
    supmol._env = _env
    supmol.verbose = 0
    supmol.output = None

    ao_loc = np.empty(nbas_ref * (1 + nket) + 1, dtype=np.int32)
    ao_loc[:nbas_ref + 1] = ref_ao_loc
    for b in range(nket):
        ao_loc[nbas_ref * (b + 1):nbas_ref * (b + 2)] = \
            (b + 1) * nao_ref_cart + ref_ao_loc[:-1]
    ao_loc[-1] = (1 + nket) * nao_ref_cart
    return supmol, ao_loc, nao_ref_cart


def _lattice_ecp_cart(cell, intor):
    '''Cross-image cartesian ECP block, folded over nothing yet.

    Returns
        blk   : CuPy ``[comp, nao_ref, nL, nao_ref]`` in the cell's own AO basis
                (cart->sph applied), where ``blk[c, i, L, j]`` =
                ``sum_{L_U} <phi_i(0) | U^c_ECP(L_U) | phi_j(L)>``.
        Ls    : ``[nL, 3]`` lattice vectors used.
    '''
    sort_fn, kernel, comp = _INTORS[intor]

    Ls = cell.get_lattice_Ls(rcut=_ecp_rcut(cell))
    nL = len(Ls)

    sorted_mol, coeff, uniq_l_ctr, l_ctr_counts = group_basis(cell)
    uniq_l_ctr, l_ctr_counts = _collapse_l_groups(uniq_l_ctr, l_ctr_counts)
    l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
    n_groups = len(uniq_l_ctr)
    nbas_ref = sorted_mol.nbas

    sorted_ecpbas, uniq_lecp, lecp_counts, ref_ecp_loc = sort_fn(
        sorted_mol._ecpbas)
    if len(sorted_ecpbas) == 0:
        raise ValueError(f'cell has no {intor} projectors')
    n_ref_ecp_groups = len(ref_ecp_loc) - 1
    n_ref_ecpbas = len(sorted_ecpbas)
    lecp_offsets = np.append(0, np.cumsum(lecp_counts))
    n_ecp_l_types = len(uniq_lecp)

    ref_ao_loc = sorted_mol.ao_loc_nr(cart=True)
    nao_ref_cart = int(ref_ao_loc[-1])
    expcutoff = _ecp_expcutoff(cell)
    do_screen = _mol_ecp.SCREEN_ECP

    # supmol ECP groups span every image: image m, ref group g ->
    #   supmol group m*n_ref_ecp_groups + g,  offset m*n_ref_ecpbas + ref_ecp_loc[g]
    supmol_ecp_loc = np.concatenate(
        [m * n_ref_ecpbas + ref_ecp_loc[:-1] for m in range(nL)]
        + [np.array([nL * n_ref_ecpbas])]).astype(np.int32)

    blk_cart = cp.zeros([comp, nao_ref_cart, nL, nao_ref_cart])

    bsize = _image_batch_size(nao_ref_cart, comp, nL)
    for b0 in range(0, nL, bsize):
        b1 = min(b0 + bsize, nL)
        batch = list(range(b0, b1))
        nket = len(batch)

        supmol, ao_loc, _ = _ecp_supmol(Ls, batch, sorted_mol, sorted_ecpbas)
        screen_data = _build_screen_data(supmol, supmol._ecpbas, supmol_ecp_loc)

        atm_g = cp.asarray(supmol._atm, dtype=np.int32)
        bas_g = cp.asarray(supmol._bas, dtype=np.int32)
        env_g = cp.asarray(supmol._env, dtype=np.float64)
        ecpbas_g = cp.asarray(supmol._ecpbas, dtype=np.int32)
        ecploc_g = cp.asarray(supmol_ecp_loc, dtype=np.int32)
        ao_loc_g = cp.asarray(ao_loc, dtype=np.int32)
        nao_sup = int(ao_loc[-1])

        mat1 = cp.zeros([comp, nao_sup, nao_sup])

        flat_segments = []
        launch_specs = []
        off_ints = 0
        ksh_sups = [
            np.concatenate([
                m * n_ref_ecp_groups + np.arange(
                    lecp_offsets[k], lecp_offsets[k + 1], dtype=np.int32)
                for m in range(nL)])
            for k in range(n_ecp_l_types)
        ]

        for i in range(n_groups):
            li = int(uniq_l_ctr[i, 0])
            ish_range = np.arange(
                l_ctr_offsets[i], l_ctr_offsets[i + 1], dtype=np.int32)
            for j in range(n_groups):
                lj = int(uniq_l_ctr[j, 0])
                # batch ket shells of type j across the batch images
                jsh_range = np.concatenate([
                    np.arange(nbas_ref * (b + 1) + l_ctr_offsets[j],
                              nbas_ref * (b + 1) + l_ctr_offsets[j + 1],
                              dtype=np.int32)
                    for b in range(nket)])
                for k in range(n_ecp_l_types):
                    lk = int(uniq_lecp[k])
                    ksh_sup = ksh_sups[k]
                    if do_screen:
                        cols = _screen_block(
                            ish_range, jsh_range, ksh_sup,
                            screen_data, expcutoff,
                            triangular=False, return_columns=True)
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
                            ish_range, jsh_range, ksh_sup, triangular=False)
                        if ii.size == 0:
                            continue
                    ntasks = int(ii.size)
                    flat_segments.extend((ii, jj, kk))
                    launch_specs.append((off_ints, ntasks, li, lj, lk))
                    off_ints += 3 * ntasks

        if launch_specs:
            tasks_flat_g = cp.asarray(
                np.concatenate(flat_segments), dtype=np.int32)
            base_ptr = tasks_flat_g.data.ptr
            for off_ints, ntasks, li, lj, lk in launch_specs:
                err = kernel(
                    mat1.data.ptr, ao_loc_g.data.ptr, nao_sup,
                    base_ptr + off_ints * 4, ntasks,
                    ecpbas_g.data.ptr, ecploc_g.data.ptr,
                    atm_g.data.ptr, bas_g.data.ptr, env_g.data.ptr,
                    li, lj, lk)
                if err != 0:
                    raise RuntimeError(f'PBC {intor} CUDA kernel failed.')

        # bra x ket block -> [comp, nao_ref_cart, nket, nao_ref_cart]
        sub = mat1[:, :nao_ref_cart, nao_ref_cart:].reshape(
            comp, nao_ref_cart, nket, nao_ref_cart)
        blk_cart[:, :, b0:b1, :] = sub
        del mat1, sub
        cp.get_default_memory_pool().free_all_blocks()

    # cart -> sph (cell AO basis) on bra and ket
    coeff_g = cp.asarray(coeff)                         # [nao_cart_sorted, nao]
    blk = contract('ip,ciLj->cpLj', coeff_g, blk_cart)
    blk = contract('cpLi,iq->cpLq', blk, coeff_g)
    return blk, Ls


def ecp_int(cell, kpts=None, intor='ECPscalar'):
    '''Periodic short-range ECP integrals.

    Args:
        cell : :class:`pyscf.pbc.gto.Cell` with ``cell._ecpbas``.
        kpts : (nkpts, 3) array or None (Gamma).
        intor: ``'ECPscalar'`` -> ``[nao, nao]`` (Gamma, real) /
               ``[nkpts, nao, nao]``;
               ``'ECPso'`` -> spinor operator ``[2*nao, 2*nao]`` /
               ``[nkpts, 2*nao, 2*nao]`` (complex),
               matching ``pyscf.pbc.gto.ecp.ecp_int(cell, kpts, intor)``.
    '''
    if intor not in _INTORS:
        if intor == 'ECPso':
            raise NotImplementedError(
                'libgecp was built without ECP_so_cart; rebuild gpu4pyscf/lib.')
        raise ValueError(f'intor must be one of {list(_INTORS)}, got {intor!r}')
    if len(cell._ecpbas) == 0:
        raise ValueError('cell has no ECP basis')
    has_so = bool(np.any(cell._ecpbas[:, gto.SO_TYPE_OF] == 1))
    if intor == 'ECPso' and not has_so:
        raise ValueError('cell has no spin-orbit ECP projectors')

    blk, Ls = _lattice_ecp_cart(cell, intor)            # [comp, nao, nL, nao]
    nao = blk.shape[1]

    if kpts is None:
        phase = cp.ones((1, len(Ls)))
        single = True
    else:
        kpts_2d = np.asarray(kpts).reshape(-1, 3)
        single = (np.asarray(kpts).ndim == 1)
        phase = cp.asarray(np.exp(1j * kpts_2d @ Ls.T))    # [nkpts, nL]

    # fold lattice images -> [nkpts, comp, nao, nao]
    out = contract('kL,cpLq->kcpq', phase, blk.astype(cp.complex128))

    if intor == 'ECPscalar':
        out = out[:, 0]                                    # [nkpts, nao, nao]
        if kpts is None:
            out = out.real
        return out[0] if single else out

    # ECPso: i/2 sigma . <l U>  ->  [nkpts, 2nao, 2nao]
    # (cp.einsum for the tiny [3,2,2] x [nk,3,nao,nao] complex contraction, as in
    #  gpu4pyscf.gto.ecp.get_soc_1e)
    pauli = cp.asarray(lib.PauliMatrices)                  # [3, 2, 2]
    hso = cp.einsum('sxy,kspq->kxpyq', -1j * 0.5 * pauli, out)
    hso = hso.reshape(-1, 2 * nao, 2 * nao)
    return hso[0] if single else hso

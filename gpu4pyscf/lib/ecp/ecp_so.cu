/*
 * Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Spin-orbit ECP integrals, real cartesian-component form:
 *
 *     gctr[a, i, j] = < i | l_a  dU^SO(r) | j >,   a in {x, y, z}
 *
 * This is the type-2 (semilocal) machinery of ecp_type2.cu with two changes,
 * matching ECPtype_so_cart() in pyscf/lib/gto/nr_ecp.c:
 *
 *   1. the shell-j angular factor is contracted with the angular-momentum
 *      operator L^a_{m'm} (so_ang_matrix.cu, real & antisymmetric) on the
 *      projector spherical index -- producing three components instead of one.
 *      type2_ang() is linear in `omega`, so we apply L^a to omegaj up front
 *      (transform_omega_lop) and reuse the scalar contraction unchanged;
 *   2. the driver feeds only SO_TYPE_OF == 1 projectors, with the `ul` term
 *      (lc == -1) already rewritten to max_l(atom)+1 on the host side.
 *
 * The prefactor is identical to the scalar type-2 kernel: the `prad[i] *= .5`
 * in ECPtype_so_cart is part of the CPU's iterative multi-level quadrature
 * refinement (compensating point-doubling between levels), not a physical
 * factor -- see the `//common_fac *= .5` comment there. The GPU uses a single
 * fixed 128-point rule, so no extra 1/2.
 *
 * The output is real and antisymmetric in (i, j): < j | l_a U | i > =
 * -< i | l_a U | j >.  The task list is triangular (ish <= jsh); the off
 * -diagonal transpose is written with the opposite sign.
 *
 * The complex/Pauli (spinor) assembly is done in Python (gto/ecp.py:get_soc_1e).
 */

// jmm[a] : L^a on the (2*LC+1) projector index.  out[r, m'] = sum_m L[m',m] in[r, m].
__device__
static void transform_omega_lop(double *out, const double *in,
                                const int nrow, const int LC, const int a)
{
    const int dlc = 2*LC + 1;
    const double *L = _l_op[LC] + a*dlc*dlc;
    for (int r = threadIdx.x; r < nrow; r += blockDim.x){
        const double *win = in + (size_t)r*dlc;
        double *wout = out + (size_t)r*dlc;
        for (int mp = 0; mp < dlc; mp++){
            double s = 0.0;
            for (int m = 0; m < dlc; m++){
                s += L[mp*dlc + m] * win[m];
            }
            wout[mp] = s;
        }
    }
}

__global__
void so_cart(double * __restrict__ gctr,
             const int LI, const int LJ, const int LC,
             const int *ao_loc, const int nao,
             const int *tasks, const int ntasks,
             const int *ecpbas, const int *ecploc,
             const int *atm, const int *bas, const double *env)
{
    const int task_id = blockIdx.x;
    if (task_id >= ntasks){
        return;
    }

    const int ish = tasks[task_id];
    const int jsh = tasks[task_id + ntasks];
    const int ksh = tasks[task_id + 2*ntasks];

    extern __shared__ double smem[];

    const double *ri = env + atm[PTR_COORD+bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS];
    const double *rj = env + atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];

    const int atm_id = ecpbas[ATOM_OF+ecploc[ksh]*BAS_SLOTS];
    const double *rc = env + atm[PTR_COORD+atm_id*ATM_SLOTS];

    double ur = 0.0;
    for (int kbas = ecploc[ksh]; kbas < ecploc[ksh+1]; kbas++){
        ur += rad_part(kbas, ecpbas, env);
    }

    double rca[3], rcb[3];
    rca[0] = rc[0] - ri[0];
    rca[1] = rc[1] - ri[1];
    rca[2] = rc[2] - ri[2];
    rcb[0] = rc[0] - rj[0];
    rcb[1] = rc[1] - rj[1];
    rcb[2] = rc[2] - rj[2];

    const int LIC1 = LI + LC + 1;
    const int LJC1 = LJ + LC + 1;
    const int nfi = (LI+1) * (LI+2) / 2;
    const int nfj = (LJ+1) * (LJ+2) / 2;

    // smem layout:  rad_all | omegai | omegaj | omegaj_a | angi | angj
    const int omegai_sz = (LI+LC+2)/2 * (LI+1)*(LI+2)*(LI+3)/6 * (2*LC+1);
    const int omegaj_rows = (LJ+LC+2)/2 * ((LJ+1)*(LJ+2)*(LJ+3)/6);
    const int omegaj_sz = omegaj_rows * (2*LC+1);

    double *rad_all  = smem;
    double *omegai   = rad_all + (LI+LJ+1) * LIC1 * LJC1;
    double *omegaj   = omegai + omegai_sz;
    double *omegaj_a = omegaj + omegaj_sz;
    double *angi     = omegaj_a + omegaj_sz;
    double *angj     = angi + (LI+1)*nfi*LIC1;

    type2_facs_omega(omegai, LI, LC, rca);
    type2_facs_omega(omegaj, LJ, LC, rcb);
    __syncthreads();

    const int npi = bas[NPRIM_OF+ish*BAS_SLOTS];
    const int npj = bas[NPRIM_OF+jsh*BAS_SLOTS];
    const double *ai = env + bas[PTR_EXP+ish*BAS_SLOTS];
    const double *aj = env + bas[PTR_EXP+jsh*BAS_SLOTS];
    const double *ci = env + bas[PTR_COEFF+ish*BAS_SLOTS];
    const double *cj = env + bas[PTR_COEFF+jsh*BAS_SLOTS];

    double radi[AO_LMAX+ECP_LMAX+1];
    const double dca = norm3d(rca[0], rca[1], rca[2]);
    type2_facs_rad<0>(radi, LI+LC, npi, dca, ci, ai);

    double radj[AO_LMAX+ECP_LMAX+1];
    const double dcb = norm3d(rcb[0], rcb[1], rcb[2]);
    type2_facs_rad<0>(radj, LJ+LC, npj, dcb, cj, aj);

    double root = 0.0;
    if (threadIdx.x < NGAUSS){
        root = r128[threadIdx.x];
    }
    set_shared_memory(rad_all, (LI+LJ+1)*LIC1*LJC1);
    for (int p = 0; p <= LI+LJ; p++){
        double *prad = rad_all + p*LIC1*LJC1;
        for (int i = 0; i <= LI+LC; i++){
            const double ri_ur = radi[i] * ur;
            for (int j = (p + i) & 1; j <= LJ+LC; j += 2){
                block_reduce_NoSync(ri_ur * radj[j], prad + i*LJC1 + j);
            }
        }
        ur *= root;
    }
    __syncthreads();

    // same prefactor as the scalar type-2 kernel (no extra 1/2 -- see header)
    const double fac = 16.0 * M_PI * M_PI * _common_fac[LI] * _common_fac[LJ];

    constexpr int nreg = (NF_MAX*NF_MAX + THREADS - 1)/THREADS;
    const int ioff = ao_loc[ish];
    const int joff = ao_loc[jsh];

    for (int a = 0; a < 3; a++){
        transform_omega_lop(omegaj_a, omegaj, omegaj_rows, LC, a);
        __syncthreads();

        double reg_gctr[nreg];
        for (int r = 0; r < nreg; r++){
            reg_gctr[r] = 0.0;
        }

        // (k+l)pq, k i m p, l j m q -> i j     (m: projector spherical index)
        for (int m = 0; m < 2*LC+1; m++){
            type2_ang(angi, LI, LC, rca, omegai   + m);
            type2_ang(angj, LJ, LC, rcb, omegaj_a + m);
            __syncthreads();

            for (int ij = threadIdx.x; ij < nfi*nfj; ij += blockDim.x){
                const int i = ij % nfi;
                const int j = ij / nfi;
                double s = 0.0;
                for (int k = 0; k <= LI; k++){
                for (int l = 0; l <= LJ; l++){
                    const double *pangi = angi + k*nfi*LIC1 + i*LIC1;
                    const double *pangj = angj + l*nfj*LJC1 + j*LJC1;
                    const double *prad  = rad_all + (k+l)*LIC1*LJC1;
                    for (int p = (LC + k) & 1; p <= k + LC; p += 2){
                        const double aip = pangi[p];
                        const double *prad_p = prad + p*LJC1;
                        for (int q = (LC + l) & 1; q <= l + LC; q += 2){
                            s += prad_p[q] * aip * pangj[q];
                        }
                    }
                }}
                reg_gctr[ij/THREADS] += fac * s;
            }
            __syncthreads();
        }

        // gctr[a, bra=i, ket=j] (row-major); transpose block gets -1 (antisym).
        double *g = gctr + (size_t)a * nao * nao;
        for (int ij = threadIdx.x; ij < nfi*nfj; ij += blockDim.x){
            const int i = ij % nfi;
            const int j = ij / nfi;
            const double tmp = reg_gctr[ij/THREADS];
            atomicAdd(g + (size_t)(i+ioff)*nao + (j+joff), tmp);
            if (ish != jsh){
                atomicAdd(g + (size_t)(j+joff)*nao + (i+ioff), -tmp);
            }
        }
        __syncthreads();
    }
    return;
}

template <int LI, int LJ, int LC> __global__
void so_cart(double * __restrict__ gctr,
             const int *ao_loc, const int nao,
             const int *tasks, const int ntasks,
             const int *ecpbas, const int *ecploc,
             const int *atm, const int *bas, const double *env)
{
    const int task_id = blockIdx.x;
    if (task_id >= ntasks || LC < 1){
        return;
    }

    const int ish = tasks[task_id];
    const int jsh = tasks[task_id + ntasks];
    const int ksh = tasks[task_id + 2*ntasks];

    const double *ri = env + atm[PTR_COORD+bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS];
    const double *rj = env + atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];
    const int atm_id = ecpbas[ATOM_OF+ecploc[ksh]*BAS_SLOTS];
    const double *rc = env + atm[PTR_COORD+atm_id*ATM_SLOTS];

    double ur = 0.0;
    for (int kbas = ecploc[ksh]; kbas < ecploc[ksh+1]; kbas++){
        ur += rad_part(kbas, ecpbas, env);
    }

    double rca[3], rcb[3];
    rca[0] = rc[0] - ri[0];
    rca[1] = rc[1] - ri[1];
    rca[2] = rc[2] - ri[2];
    rcb[0] = rc[0] - rj[0];
    rcb[1] = rc[1] - rj[1];
    rcb[2] = rc[2] - rj[2];

    constexpr int LI1 = LI + 1;
    constexpr int LJ1 = LJ + 1;
    constexpr int LIC1 = LI + LC + 1;
    constexpr int LJC1 = LJ + LC + 1;
    constexpr int LCC1 = 2 * LC + 1;
    constexpr int nfi = (LI+1) * (LI+2) / 2;
    constexpr int nfj = (LJ+1) * (LJ+2) / 2;
    constexpr int BLKI = (LIC1+1)/2 * LCC1;
    constexpr int omegaj_rows = (LJC1+1)/2 * (LJ1*(LJ1+1)*(LJ1+2)/6);
    constexpr int BLKJ = (LJC1+1)/2 * LCC1;

    __shared__ double rad_all[(LI+LJ+1) * LIC1 * LJC1];
    __shared__ double omegai[LI1*(LI1+1)*(LI1+2)/6 * BLKI];
    __shared__ double omegaj[LJ1*(LJ1+1)*(LJ1+2)/6 * BLKJ];
    __shared__ double omegaj_a[LJ1*(LJ1+1)*(LJ1+2)/6 * BLKJ];
    __shared__ double angi[LI1*nfi*LIC1];
    __shared__ double angj[LJ1*nfj*LJC1];

    type2_facs_omega<LI, LC>(omegai, rca);
    type2_facs_omega<LJ, LC>(omegaj, rcb);
    __syncthreads();

    const int npi = bas[NPRIM_OF+ish*BAS_SLOTS];
    const int npj = bas[NPRIM_OF+jsh*BAS_SLOTS];
    const double *ai = env + bas[PTR_EXP+ish*BAS_SLOTS];
    const double *aj = env + bas[PTR_EXP+jsh*BAS_SLOTS];
    const double *ci = env + bas[PTR_COEFF+ish*BAS_SLOTS];
    const double *cj = env + bas[PTR_COEFF+jsh*BAS_SLOTS];

    double radi[LIC1];
    const double dca = norm3d(rca[0], rca[1], rca[2]);
    type2_facs_rad<0>(radi, LI+LC, npi, dca, ci, ai);

    double radj[LJC1];
    const double dcb = norm3d(rcb[0], rcb[1], rcb[2]);
    type2_facs_rad<0>(radj, LJ+LC, npj, dcb, cj, aj);

    double root = 0.0;
    if (threadIdx.x < NGAUSS){
        root = r128[threadIdx.x];
    }
    set_shared_memory(rad_all, (LI+LJ+1)*LIC1*LJC1);
    for (int p = 0; p <= LI+LJ; p++){
        double *prad = rad_all + p*LIC1*LJC1;
        for (int i = 0; i <= LI+LC; i++){
            const double ri_ur = radi[i] * ur;
            for (int j = (p + i) & 1; j <= LJ+LC; j += 2){
                block_reduce_NoSync(ri_ur * radj[j], prad + i*LJC1 + j);
            }
        }
        ur *= root;
    }
    __syncthreads();

    const double fac = 16.0 * M_PI * M_PI * _common_fac[LI] * _common_fac[LJ];
    constexpr int nreg = (nfi*nfj + THREADS - 1)/THREADS;
    const int ioff = ao_loc[ish];
    const int joff = ao_loc[jsh];

    for (int a = 0; a < 3; a++){
        transform_omega_lop(omegaj_a, omegaj, omegaj_rows, LC, a);
        __syncthreads();

        double reg_gctr[nreg];
        for (int r = 0; r < nreg; r++){
            reg_gctr[r] = 0.0;
        }

        for (int m = 0; m < LCC1; m++){
            type2_ang<LI, LC>(angi, rca, omegai   + m);
            type2_ang<LJ, LC>(angj, rcb, omegaj_a + m);
            __syncthreads();

            for (int ij = threadIdx.x; ij < nfi*nfj; ij += blockDim.x){
                const int i = ij % nfi;
                const int j = ij / nfi;
                double s = 0.0;
                for (int k = 0; k <= LI; k++){
                for (int l = 0; l <= LJ; l++){
                    const double *pangi = angi + k*nfi*LIC1 + i*LIC1;
                    const double *pangj = angj + l*nfj*LJC1 + j*LJC1;
                    const double *prad  = rad_all + (k+l)*LIC1*LJC1;
                    for (int p = (LC + k) & 1; p <= k + LC; p += 2){
                        const double aip = pangi[p];
                        const double *prad_p = prad + p*LJC1;
                        for (int q = (LC + l) & 1; q <= l + LC; q += 2){
                            s += prad_p[q] * aip * pangj[q];
                        }
                    }
                }}
                reg_gctr[ij/THREADS] += fac * s;
            }
            __syncthreads();
        }

        double *g = gctr + (size_t)a * nao * nao;
        for (int ij = threadIdx.x; ij < nfi*nfj; ij += blockDim.x){
            const int i = ij % nfi;
            const int j = ij / nfi;
            const double tmp = reg_gctr[ij/THREADS];
            atomicAdd(g + (size_t)(i+ioff)*nao + (j+joff), tmp);
            if (ish != jsh){
                atomicAdd(g + (size_t)(j+joff)*nao + (i+ioff), -tmp);
            }
        }
        __syncthreads();
    }
    return;
}

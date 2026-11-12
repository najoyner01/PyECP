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


template <int LI, int LJ, int LC> __device__
void type2_cart_unrolled_sub(
    double *gctr, double *rca, double *rcb,
    const double *radi, const double *radj, const double ur,
    double *omegai, double *omegaj, double *rad_all,
    double *angi, double *angj)
{
    constexpr int LI1 = LI+1;
    constexpr int LIC1 = LI+LC+1;
    constexpr int LJC1 = LJ+LC+1;
    constexpr int LCC1 = (2*LC+1);

    type2_facs_omega<LI, LC>(omegai, rca);
    set_shared_memory(rad_all, (LI+LJ+1)*LIC1*LJC1);

    double root = 0.0;
    if (threadIdx.x < NGAUSS){
        root = r128[threadIdx.x];
    }
    double ur_tmp = ur;
    for (int p = 0; p <= LI+LJ; p++){
        double *prad = rad_all + p*LIC1*LJC1;
        for (int i = 0; i <= LI+LC; i++){
            const double ri_ur = radi[i] * ur_tmp;
            for (int j = (p + i) & 1; j <= LJ+LC; j += 2){
                block_reduce_NoSync(ri_ur * radj[j], prad + i*LJC1 + j);
            }
        }
        ur_tmp *= root;
    }
    __syncthreads();

    constexpr int nfi = (LI+1) * (LI+2) / 2;
    constexpr int nfj = (LJ+1) * (LJ+2) / 2;
    const double fac = 16.0 * M_PI * M_PI * _common_fac[LI] * _common_fac[LJ];

    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        gctr[ij] = 0.0;
    }
    __syncthreads();

    // (k+l)pq,kimp,ljmq->ij
    for (int m = 0; m < LCC1; m++){
        type2_ang<LI, LC>(angi, rca, omegai+m);
        type2_ang<LJ, LC>(angj, rcb, omegaj+m);
        __syncthreads();
        for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
            const int i = ij%nfi;
            const int j = ij/nfi;
            double s = 0;
            for (int k = 0; k <= LI; k++){
            for (int l = 0; l <= LJ; l++){
                const double *pangi = angi + k*nfi*LIC1 + i*LIC1;
                const double *pangj = angj + l*nfj*LJC1 + j*LJC1;
                const double *prad = rad_all + (k+l)*LIC1*LJC1;
                for (int p = (LC + k) & 1; p <= k + LC; p += 2){
                    const double aip = pangi[p];
                    const double *prad_p = prad + p*LJC1;
                    for (int q = (LC + l) & 1; q <= l + LC; q += 2){
                        s += prad_p[q] * aip * pangj[q];
                    }
                }
            }}
            gctr[ij] += fac*s;
        }
        __syncthreads();
    }
}

__device__
void type2_cart_kernel_precomp(
    double *gctr,
    const int LI, const int LJ, const int LC,
    double *rca, double *rcb,
    const double *radi, const double *radj, const double ur,
    const bool compute_omegaj = true)
{
    extern __shared__ double smem[];

    const int LIC1 = LI+LC+1;
    const int LJC1 = LJ+LC+1;
    double* rad_all = smem;
    double* omegai = rad_all + (LI+LJ+1) * LIC1 * LJC1;
    double* omegaj = omegai + (LI+LC+2)/2 * (LI+1)*(LI+2)*(LI+3)/6 * (2*LC+1);

    type2_facs_omega(omegai, LI, LC, rca);
    if (compute_omegaj){
        type2_facs_omega(omegaj, LJ, LC, rcb);
    }
    set_shared_memory(rad_all, (LI+LJ+1)*LIC1*LJC1);

    double root = 0.0;
    if (threadIdx.x < NGAUSS){
        root = r128[threadIdx.x];
    }
    double ur_tmp = ur;
    for (int p = 0; p <= LI+LJ; p++){
        double *prad = rad_all + p*LIC1*LJC1;
        for (int i = 0; i <= LI+LC; i++){
            const double ri_ur = radi[i] * ur_tmp;
            for (int j = (p + i) & 1; j <= LJ+LC; j += 2){
                block_reduce_NoSync(ri_ur * radj[j], prad + i*LJC1 + j);
            }
        }
        ur_tmp *= root;
    }
    __syncthreads();

    const int nfi = (LI+1) * (LI+2) / 2;
    const int nfj = (LJ+1) * (LJ+2) / 2;
    double* angi = omegaj + (LJ+LC+2)/2 * (LJ+1)*(LJ+2)*(LJ+3)/6 * (2*LC+1);
    double* angj = angi + (LI+1)*nfi*LIC1;

    const double fac = 16.0 * M_PI * M_PI * _common_fac[LI] * _common_fac[LJ];

    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        gctr[ij] = 0.0;
    }
    __syncthreads();

    // (k+l)pq,kimp,ljmq->ij
    for (int m = 0; m < 2*LC+1; m++){
        type2_ang(angi, LI, LC, rca, omegai+m);
        type2_ang(angj, LJ, LC, rcb, omegaj+m);
        __syncthreads();
        for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
            const int i = ij%nfi;
            const int j = ij/nfi;
            double s = 0;
            for (int k = 0; k <= LI; k++){
            for (int l = 0; l <= LJ; l++){
                const double *pangi = angi + k*nfi*LIC1 + i*LIC1;
                const double *pangj = angj + l*nfj*LJC1 + j*LJC1;
                const double *prad = rad_all + (k+l)*LIC1*LJC1;
                for (int p = (LC + k) & 1; p <= k + LC; p += 2){
                    const double aip = pangi[p];
                    const double *prad_p = prad + p*LJC1;
                    for (int q = (LC + l) & 1; q <= l + LC; q += 2){
                        s += prad_p[q] * aip * pangj[q];
                    }
                }
            }}
            gctr[ij] += fac*s;
        }
        __syncthreads();
    }
}

// Unrolling case
template <int LI, int LJ, int LC> __global__
void type2_cart_ip1(double *gctr,
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
    const int ioff = ao_loc[ish];
    const int joff = ao_loc[jsh];
    const int ecp_id = ecpbas[ECP_ATOM_ID+ecploc[ksh]*BAS_SLOTS];
    gctr += 3*ecp_id*nao*nao + ioff*nao + joff;

    const double *ri = env + atm[PTR_COORD+bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS];
    const double *rj = env + atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];
    const int atm_id = ecpbas[ATOM_OF+ecploc[ksh]*BAS_SLOTS];
    const double *rc = env + atm[PTR_COORD+atm_id*ATM_SLOTS];

    double rca[3], rcb[3];
    rca[0] = rc[0] - ri[0];
    rca[1] = rc[1] - ri[1];
    rca[2] = rc[2] - ri[2];
    rcb[0] = rc[0] - rj[0];
    rcb[1] = rc[1] - rj[1];
    rcb[2] = rc[2] - rj[2];

    const int npi = bas[NPRIM_OF+ish*BAS_SLOTS];
    const double *ai = env + bas[PTR_EXP+ish*BAS_SLOTS];
    const double *ci = env + bas[PTR_COEFF+ish*BAS_SLOTS];
    const double dca = norm3d(rca[0], rca[1], rca[2]);

    const int npj = bas[NPRIM_OF+jsh*BAS_SLOTS];
    const double *aj = env + bas[PTR_EXP+jsh*BAS_SLOTS];
    const double *cj = env + bas[PTR_COEFF+jsh*BAS_SLOTS];
    const double dcb = norm3d(rcb[0], rcb[1], rcb[2]);

    constexpr int LIC1_MAX = (LI+1)+LC+1;
    constexpr int LIC0_MAX = (LI > 0) ? (LI-1+LC) : 0;
    constexpr int LJC1 = LJ+LC+1;

    double radi_0[LIC0_MAX+1];
    double radi_1[LIC1_MAX];
    if constexpr (LI > 0){
        type2_facs_rad_01(radi_0, radi_1, LI-1+LC, LI+1+LC, npi, dca, ci, ai);
    } else {
        type2_facs_rad<1>(radi_1, LI+1+LC, npi, dca, ci, ai);
    }
    double radj[LJC1];
    type2_facs_rad<0>(radj, LJ+LC, npj, dcb, cj, aj);

    double ur = 0.0;
    for (int kbas = ecploc[ksh]; kbas < ecploc[ksh+1]; kbas++){
        ur += rad_part(kbas, ecpbas, env);
    }

    constexpr int nfi = (LI+1) * (LI+2) / 2;
    constexpr int nfj = (LJ+1) * (LJ+2) / 2;
    constexpr int nfi1 = (LI+2) * (LI+3) / 2;
    constexpr int LI1_MAX = LI + 2;
    constexpr int LJ1 = LJ + 1;
    constexpr int LCC1 = 2*LC + 1;
    constexpr int BLKI_MAX = (LIC1_MAX+1)/2 * LCC1;
    constexpr int BLKJ = (LJC1+1)/2 * LCC1;

    __shared__ double gctr_smem[nfi*nfj*3];
    __shared__ double buf[nfi1*nfj];
    __shared__ double omegai[LI1_MAX*(LI1_MAX+1)*(LI1_MAX+2)/6 * BLKI_MAX];
    __shared__ double omegaj[LJ1*(LJ1+1)*(LJ1+2)/6 * BLKJ];
    __shared__ double rad_all[(LI+1+LJ+1)*LIC1_MAX*LJC1];
    __shared__ double angi[LI1_MAX*nfi1*LIC1_MAX];
    __shared__ double angj[LJ1*nfj*LJC1];

    for (int ij = threadIdx.x; ij < nfi*nfj*3; ij+=blockDim.x){
        gctr_smem[ij] = 0.0;
    }
    type2_facs_omega<LJ, LC>(omegaj, rcb);
    __syncthreads();

    type2_cart_unrolled_sub<LI+1, LJ, LC>(
        buf, rca, rcb, radi_1, radj, ur,
        omegai, omegaj, rad_all, angi, angj);
    _li_down(gctr_smem, buf, LI, LJ);
    __syncthreads();

    if constexpr (LI > 0){
        type2_cart_unrolled_sub<LI-1, LJ, LC>(
            buf, rca, rcb, radi_0, radj, ur,
            omegai, omegaj, rad_all, angi, angj);
        _li_up(gctr_smem, buf, LI, LJ);
        __syncthreads();
    }

    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        const int i = ij%nfi;
        const int j = ij/nfi;
        double *gx = gctr;
        double *gy = gctr +   nao*nao;
        double *gz = gctr + 2*nao*nao;
        atomicAdd(gx + i*nao + j, gctr_smem[ij]);
        atomicAdd(gy + i*nao + j, gctr_smem[ij+nfi*nfj]);
        atomicAdd(gz + i*nao + j, gctr_smem[ij+2*nfi*nfj]);
    }
    return;
}


__global__
void type2_cart_ip1_general(double *gctr,
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
    const int ioff = ao_loc[ish];
    const int joff = ao_loc[jsh];
    const int ecp_id = ecpbas[ECP_ATOM_ID+ecploc[ksh]*BAS_SLOTS];
    gctr += 3*ecp_id*nao*nao + ioff*nao + joff;

    const double *ri = env + atm[PTR_COORD+bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS];
    const double *rj = env + atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];
    const int atm_id = ecpbas[ATOM_OF+ecploc[ksh]*BAS_SLOTS];
    const double *rc = env + atm[PTR_COORD+atm_id*ATM_SLOTS];

    double rca[3], rcb[3];
    rca[0] = rc[0] - ri[0];
    rca[1] = rc[1] - ri[1];
    rca[2] = rc[2] - ri[2];
    rcb[0] = rc[0] - rj[0];
    rcb[1] = rc[1] - rj[1];
    rcb[2] = rc[2] - rj[2];

    const int npi = bas[NPRIM_OF+ish*BAS_SLOTS];
    const int npj = bas[NPRIM_OF+jsh*BAS_SLOTS];
    const double *ai = env + bas[PTR_EXP+ish*BAS_SLOTS];
    const double *aj = env + bas[PTR_EXP+jsh*BAS_SLOTS];
    const double *ci = env + bas[PTR_COEFF+ish*BAS_SLOTS];
    const double *cj = env + bas[PTR_COEFF+jsh*BAS_SLOTS];

    const double dca = norm3d(rca[0], rca[1], rca[2]);
    const double dcb = norm3d(rcb[0], rcb[1], rcb[2]);

    double radi_0[AO_LMAX+ECP_LMAX+1];
    double radi_1[AO_LMAX+ECP_LMAX+2];
    if (LI > 0){
        type2_facs_rad_01(radi_0, radi_1, LI-1+LC, LI+1+LC, npi, dca, ci, ai);
    } else {
        type2_facs_rad<1>(radi_1, LI+1+LC, npi, dca, ci, ai);
    }
    double radj[AO_LMAX+ECP_LMAX+1];
    type2_facs_rad<0>(radj, LJ+LC, npj, dcb, cj, aj);

    double ur = 0.0;
    for (int kbas = ecploc[ksh]; kbas < ecploc[ksh+1]; kbas++){
        ur += rad_part(kbas, ecpbas, env);
    }

    __shared__ double gctr_smem[NF_MAX*NF_MAX*3];
    for (int ij = threadIdx.x; ij < NF_MAX*NF_MAX*3; ij+=blockDim.x){
        gctr_smem[ij] = 0.0;
    }
    __syncthreads();

    constexpr int NFI_MAX = (AO_LMAX+2)*(AO_LMAX+3)/2;
    constexpr int NFJ_MAX = (AO_LMAX+1)*(AO_LMAX+2)/2;
    __shared__ double buf[NFI_MAX*NFJ_MAX];

    type2_cart_kernel_precomp(buf, LI+1, LJ, LC, rca, rcb, radi_1, radj, ur, true);
    _li_down(gctr_smem, buf, LI, LJ);
    __syncthreads();
    if (LI > 0){
        type2_cart_kernel_precomp(buf, LI-1, LJ, LC, rca, rcb, radi_0, radj, ur, true);
        _li_up(gctr_smem, buf, LI, LJ);
        __syncthreads();
    }

    const int nfi = (LI+1) * (LI+2) / 2;
    const int nfj = (LJ+1) * (LJ+2) / 2;
    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        const int i = ij%nfi;
        const int j = ij/nfi;
        double *gx = gctr;
        double *gy = gctr +   nao*nao;
        double *gz = gctr + 2*nao*nao;
        atomicAdd(gx + i*nao + j, gctr_smem[ij]);
        atomicAdd(gy + i*nao + j, gctr_smem[ij+nfi*nfj]);
        atomicAdd(gz + i*nao + j, gctr_smem[ij+2*nfi*nfj]);
    }
    return;
}

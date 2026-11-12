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


__global__
void type2_cart_ipipv(double *gctr,
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
    gctr += ioff*nao + joff + 9*ecp_id*nao*nao;

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
    double radi_2[AO_LMAX+ECP_LMAX+3];
    type2_facs_rad_012(
        radi_0, radi_1, radi_2,
        (LI > 1) ? (LI-2+LC) : 0, LI+LC, LI+2+LC,
        npi, dca, ci, ai);
    double radj[AO_LMAX+ECP_LMAX+1];
    type2_facs_rad<0>(radj, LJ+LC, npj, dcb, cj, aj);

    double ur = 0.0;
    for (int kbas = ecploc[ksh]; kbas < ecploc[ksh+1]; kbas++){
        ur += rad_part(kbas, ecpbas, env);
    }

    constexpr int nfi2_max = (AO_LMAX+3)*(AO_LMAX+4)/2;
    constexpr int nfj_max = (AO_LMAX+1)*(AO_LMAX+2)/2;
    __shared__ double buf1[nfi2_max*nfj_max];
    type2_cart_kernel_precomp(buf1, LI+2, LJ, LC, rca, rcb, radi_2, radj, ur, true);

    constexpr int nfi1_max = (AO_LMAX+2)*(AO_LMAX+3)/2;
    extern __shared__ double smem[];
    double *buf = smem;
    set_shared_memory(buf, 3*nfi1_max*nfj_max);
    _li_down(buf, buf1, LI+1, LJ);
    __syncthreads();
    _li_down_and_write(gctr, buf, LI, LJ, nao);
    __syncthreads();

    type2_cart_kernel_precomp(buf1, LI, LJ, LC, rca, rcb, radi_1, radj, ur, true);
    set_shared_memory(buf, 3*nfi1_max*nfj_max);
    _li_up(buf, buf1, LI+1, LJ);
    __syncthreads();
    _li_down_and_write(gctr, buf, LI, LJ, nao);
    __syncthreads();

    if (LI > 0){
        set_shared_memory(buf, 3*nfi1_max*nfj_max);
        _li_down(buf, buf1, LI-1, LJ);
        __syncthreads();
        _li_up_and_write(gctr, buf, LI, LJ, nao);
        __syncthreads();
        if (LI > 1){
            type2_cart_kernel_precomp(buf1, LI-2, LJ, LC, rca, rcb, radi_0, radj, ur, true);
            set_shared_memory(buf, 3*nfi1_max*nfj_max);
            _li_up(buf, buf1, LI-1, LJ);
            __syncthreads();
            _li_up_and_write(gctr, buf, LI, LJ, nao);
        }
    }
    return;
}

__global__
void type2_cart_ipvip(double *gctr,
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
    gctr += ioff*nao + joff + 9*ecp_id*nao*nao;

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

    double radj_0[AO_LMAX+ECP_LMAX+1];
    double radj_1[AO_LMAX+ECP_LMAX+2];
    if (LJ > 0){
        type2_facs_rad_01(radj_0, radj_1, LJ-1+LC, LJ+1+LC, npj, dcb, cj, aj);
    } else {
        type2_facs_rad<1>(radj_1, LJ+1+LC, npj, dcb, cj, aj);
    }

    double ur = 0.0;
    for (int kbas = ecploc[ksh]; kbas < ecploc[ksh+1]; kbas++){
        ur += rad_part(kbas, ecpbas, env);
    }

    constexpr int nfi1_max = (AO_LMAX+2)*(AO_LMAX+3)/2;
    constexpr int nfj1_max = (AO_LMAX+2)*(AO_LMAX+3)/2;
    __shared__ double buf1[nfi1_max*nfj1_max];
    type2_cart_kernel_precomp(buf1, LI+1, LJ+1, LC, rca, rcb, radi_1, radj_1, ur, true);

    constexpr int nfi_max = (AO_LMAX+1)*(AO_LMAX+2)/2;
    extern __shared__ double smem[];
    double *buf = smem;
    set_shared_memory(buf, 3*nfi_max*nfj1_max);
    _li_down(buf, buf1, LI, LJ+1);
    __syncthreads();
    _lj_down_and_write(gctr, buf, LI, LJ, nao);
    __syncthreads();
    if (LI > 0){
        type2_cart_kernel_precomp(buf1, LI-1, LJ+1, LC, rca, rcb, radi_0, radj_1, ur, true);
        set_shared_memory(buf, 3*nfi_max*nfj1_max);
        _li_up(buf, buf1, LI, LJ+1);
        __syncthreads();
        _lj_down_and_write(gctr, buf, LI, LJ, nao);
        __syncthreads();
    }

    if (LJ > 0){
        type2_cart_kernel_precomp(buf1, LI+1, LJ-1, LC, rca, rcb, radi_1, radj_0, ur, true);
        set_shared_memory(buf, 3*nfi_max*nfj1_max);
        _li_down(buf, buf1, LI, LJ-1);
        __syncthreads();
        _lj_up_and_write(gctr, buf, LI, LJ, nao);
        __syncthreads();
        if (LI > 0){
            type2_cart_kernel_precomp(buf1, LI-1, LJ-1, LC, rca, rcb, radi_0, radj_0, ur, true);
            set_shared_memory(buf, 3*nfi_max*nfj1_max);
            _li_up(buf, buf1, LI, LJ-1);
            __syncthreads();
            _lj_up_and_write(gctr, buf, LI, LJ, nao);
            __syncthreads();
        }
    }
    return;
}

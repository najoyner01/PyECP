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

#include <stdio.h>
#include "ecp.h"
#include "bessel.cu"
#include "cart2sph.cu"
#include "gauss_chebyshev.cu"
#include "common.cu"
#include "type1_ang_nuc.cu"
#include "type2_ang_nuc.cu"
#include "ecp_type1.cu"
#include "ecp_type2.cu"
#include "ecp_type1_ip.cu"
#include "ecp_type2_ip.cu"
#include "ecp_type1_ipip.cu"
#include "ecp_type2_ipip.cu"
#include "so_ang_matrix.cu"
#include "ecp_so.cu"

static inline cudaError_t ensure_max_dynamic_smem(
    const void *func, int *cached_bytes, int *cached_dev, int req_bytes)
{
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev != *cached_dev) {
        *cached_dev = dev;
        *cached_bytes = 0;
    }
    if (req_bytes > *cached_bytes) {
        cudaError_t err = cudaFuncSetAttribute(
            func, cudaFuncAttributeMaxDynamicSharedMemorySize, req_bytes);
        if (err != cudaSuccess) {
            return err;
        }
        *cached_bytes = req_bytes;
    }
    return cudaSuccess;
}

extern "C" {
int ECP_cart(double *gctr,
            const int *ao_loc, const int nao,
            const int *tasks, const int ntasks,
            const int *ecpbas, const int *ecploc,
            const int *atm, const int *bas, const double *env,
            const int li, const int lj, const int lc){
    // one task per thread block
    // Screening can empty a task group; a 0-block launch is not a legal
    // configuration, so treat it as a no-op.
    if (ntasks == 0){
        return 0;
    }
    dim3 threads(THREADS);
    dim3 blocks(ntasks);
    if (lc >= 0){
        int task_type = li * 100 + lj * 10 + lc;
        switch (task_type) {
        #define LAUNCH_T2(LI, LJ, LC) \
            case ((LI)*100 + (LJ)*10 + (LC)): \
                type2_cart<LI, LJ, LC><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); \
                break
        #define LAUNCH_T2_LC(LI, LJ) \
            LAUNCH_T2(LI, LJ, 0); LAUNCH_T2(LI, LJ, 1); LAUNCH_T2(LI, LJ, 2); LAUNCH_T2(LI, LJ, 3)
        #define LAUNCH_T2_LJ(LI) \
            LAUNCH_T2_LC(LI, 0); LAUNCH_T2_LC(LI, 1); LAUNCH_T2_LC(LI, 2); LAUNCH_T2_LC(LI, 3)
        LAUNCH_T2_LJ(0);
        LAUNCH_T2_LJ(1);
        LAUNCH_T2_LJ(2);
        LAUNCH_T2_LJ(3);
        #undef LAUNCH_T2_LJ
        #undef LAUNCH_T2_LC
        #undef LAUNCH_T2

        // General kernel
        default: {
            const int li1 = li+1;
            const int lj1 = lj+1;
            const int nfi = (li+1)*(li+2)/2;
            const int nfj = (lj+1)*(lj+2)/2;
            const int lic1 = li+lc+1;
            const int ljc1 = lj+lc+1;
            const int lcc1 = 2*lc+1;
            const int blki = (lic1+1)/2 * lcc1;
            const int blkj = (ljc1+1)/2 * lcc1;

            int smem_size0 = (li+lj+1) * lic1 * ljc1; // rad_all
            int smem_size1 = li1*(li1+1)*(li1+2)/6 * blki; // omegai
            int smem_size2 = lj1*(lj1+1)*(lj1+2)/6 * blkj; // omegaj
            int smem_size3 = li1*nfi*lic1; // angi
            int smem_size4 = lj1*nfj*ljc1; // angj
            int smem_size = smem_size0 + smem_size1 + smem_size2 + smem_size3 + smem_size4;

            static int max_smem_t2 = 0, dev_t2 = -1;
            void (*t2_gen)(double*, const int, const int, const int, const int*, const int, const int*, const int, const int*, const int*, const int*, const int*, const double*) = type2_cart;
            cudaError_t attr_err = ensure_max_dynamic_smem(
                (const void*)t2_gen, &max_smem_t2, &dev_t2, smem_size*sizeof(double));
            if (attr_err != cudaSuccess) {
                fprintf(stderr, "CUDA Error in cudaFuncSetAttribute %s (li,lj,lc = %d,%d,%d): %s\n", __func__, li,lj,lc, cudaGetErrorString(attr_err));
                return 1;
            }

            type2_cart<<<blocks, threads, smem_size*sizeof(double)>>>(
                gctr,
                li, lj, lc,
                ao_loc, nao,
                tasks, ntasks,
                ecpbas, ecploc,
                atm, bas, env);
        }}
    } else {
        int task_type = li * 10 + lj;
        switch (task_type)
        {
        #define LAUNCH_T1(LI, LJ) \
            case ((LI)*10 + (LJ)): \
                type1_cart<LI, LJ><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); \
                break
        #define LAUNCH_T1_LJ(LI) \
            LAUNCH_T1(LI, 0); LAUNCH_T1(LI, 1); LAUNCH_T1(LI, 2); LAUNCH_T1(LI, 3); LAUNCH_T1(LI, 4)
        LAUNCH_T1_LJ(0);
        LAUNCH_T1_LJ(1);
        LAUNCH_T1_LJ(2);
        LAUNCH_T1_LJ(3);
        LAUNCH_T1_LJ(4);
        #undef LAUNCH_T1_LJ
        #undef LAUNCH_T1
        default: {
            const int lij1 = li+lj+1;
            const int lij3 = lij1*lij1*lij1;

            int smem_size = 0;
            smem_size += lij3;      // rad_ang
            smem_size += lij1*lij1; // rad_all
            type1_cart<<<blocks, threads, smem_size*sizeof(double)>>>(
                gctr, li, lj,
                ao_loc, nao,
                tasks, ntasks,
                ecpbas, ecploc,
                atm, bas, env);
        }
        }
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s (li,lj,lc = %d,%d,%d): %s\n", __func__, li,lj,lc, cudaGetErrorString(err));
        return 1;
    }
    return 0;
    }

int ECP_ip_cart(double *gctr,
            const int *ao_loc, const int nao,
            const int *tasks, const int ntasks,
            const int *ecpbas, const int *ecploc,
            const int *atm, const int *bas, const double *env,
            const int li, const int lj, const int lc){
    // one task per thread block
    // Screening can empty a task group; a 0-block launch is not a legal
    // configuration, so treat it as a no-op.
    if (ntasks == 0){
        return 0;
    }
    dim3 threads(THREADS);
    dim3 blocks(ntasks);
    if (lc < 0){
        int task_type = li * 10 + lj;
        switch (task_type) {
        #define LAUNCH_T1_IP1(LI, LJ) \
            case ((LI)*10 + (LJ)): \
                type1_cart_ip1<LI, LJ><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); \
                break
        #define LAUNCH_T1_IP1_LJ(LI) \
            LAUNCH_T1_IP1(LI, 0); LAUNCH_T1_IP1(LI, 1); LAUNCH_T1_IP1(LI, 2); LAUNCH_T1_IP1(LI, 3); LAUNCH_T1_IP1(LI, 4)
        LAUNCH_T1_IP1_LJ(0);
        LAUNCH_T1_IP1_LJ(1);
        LAUNCH_T1_IP1_LJ(2);
        LAUNCH_T1_IP1_LJ(3);
        LAUNCH_T1_IP1_LJ(4);
        #undef LAUNCH_T1_IP1_LJ
        #undef LAUNCH_T1_IP1
        default: {
            const int lij1 = li+lj+2;
            const int lij3 = lij1*lij1*lij1;

            int smem_size = 0;
            smem_size += lij3;      // rad_ang
            smem_size += lij1*lij1; // rad_all
            type1_cart_ip1_general<<<blocks, threads, smem_size*sizeof(double)>>>(
                gctr, li, lj,
                ao_loc, nao,
                tasks, ntasks,
                ecpbas, ecploc,
                atm, bas, env);
        }}
    } else {
        int task_type = li * 100 + lj * 10 + lc;
        switch (task_type) {
        #define LAUNCH_T2_IP1(LI, LJ, LC) \
            case ((LI)*100 + (LJ)*10 + (LC)): \
                type2_cart_ip1<LI, LJ, LC><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); \
                break
        #define LAUNCH_T2_IP1_LC(LI, LJ) \
            LAUNCH_T2_IP1(LI, LJ, 0); LAUNCH_T2_IP1(LI, LJ, 1); LAUNCH_T2_IP1(LI, LJ, 2); LAUNCH_T2_IP1(LI, LJ, 3)
        #define LAUNCH_T2_IP1_LJ(LI) \
            LAUNCH_T2_IP1_LC(LI, 0); LAUNCH_T2_IP1_LC(LI, 1); LAUNCH_T2_IP1_LC(LI, 2); LAUNCH_T2_IP1_LC(LI, 3)
        LAUNCH_T2_IP1_LJ(0);
        LAUNCH_T2_IP1_LJ(1);
        LAUNCH_T2_IP1_LJ(2);
        LAUNCH_T2_IP1_LJ(3);
        #undef LAUNCH_T2_IP1_LJ
        #undef LAUNCH_T2_IP1_LC
        #undef LAUNCH_T2_IP1

        // General kernel
        default: {
            const int li1 = li+2;
            const int lj1 = lj+1;
            const int lij1 = (li+1)+lj+1;
            const int nfi = (li+2)*(li+3)/2;
            const int nfj = (lj+1)*(lj+2)/2;
            const int lic1 = li1+lc+1;
            const int ljc1 = lj1+lc+1;
            const int lcc1 = 2*lc+1;
            const int blki = (lic1+1)/2 * lcc1;
            const int blkj = (ljc1+1)/2 * lcc1;

            int smem_size0 = lij1 * lic1 * ljc1; // rad_all
            int smem_size1 = li1*(li1+1)*(li1+2)/6 * blki; // omegai
            int smem_size2 = lj1*(lj1+1)*(lj1+2)/6 * blkj; // omegaj
            int smem_size3 = li1*lic1*nfi; // angi
            int smem_size4 = lj1*ljc1*nfj; // angj

            int dynamic_smem_size = smem_size0 + smem_size1 + smem_size2 + smem_size3 + smem_size4;
            static int max_smem_ip1 = 0, dev_ip1 = -1;
            cudaError_t err = ensure_max_dynamic_smem(
                (const void*)type2_cart_ip1_general,
                &max_smem_ip1, &dev_ip1,
                dynamic_smem_size*sizeof(double));

            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA Error in cudaFuncSetAttribute %s (li,lj,lc = %d,%d,%d): %s\n", __func__, li,lj,lc, cudaGetErrorString(err));
                return 1;
            }

            type2_cart_ip1_general<<<blocks, threads, dynamic_smem_size*sizeof(double)>>>(
                gctr, li, lj, lc,
                ao_loc, nao,
                tasks, ntasks,
                ecpbas, ecploc,
                atm, bas, env);
        }}
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s (li,lj,lc = %d,%d,%d): %s\n", __func__, li,lj,lc, cudaGetErrorString(err));
        return 1;
    }
    return 0;
    }

int ECP_ipipv_cart(double *gctr,
            const int *ao_loc, const int nao,
            const int *tasks, const int ntasks,
            const int *ecpbas, const int *ecploc,
            const int *atm, const int *bas, const double *env,
            const int li, const int lj, const int lc){
    // one task per thread block
    // Screening can empty a task group; a 0-block launch is not a legal
    // configuration, so treat it as a no-op.
    if (ntasks == 0){
        return 0;
    }
    dim3 threads(THREADS);
    dim3 blocks(ntasks);
    
    if (lc < 0){
        const int lij1 = li+lj+3; //
        const int lij3 = lij1*lij1*lij1;

        int smem_size = 0;
        smem_size += lij3;      // rad_ang
        smem_size += lij1*lij1; // rad_all
        type1_cart_ipipv<<<blocks, threads, smem_size*sizeof(double)>>>(
            gctr, li, lj,
            ao_loc, nao,
            tasks, ntasks,
            ecpbas, ecploc,
            atm, bas, env);

    } else {
        const int li1 = li+3;
        const int lj1 = lj+1;
        const int lij1 = li1+lj;
        const int nfi =  li1*(li1+1)/2;
        const int nfj = lj1*(lj1+1)/2;
        const int lic1 = li1+lc;
        const int ljc1 = lj1+lc;
        const int lcc1 = 2*lc+1;
        const int blki = (lic1+1)/2 * lcc1;
        const int blkj = (ljc1+1)/2 * lcc1;

        int smem_size0 = lij1 * lic1 * ljc1; // rad_all
        int smem_size1 = li1*(li1+1)*(li1+2)/6 * blki; // omegai
        int smem_size2 = lj1*(lj1+1)*(lj1+2)/6 * blkj; // omegaj
        int smem_size3 = li1*lic1*nfi; // angi
        int smem_size4 = lj1*ljc1*nfj; // angj

        int NF1_MAX = (AO_LMAX+2)*(AO_LMAX+3)/2;
        int NF0_MAX = (AO_LMAX+1)*(AO_LMAX+2)/2;
        int dynamic_smem_size = smem_size0 + smem_size1 + smem_size2 + smem_size3 + smem_size4;
        dynamic_smem_size = max(dynamic_smem_size, 3*NF1_MAX*NF0_MAX);

        static int max_smem_ipipv = 0, dev_ipipv = -1;
        cudaError_t err = ensure_max_dynamic_smem(
            (const void*)type2_cart_ipipv,
            &max_smem_ipipv, &dev_ipipv,
            dynamic_smem_size*sizeof(double));
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error in cudaFuncSetAttribute %s (li,lj,lc = %d,%d,%d): %s\n", __func__, li,lj,lc, cudaGetErrorString(err));
            return 1;
        }

        type2_cart_ipipv<<<blocks, threads, dynamic_smem_size*sizeof(double)>>>(
            gctr, li, lj, lc,
            ao_loc, nao,
            tasks, ntasks,
            ecpbas, ecploc,
            atm, bas, env);

    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s (li,lj,lc = %d,%d,%d): %s\n", __func__, li,lj,lc, cudaGetErrorString(err));
        return 1;
    }
    return 0;
    }

int ECP_ipvip_cart(double *gctr,
            const int *ao_loc, const int nao,
            const int *tasks, const int ntasks,
            const int *ecpbas, const int *ecploc,
            const int *atm, const int *bas, const double *env,
            const int li, const int lj, const int lc){
    // one task per thread block
    // Screening can empty a task group; a 0-block launch is not a legal
    // configuration, so treat it as a no-op.
    if (ntasks == 0){
        return 0;
    }
    dim3 threads(THREADS);
    dim3 blocks(ntasks);

    if (lc < 0){
        const int lij1 = li+lj+3; //
        const int lij3 = lij1*lij1*lij1;

        int smem_size = 0;
        smem_size += lij3;      // rad_ang
        smem_size += lij1*lij1; // rad_all
        type1_cart_ipvip<<<blocks, threads, smem_size*sizeof(double)>>>(
            gctr, li, lj,
            ao_loc, nao,
            tasks, ntasks,
            ecpbas, ecploc,
            atm, bas, env);
    } else {
        const int li1 = li+2;
        const int lj1 = lj+2;
        const int lij1 = li1+lj1-1;
        const int nfi = li1*(li1+1)/2;
        const int nfj = lj1*(lj1+1)/2;
        const int lic1 = li1+lc;
        const int ljc1 = lj1+lc;
        const int lcc1 = 2*lc+1;
        const int blki = (lic1+1)/2 * lcc1;
        const int blkj = (ljc1+1)/2 * lcc1;

        int smem_size0 = lij1 * lic1 * ljc1; // rad_all
        int smem_size1 = li1*(li1+1)*(li1+2)/6 * blki; // omegai
        int smem_size2 = lj1*(lj1+1)*(lj1+2)/6 * blkj; // omegaj
        int smem_size3 = li1*lic1*nfi; // angi
        int smem_size4 = lj1*ljc1*nfj; // angj

        int NF1_MAX = (AO_LMAX+2)*(AO_LMAX+3)/2;
        int NF0_MAX = (AO_LMAX+1)*(AO_LMAX+2)/2;
        int dynamic_smem_size = smem_size0 + smem_size1 + smem_size2 + smem_size3 + smem_size4;
        dynamic_smem_size = max(dynamic_smem_size, 3*NF0_MAX*NF1_MAX);

        static int max_smem_ipvip = 0, dev_ipvip = -1;
        cudaError_t err = ensure_max_dynamic_smem(
            (const void*)type2_cart_ipvip,
            &max_smem_ipvip, &dev_ipvip,
            dynamic_smem_size*sizeof(double));
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error in cudaFuncSetAttribute %s (li,lj,lc = %d,%d,%d): %s\n", __func__, li,lj,lc, cudaGetErrorString(err));
            return 1;
        }

        type2_cart_ipvip<<<blocks, threads, dynamic_smem_size*sizeof(double)>>>(
            gctr, li, lj, lc,
            ao_loc, nao,
            tasks, ntasks,
            ecpbas, ecploc,
            atm, bas, env);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s (li,lj,lc = %d,%d,%d): %s\n", __func__, li,lj,lc, cudaGetErrorString(err));
        return 1;
    }
    return 0;
    }

int ECP_so_cart(double *gctr,
            const int *ao_loc, const int nao,
            const int *tasks, const int ntasks,
            const int *ecpbas, const int *ecploc,
            const int *atm, const int *bas, const double *env,
            const int li, const int lj, const int lc){
    // Spin-orbit ECP: gctr has layout [3, nao, nao] (component, bra, ket).
    // The host passes only SO_TYPE_OF==1 projectors, with the `ul` term already
    // rewritten to max_l(atom)+1, so lc >= 0 always here.
    // Screening can empty a task group; a 0-block launch is not a legal
    // configuration, so treat it as a no-op.
    if (ntasks == 0 || lc == 0){
        return 0;
    }
    dim3 threads(THREADS);
    dim3 blocks(ntasks);

    if (lc < 0 || lc > ECP_LMAX){
        fprintf(stderr, "ECP_so_cart: unsupported projector lc=%d (need 0..%d)\n",
                lc, ECP_LMAX);
        return 1;
    }

    int task_type = li * 100 + lj * 10 + lc;
    switch (task_type) {
    #define LAUNCH_SO(LI, LJ, LC) \
        case ((LI)*100 + (LJ)*10 + (LC)): \
            so_cart<LI, LJ, LC><<<blocks, threads>>>(gctr, ao_loc, nao, tasks, ntasks, ecpbas, ecploc, atm, bas, env); \
            break
    #define LAUNCH_SO_LC(LI, LJ) \
        LAUNCH_SO(LI, LJ, 1); LAUNCH_SO(LI, LJ, 2); LAUNCH_SO(LI, LJ, 3)
    #define LAUNCH_SO_LJ(LI) \
        LAUNCH_SO_LC(LI, 0); LAUNCH_SO_LC(LI, 1); LAUNCH_SO_LC(LI, 2)
    LAUNCH_SO_LJ(0);
    LAUNCH_SO_LJ(1);
    LAUNCH_SO_LJ(2);
    #undef LAUNCH_SO_LJ
    #undef LAUNCH_SO_LC
    #undef LAUNCH_SO

    default: {
        const int li1 = li+1;
        const int lj1 = lj+1;
        const int nfi = (li+1)*(li+2)/2;
        const int nfj = (lj+1)*(lj+2)/2;
        const int lic1 = li+lc+1;
        const int ljc1 = lj+lc+1;
        const int lcc1 = 2*lc+1;
        const int blki = (lic1+1)/2 * lcc1;
        const int blkj = (ljc1+1)/2 * lcc1;

        int smem_size0 = (li+lj+1) * lic1 * ljc1;         // rad_all
        int smem_size1 = li1*(li1+1)*(li1+2)/6 * blki;    // omegai
        int smem_size2 = lj1*(lj1+1)*(lj1+2)/6 * blkj;    // omegaj (and omegaj_a)
        int smem_size3 = li1*nfi*lic1;                    // angi
        int smem_size4 = lj1*nfj*ljc1;                    // angj
        int smem_size = smem_size0 + smem_size1 + 2*smem_size2 + smem_size3 + smem_size4;

        static int max_smem_so = 0, dev_so = -1;
        void (*so_gen)(double*, const int, const int, const int, const int*, const int, const int*, const int, const int*, const int*, const int*, const int*, const double*) = so_cart;
        cudaError_t attr_err = ensure_max_dynamic_smem(
            (const void*)so_gen, &max_smem_so, &dev_so, smem_size*sizeof(double));
        if (attr_err != cudaSuccess) {
            fprintf(stderr, "CUDA Error in cudaFuncSetAttribute %s (li,lj,lc = %d,%d,%d): %s\n",
                    __func__, li, lj, lc, cudaGetErrorString(attr_err));
            return 1;
        }

        so_cart<<<blocks, threads, smem_size*sizeof(double)>>>(
            gctr, li, lj, lc,
            ao_loc, nao,
            tasks, ntasks,
            ecpbas, ecploc,
            atm, bas, env);
    }}

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s (li,lj,lc = %d,%d,%d): %s\n", __func__, li,lj,lc, cudaGetErrorString(err));
        return 1;
    }
    return 0;
    }
}

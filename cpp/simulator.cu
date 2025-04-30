#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

__global__ void kernel(double* accelerations, double* masses, double* positions, int n){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
}

void calculate_acceleration(double* accelerations, double* masses, double* positions, int n){
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(1, 16);
    thrust::host_vector<double> h_vec(32 << 20);
    thrust::device_vector<double> d_vec = h_vec;
    double x = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());
}

double *calculate_acceleration(double *masses, double *positions, int n){
	double G = 6.67e-11;
    double *acceleration = new double[n*3];
    for(int i = 0; i<n*3; i++){
    	acceleration[i] = 0;
    }
    for(int i = 0; i<n; i++){
	    for(int j = i+1; j<n; j++){
        	double r = 0;
        	for(int k = 0; k<3; k++){
        		r += pow(positions[i*3+k]-positions[j*3+k], 2);
        	}
        	double magnitude = pow(r, -1.5);
        	for(int k = 0; k<3; k++){
        		double directional_magnitude = magnitude*(positions[j*3+k] - positions[i*3+k]);
        		acceleration[i*3+k] += masses[j]*directional_magnitude;
        		acceleration[j*3+k] -= masses[i]*directional_magnitude;
        	}
    	}
    }
    for(int i = 0; i<n*3; i++){
    	acceleration[i] *= G;
    }
    return acceleration;
}

/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

using data_type = double;

void calculate_acceleration(double* accelerations, double* masses, double* positions, int n) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    const std::vector<data_type> A = {1.0, 2.0, 3.0, 4.0};
    std::vector<data_type> B = {5.0, 6.0, 7.0, 8.0};
    const data_type alpha = 2.1;
    const int incx = 1;
    const int incy = 1;

    data_type *d_A = nullptr;
    data_type *d_B = nullptr;

    /* step 1: create cublas handle, bind a stream */
    cublasCreate(&cublasH);

    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cublasSetStream(cublasH, stream);

    /* step 2: copy data to device */
    cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * 3 * n);
    cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * 3 * n);

    cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * 3 * n, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * 3 * n, cudaMemcpyHostToDevice, stream);

    /* step 3: compute */
    cublasDaxpy(cublasH, A.size(), &alpha, d_A, incx, d_B, incy);

    /* step 4: copy data to host */
    cudaMemcpyAsync(B.data(), d_B, sizeof(data_type) * B.size(), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    /*
     *   B = | 7.10 10.20 13.30 16.40 |
     */

    printf("B\n");
    print_vector(B.size(), B.data());
    printf("=====\n");

    /* free resources */
    cudaFree(d_A);
    cudaFree(d_B);

    cublasDestroy(cublasH);

    cudaStreamDestroy(stream);

    cudaDeviceReset();

    return EXIT_SUCCESS;
}
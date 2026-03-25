#include "cuda_print.h"

#include <cuda_runtime.h>
#include <iostream>

using namespace std;

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            cerr << "CUDA Error: " << cudaGetErrorString(err)                \
                 << " at line " << __LINE__ << endl;                         \
            return false;                                                    \
        }                                                                    \
    } while (0)

__global__ void printGraphCSRKernel(const int *row_ptr,
                                    const int *col_ind,
                                    const int *weights,
                                    int numNodes,
                                    int numObjectives) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int u = 0; u < numNodes; ++u) {
            printf("Node %d -> ", u);
            for (int e = row_ptr[u]; e < row_ptr[u + 1]; ++e) {
                printf("(%d, w=[", col_ind[e]);
                for (int k = 0; k < numObjectives; ++k) {
                    printf("%d", weights[e * numObjectives + k]);
                    if (k + 1 < numObjectives) {
                        printf(", ");
                    }
                }
                printf("]) ");
            }
            printf("\n");
        }
    }
}

bool printCSRFromGPU(const CSRGraph &csr) {
    int *row_ptr = nullptr;
    int *col_ind = nullptr;
    int *weights = nullptr;

    CUDA_CHECK(cudaMallocManaged(&row_ptr, csr.row_ptr.size() * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&col_ind, csr.col_ind.size() * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&weights, csr.weights.size() * sizeof(int)));

    for (size_t i = 0; i < csr.row_ptr.size(); ++i) {
        row_ptr[i] = csr.row_ptr[i];
    }

    for (size_t i = 0; i < csr.col_ind.size(); ++i) {
        col_ind[i] = csr.col_ind[i];
    }

    for (size_t i = 0; i < csr.weights.size(); ++i) {
        weights[i] = csr.weights[i];
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    printGraphCSRKernel<<<1, 1>>>(row_ptr, col_ind, weights,
                                  csr.numNodes, csr.numObjectives);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(row_ptr);
    cudaFree(col_ind);
    cudaFree(weights);

    return true;
}
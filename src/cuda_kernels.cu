#include "../headers/cuda_kernels.cuh"
#include <cuda_runtime.h>
#include <climits>

/**
 * @file cuda_kernels.cu
 * @brief Implements CUDA kernel helpers for CSR graphs.
 */

__global__ void computeOutDegreesKernel(int n, const int* rowPtr, int* degrees) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;

    if (u < n) {
        degrees[u] = rowPtr[u + 1] - rowPtr[u];
    }
}

__global__ void recomputeBestParentsKernel(int n,
                                           const int* incomingRowPtr,
                                           const int* incomingColInd,
                                           const int* incomingWeight,
                                           const int* currentDistances,
                                           int sourceVertex,
                                           int* newDistances,
                                           int* newParents) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;

    if (v >= n) {
        return;
    }

    if (v == sourceVertex) {
        newDistances[v] = 0;
        newParents[v] = -1;
        return;
    }

    int bestDistance = INT_MAX;
    int bestParent = -1;

    int start = incomingRowPtr[v];
    int end = incomingRowPtr[v + 1];

    for (int i = start; i < end; ++i) {
        int u = incomingColInd[i];
        int w = incomingWeight[i];
        int distU = currentDistances[u];

        if (distU == INT_MAX) {
            continue;
        }

        int candidateDistance = distU + w;

        if (candidateDistance < bestDistance) {
            bestDistance = candidateDistance;
            bestParent = u;
        }
    }

    newDistances[v] = bestDistance;
    newParents[v] = bestParent;
}

__global__ void recomputeCandidatesKernel(int candidateCount,
                                          const int* candidateVertices,
                                          const int* incomingRowPtr,
                                          const int* incomingColInd,
                                          const int* incomingWeight,
                                          const int* currentDistances,
                                          int sourceVertex,
                                          int* newDistances,
                                          int* newParents) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= candidateCount) {
        return;
    }

    int v = candidateVertices[idx];

    if (v == sourceVertex) {
        newDistances[v] = 0;
        newParents[v] = -1;
        return;
    }

    int bestDistance = INT_MAX;
    int bestParent = -1;

    int start = incomingRowPtr[v];
    int end = incomingRowPtr[v + 1];

    for (int i = start; i < end; ++i) {
        int u = incomingColInd[i];
        int w = incomingWeight[i];
        int distU = currentDistances[u];

        if (distU == INT_MAX) {
            continue;
        }

        int candidateDistance = distU + w;

        if (candidateDistance < bestDistance) {
            bestDistance = candidateDistance;
            bestParent = u;
        }
    }

    newDistances[v] = bestDistance;
    newParents[v] = bestParent;
}

bool computeOutDegreesOnDevice(const DeviceCsrGraph& deviceGraph, std::vector<int>& degrees) {
    degrees.assign(deviceGraph.n, 0);

    int* d_degrees = nullptr;
    cudaError_t err;

    err = cudaMalloc(reinterpret_cast<void**>(&d_degrees), deviceGraph.n * sizeof(int));
    if (err != cudaSuccess) {
        return false;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (deviceGraph.n + threadsPerBlock - 1) / threadsPerBlock;

    computeOutDegreesKernel<<<blocksPerGrid, threadsPerBlock>>>(
        deviceGraph.n, deviceGraph.rowPtr, d_degrees
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_degrees);
        return false;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_degrees);
        return false;
    }

    err = cudaMemcpy(degrees.data(), d_degrees,
                     deviceGraph.n * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_degrees);
        return false;
    }

    cudaFree(d_degrees);
    return true;
}

bool recomputeBestParentsOnDevice(const DeviceCsrGraph& incomingGraph,
                                  const std::vector<int>& currentDistances,
                                  int sourceVertex,
                                  std::vector<int>& newDistances,
                                  std::vector<int>& newParents) {
    if (static_cast<int>(currentDistances.size()) != incomingGraph.n) {
        return false;
    }

    newDistances.assign(incomingGraph.n, INT_MAX);
    newParents.assign(incomingGraph.n, -1);

    int* d_currentDistances = nullptr;
    int* d_newDistances = nullptr;
    int* d_newParents = nullptr;
    cudaError_t err;

    err = cudaMalloc(reinterpret_cast<void**>(&d_currentDistances), incomingGraph.n * sizeof(int));
    if (err != cudaSuccess) {
        return false;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&d_newDistances), incomingGraph.n * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_currentDistances);
        return false;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&d_newParents), incomingGraph.n * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_currentDistances);
        cudaFree(d_newDistances);
        return false;
    }

    err = cudaMemcpy(d_currentDistances, currentDistances.data(),
                     incomingGraph.n * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_currentDistances);
        cudaFree(d_newDistances);
        cudaFree(d_newParents);
        return false;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (incomingGraph.n + threadsPerBlock - 1) / threadsPerBlock;

    recomputeBestParentsKernel<<<blocksPerGrid, threadsPerBlock>>>(
        incomingGraph.n,
        incomingGraph.rowPtr,
        incomingGraph.colInd,
        incomingGraph.weight,
        d_currentDistances,
        sourceVertex,
        d_newDistances,
        d_newParents
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_currentDistances);
        cudaFree(d_newDistances);
        cudaFree(d_newParents);
        return false;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_currentDistances);
        cudaFree(d_newDistances);
        cudaFree(d_newParents);
        return false;
    }

    err = cudaMemcpy(newDistances.data(), d_newDistances,
                     incomingGraph.n * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_currentDistances);
        cudaFree(d_newDistances);
        cudaFree(d_newParents);
        return false;
    }

    err = cudaMemcpy(newParents.data(), d_newParents,
                     incomingGraph.n * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_currentDistances);
        cudaFree(d_newDistances);
        cudaFree(d_newParents);
        return false;
    }

    cudaFree(d_currentDistances);
    cudaFree(d_newDistances);
    cudaFree(d_newParents);
    return true;
}

bool recomputeCandidatesOnDevice(const DeviceCsrGraph& incomingGraph,
                                 const std::vector<int>& currentDistances,
                                 const std::vector<int>& candidateVertices,
                                 int sourceVertex,
                                 std::vector<int>& newDistances,
                                 std::vector<int>& newParents) {
    if (static_cast<int>(currentDistances.size()) != incomingGraph.n) {
        return false;
    }

    newDistances = currentDistances;
    newParents.assign(incomingGraph.n, -1);

    if (candidateVertices.empty()) {
        return true;
    }

    int* d_currentDistances = nullptr;
    int* d_newDistances = nullptr;
    int* d_newParents = nullptr;
    int* d_candidateVertices = nullptr;
    cudaError_t err;

    err = cudaMalloc(reinterpret_cast<void**>(&d_currentDistances), incomingGraph.n * sizeof(int));
    if (err != cudaSuccess) {
        return false;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&d_newDistances), incomingGraph.n * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_currentDistances);
        return false;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&d_newParents), incomingGraph.n * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_currentDistances);
        cudaFree(d_newDistances);
        return false;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&d_candidateVertices),
                     static_cast<int>(candidateVertices.size()) * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_currentDistances);
        cudaFree(d_newDistances);
        cudaFree(d_newParents);
        return false;
    }

    err = cudaMemcpy(d_currentDistances, currentDistances.data(),
                     incomingGraph.n * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_currentDistances);
        cudaFree(d_newDistances);
        cudaFree(d_newParents);
        cudaFree(d_candidateVertices);
        return false;
    }

    err = cudaMemcpy(d_newDistances, newDistances.data(),
                     incomingGraph.n * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_currentDistances);
        cudaFree(d_newDistances);
        cudaFree(d_newParents);
        cudaFree(d_candidateVertices);
        return false;
    }

    err = cudaMemcpy(d_newParents, newParents.data(),
                     incomingGraph.n * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_currentDistances);
        cudaFree(d_newDistances);
        cudaFree(d_newParents);
        cudaFree(d_candidateVertices);
        return false;
    }

    err = cudaMemcpy(d_candidateVertices, candidateVertices.data(),
                     static_cast<int>(candidateVertices.size()) * sizeof(int),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_currentDistances);
        cudaFree(d_newDistances);
        cudaFree(d_newParents);
        cudaFree(d_candidateVertices);
        return false;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (static_cast<int>(candidateVertices.size()) + threadsPerBlock - 1) / threadsPerBlock;

    recomputeCandidatesKernel<<<blocksPerGrid, threadsPerBlock>>>(
        static_cast<int>(candidateVertices.size()),
        d_candidateVertices,
        incomingGraph.rowPtr,
        incomingGraph.colInd,
        incomingGraph.weight,
        d_currentDistances,
        sourceVertex,
        d_newDistances,
        d_newParents
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_currentDistances);
        cudaFree(d_newDistances);
        cudaFree(d_newParents);
        cudaFree(d_candidateVertices);
        return false;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_currentDistances);
        cudaFree(d_newDistances);
        cudaFree(d_newParents);
        cudaFree(d_candidateVertices);
        return false;
    }

    err = cudaMemcpy(newDistances.data(), d_newDistances,
                     incomingGraph.n * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_currentDistances);
        cudaFree(d_newDistances);
        cudaFree(d_newParents);
        cudaFree(d_candidateVertices);
        return false;
    }

    err = cudaMemcpy(newParents.data(), d_newParents,
                     incomingGraph.n * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_currentDistances);
        cudaFree(d_newDistances);
        cudaFree(d_newParents);
        cudaFree(d_candidateVertices);
        return false;
    }

    cudaFree(d_currentDistances);
    cudaFree(d_newDistances);
    cudaFree(d_newParents);
    cudaFree(d_candidateVertices);
    return true;
}
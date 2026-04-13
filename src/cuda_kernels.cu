#include "../headers/cuda_kernels.cuh"

#include <cuda_runtime.h>
#include <climits>
#include <utility>
#include <vector>

/**
 * @file cuda_kernels.cu
 * @brief CUDA kernel helpers for CSR graphs.
 */

__device__ __forceinline__ bool betterParentCandidate(int candidateDistance,
                                                      int candidateParent,
                                                      int bestDistance,
                                                      int bestParent) {
    return (candidateDistance < bestDistance) ||                         // smaller distance wins
           (candidateDistance == bestDistance && candidateParent > bestParent); // tie: larger parent id wins
}
__global__ void computeOutDegreesKernel(int n,
                                        const int* rowPtr,
                                        int* degrees) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;         // vertex id
    if (u < n) {
        degrees[u] = rowPtr[u + 1] - rowPtr[u];            // out-degree
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
    int v = blockIdx.x * blockDim.x + threadIdx.x;         // vertex id
    if (v >= n) return;                                    // skip out of range

    if (v == sourceVertex) {
        newDistances[v] = 0;                               // source distance
        newParents[v] = -1;                                // source parent
        return;
    }

    int bestDistance = INT_MAX;                            // best dist so far
    int bestParent = -1;                                   // best parent so far

    int start = incomingRowPtr[v];                         // incoming start
    int end = incomingRowPtr[v + 1];                       // incoming end

    for (int i = start; i < end; ++i) {
        int u = incomingColInd[i];                         // predecessor
        int w = incomingWeight[i];                         // edge weight
        int distU = currentDistances[u];                   // known dist of u

        if (distU == INT_MAX) continue;                    // ignore INF

        int candidateDistance = distU + w;                 // candidate path
        if (candidateDistance < bestDistance) {
            bestDistance = candidateDistance;              // better distance
            bestParent = u;                                // better parent
        }
    }

    newDistances[v] = bestDistance;                        // save result
    newParents[v] = bestParent;                            // save parent
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;       // candidate index
    if (idx >= candidateCount) return;                     // skip out of range

    int v = candidateVertices[idx];                        // actual vertex

    if (v == sourceVertex) {
        newDistances[v] = 0;                               // source distance
        newParents[v] = -1;                                // source parent
        return;
    }

    int bestDistance = INT_MAX;                            // best dist so far
    int bestParent = -1;                                   // best parent so far

    int start = incomingRowPtr[v];                         // incoming start
    int end = incomingRowPtr[v + 1];                       // incoming end

    for (int i = start; i < end; ++i) {
        int u = incomingColInd[i];                         // predecessor
        int w = incomingWeight[i];                         // edge weight
        int distU = currentDistances[u];                   // known dist of u

        if (distU == INT_MAX) continue;                    // ignore INF

        int candidateDistance = distU + w;                 // candidate path
        if (candidateDistance < bestDistance) {
            bestDistance = candidateDistance;              // better distance
            bestParent = u;                                // better parent
        }
    }

    newDistances[v] = bestDistance;                        // save result
    newParents[v] = bestParent;                            // save parent
}

__global__ void detectChangedCandidatesKernel(int candidateCount,
                                              const int* candidateVertices,
                                              const int* oldDistances,
                                              const int* newDistances,
                                              int* changedFlags) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;       // candidate index
    if (idx >= candidateCount) return;                     // skip out of range

    int v = candidateVertices[idx];                        // actual vertex
    changedFlags[idx] = (oldDistances[v] != newDistances[v]) ? 1 : 0;
}

__global__ void markEdgeEndpointsKernel(int edgeCount,
                                        const int* edgeU,
                                        const int* edgeV,
                                        int numVertices,
                                        int* flags) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;       // edge index
    if (idx >= edgeCount) return;                          // skip out of range

    int u = edgeU[idx];                                    // tail
    int v = edgeV[idx];                                    // head

    if (u >= 0 && u < numVertices) atomicExch(&flags[u], 1); // mark tail
    if (v >= 0 && v < numVertices) atomicExch(&flags[v], 1); // mark head
}

__global__ void markOutgoingNeighborsKernel(int changedCount,
                                            const int* changedVertices,
                                            int numVertices,
                                            const int* rowPtr,
                                            const int* colInd,
                                            int* flags) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;       // changed index
    if (idx >= changedCount) return;                       // skip out of range

    int u = changedVertices[idx];                          // changed vertex
    if (u < 0 || u >= numVertices) return;                 // skip bad vertex

    int start = rowPtr[u];                                 // row start
    int end = rowPtr[u + 1];                               // row end

    for (int i = start; i < end; ++i) {
        int v = colInd[i];                                 // outgoing neighbor
        if (v >= 0 && v < numVertices) {
            atomicExch(&flags[v], 1);                      // mark neighbor
        }
    }
}

__global__ void markDirectlyDeletedTreeChildrenKernel(int deleteCount,
                                                      const int* deleteU,
                                                      const int* deleteV,
                                                      const int* initialParents,
                                                      int sourceVertex,
                                                      int* invalidFlags) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;       // deleted edge index
    if (idx >= deleteCount) return;                        // skip out of range

    int u = deleteU[idx];                                  // deleted parent
    int v = deleteV[idx];                                  // deleted child

    if (v == sourceVertex) return;                         // source never invalidated
    if (v < 0) return;                                     // skip bad child

    if (initialParents[v] == u) {
        atomicExch(&invalidFlags[v], 1);                   // direct tree break
    }
}

__global__ void propagateInvalidByParentKernel(int n,
                                               const int* initialParents,
                                               int sourceVertex,
                                               int* invalidFlags,
                                               int* changed) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;         // vertex id
    if (v >= n) return;                                    // skip out of range
    if (v == sourceVertex) return;                         // keep source safe
    if (invalidFlags[v]) return;                           // already invalid

    int p = initialParents[v];                             // parent of v
    if (p >= 0 && p < n && invalidFlags[p]) {
        invalidFlags[v] = 1;                               // parent invalid => child invalid
        atomicExch(changed, 1);                            // signal progress
    }
}

__global__ void markVerticesFromListKernel(int count,
                                           const int* vertices,
                                           int n,
                                           int* flags,
                                           int* flaggedCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;       // list index
    if (idx >= count) return;                              // skip out of range

    int v = vertices[idx];                                 // actual vertex
    if (v < 0 || v >= n) return;                           // skip bad vertex

    if (atomicExch(&flags[v], 1) == 0) {
        atomicAdd(flaggedCount, 1);                        // count unique frontier vertex
    }
}

__global__ void recomputeFrontierFlagsKernel(int n,
                                             const int* frontierFlags,
                                             const int* incomingRowPtr,
                                             const int* incomingColInd,
                                             const int* incomingWeight,
                                             const int* currentDistances,
                                             int sourceVertex,
                                             int* newDistances,
                                             int* newParents) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;         // vertex id
    if (v >= n) return;                                    // skip out of range
    if (frontierFlags[v] == 0) return;                     // not in frontier

    if (v == sourceVertex) {
        newDistances[v] = 0;                               // source distance
        newParents[v] = -1;                                // source parent
        return;
    }

    int bestDistance = INT_MAX;                            // best dist so far
    int bestParent = -1;                                   // best parent so far

    int start = incomingRowPtr[v];                         // incoming start
    int end = incomingRowPtr[v + 1];                       // incoming end

    for (int i = start; i < end; ++i) {
        int u = incomingColInd[i];                         // predecessor
        int w = incomingWeight[i];                         // edge weight
        int distU = currentDistances[u];                   // known dist of u

        if (distU == INT_MAX) continue;                    // ignore INF

        int candidateDistance = distU + w;                 // candidate path
        if (candidateDistance < bestDistance) {
            bestDistance = candidateDistance;              // better distance
            bestParent = u;                                // better parent
        }
    }

    newDistances[v] = bestDistance;                        // save result
    newParents[v] = bestParent;                            // save parent
}

__global__ void markChangedFromFrontierKernel(int n,
                                              const int* frontierFlags,
                                              const int* oldDistances,
                                              const int* newDistances,
                                              int* changedFlags) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;         // vertex id
    if (v >= n) return;                                    // skip out of range

    if (frontierFlags[v] != 0 && oldDistances[v] != newDistances[v]) {
        changedFlags[v] = 1;                               // vertex changed
    }
}

__global__ void buildNextFrontierFromChangedKernel(int n,
                                                   const int* changedFlags,
                                                   const int* rowPtr,
                                                   const int* colInd,
                                                   int* nextFrontierFlags,
                                                   int* nextFrontierCount) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;         // source vertex
    if (u >= n) return;                                    // skip out of range
    if (changedFlags[u] == 0) return;                      // only changed vertices expand

    int start = rowPtr[u];                                 // row start
    int end = rowPtr[u + 1];                               // row end

    for (int i = start; i < end; ++i) {
        int v = colInd[i];                                 // outgoing neighbor
        if (v >= 0 && v < n) {
            if (atomicExch(&nextFrontierFlags[v], 1) == 0) {
                atomicAdd(nextFrontierCount, 1);           // count unique frontier vertex
            }
        }
    }
}

__global__ void applyInvalidFlagsKernel(int n,
                                        const int* invalidFlags,
                                        int sourceVertex,
                                        int* currentDistances,
                                        int* currentParents,
                                        int* frontierFlags,
                                        int* frontierCount) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;         // vertex id
    if (v >= n) return;                                    // skip out of range
    if (v == sourceVertex) return;                         // keep source safe
    if (invalidFlags[v] == 0) return;                      // only invalid nodes

    currentDistances[v] = INT_MAX;                         // reset distance
    currentParents[v] = -1;                                // reset parent

    if (atomicExch(&frontierFlags[v], 1) == 0) {
        atomicAdd(frontierCount, 1);                       // add to frontier once
    }
}

static bool compactFlagsToVertices(const std::vector<int>& flags,
                                   std::vector<int>& vertices) {
    vertices.clear();                                      // reset output
    for (int v = 0; v < static_cast<int>(flags.size()); ++v) {
        if (flags[v] != 0) vertices.push_back(v);          // keep marked vertex
    }
    return true;
}

bool computeOutDegreesOnDevice(const DeviceCsrGraph& deviceGraph,
                               std::vector<int>& degrees) {
    degrees.assign(deviceGraph.n, 0);                      // output size

    int* d_degrees = nullptr;                              // device degrees
    cudaError_t err;

    err = cudaMalloc(reinterpret_cast<void**>(&d_degrees),
                     deviceGraph.n * sizeof(int));
    if (err != cudaSuccess) return false;

    int threadsPerBlock = 256;                             // CUDA block size
    int blocksPerGrid = (deviceGraph.n + threadsPerBlock - 1) / threadsPerBlock;

    computeOutDegreesKernel<<<blocksPerGrid, threadsPerBlock>>>(
        deviceGraph.n, deviceGraph.rowPtr, d_degrees);

    err = cudaGetLastError();                              // launch check
    if (err != cudaSuccess) {
        cudaFree(d_degrees);
        return false;
    }

    err = cudaDeviceSynchronize();                         // finish kernel
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

    cudaFree(d_degrees);                                   // free device memory
    return true;
}

bool recomputeBestParentsOnDevice(const DeviceCsrGraph& incomingGraph,
                                  const std::vector<int>& currentDistances,
                                  int sourceVertex,
                                  std::vector<int>& newDistances,
                                  std::vector<int>& newParents) {
    if (static_cast<int>(currentDistances.size()) != incomingGraph.n) return false;

    newDistances.assign(incomingGraph.n, INT_MAX);         // output dist size
    newParents.assign(incomingGraph.n, -1);                // output parent size

    int* d_currentDistances = nullptr;                     // device old dist
    int* d_newDistances = nullptr;                         // device new dist
    int* d_newParents = nullptr;                           // device new parent
    cudaError_t err;

    err = cudaMalloc(reinterpret_cast<void**>(&d_currentDistances),
                     incomingGraph.n * sizeof(int));
    if (err != cudaSuccess) return false;

    err = cudaMalloc(reinterpret_cast<void**>(&d_newDistances),
                     incomingGraph.n * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_currentDistances);
        return false;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&d_newParents),
                     incomingGraph.n * sizeof(int));
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

    int threadsPerBlock = 256;                             // CUDA block size
    int blocksPerGrid = (incomingGraph.n + threadsPerBlock - 1) / threadsPerBlock;

    recomputeBestParentsKernel<<<blocksPerGrid, threadsPerBlock>>>(
        incomingGraph.n,
        incomingGraph.rowPtr,
        incomingGraph.colInd,
        incomingGraph.weight,
        d_currentDistances,
        sourceVertex,
        d_newDistances,
        d_newParents);

    err = cudaGetLastError();                              // launch check
    if (err != cudaSuccess) {
        cudaFree(d_currentDistances);
        cudaFree(d_newDistances);
        cudaFree(d_newParents);
        return false;
    }

    err = cudaDeviceSynchronize();                         // finish kernel
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

    cudaFree(d_currentDistances);                          // free device memory
    cudaFree(d_newDistances);                              // free device memory
    cudaFree(d_newParents);                                // free device memory
    return true;
}

bool recomputeCandidatesOnDevice(const DeviceCsrGraph& incomingGraph,
                                 const std::vector<int>& currentDistances,
                                 const std::vector<int>& candidateVertices,
                                 int sourceVertex,
                                 std::vector<int>& newDistances,
                                 std::vector<int>& newParents) {
    if (static_cast<int>(currentDistances.size()) != incomingGraph.n) return false;

    newDistances = currentDistances;                       // start from old dist
    newParents.assign(incomingGraph.n, -1);                // fresh parent array

    if (candidateVertices.empty()) return true;            // nothing to do

    int* d_currentDistances = nullptr;                     // device old dist
    int* d_newDistances = nullptr;                         // device new dist
    int* d_newParents = nullptr;                           // device new parent
    int* d_candidateVertices = nullptr;                    // device candidates
    cudaError_t err;

    err = cudaMalloc(reinterpret_cast<void**>(&d_currentDistances),
                     incomingGraph.n * sizeof(int));
    if (err != cudaSuccess) return false;

    err = cudaMalloc(reinterpret_cast<void**>(&d_newDistances),
                     incomingGraph.n * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_currentDistances);
        return false;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&d_newParents),
                     incomingGraph.n * sizeof(int));
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

    int threadsPerBlock = 256;                             // CUDA block size
    int blocksPerGrid =
        (static_cast<int>(candidateVertices.size()) + threadsPerBlock - 1) / threadsPerBlock;

    recomputeCandidatesKernel<<<blocksPerGrid, threadsPerBlock>>>(
        static_cast<int>(candidateVertices.size()),
        d_candidateVertices,
        incomingGraph.rowPtr,
        incomingGraph.colInd,
        incomingGraph.weight,
        d_currentDistances,
        sourceVertex,
        d_newDistances,
        d_newParents);

    err = cudaGetLastError();                              // launch check
    if (err != cudaSuccess) {
        cudaFree(d_currentDistances);
        cudaFree(d_newDistances);
        cudaFree(d_newParents);
        cudaFree(d_candidateVertices);
        return false;
    }

    err = cudaDeviceSynchronize();                         // finish kernel
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

    cudaFree(d_currentDistances);                          // free device memory
    cudaFree(d_newDistances);                              // free device memory
    cudaFree(d_newParents);                                // free device memory
    cudaFree(d_candidateVertices);                         // free device memory
    return true;
}

bool detectChangedCandidatesOnDevice(const std::vector<int>& oldDistances,
                                     const std::vector<int>& newDistances,
                                     const std::vector<int>& candidateVertices,
                                     std::vector<int>& changedVertices) {
    if (oldDistances.size() != newDistances.size()) return false;

    changedVertices.clear();                               // reset output
    if (candidateVertices.empty()) return true;            // nothing to do

    int n = static_cast<int>(oldDistances.size());         // vertex count
    int candidateCount = static_cast<int>(candidateVertices.size());

    int* d_oldDistances = nullptr;                         // device old dist
    int* d_newDistances = nullptr;                         // device new dist
    int* d_candidateVertices = nullptr;                    // device candidates
    int* d_changedFlags = nullptr;                         // device flags
    cudaError_t err;

    err = cudaMalloc(reinterpret_cast<void**>(&d_oldDistances), n * sizeof(int));
    if (err != cudaSuccess) return false;

    err = cudaMalloc(reinterpret_cast<void**>(&d_newDistances), n * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_oldDistances);
        return false;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&d_candidateVertices),
                     candidateCount * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_oldDistances);
        cudaFree(d_newDistances);
        return false;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&d_changedFlags),
                     candidateCount * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_oldDistances);
        cudaFree(d_newDistances);
        cudaFree(d_candidateVertices);
        return false;
    }

    err = cudaMemcpy(d_oldDistances, oldDistances.data(),
                     n * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_oldDistances);
        cudaFree(d_newDistances);
        cudaFree(d_candidateVertices);
        cudaFree(d_changedFlags);
        return false;
    }

    err = cudaMemcpy(d_newDistances, newDistances.data(),
                     n * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_oldDistances);
        cudaFree(d_newDistances);
        cudaFree(d_candidateVertices);
        cudaFree(d_changedFlags);
        return false;
    }

    err = cudaMemcpy(d_candidateVertices, candidateVertices.data(),
                     candidateCount * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_oldDistances);
        cudaFree(d_newDistances);
        cudaFree(d_candidateVertices);
        cudaFree(d_changedFlags);
        return false;
    }

    int threadsPerBlock = 256;                             // CUDA block size
    int blocksPerGrid = (candidateCount + threadsPerBlock - 1) / threadsPerBlock;

    detectChangedCandidatesKernel<<<blocksPerGrid, threadsPerBlock>>>(
        candidateCount,
        d_candidateVertices,
        d_oldDistances,
        d_newDistances,
        d_changedFlags);

    err = cudaGetLastError();                              // launch check
    if (err != cudaSuccess) {
        cudaFree(d_oldDistances);
        cudaFree(d_newDistances);
        cudaFree(d_candidateVertices);
        cudaFree(d_changedFlags);
        return false;
    }

    err = cudaDeviceSynchronize();                         // finish kernel
    if (err != cudaSuccess) {
        cudaFree(d_oldDistances);
        cudaFree(d_newDistances);
        cudaFree(d_candidateVertices);
        cudaFree(d_changedFlags);
        return false;
    }

    std::vector<int> changedFlags(candidateCount, 0);      // host flags
    err = cudaMemcpy(changedFlags.data(), d_changedFlags,
                     candidateCount * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_oldDistances);
        cudaFree(d_newDistances);
        cudaFree(d_candidateVertices);
        cudaFree(d_changedFlags);
        return false;
    }

    for (int i = 0; i < candidateCount; ++i) {
        if (changedFlags[i] != 0) {
            changedVertices.push_back(candidateVertices[i]); // keep changed vertex
        }
    }

    cudaFree(d_oldDistances);                              // free device memory
    cudaFree(d_newDistances);                              // free device memory
    cudaFree(d_candidateVertices);                         // free device memory
    cudaFree(d_changedFlags);                              // free device memory
    return true;
}

bool buildInitialCandidatesOnDevice(int numVertices,
                                    const std::vector<int>& insertU,
                                    const std::vector<int>& insertV,
                                    const std::vector<int>& deleteU,
                                    const std::vector<int>& deleteV,
                                    std::vector<int>& initialCandidates) {
    if (numVertices < 0) return false;                     // bad graph size
    if (insertU.size() != insertV.size()) return false;    // bad insert data
    if (deleteU.size() != deleteV.size()) return false;    // bad delete data

    initialCandidates.clear();                             // reset output
    if (numVertices == 0) return true;                     // nothing to do

    int* d_flags = nullptr;                                // device flags
    int* d_insertU = nullptr;                              // device insert tail
    int* d_insertV = nullptr;                              // device insert head
    int* d_deleteU = nullptr;                              // device delete tail
    int* d_deleteV = nullptr;                              // device delete head
    cudaError_t err;

    err = cudaMalloc(reinterpret_cast<void**>(&d_flags),
                     numVertices * sizeof(int));
    if (err != cudaSuccess) return false;

    err = cudaMemset(d_flags, 0, numVertices * sizeof(int)); // clear flags
    if (err != cudaSuccess) {
        cudaFree(d_flags);
        return false;
    }

    int threadsPerBlock = 256;                             // CUDA block size

    if (!insertU.empty()) {
        err = cudaMalloc(reinterpret_cast<void**>(&d_insertU),
                         static_cast<int>(insertU.size()) * sizeof(int));
        if (err != cudaSuccess) {
            cudaFree(d_flags);
            return false;
        }

        err = cudaMalloc(reinterpret_cast<void**>(&d_insertV),
                         static_cast<int>(insertV.size()) * sizeof(int));
        if (err != cudaSuccess) {
            cudaFree(d_flags);
            cudaFree(d_insertU);
            return false;
        }

        err = cudaMemcpy(d_insertU, insertU.data(),
                         static_cast<int>(insertU.size()) * sizeof(int),
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_flags);
            cudaFree(d_insertU);
            cudaFree(d_insertV);
            return false;
        }

        err = cudaMemcpy(d_insertV, insertV.data(),
                         static_cast<int>(insertV.size()) * sizeof(int),
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_flags);
            cudaFree(d_insertU);
            cudaFree(d_insertV);
            return false;
        }

        int blocksPerGrid =
            (static_cast<int>(insertU.size()) + threadsPerBlock - 1) / threadsPerBlock;

        markEdgeEndpointsKernel<<<blocksPerGrid, threadsPerBlock>>>(
            static_cast<int>(insertU.size()),
            d_insertU,
            d_insertV,
            numVertices,
            d_flags);

        err = cudaGetLastError();                          // launch check
        if (err != cudaSuccess) {
            cudaFree(d_flags);
            cudaFree(d_insertU);
            cudaFree(d_insertV);
            return false;
        }
    }

    if (!deleteU.empty()) {
        err = cudaMalloc(reinterpret_cast<void**>(&d_deleteU),
                         static_cast<int>(deleteU.size()) * sizeof(int));
        if (err != cudaSuccess) {
            cudaFree(d_flags);
            if (d_insertU) cudaFree(d_insertU);
            if (d_insertV) cudaFree(d_insertV);
            return false;
        }

        err = cudaMalloc(reinterpret_cast<void**>(&d_deleteV),
                         static_cast<int>(deleteV.size()) * sizeof(int));
        if (err != cudaSuccess) {
            cudaFree(d_flags);
            if (d_insertU) cudaFree(d_insertU);
            if (d_insertV) cudaFree(d_insertV);
            cudaFree(d_deleteU);
            return false;
        }

        err = cudaMemcpy(d_deleteU, deleteU.data(),
                         static_cast<int>(deleteU.size()) * sizeof(int),
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_flags);
            if (d_insertU) cudaFree(d_insertU);
            if (d_insertV) cudaFree(d_insertV);
            cudaFree(d_deleteU);
            cudaFree(d_deleteV);
            return false;
        }

        err = cudaMemcpy(d_deleteV, deleteV.data(),
                         static_cast<int>(deleteV.size()) * sizeof(int),
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_flags);
            if (d_insertU) cudaFree(d_insertU);
            if (d_insertV) cudaFree(d_insertV);
            cudaFree(d_deleteU);
            cudaFree(d_deleteV);
            return false;
        }

        int blocksPerGrid =
            (static_cast<int>(deleteU.size()) + threadsPerBlock - 1) / threadsPerBlock;

        markEdgeEndpointsKernel<<<blocksPerGrid, threadsPerBlock>>>(
            static_cast<int>(deleteU.size()),
            d_deleteU,
            d_deleteV,
            numVertices,
            d_flags);

        err = cudaGetLastError();                          // launch check
        if (err != cudaSuccess) {
            cudaFree(d_flags);
            if (d_insertU) cudaFree(d_insertU);
            if (d_insertV) cudaFree(d_insertV);
            cudaFree(d_deleteU);
            cudaFree(d_deleteV);
            return false;
        }
    }

    err = cudaDeviceSynchronize();                         // finish kernels
    if (err != cudaSuccess) {
        cudaFree(d_flags);
        if (d_insertU) cudaFree(d_insertU);
        if (d_insertV) cudaFree(d_insertV);
        if (d_deleteU) cudaFree(d_deleteU);
        if (d_deleteV) cudaFree(d_deleteV);
        return false;
    }

    std::vector<int> flags(numVertices, 0);                // host flags
    err = cudaMemcpy(flags.data(), d_flags,
                     numVertices * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_flags);
        if (d_insertU) cudaFree(d_insertU);
        if (d_insertV) cudaFree(d_insertV);
        if (d_deleteU) cudaFree(d_deleteU);
        if (d_deleteV) cudaFree(d_deleteV);
        return false;
    }

    compactFlagsToVertices(flags, initialCandidates);      // compact on host

    cudaFree(d_flags);                                     // free device memory
    if (d_insertU) cudaFree(d_insertU);
    if (d_insertV) cudaFree(d_insertV);
    if (d_deleteU) cudaFree(d_deleteU);
    if (d_deleteV) cudaFree(d_deleteV);
    return true;
}

bool buildNextCandidatesOnDevice(const DeviceCsrGraph& outgoingGraph,
                                 const std::vector<int>& changedVertices,
                                 std::vector<int>& nextCandidates) {
    nextCandidates.clear();                                // reset output
    if (outgoingGraph.n == 0) return true;                 // nothing to do
    if (changedVertices.empty()) return true;              // empty frontier

    int* d_flags = nullptr;                                // device flags
    int* d_changedVertices = nullptr;                      // device frontier
    cudaError_t err;

    err = cudaMalloc(reinterpret_cast<void**>(&d_flags),
                     outgoingGraph.n * sizeof(int));
    if (err != cudaSuccess) return false;

    err = cudaMemset(d_flags, 0, outgoingGraph.n * sizeof(int)); // clear flags
    if (err != cudaSuccess) {
        cudaFree(d_flags);
        return false;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&d_changedVertices),
                     static_cast<int>(changedVertices.size()) * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_flags);
        return false;
    }

    err = cudaMemcpy(d_changedVertices, changedVertices.data(),
                     static_cast<int>(changedVertices.size()) * sizeof(int),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_flags);
        cudaFree(d_changedVertices);
        return false;
    }

    int threadsPerBlock = 256;                             // CUDA block size
    int blocksPerGrid =
        (static_cast<int>(changedVertices.size()) + threadsPerBlock - 1) / threadsPerBlock;

    markOutgoingNeighborsKernel<<<blocksPerGrid, threadsPerBlock>>>(
        static_cast<int>(changedVertices.size()),
        d_changedVertices,
        outgoingGraph.n,
        outgoingGraph.rowPtr,
        outgoingGraph.colInd,
        d_flags);

    err = cudaGetLastError();                              // launch check
    if (err != cudaSuccess) {
        cudaFree(d_flags);
        cudaFree(d_changedVertices);
        return false;
    }

    err = cudaDeviceSynchronize();                         // finish kernel
    if (err != cudaSuccess) {
        cudaFree(d_flags);
        cudaFree(d_changedVertices);
        return false;
    }

    std::vector<int> flags(outgoingGraph.n, 0);            // host flags
    err = cudaMemcpy(flags.data(), d_flags,
                     outgoingGraph.n * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_flags);
        cudaFree(d_changedVertices);
        return false;
    }

    compactFlagsToVertices(flags, nextCandidates);         // compact on host

    cudaFree(d_flags);                                     // free device memory
    cudaFree(d_changedVertices);                           // free device memory
    return true;
}

bool invalidateDeletedTreeSubtreesOnDevice(const std::vector<int>& initialParents,
                                           const std::vector<int>& deleteU,
                                           const std::vector<int>& deleteV,
                                           int sourceVertex,
                                           std::vector<int>& invalidFlags) {
    if (deleteU.size() != deleteV.size()) return false;    // bad delete data

    int n = static_cast<int>(initialParents.size());       // number of vertices
    invalidFlags.assign(n, 0);                             // default safe

    if (n == 0 || deleteU.empty()) return true;            // nothing to do

    int* d_initialParents = nullptr;                       // device parents
    int* d_deleteU = nullptr;                              // device delete tail
    int* d_deleteV = nullptr;                              // device delete head
    int* d_invalidFlags = nullptr;                         // device invalid flags
    int* d_changed = nullptr;                              // device progress flag
    cudaError_t err;

    err = cudaMalloc(reinterpret_cast<void**>(&d_initialParents), n * sizeof(int));
    if (err != cudaSuccess) return false;

    err = cudaMalloc(reinterpret_cast<void**>(&d_invalidFlags), n * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_initialParents);
        return false;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&d_deleteU),
                     static_cast<int>(deleteU.size()) * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_initialParents);
        cudaFree(d_invalidFlags);
        return false;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&d_deleteV),
                     static_cast<int>(deleteV.size()) * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_initialParents);
        cudaFree(d_invalidFlags);
        cudaFree(d_deleteU);
        return false;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&d_changed), sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_initialParents);
        cudaFree(d_invalidFlags);
        cudaFree(d_deleteU);
        cudaFree(d_deleteV);
        return false;
    }

    err = cudaMemcpy(d_initialParents, initialParents.data(),
                     n * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_initialParents);
        cudaFree(d_invalidFlags);
        cudaFree(d_deleteU);
        cudaFree(d_deleteV);
        cudaFree(d_changed);
        return false;
    }

    err = cudaMemcpy(d_deleteU, deleteU.data(),
                     static_cast<int>(deleteU.size()) * sizeof(int),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_initialParents);
        cudaFree(d_invalidFlags);
        cudaFree(d_deleteU);
        cudaFree(d_deleteV);
        cudaFree(d_changed);
        return false;
    }

    err = cudaMemcpy(d_deleteV, deleteV.data(),
                     static_cast<int>(deleteV.size()) * sizeof(int),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_initialParents);
        cudaFree(d_invalidFlags);
        cudaFree(d_deleteU);
        cudaFree(d_deleteV);
        cudaFree(d_changed);
        return false;
    }

    err = cudaMemset(d_invalidFlags, 0, n * sizeof(int));  // clear invalid flags
    if (err != cudaSuccess) {
        cudaFree(d_initialParents);
        cudaFree(d_invalidFlags);
        cudaFree(d_deleteU);
        cudaFree(d_deleteV);
        cudaFree(d_changed);
        return false;
    }

    int threadsPerBlock = 256;                             // CUDA block size
    int deleteBlocks =
        (static_cast<int>(deleteU.size()) + threadsPerBlock - 1) / threadsPerBlock;

    markDirectlyDeletedTreeChildrenKernel<<<deleteBlocks, threadsPerBlock>>>(
        static_cast<int>(deleteU.size()),
        d_deleteU,
        d_deleteV,
        d_initialParents,
        sourceVertex,
        d_invalidFlags);

    err = cudaGetLastError();                              // launch check
    if (err != cudaSuccess) {
        cudaFree(d_initialParents);
        cudaFree(d_invalidFlags);
        cudaFree(d_deleteU);
        cudaFree(d_deleteV);
        cudaFree(d_changed);
        return false;
    }

    int vertexBlocks = (n + threadsPerBlock - 1) / threadsPerBlock; // vertex grid

    while (true) {
        int hostChanged = 0;                               // progress flag
        err = cudaMemcpy(d_changed, &hostChanged,
                         sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_initialParents);
            cudaFree(d_invalidFlags);
            cudaFree(d_deleteU);
            cudaFree(d_deleteV);
            cudaFree(d_changed);
            return false;
        }

        propagateInvalidByParentKernel<<<vertexBlocks, threadsPerBlock>>>(
            n,
            d_initialParents,
            sourceVertex,
            d_invalidFlags,
            d_changed);

        err = cudaGetLastError();                          // launch check
        if (err != cudaSuccess) {
            cudaFree(d_initialParents);
            cudaFree(d_invalidFlags);
            cudaFree(d_deleteU);
            cudaFree(d_deleteV);
            cudaFree(d_changed);
            return false;
        }

        err = cudaDeviceSynchronize();                     // finish propagation
        if (err != cudaSuccess) {
            cudaFree(d_initialParents);
            cudaFree(d_invalidFlags);
            cudaFree(d_deleteU);
            cudaFree(d_deleteV);
            cudaFree(d_changed);
            return false;
        }

        err = cudaMemcpy(&hostChanged, d_changed,
                         sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            cudaFree(d_initialParents);
            cudaFree(d_invalidFlags);
            cudaFree(d_deleteU);
            cudaFree(d_deleteV);
            cudaFree(d_changed);
            return false;
        }

        if (hostChanged == 0) break;                       // no more propagation
    }

    err = cudaMemcpy(invalidFlags.data(), d_invalidFlags,
                     n * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_initialParents);
        cudaFree(d_invalidFlags);
        cudaFree(d_deleteU);
        cudaFree(d_deleteV);
        cudaFree(d_changed);
        return false;
    }

    cudaFree(d_initialParents);                            // free device memory
    cudaFree(d_invalidFlags);                              // free device memory
    cudaFree(d_deleteU);                                   // free device memory
    cudaFree(d_deleteV);                                   // free device memory
    cudaFree(d_changed);                                   // free device memory
    return true;
}

bool runFrontierUpdateLoopOnDevice(const DeviceCsrGraph& outgoingGraph,
                                   const DeviceCsrGraph& incomingGraph,
                                   const std::vector<int>& initialDistances,
                                   const std::vector<int>& initialParents,
                                   const std::vector<int>& initialCandidates,
                                   int sourceVertex,
                                   std::vector<int>& finalDistances,
                                   std::vector<int>& finalParents) {
    int n = incomingGraph.n;                               // number of vertices
    if (outgoingGraph.n != n) return false;                // graph size mismatch
    if (static_cast<int>(initialDistances.size()) != n) return false;
    if (static_cast<int>(initialParents.size()) != n) return false;

    finalDistances = initialDistances;                     // default output
    finalParents = initialParents;                         // default output

    if (n == 0) return true;                               // nothing to do
    if (initialCandidates.empty()) return true;            // no frontier

    int* d_currentDistances = nullptr;                     // device current dist
    int* d_currentParents = nullptr;                       // device current parent
    int* d_newDistances = nullptr;                         // device next dist
    int* d_newParents = nullptr;                           // device next parent
    int* d_frontierFlags = nullptr;                        // device current frontier
    int* d_nextFrontierFlags = nullptr;                    // device next frontier
    int* d_changedFlags = nullptr;                         // device changed flags
    int* d_frontierVertices = nullptr;                     // initial frontier list
    int* d_frontierCount = nullptr;                        // current frontier count
    int* d_nextFrontierCount = nullptr;                    // next frontier count
    cudaError_t err;

    err = cudaMalloc(reinterpret_cast<void**>(&d_currentDistances), n * sizeof(int));
    if (err != cudaSuccess) return false;

    err = cudaMalloc(reinterpret_cast<void**>(&d_currentParents), n * sizeof(int));
    if (err != cudaSuccess) goto device_loop_fail;

    err = cudaMalloc(reinterpret_cast<void**>(&d_newDistances), n * sizeof(int));
    if (err != cudaSuccess) goto device_loop_fail;

    err = cudaMalloc(reinterpret_cast<void**>(&d_newParents), n * sizeof(int));
    if (err != cudaSuccess) goto device_loop_fail;

    err = cudaMalloc(reinterpret_cast<void**>(&d_frontierFlags), n * sizeof(int));
    if (err != cudaSuccess) goto device_loop_fail;

    err = cudaMalloc(reinterpret_cast<void**>(&d_nextFrontierFlags), n * sizeof(int));
    if (err != cudaSuccess) goto device_loop_fail;

    err = cudaMalloc(reinterpret_cast<void**>(&d_changedFlags), n * sizeof(int));
    if (err != cudaSuccess) goto device_loop_fail;

    err = cudaMalloc(reinterpret_cast<void**>(&d_frontierVertices),
                     static_cast<int>(initialCandidates.size()) * sizeof(int));
    if (err != cudaSuccess) goto device_loop_fail;

    err = cudaMalloc(reinterpret_cast<void**>(&d_frontierCount), sizeof(int));
    if (err != cudaSuccess) goto device_loop_fail;

    err = cudaMalloc(reinterpret_cast<void**>(&d_nextFrontierCount), sizeof(int));
    if (err != cudaSuccess) goto device_loop_fail;

    err = cudaMemcpy(d_currentDistances, initialDistances.data(),
                     n * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto device_loop_fail;

    err = cudaMemcpy(d_currentParents, initialParents.data(),
                     n * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto device_loop_fail;

    err = cudaMemset(d_frontierFlags, 0, n * sizeof(int)); // clear frontier
    if (err != cudaSuccess) goto device_loop_fail;

    {
        int hostFrontierCount = 0;                         // initial frontier count
        err = cudaMemcpy(d_frontierCount, &hostFrontierCount,
                         sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto device_loop_fail;
    }

    err = cudaMemcpy(d_frontierVertices, initialCandidates.data(),
                     static_cast<int>(initialCandidates.size()) * sizeof(int),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto device_loop_fail;

    {
        int threadsPerBlock = 256;                         // CUDA block size
        int listBlocks =
            (static_cast<int>(initialCandidates.size()) + threadsPerBlock - 1) / threadsPerBlock;

        markVerticesFromListKernel<<<listBlocks, threadsPerBlock>>>(
            static_cast<int>(initialCandidates.size()),
            d_frontierVertices,
            n,
            d_frontierFlags,
            d_frontierCount);

        err = cudaGetLastError();                          // launch check
        if (err != cudaSuccess) goto device_loop_fail;
    }

    err = cudaDeviceSynchronize();                         // finish initialization
    if (err != cudaSuccess) goto device_loop_fail;

    {
        int hostFrontierCount = 0;                         // read frontier count
        err = cudaMemcpy(&hostFrontierCount, d_frontierCount,
                         sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) goto device_loop_fail;

        int iteration = 0;                                 // loop counter
        int maxIterations = n * 2;                         // simple guard
        int threadsPerBlock = 256;                         // CUDA block size
        int vertexBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

        while (hostFrontierCount > 0 && iteration < maxIterations) {
            ++iteration;                                   // next round

            err = cudaMemcpy(d_newDistances, d_currentDistances,
                             n * sizeof(int), cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) goto device_loop_fail;

            err = cudaMemcpy(d_newParents, d_currentParents,
                             n * sizeof(int), cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) goto device_loop_fail;

            err = cudaMemset(d_changedFlags, 0, n * sizeof(int)); // clear changed
            if (err != cudaSuccess) goto device_loop_fail;

            err = cudaMemset(d_nextFrontierFlags, 0, n * sizeof(int)); // clear next frontier
            if (err != cudaSuccess) goto device_loop_fail;

            {
                int zero = 0;                              // reset next frontier count
                err = cudaMemcpy(d_nextFrontierCount, &zero,
                                 sizeof(int), cudaMemcpyHostToDevice);
                if (err != cudaSuccess) goto device_loop_fail;
            }

            recomputeFrontierFlagsKernel<<<vertexBlocks, threadsPerBlock>>>(
                n,
                d_frontierFlags,
                incomingGraph.rowPtr,
                incomingGraph.colInd,
                incomingGraph.weight,
                d_currentDistances,
                sourceVertex,
                d_newDistances,
                d_newParents);

            err = cudaGetLastError();                      // launch check
            if (err != cudaSuccess) goto device_loop_fail;

            markChangedFromFrontierKernel<<<vertexBlocks, threadsPerBlock>>>(
                n,
                d_frontierFlags,
                d_currentDistances,
                d_newDistances,
                d_changedFlags);

            err = cudaGetLastError();                      // launch check
            if (err != cudaSuccess) goto device_loop_fail;

            buildNextFrontierFromChangedKernel<<<vertexBlocks, threadsPerBlock>>>(
                n,
                d_changedFlags,
                outgoingGraph.rowPtr,
                outgoingGraph.colInd,
                d_nextFrontierFlags,
                d_nextFrontierCount);

            err = cudaGetLastError();                      // launch check
            if (err != cudaSuccess) goto device_loop_fail;

            err = cudaDeviceSynchronize();                 // finish all kernels
            if (err != cudaSuccess) goto device_loop_fail;

            err = cudaMemcpy(&hostFrontierCount, d_nextFrontierCount,
                             sizeof(int), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) goto device_loop_fail;

            std::swap(d_currentDistances, d_newDistances); // next becomes current
            std::swap(d_currentParents, d_newParents);     // next becomes current
            std::swap(d_frontierFlags, d_nextFrontierFlags); // next frontier becomes current
        }
    }

        
    finalDistances.assign(n, INT_MAX);
    finalParents.assign(n, -1);

    err = cudaMemcpy(finalDistances.data(), d_currentDistances,
                     n * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto device_loop_fail;

    err = cudaMemcpy(finalParents.data(), d_currentParents,
                     n * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto device_loop_fail;

    cudaFree(d_currentDistances);                          // free device memory
    cudaFree(d_currentParents);                            // free device memory
    cudaFree(d_newDistances);                              // free device memory
    cudaFree(d_newParents);                                // free device memory
    cudaFree(d_frontierFlags);                             // free device memory
    cudaFree(d_nextFrontierFlags);                         // free device memory
    cudaFree(d_changedFlags);                              // free device memory
    cudaFree(d_frontierVertices);                          // free device memory
    cudaFree(d_frontierCount);                             // free device memory
    cudaFree(d_nextFrontierCount);                         // free device memory
    return true;

device_loop_fail:
    if (d_currentDistances) cudaFree(d_currentDistances);
    if (d_currentParents) cudaFree(d_currentParents);
    if (d_newDistances) cudaFree(d_newDistances);
    if (d_newParents) cudaFree(d_newParents);
    if (d_frontierFlags) cudaFree(d_frontierFlags);
    if (d_nextFrontierFlags) cudaFree(d_nextFrontierFlags);
    if (d_changedFlags) cudaFree(d_changedFlags);
    if (d_frontierVertices) cudaFree(d_frontierVertices);
    if (d_frontierCount) cudaFree(d_frontierCount);
    if (d_nextFrontierCount) cudaFree(d_nextFrontierCount);
    return false;
}

bool runIncrementalFrontierUpdateLoopOnDevice(const DeviceCsrGraph& outgoingGraph,
                                              const DeviceCsrGraph& incomingGraph,
                                              const std::vector<int>& initialDistances,
                                              const std::vector<int>& initialParents,
                                              const std::vector<int>& initialCandidates,
                                              const std::vector<int>& deleteU,
                                              const std::vector<int>& deleteV,
                                              int sourceVertex,
                                              std::vector<int>& finalDistances,
                                              std::vector<int>& finalParents) {
    int n = incomingGraph.n;
    if (outgoingGraph.n != n) return false;
    if (static_cast<int>(initialDistances.size()) != n) return false;
    if (static_cast<int>(initialParents.size()) != n) return false;
    if (deleteU.size() != deleteV.size()) return false;

    finalDistances = initialDistances;
    finalParents = initialParents;

    if (n == 0) return true;

    int* d_currentDistances = nullptr;
    int* d_currentParents = nullptr;
    int* d_newDistances = nullptr;
    int* d_newParents = nullptr;
    int* d_frontierFlags = nullptr;
    int* d_nextFrontierFlags = nullptr;
    int* d_changedFlags = nullptr;
    int* d_frontierVertices = nullptr;
    int* d_frontierCount = nullptr;
    int* d_nextFrontierCount = nullptr;
    int* d_initialParents = nullptr;
    int* d_deleteU = nullptr;
    int* d_deleteV = nullptr;
    int* d_invalidFlags = nullptr;
    int* d_invalidationChanged = nullptr;
    cudaError_t err;
    int frontierAllocCount = static_cast<int>(initialCandidates.size());
    if (frontierAllocCount < 1) frontierAllocCount = 1;

    err = cudaMalloc(reinterpret_cast<void**>(&d_currentDistances), n * sizeof(int));
    if (err != cudaSuccess) return false;

    err = cudaMalloc(reinterpret_cast<void**>(&d_currentParents), n * sizeof(int));
    if (err != cudaSuccess) goto incremental_loop_fail;

    err = cudaMalloc(reinterpret_cast<void**>(&d_newDistances), n * sizeof(int));
    if (err != cudaSuccess) goto incremental_loop_fail;

    err = cudaMalloc(reinterpret_cast<void**>(&d_newParents), n * sizeof(int));
    if (err != cudaSuccess) goto incremental_loop_fail;

    err = cudaMalloc(reinterpret_cast<void**>(&d_frontierFlags), n * sizeof(int));
    if (err != cudaSuccess) goto incremental_loop_fail;

    err = cudaMalloc(reinterpret_cast<void**>(&d_nextFrontierFlags), n * sizeof(int));
    if (err != cudaSuccess) goto incremental_loop_fail;

    err = cudaMalloc(reinterpret_cast<void**>(&d_changedFlags), n * sizeof(int));
    if (err != cudaSuccess) goto incremental_loop_fail;

    err = cudaMalloc(reinterpret_cast<void**>(&d_frontierVertices),
                     frontierAllocCount * static_cast<int>(sizeof(int)));
    if (err != cudaSuccess) goto incremental_loop_fail;

    err = cudaMalloc(reinterpret_cast<void**>(&d_frontierCount), sizeof(int));
    if (err != cudaSuccess) goto incremental_loop_fail;

    err = cudaMalloc(reinterpret_cast<void**>(&d_nextFrontierCount), sizeof(int));
    if (err != cudaSuccess) goto incremental_loop_fail;

    err = cudaMalloc(reinterpret_cast<void**>(&d_initialParents), n * sizeof(int));
    if (err != cudaSuccess) goto incremental_loop_fail;

    err = cudaMalloc(reinterpret_cast<void**>(&d_invalidFlags), n * sizeof(int));
    if (err != cudaSuccess) goto incremental_loop_fail;

    err = cudaMalloc(reinterpret_cast<void**>(&d_invalidationChanged), sizeof(int));
    if (err != cudaSuccess) goto incremental_loop_fail;

    err = cudaMemcpy(d_currentDistances, initialDistances.data(),
                     n * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto incremental_loop_fail;

    err = cudaMemcpy(d_currentParents, initialParents.data(),
                     n * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto incremental_loop_fail;

    err = cudaMemcpy(d_initialParents, initialParents.data(),
                     n * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto incremental_loop_fail;

    err = cudaMemset(d_invalidFlags, 0, n * sizeof(int));
    if (err != cudaSuccess) goto incremental_loop_fail;

    if (!deleteU.empty()) {
        err = cudaMalloc(reinterpret_cast<void**>(&d_deleteU),
                         static_cast<int>(deleteU.size()) * sizeof(int));
        if (err != cudaSuccess) goto incremental_loop_fail;

        err = cudaMalloc(reinterpret_cast<void**>(&d_deleteV),
                         static_cast<int>(deleteV.size()) * sizeof(int));
        if (err != cudaSuccess) goto incremental_loop_fail;

        err = cudaMemcpy(d_deleteU, deleteU.data(),
                         static_cast<int>(deleteU.size()) * sizeof(int),
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto incremental_loop_fail;

        err = cudaMemcpy(d_deleteV, deleteV.data(),
                         static_cast<int>(deleteV.size()) * sizeof(int),
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto incremental_loop_fail;

        int threadsPerBlock = 256;
        int deleteBlocks =
            (static_cast<int>(deleteU.size()) + threadsPerBlock - 1) / threadsPerBlock;

        markDirectlyDeletedTreeChildrenKernel<<<deleteBlocks, threadsPerBlock>>>(
            static_cast<int>(deleteU.size()),
            d_deleteU,
            d_deleteV,
            d_initialParents,
            sourceVertex,
            d_invalidFlags);

        err = cudaGetLastError();
        if (err != cudaSuccess) goto incremental_loop_fail;

        int vertexBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

        while (true) {
            int hostChanged = 0;
            err = cudaMemcpy(d_invalidationChanged, &hostChanged,
                             sizeof(int), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) goto incremental_loop_fail;

            propagateInvalidByParentKernel<<<vertexBlocks, threadsPerBlock>>>(
                n,
                d_initialParents,
                sourceVertex,
                d_invalidFlags,
                d_invalidationChanged);

            err = cudaGetLastError();
            if (err != cudaSuccess) goto incremental_loop_fail;

            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) goto incremental_loop_fail;

            err = cudaMemcpy(&hostChanged, d_invalidationChanged,
                             sizeof(int), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) goto incremental_loop_fail;

            if (hostChanged == 0) break;
        }
    }

    err = cudaMemset(d_frontierFlags, 0, n * sizeof(int));
    if (err != cudaSuccess) goto incremental_loop_fail;

    {
        int zero = 0;
        err = cudaMemcpy(d_frontierCount, &zero,
                         sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto incremental_loop_fail;
    }

    if (!initialCandidates.empty()) {
        err = cudaMemcpy(d_frontierVertices, initialCandidates.data(),
                         static_cast<int>(initialCandidates.size()) * sizeof(int),
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto incremental_loop_fail;

        int threadsPerBlock = 256;
        int listBlocks =
            (static_cast<int>(initialCandidates.size()) + threadsPerBlock - 1) / threadsPerBlock;

        markVerticesFromListKernel<<<listBlocks, threadsPerBlock>>>(
            static_cast<int>(initialCandidates.size()),
            d_frontierVertices,
            n,
            d_frontierFlags,
            d_frontierCount);

        err = cudaGetLastError();
        if (err != cudaSuccess) goto incremental_loop_fail;
    }

    {
        int threadsPerBlock = 256;
        int vertexBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

        applyInvalidFlagsKernel<<<vertexBlocks, threadsPerBlock>>>(
            n,
            d_invalidFlags,
            sourceVertex,
            d_currentDistances,
            d_currentParents,
            d_frontierFlags,
            d_frontierCount);

        err = cudaGetLastError();
        if (err != cudaSuccess) goto incremental_loop_fail;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto incremental_loop_fail;

    {
        int hostFrontierCount = 0;
        err = cudaMemcpy(&hostFrontierCount, d_frontierCount,
                         sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) goto incremental_loop_fail;

        int iteration = 0;
        int maxIterations = n * 2;
        int threadsPerBlock = 256;
        int vertexBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

        while (hostFrontierCount > 0 && iteration < maxIterations) {
            ++iteration;

            err = cudaMemcpy(d_newDistances, d_currentDistances,
                             n * sizeof(int), cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) goto incremental_loop_fail;

            err = cudaMemcpy(d_newParents, d_currentParents,
                             n * sizeof(int), cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) goto incremental_loop_fail;

            err = cudaMemset(d_changedFlags, 0, n * sizeof(int));
            if (err != cudaSuccess) goto incremental_loop_fail;

            err = cudaMemset(d_nextFrontierFlags, 0, n * sizeof(int));
            if (err != cudaSuccess) goto incremental_loop_fail;

            {
                int zero = 0;
                err = cudaMemcpy(d_nextFrontierCount, &zero,
                                 sizeof(int), cudaMemcpyHostToDevice);
                if (err != cudaSuccess) goto incremental_loop_fail;
            }

            recomputeFrontierFlagsKernel<<<vertexBlocks, threadsPerBlock>>>(
                n,
                d_frontierFlags,
                incomingGraph.rowPtr,
                incomingGraph.colInd,
                incomingGraph.weight,
                d_currentDistances,
                sourceVertex,
                d_newDistances,
                d_newParents);

            err = cudaGetLastError();
            if (err != cudaSuccess) goto incremental_loop_fail;

            markChangedFromFrontierKernel<<<vertexBlocks, threadsPerBlock>>>(
                n,
                d_frontierFlags,
                d_currentDistances,
                d_newDistances,
                d_changedFlags);

            err = cudaGetLastError();
            if (err != cudaSuccess) goto incremental_loop_fail;

            buildNextFrontierFromChangedKernel<<<vertexBlocks, threadsPerBlock>>>(
                n,
                d_changedFlags,
                outgoingGraph.rowPtr,
                outgoingGraph.colInd,
                d_nextFrontierFlags,
                d_nextFrontierCount);

            err = cudaGetLastError();
            if (err != cudaSuccess) goto incremental_loop_fail;

            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) goto incremental_loop_fail;

            err = cudaMemcpy(&hostFrontierCount, d_nextFrontierCount,
                             sizeof(int), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) goto incremental_loop_fail;

            std::swap(d_currentDistances, d_newDistances);
            std::swap(d_currentParents, d_newParents);
            std::swap(d_frontierFlags, d_nextFrontierFlags);
        }
    }

        

    finalDistances.assign(n, INT_MAX);
    finalParents.assign(n, -1);

    err = cudaMemcpy(finalDistances.data(), d_currentDistances,
                     n * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto incremental_loop_fail;

    err = cudaMemcpy(finalParents.data(), d_currentParents,
                     n * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto incremental_loop_fail;

    cudaFree(d_currentDistances);
    cudaFree(d_currentParents);
    cudaFree(d_newDistances);
    cudaFree(d_newParents);
    cudaFree(d_frontierFlags);
    cudaFree(d_nextFrontierFlags);
    cudaFree(d_changedFlags);
    cudaFree(d_frontierVertices);
    cudaFree(d_frontierCount);
    cudaFree(d_nextFrontierCount);
    cudaFree(d_initialParents);
    if (d_deleteU) cudaFree(d_deleteU);
    if (d_deleteV) cudaFree(d_deleteV);
    cudaFree(d_invalidFlags);
    cudaFree(d_invalidationChanged);
    return true;

incremental_loop_fail:
    if (d_currentDistances) cudaFree(d_currentDistances);
    if (d_currentParents) cudaFree(d_currentParents);
    if (d_newDistances) cudaFree(d_newDistances);
    if (d_newParents) cudaFree(d_newParents);
    if (d_frontierFlags) cudaFree(d_frontierFlags);
    if (d_nextFrontierFlags) cudaFree(d_nextFrontierFlags);
    if (d_changedFlags) cudaFree(d_changedFlags);
    if (d_frontierVertices) cudaFree(d_frontierVertices);
    if (d_frontierCount) cudaFree(d_frontierCount);
    if (d_nextFrontierCount) cudaFree(d_nextFrontierCount);
    if (d_initialParents) cudaFree(d_initialParents);
    if (d_deleteU) cudaFree(d_deleteU);
    if (d_deleteV) cudaFree(d_deleteV);
    if (d_invalidFlags) cudaFree(d_invalidFlags);
    if (d_invalidationChanged) cudaFree(d_invalidationChanged);
    return false;
}
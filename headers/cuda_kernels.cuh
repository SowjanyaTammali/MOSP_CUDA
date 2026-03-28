#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include "cuda_graph.cuh"
#include <vector>

/**
 * @file cuda_kernels.cuh
 * @brief Declares CUDA kernel helper functions for CSR graphs.
 */

/**
 * @brief Computes out-degrees of all vertices on the GPU.
 * @param deviceGraph Input CSR graph stored on the GPU.
 * @param degrees Output vector on the CPU containing one out-degree per vertex.
 * @return true if successful, false otherwise.
 */
bool computeOutDegreesOnDevice(const DeviceCsrGraph& deviceGraph, std::vector<int>& degrees);

/**
 * @brief Recomputes best parent and best distance for every vertex using incoming CSR.
 * @param incomingGraph Input incoming CSR graph stored on the GPU.
 * @param currentDistances Input distance array on the CPU.
 * @param sourceVertex Source vertex whose distance remains zero.
 * @param newDistances Output distance array on the CPU.
 * @param newParents Output parent array on the CPU.
 * @return true if successful, false otherwise.
 */
bool recomputeBestParentsOnDevice(const DeviceCsrGraph& incomingGraph,
                                  const std::vector<int>& currentDistances,
                                  int sourceVertex,
                                  std::vector<int>& newDistances,
                                  std::vector<int>& newParents);

/**
 * @brief Recomputes best parent and best distance only for candidate vertices.
 * @param incomingGraph Input incoming CSR graph stored on the GPU.
 * @param currentDistances Input distance array on the CPU.
 * @param candidateVertices List of vertices to recompute.
 * @param sourceVertex Source vertex whose distance remains zero.
 * @param newDistances Output distance array on the CPU.
 * @param newParents Output parent array on the CPU.
 * @return true if successful, false otherwise.
 */
bool recomputeCandidatesOnDevice(const DeviceCsrGraph& incomingGraph,
                                 const std::vector<int>& currentDistances,
                                 const std::vector<int>& candidateVertices,
                                 int sourceVertex,
                                 std::vector<int>& newDistances,
                                 std::vector<int>& newParents);

#endif
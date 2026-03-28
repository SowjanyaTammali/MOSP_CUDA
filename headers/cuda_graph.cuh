#ifndef CUDA_GRAPH_CUH
#define CUDA_GRAPH_CUH

#include "read.h"
#include <vector>

/**
 * @file cuda_graph.cuh
 * @brief Declares host-side and device-side CSR graph structures and helper
 *        functions for CUDA-based graph processing.
 */

/// Host-side CSR graph for one selected objective.
struct HostCsrGraph {
    int n = 0;                    ///< Number of vertices
    int m = 0;                    ///< Number of edges
    std::vector<int> rowPtr;      ///< CSR row pointer array
    std::vector<int> colInd;      ///< CSR column index array
    std::vector<int> weight;      ///< Edge weights for one objective
};

/// Device-side CSR graph stored in GPU memory.
struct DeviceCsrGraph {
    int n = 0;                    ///< Number of vertices
    int m = 0;                    ///< Number of edges
    int* rowPtr = nullptr;        ///< Device pointer to CSR row pointer array
    int* colInd = nullptr;        ///< Device pointer to CSR column index array
    int* weight = nullptr;        ///< Device pointer to edge weight array
};

/**
 * @brief Builds outgoing CSR from the current graph representation.
 * @param graph Input graph in adjacency-list form.
 * @param objectiveIndex Selected objective index.
 * @param csr Output outgoing CSR graph.
 * @return true if successful, false otherwise.
 */
bool buildOutgoingCSR(const Graph& graph, int objectiveIndex, HostCsrGraph& csr);

/**
 * @brief Builds incoming CSR from the current graph representation.
 * @param graph Input graph in adjacency-list form.
 * @param objectiveIndex Selected objective index.
 * @param csr Output incoming CSR graph.
 * @return true if successful, false otherwise.
 */
bool buildIncomingCSR(const Graph& graph, int objectiveIndex, HostCsrGraph& csr);

/**
 * @brief Allocates GPU memory and copies a host CSR graph to device memory.
 * @param hostGraph Input CSR graph stored on the CPU.
 * @param deviceGraph Output CSR graph stored on the GPU.
 * @return true if successful, false otherwise.
 */
bool copyHostCsrToDevice(const HostCsrGraph& hostGraph, DeviceCsrGraph& deviceGraph);

/**
 * @brief Frees GPU memory owned by a device CSR graph.
 * @param deviceGraph Device CSR graph whose memory should be released.
 */
void freeDeviceCsr(DeviceCsrGraph& deviceGraph);

#endif
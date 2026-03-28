#include "../headers/cuda_graph.cuh"
#include <cuda_runtime.h>

/**
 * @file cuda_graph.cu
 * @brief Builds host-side CSR graphs and manages device-side CSR memory.
 */

/**
 * @brief Builds outgoing CSR for one selected objective.
 * @param graph Input graph in adjacency-list form.
 * @param objectiveIndex Selected objective index.
 * @param csr Output outgoing CSR graph.
 * @return true if successful, false otherwise.
 */
bool buildOutgoingCSR(const Graph& graph, int objectiveIndex, HostCsrGraph& csr) {
    csr.n = static_cast<int>(graph.size());
    csr.m = 0;
    csr.rowPtr.assign(csr.n + 1, 0);
    csr.colInd.clear();
    csr.weight.clear();

    for (int u = 0; u < csr.n; ++u) {
        csr.rowPtr[u] = csr.m;

        for (const auto& edge : graph[u]) {
            if (objectiveIndex < 0 || objectiveIndex >= static_cast<int>(edge.weights.size())) {
                return false;
            }

            csr.colInd.push_back(edge.to);
            csr.weight.push_back(edge.weights[objectiveIndex]);
            ++csr.m;
        }
    }

    csr.rowPtr[csr.n] = csr.m;
    return true;
}

/**
 * @brief Builds incoming CSR for one selected objective.
 * @param graph Input graph in adjacency-list form.
 * @param objectiveIndex Selected objective index.
 * @param csr Output incoming CSR graph.
 * @return true if successful, false otherwise.
 */
bool buildIncomingCSR(const Graph& graph, int objectiveIndex, HostCsrGraph& csr) {
    csr.n = static_cast<int>(graph.size());
    csr.m = 0;
    csr.rowPtr.assign(csr.n + 1, 0);
    csr.colInd.clear();
    csr.weight.clear();

    std::vector<int> inDegree(csr.n, 0);

    for (int u = 0; u < csr.n; ++u) {
        for (const auto& edge : graph[u]) {
            if (objectiveIndex < 0 || objectiveIndex >= static_cast<int>(edge.weights.size())) {
                return false;
            }

            if (edge.to < 0 || edge.to >= csr.n) {
                return false;
            }

            ++inDegree[edge.to];
            ++csr.m;
        }
    }

    csr.rowPtr[0] = 0;
    for (int v = 0; v < csr.n; ++v) {
        csr.rowPtr[v + 1] = csr.rowPtr[v] + inDegree[v];
    }

    csr.colInd.assign(csr.m, 0);
    csr.weight.assign(csr.m, 0);

    std::vector<int> nextPosition = csr.rowPtr;

    for (int u = 0; u < csr.n; ++u) {
        for (const auto& edge : graph[u]) {
            int v = edge.to;
            int pos = nextPosition[v]++;

            csr.colInd[pos] = u;
            csr.weight[pos] = edge.weights[objectiveIndex];
        }
    }

    return true;
}

/**
 * @brief Allocates GPU memory and copies a host CSR graph to device memory.
 * @param hostGraph Input CSR graph stored on the CPU.
 * @param deviceGraph Output CSR graph stored on the GPU.
 * @return true if successful, false otherwise.
 */
bool copyHostCsrToDevice(const HostCsrGraph& hostGraph, DeviceCsrGraph& deviceGraph) {
    deviceGraph.n = hostGraph.n;
    deviceGraph.m = hostGraph.m;
    deviceGraph.rowPtr = nullptr;
    deviceGraph.colInd = nullptr;
    deviceGraph.weight = nullptr;

    cudaError_t err;
   
    err = cudaMalloc(reinterpret_cast<void**>(&deviceGraph.rowPtr),
                     (hostGraph.n + 1) * sizeof(int));
    if (err != cudaSuccess) {
        return false;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&deviceGraph.colInd),
                     hostGraph.m * sizeof(int));
    if (err != cudaSuccess) {
        freeDeviceCsr(deviceGraph);
        return false;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&deviceGraph.weight),
                     hostGraph.m * sizeof(int));
    if (err != cudaSuccess) {
        freeDeviceCsr(deviceGraph);
        return false;
    }

    err = cudaMemcpy(deviceGraph.rowPtr, hostGraph.rowPtr.data(),
                     (hostGraph.n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        freeDeviceCsr(deviceGraph);
        return false;
    }

    err = cudaMemcpy(deviceGraph.colInd, hostGraph.colInd.data(),
                     hostGraph.m * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        freeDeviceCsr(deviceGraph);
        return false;
    }

    err = cudaMemcpy(deviceGraph.weight, hostGraph.weight.data(),
                     hostGraph.m * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        freeDeviceCsr(deviceGraph);
        return false;
    }

    return true;
}

/**
 * @brief Frees GPU memory owned by a device CSR graph.
 * @param deviceGraph Device CSR graph whose memory should be released.
 */
void freeDeviceCsr(DeviceCsrGraph& deviceGraph) {
    if (deviceGraph.rowPtr != nullptr) {
        cudaFree(deviceGraph.rowPtr);
        deviceGraph.rowPtr = nullptr;
    }

    if (deviceGraph.colInd != nullptr) {
        cudaFree(deviceGraph.colInd);
        deviceGraph.colInd = nullptr;
    }

    if (deviceGraph.weight != nullptr) {
        cudaFree(deviceGraph.weight);
        deviceGraph.weight = nullptr;
    }

    deviceGraph.n = 0;
    deviceGraph.m = 0;
}
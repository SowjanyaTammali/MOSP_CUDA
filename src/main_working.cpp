#include "../headers/read.h"
#include "../headers/cuda_graph.cuh"
#include "../headers/cuda_kernels.cuh"
#include "../headers/cuda_sosp_update.cuh"

#include <climits>
#include <iostream>
#include <string>
#include <vector>

/**
 * @file main.cpp
 * @brief Minimal driver for CSR setup and hybrid CUDA SOSP update.
 */

void printCSR(const std::string& name, const HostCsrGraph& csr) {
    std::cout << "\n--- " << name << " ---\n";
    std::cout << "n = " << csr.n << ", m = " << csr.m << "\n";

    std::cout << "rowPtr: ";
    for (int x : csr.rowPtr) std::cout << x << " ";
    std::cout << "\n";

    std::cout << "colInd: ";
    for (int x : csr.colInd) std::cout << x << " ";
    std::cout << "\n";

    std::cout << "weight: ";
    for (int x : csr.weight) std::cout << x << " ";
    std::cout << "\n";
}

void printVector(const std::string& name, const std::vector<int>& values) {
    std::cout << "\n--- " << name << " ---\n";
    for (int x : values) {
        if (x == INT_MAX) std::cout << "INF ";
        else std::cout << x << " ";
    }
    std::cout << "\n";
}

bool vectorsEqual(const std::vector<int>& a, const std::vector<int>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

void computeOutDegreesOnCPU(const HostCsrGraph& graph, std::vector<int>& degrees) {
    degrees.assign(graph.n, 0);
    for (int u = 0; u < graph.n; ++u) {
        degrees[u] = graph.rowPtr[u + 1] - graph.rowPtr[u];
    }
}

/**
 * @brief Entry point for minimal CUDA SOSP prototype.
 * @return 0 on success, 1 on failure.
 */
int main() {
    Graph graph;
    int numberOfObjectives = 0;

    std::string basePath = "data/originalGraph/graphCsr";
    int objectiveIndex = 0;
    int sourceVertex = 0;

    if (!readCSR(basePath, graph, numberOfObjectives)) {
        std::cerr << "Error: failed to read CSR graph from files.\n";
        return 1;
    }

    if (objectiveIndex < 0 || objectiveIndex >= numberOfObjectives) {
        std::cerr << "Error: invalid objective index.\n";
        return 1;
    }

    HostCsrGraph outgoingCSR;
    HostCsrGraph incomingCSR;

    if (!buildOutgoingCSR(graph, objectiveIndex, outgoingCSR)) {
        std::cerr << "Error: failed to build outgoing CSR.\n";
        return 1;
    }

    if (!buildIncomingCSR(graph, objectiveIndex, incomingCSR)) {
        std::cerr << "Error: failed to build incoming CSR.\n";
        return 1;
    }

    DeviceCsrGraph deviceOutgoingCSR;
    DeviceCsrGraph deviceIncomingCSR;

    if (!copyHostCsrToDevice(outgoingCSR, deviceOutgoingCSR)) {
        std::cerr << "Error: failed to copy outgoing CSR to device.\n";
        return 1;
    }

    if (!copyHostCsrToDevice(incomingCSR, deviceIncomingCSR)) {
        std::cerr << "Error: failed to copy incoming CSR to device.\n";
        freeDeviceCsr(deviceOutgoingCSR);
        return 1;
    }

    std::vector<int> cpuOutDegrees;
    std::vector<int> gpuOutDegrees;

    computeOutDegreesOnCPU(outgoingCSR, cpuOutDegrees);

    if (!computeOutDegreesOnDevice(deviceOutgoingCSR, gpuOutDegrees)) {
        std::cerr << "Error: failed to compute out-degrees on device.\n";
        freeDeviceCsr(deviceOutgoingCSR);
        freeDeviceCsr(deviceIncomingCSR);
        return 1;
    }

    if (!vectorsEqual(cpuOutDegrees, gpuOutDegrees)) {
        std::cerr << "Error: CPU and GPU out-degrees do not match.\n";
        freeDeviceCsr(deviceOutgoingCSR);
        freeDeviceCsr(deviceIncomingCSR);
        return 1;
    }

    std::cout << "CPU and GPU out-degrees match.\n";

    std::vector<int> finalDistances;
    std::vector<int> finalParents;

    if (!runHybridSOSPUpdate(outgoingCSR,
                             incomingCSR,
                             deviceIncomingCSR,
                             sourceVertex,
                             finalDistances,
                             finalParents)) {
        std::cerr << "Error: hybrid SOSP update failed.\n";
        freeDeviceCsr(deviceOutgoingCSR);
        freeDeviceCsr(deviceIncomingCSR);
        return 1;
    }

    printVector("Final Distances", finalDistances);
    printVector("Final Parents", finalParents);

    freeDeviceCsr(deviceOutgoingCSR);
    freeDeviceCsr(deviceIncomingCSR);
    return 0;
}
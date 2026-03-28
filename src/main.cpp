#include "../headers/read.h"
#include "../headers/cuda_graph.cuh"
#include "../headers/cuda_kernels.cuh"
#include "../headers/cuda_sosp_update.cuh"
#include "../headers/updateGraphCSR.h"
#include "../headers/sequentialSOSPUpdate.h"
#include "../headers/cudaCombinedGraph.cuh"
#include "../headers/cudaMOSPWorkflow.cuh"

#include <climits>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

/**
 * @file main.cpp
 * @brief Runs CUDA SOSP, CPU sequential SOSP baseline, CUDA incremental update,
 *        or CUDA combined graph mode.
 */

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

bool writeVectorToFile(const std::string& path, const std::vector<int>& values) {
    std::filesystem::path outPath(path);
    if (!outPath.parent_path().empty()) {
        std::filesystem::create_directories(outPath.parent_path());
    }

    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "Error: could not write file: " << path << "\n";
        return false;
    }

    for (size_t i = 0; i < values.size(); ++i) {
        out << i << " ";
        if (values[i] == INT_MAX) out << "INF";
        else out << values[i];
        out << "\n";
    }

    return true;
}

bool readVectorFromFile(const std::string& path, std::vector<int>& values) {
    std::ifstream in(path);
    if (!in.is_open()) {
        std::cerr << "Error: could not read file: " << path << "\n";
        return false;
    }

    values.clear();

    int vertex;
    std::string token;
    while (in >> vertex >> token) {
        if (vertex != static_cast<int>(values.size())) {
            values.resize(vertex + 1, INT_MAX);
        } else {
            values.push_back(INT_MAX);
        }

        if (token == "INF") values[vertex] = INT_MAX;
        else values[vertex] = std::stoi(token);
    }

    return true;
}

bool prepareGraphs(const std::string& prefix,
                   int objectiveIndex,
                   HostCsrGraph& outgoingCSR,
                   HostCsrGraph& incomingCSR,
                   DeviceCsrGraph& deviceIncomingCSR) {
    Graph graph;
    int numberOfObjectives = 0;

    if (!readCSR(prefix, graph, numberOfObjectives)) {
        std::cerr << "Error: failed to read CSR graph from " << prefix << "\n";
        return false;
    }

    if (objectiveIndex < 0 || objectiveIndex >= numberOfObjectives) {
        std::cerr << "Error: invalid objective index.\n";
        return false;
    }

    if (!buildOutgoingCSR(graph, objectiveIndex, outgoingCSR)) {
        std::cerr << "Error: failed to build outgoing CSR.\n";
        return false;
    }

    if (!buildIncomingCSR(graph, objectiveIndex, incomingCSR)) {
        std::cerr << "Error: failed to build incoming CSR.\n";
        return false;
    }

    if (!copyHostCsrToDevice(incomingCSR, deviceIncomingCSR)) {
        std::cerr << "Error: failed to copy incoming CSR to device.\n";
        return false;
    }

    return true;
}

bool loadInitialCandidatesFromChanges(const std::string& insertPath,
                                      const std::string& deletePath,
                                      std::vector<int>& initialCandidates) {
    std::vector<char> seen(100000, 0);
    initialCandidates.clear();

    auto addVertex = [&](int v) {
        if (v < 0) return;
        if (v >= static_cast<int>(seen.size())) {
            seen.resize(v + 1, 0);
        }
        if (!seen[v]) {
            seen[v] = 1;
            initialCandidates.push_back(v);
        }
    };

    {
        std::ifstream in(insertPath);
        if (!in.is_open()) {
            std::cerr << "Error: could not open insert file: " << insertPath << "\n";
            return false;
        }

        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            int u, v;
            if (!(iss >> u >> v)) continue;
            addVertex(v);
        }
    }

    {
        std::ifstream in(deletePath);
        if (!in.is_open()) {
            std::cerr << "Error: could not open delete file: " << deletePath << "\n";
            return false;
        }

        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            int u, v;
            if (!(iss >> u >> v)) continue;
            addVertex(v);
        }
    }

    return true;
}

int main(int argc, char* argv[]) {
    if (argc >= 2 && std::string(argv[1]) == "workflow") {
        if (argc < 7) {
            std::cerr << "Usage: " << argv[0]
                      << " workflow <originalPrefix> <updatedPrefix> <insertFile> <deleteFile> <source>\n";
            return 1;
        }

        std::string originalPrefix = argv[2];
        std::string updatedPrefix = argv[3];
        std::string insertPath = argv[4];
        std::string deletePath = argv[5];
        int source = std::stoi(argv[6]);

        if (!runCudaMOSPWorkflow(originalPrefix, updatedPrefix, insertPath, deletePath, source)) {
            std::cerr << "Error: runCudaMOSPWorkflow failed.\n";
            return 1;
        }

        return 0;
    }

    if (argc >= 2 && std::string(argv[1]) == "combined") {
        if (argc < 6) {
            std::cerr << "Usage: " << argv[0]
                      << " combined <originalPrefix> <K> <source> <tree1> <tree2> ... <treeK>\n";
            return 1;
        }

        std::string originalPrefix = argv[2];
        int K = std::stoi(argv[3]);
        int source = std::stoi(argv[4]);

        if (argc < 5 + K) {
            std::cerr << "Error: not enough tree paths for K.\n";
            return 1;
        }

        std::vector<std::string> treeInputPaths;
        for (int i = 0; i < K; ++i) {
            treeInputPaths.push_back(argv[5 + i]);
        }

        if (!cudaCombinedGraph(originalPrefix, treeInputPaths, K, source)) {
            std::cerr << "Error: cudaCombinedGraph failed.\n";
            return 1;
        }

        return 0;
    }

    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
                  << " <originalPrefix> <updatedPrefix> <insertFile> <deleteFile> <objectiveIndex> <sourceVertex>\n";
        std::cerr << "Or:    " << argv[0]
                  << " combined <originalPrefix> <K> <source> <tree1> ... <treeK>\n";
        std::cerr << "Or:    " << argv[0]
                  << " workflow <originalPrefix> <updatedPrefix> <insertFile> <deleteFile> <source>\n";
        return 1;
    }
    std::string originalPrefix = argv[1];
    std::string updatedPrefix  = argv[2];
    std::string insertPath     = argv[3];
    std::string deletePath     = argv[4];
    int objectiveIndex         = std::stoi(argv[5]);
    int sourceVertex           = std::stoi(argv[6]);

    std::string outputDir = "output";
    std::string originalDistancesPath   = outputDir + "/originalCuda/distances.txt";
    std::string originalParentsPath     = outputDir + "/originalCuda/parents.txt";
    std::string cpuUpdatedDistancesPath = outputDir + "/cpuBaseline/updatedDistances.txt";
    std::string cpuUpdatedParentsPath   = outputDir + "/cpuBaseline/updatedParents.txt";

    std::cout << "Original Prefix : " << originalPrefix << "\n";
    std::cout << "Updated Prefix  : " << updatedPrefix << "\n";
    std::cout << "Insert File     : " << insertPath << "\n";
    std::cout << "Delete File     : " << deletePath << "\n";
    std::cout << "Objective Index : " << objectiveIndex << "\n";
    std::cout << "Source Vertex   : " << sourceVertex << "\n";

    HostCsrGraph originalOutgoingCSR;
    HostCsrGraph originalIncomingCSR;
    DeviceCsrGraph deviceOriginalIncomingCSR;

    if (!prepareGraphs(originalPrefix,
                       objectiveIndex,
                       originalOutgoingCSR,
                       originalIncomingCSR,
                       deviceOriginalIncomingCSR)) {
        return 1;
    }

    std::vector<int> originalDistances;
    std::vector<int> originalParents;

    if (!runHybridSOSPUpdate(originalOutgoingCSR,
                             originalIncomingCSR,
                             deviceOriginalIncomingCSR,
                             sourceVertex,
                             originalDistances,
                             originalParents)) {
        std::cerr << "Error: hybrid SOSP update failed on original graph.\n";
        freeDeviceCsr(deviceOriginalIncomingCSR);
        return 1;
    }

    printVector("Original Graph Distances (CUDA)", originalDistances);
    printVector("Original Graph Parents (CUDA)", originalParents);

    freeDeviceCsr(deviceOriginalIncomingCSR);

    if (!writeVectorToFile(originalDistancesPath, originalDistances)) {
        return 1;
    }

    if (!writeVectorToFile(originalParentsPath, originalParents)) {
        return 1;
    }

    if (!sequentialSOSPUpdate(originalPrefix,
                              originalDistancesPath,
                              originalParentsPath,
                              insertPath,
                              deletePath,
                              objectiveIndex,
                              sourceVertex,
                              cpuUpdatedDistancesPath,
                              cpuUpdatedParentsPath)) {
        std::cerr << "Error: sequentialSOSPUpdate failed.\n";
        return 1;
    }

    std::vector<int> cpuUpdatedDistances;
    std::vector<int> cpuUpdatedParents;

    if (!readVectorFromFile(cpuUpdatedDistancesPath, cpuUpdatedDistances)) {
        return 1;
    }

    if (!readVectorFromFile(cpuUpdatedParentsPath, cpuUpdatedParents)) {
        return 1;
    }

    printVector("Updated Distances (CPU Baseline)", cpuUpdatedDistances);
    printVector("Updated Parents (CPU Baseline)", cpuUpdatedParents);

    if (!updateGraphCSR(originalPrefix, updatedPrefix, insertPath, deletePath, true)) {
        std::cerr << "Error: failed to create updated CSR graph.\n";
        return 1;
    }

    HostCsrGraph updatedOutgoingCSR;
    HostCsrGraph updatedIncomingCSR;
    DeviceCsrGraph deviceUpdatedIncomingCSR;

    if (!prepareGraphs(updatedPrefix,
                       objectiveIndex,
                       updatedOutgoingCSR,
                       updatedIncomingCSR,
                       deviceUpdatedIncomingCSR)) {
        return 1;
    }

    std::vector<int> initialCandidates;
    if (!loadInitialCandidatesFromChanges(insertPath, deletePath, initialCandidates)) {
        freeDeviceCsr(deviceUpdatedIncomingCSR);
        return 1;
    }

    std::cout << "\nInitial affected vertices for CUDA incremental update: ";
    for (int v : initialCandidates) std::cout << v << " ";
    std::cout << "\n";

    std::vector<int> updatedDistances;
    std::vector<int> updatedParents;

    if (!runHybridIncrementalSOSPUpdate(updatedOutgoingCSR,
                                        updatedIncomingCSR,
                                        deviceUpdatedIncomingCSR,
                                        originalDistances,
                                        originalParents,
                                        initialCandidates,
                                        deletePath,
                                        sourceVertex,
                                        updatedDistances,
                                        updatedParents)) {
        std::cerr << "Error: hybrid incremental SOSP update failed on updated graph.\n";
        freeDeviceCsr(deviceUpdatedIncomingCSR);
        return 1;
    }

    printVector("Updated Graph Distances (CUDA Incremental)", updatedDistances);
    printVector("Updated Graph Parents (CUDA Incremental)", updatedParents);

    freeDeviceCsr(deviceUpdatedIncomingCSR);

    if (!vectorsEqual(cpuUpdatedDistances, updatedDistances)) {
        std::cerr << "Error: CPU and CUDA updated distances do not match.\n";
        return 1;
    }

    if (!vectorsEqual(cpuUpdatedParents, updatedParents)) {
        std::cerr << "Error: CPU and CUDA updated parents do not match.\n";
        return 1;
    }

    std::cout << "\nCPU baseline and CUDA incremental update match.\n";
    return 0;
}
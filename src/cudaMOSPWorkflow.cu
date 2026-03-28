#include "../headers/cudaMOSPWorkflow.cuh"
#include "../headers/read.h"
#include "../headers/cuda_graph.cuh"
#include "../headers/cuda_sosp_update.cuh"
#include "../headers/cudaCombinedGraph.cuh"
#include "../headers/updateGraphCSR.h"

#include <climits>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

/**
 * @file cudaMOSPWorkflow.cu
 * @brief Implements an automated CUDA MOSP workflow wrapper using updated trees.
 */

namespace {

bool writeVectorToFile(const std::string& path, const std::vector<int>& values) {
    std::filesystem::path outPath(path);
    if (!outPath.parent_path().empty()) {
        std::filesystem::create_directories(outPath.parent_path());
    }

    std::ofstream out(path);
    if (!out.is_open()) {
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

bool writeParentsToFile(const std::string& path, const std::vector<int>& parents) {
    std::filesystem::path outPath(path);
    if (!outPath.parent_path().empty()) {
        std::filesystem::create_directories(outPath.parent_path());
    }

    std::ofstream out(path);
    if (!out.is_open()) {
        return false;
    }

    for (size_t i = 0; i < parents.size(); ++i) {
        out << i << " " << parents[i] << "\n";
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
            return false;
        }

        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            int u, v;
            if (!(iss >> u >> v)) continue;
            addVertex(u);
            addVertex(v);
        }
    }

    {
        std::ifstream in(deletePath);
        if (!in.is_open()) {
            return false;
        }

        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            int u, v;
            if (!(iss >> u >> v)) continue;
            addVertex(u);
            addVertex(v);
        }
    }

    return true;
}

bool prepareGraphsForObjective(const std::string& prefix,
                               int objectiveIndex,
                               HostCsrGraph& outgoingCSR,
                               HostCsrGraph& incomingCSR,
                               DeviceCsrGraph& deviceIncomingCSR,
                               int& numberOfObjectives) {
    Graph graph;

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

} // namespace

bool runCudaMOSPWorkflow(const std::string& originalCsrPrefix,
                         const std::string& updatedCsrPrefix,
                         const std::string& insertPath,
                         const std::string& deletePath,
                         int source,
                         const std::string& workDir) {
    std::filesystem::create_directories(workDir);
    std::filesystem::create_directories(workDir + "/trees");
    std::filesystem::create_directories(workDir + "/distances");
    std::filesystem::create_directories(workDir + "/originalDistances");
    std::filesystem::create_directories(workDir + "/originalTrees");

    Graph graph;
    int numberOfObjectives = 0;

    if (!readCSR(originalCsrPrefix, graph, numberOfObjectives)) {
        std::cerr << "Error: failed to read original CSR graph.\n";
        return false;
    }

    const int numberOfNodes = static_cast<int>(graph.size());
    if (numberOfNodes == 0) {
        std::cerr << "Error: graph has no vertices.\n";
        return false;
    }

    if (source < 0 || source >= numberOfNodes) {
        std::cerr << "Error: source out of range.\n";
        return false;
    }

    if (!updateGraphCSR(originalCsrPrefix, updatedCsrPrefix, insertPath, deletePath, true)) {
        std::cerr << "Error: failed to build updated CSR graph for workflow.\n";
        return false;
    }

    std::vector<int> initialCandidates;
    if (!loadInitialCandidatesFromChanges(insertPath, deletePath, initialCandidates)) {
        std::cerr << "Error: failed to load initial candidates from change files.\n";
        return false;
    }

    std::vector<std::string> treeFiles;
    treeFiles.reserve(numberOfObjectives);

    for (int objectiveIndex = 0; objectiveIndex < numberOfObjectives; ++objectiveIndex) {
        HostCsrGraph originalOutgoingCSR;
        HostCsrGraph originalIncomingCSR;
        DeviceCsrGraph deviceOriginalIncomingCSR;
        int detectedObjectivesOriginal = 0;

        if (!prepareGraphsForObjective(originalCsrPrefix,
                                       objectiveIndex,
                                       originalOutgoingCSR,
                                       originalIncomingCSR,
                                       deviceOriginalIncomingCSR,
                                       detectedObjectivesOriginal)) {
            return false;
        }

        std::vector<int> originalDistances;
        std::vector<int> originalParents;

        bool okOriginal = runHybridSOSPUpdate(originalOutgoingCSR,
                                              originalIncomingCSR,
                                              deviceOriginalIncomingCSR,
                                              source,
                                              originalDistances,
                                              originalParents);

        freeDeviceCsr(deviceOriginalIncomingCSR);

        if (!okOriginal) {
            std::cerr << "Error: CUDA SOSP failed on original graph for objective "
                      << objectiveIndex << ".\n";
            return false;
        }

        std::string originalDistancePath =
            workDir + "/originalDistances/dist_obj" + std::to_string(objectiveIndex) + ".txt";
        std::string originalTreePath =
            workDir + "/originalTrees/tree_obj" + std::to_string(objectiveIndex) + ".txt";

        if (!writeVectorToFile(originalDistancePath, originalDistances)) {
            std::cerr << "Error: failed to write original distances for objective "
                      << objectiveIndex << ".\n";
            return false;
        }

        if (!writeParentsToFile(originalTreePath, originalParents)) {
            std::cerr << "Error: failed to write original tree for objective "
                      << objectiveIndex << ".\n";
            return false;
        }

        HostCsrGraph updatedOutgoingCSR;
        HostCsrGraph updatedIncomingCSR;
        DeviceCsrGraph deviceUpdatedIncomingCSR;
        int detectedObjectivesUpdated = 0;

        if (!prepareGraphsForObjective(updatedCsrPrefix,
                                       objectiveIndex,
                                       updatedOutgoingCSR,
                                       updatedIncomingCSR,
                                       deviceUpdatedIncomingCSR,
                                       detectedObjectivesUpdated)) {
            return false;
        }

        std::vector<int> updatedDistances;
        std::vector<int> updatedParents;

        bool okUpdated = runHybridIncrementalSOSPUpdate(updatedOutgoingCSR,
                                                        updatedIncomingCSR,
                                                        deviceUpdatedIncomingCSR,
                                                        originalDistances,
                                                        originalParents,
                                                        initialCandidates,
                                                        deletePath,
                                                        source,
                                                        updatedDistances,
                                                        updatedParents);

        freeDeviceCsr(deviceUpdatedIncomingCSR);

        if (!okUpdated) {
            std::cerr << "Error: CUDA incremental SOSP failed on updated graph for objective "
                      << objectiveIndex << ".\n";
            return false;
        }

        std::string updatedDistancePath =
            workDir + "/distances/dist_obj" + std::to_string(objectiveIndex) + ".txt";
        std::string updatedTreePath =
            workDir + "/trees/tree_obj" + std::to_string(objectiveIndex) + ".txt";

        if (!writeVectorToFile(updatedDistancePath, updatedDistances)) {
            std::cerr << "Error: failed to write updated distances for objective "
                      << objectiveIndex << ".\n";
            return false;
        }

        if (!writeParentsToFile(updatedTreePath, updatedParents)) {
            std::cerr << "Error: failed to write updated tree for objective "
                      << objectiveIndex << ".\n";
            return false;
        }

        treeFiles.push_back(updatedTreePath);

        std::cout << "Objective " << objectiveIndex << " completed.\n";
        std::cout << "  Original distances: " << originalDistancePath << "\n";
        std::cout << "  Original parents:   " << originalTreePath << "\n";
        std::cout << "  Updated distances:  " << updatedDistancePath << "\n";
        std::cout << "  Updated parents:    " << updatedTreePath << "\n";
    }

    std::string combinedDir = workDir + "/combined";

    if (!cudaCombinedGraph(originalCsrPrefix,
                           treeFiles,
                           numberOfObjectives,
                           source,
                           combinedDir,
                           combinedDir + "/distancesCsr.txt",
                           combinedDir + "/SSSPTreeCsr.txt")) {
        std::cerr << "Error: cudaCombinedGraph failed.\n";
        return false;
    }

    std::cout << "\nCUDA MOSP workflow completed successfully.\n";
    std::cout << "Combined graph output directory: " << combinedDir << "\n";

    return true;
}
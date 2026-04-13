#include "../headers/cudaParallelSOSPUpdate.cuh"
#include "../headers/cuda_kernels.cuh"
#include "../headers/read.h"
#include "../headers/updateGraphCSR.h"

#include <climits>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

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

bool writeVectorToFile(const std::string& path,
                       const std::vector<int>& values) {
    std::filesystem::path outPath(path);                   // output path
    if (!outPath.parent_path().empty()) {
        std::filesystem::create_directories(outPath.parent_path()); // make dir
    }

    std::ofstream out(path);                               // open output
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

bool readValueFile(const std::string& path,
                   std::vector<int>& values) {
    std::ifstream in(path);                                // open input
    if (!in.is_open()) {
        std::cerr << "Error: could not read file: " << path << "\n";
        return false;
    }

    values.clear();                                        // reset output
    int vertex = 0;                                        // vertex id
    std::string token;                                     // value token

    while (in >> vertex >> token) {
        if (vertex < 0) continue;                          // skip bad row
        if (vertex >= static_cast<int>(values.size())) {
            values.resize(vertex + 1, INT_MAX);            // grow vector
        }

        if (token == "INF") values[vertex] = INT_MAX;      // unreachable
        else values[vertex] = std::stoi(token);            // normal value
    }

    return true;
}

bool loadChangeVertices(const std::string& insertPath,
                        const std::string& deletePath,
                        int numVertices,
                        std::vector<int>& changeVertices) {
    std::vector<int> insertU;
    std::vector<int> insertV;
    std::vector<int> deleteU;
    std::vector<int> deleteV;

    {
        std::ifstream in(insertPath);                      // open insert file
        if (!in.is_open()) {
            std::cerr << "Error: could not open insert file: " << insertPath << "\n";
            return false;
        }

        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            int u = 0, v = 0;
            if (!(iss >> u >> v)) continue;
            insertU.push_back(u);
            insertV.push_back(v);
        }
    }

    {
        std::ifstream in(deletePath);                      // open delete file
        if (!in.is_open()) {
            std::cerr << "Error: could not open delete file: " << deletePath << "\n";
            return false;
        }

        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            int u = 0, v = 0;
            if (!(iss >> u >> v)) continue;
            deleteU.push_back(u);
            deleteV.push_back(v);
        }
    }

    return buildInitialCandidatesOnDevice(numVertices,
                                          insertU, insertV,
                                          deleteU, deleteV,
                                          changeVertices);
}

bool loadDeletedEdges(const std::string& deletePath,
                      std::vector<int>& deleteU,
                      std::vector<int>& deleteV) {
    std::ifstream in(deletePath);                          // open delete file
    if (!in.is_open()) {
        std::cerr << "Error: could not open delete file: " << deletePath << "\n";
        return false;
    }

    deleteU.clear();                                       // reset output
    deleteV.clear();
    std::string line;

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        int u = 0, v = 0;
        if (!(iss >> u >> v)) continue;
        deleteU.push_back(u);
        deleteV.push_back(v);
    }

    return true;
}

std::string buildTemporaryUpdatedPrefix(const std::string& distancesOutputPath,
                                        const std::string& treeOutputPath,
                                        int objectiveIndex) {
    std::filesystem::path workDir = std::filesystem::path(distancesOutputPath).parent_path(); // try dist dir
    if (workDir.empty()) {
        workDir = std::filesystem::path(treeOutputPath).parent_path(); // fallback tree dir
    }
    if (workDir.empty()) {
        workDir = std::filesystem::path("output") /
                  ("parallelSospObj" + std::to_string(objectiveIndex)); // final fallback
    }

    std::filesystem::create_directories(workDir);          // make directory
    return (workDir / "updatedGraphCsr").string();         // temp updated prefix
}

} // namespace

bool runCudaSOSPFromScratchInternal(const HostCsrGraph& outgoingCSR,
                                    const HostCsrGraph& incomingCSR,
                                    const DeviceCsrGraph& deviceIncomingCSR,
                                    int sourceVertex,
                                    std::vector<int>& finalDistances,
                                    std::vector<int>& finalParents) {
    (void)incomingCSR;                                     // not needed directly

    std::vector<int> initialDistances(outgoingCSR.n, INT_MAX); // start INF
    std::vector<int> initialParents(outgoingCSR.n, -1);        // no parent
    initialDistances[sourceVertex] = 0;                       // source dist

    DeviceCsrGraph deviceOutgoingCSR;
    if (!copyHostCsrToDevice(outgoingCSR, deviceOutgoingCSR)) {
        std::cerr << "Error: failed to copy outgoing CSR to device.\n";
        return false;
    }

    std::vector<int> seedVertices(1, sourceVertex);        // start from source
    std::vector<int> initialCandidates;
    if (!buildNextCandidatesOnDevice(deviceOutgoingCSR,
                                     seedVertices,
                                     initialCandidates)) {
        freeDeviceCsr(deviceOutgoingCSR);
        return false;
    }

    bool ok = runFrontierUpdateLoopOnDevice(deviceOutgoingCSR,
                                            deviceIncomingCSR,
                                            initialDistances,
                                            initialParents,
                                            initialCandidates,
                                            sourceVertex,
                                            finalDistances,
                                            finalParents);

    freeDeviceCsr(deviceOutgoingCSR);
    return ok;
}

bool runCudaIncrementalSOSPUpdateInternal(const HostCsrGraph& outgoingCSR,
                                          const HostCsrGraph& incomingCSR,
                                          const DeviceCsrGraph& deviceIncomingCSR,
                                          const std::vector<int>& initialDistances,
                                          const std::vector<int>& initialParents,
                                          const std::vector<int>& initialCandidates,
                                          const std::string& deletePath,
                                          int sourceVertex,
                                          std::vector<int>& finalDistances,
                                          std::vector<int>& finalParents) {
    (void)incomingCSR;                                     // not needed directly

    if (static_cast<int>(initialDistances.size()) != outgoingCSR.n ||
        static_cast<int>(initialParents.size()) != outgoingCSR.n) {
        std::cerr << "Error: initial arrays do not match graph size.\n";
        return false;
    }

    std::vector<int> deleteU;
    std::vector<int> deleteV;
    if (!loadDeletedEdges(deletePath, deleteU, deleteV)) {
        return false;
    }

    DeviceCsrGraph deviceOutgoingCSR;
    if (!copyHostCsrToDevice(outgoingCSR, deviceOutgoingCSR)) {
        std::cerr << "Error: failed to copy outgoing CSR to device.\n";
        return false;
    }

    bool ok = runIncrementalFrontierUpdateLoopOnDevice(deviceOutgoingCSR,
                                                       deviceIncomingCSR,
                                                       initialDistances,
                                                       initialParents,
                                                       initialCandidates,
                                                       deleteU,
                                                       deleteV,
                                                       sourceVertex,
                                                       finalDistances,
                                                       finalParents);

    freeDeviceCsr(deviceOutgoingCSR);
    return ok;
}

bool cudaParallelSOSPUpdate(const std::string& originalCsrPrefix,
                            const std::string& distancesInputPath,
                            const std::string& treeInputPath,
                            const std::string& insertPath,
                            const std::string& deletePath,
                            int objectiveIndex,
                            int sourceVertex,
                            const std::string& distancesOutputPath,
                            const std::string& treeOutputPath) {
    std::vector<int> initialDistances;
    std::vector<int> initialParents;

    if (!readValueFile(distancesInputPath, initialDistances)) return false; // read original dist
    if (!readValueFile(treeInputPath, initialParents)) return false;        // read original tree

    const std::string tempUpdatedPrefix =                     // build updated graph internally
        buildTemporaryUpdatedPrefix(distancesOutputPath, treeOutputPath, objectiveIndex);

    if (!updateGraphCSR(originalCsrPrefix,                    // OpenMP-style semantics
                        tempUpdatedPrefix,
                        insertPath,
                        deletePath,
                        true)) {
        std::cerr << "Error: failed to build temporary updated CSR graph.\n";
        return false;
    }

    HostCsrGraph updatedOutgoingCSR;
    HostCsrGraph updatedIncomingCSR;
    DeviceCsrGraph deviceUpdatedIncomingCSR;

    if (!prepareGraphs(tempUpdatedPrefix,
                       objectiveIndex,
                       updatedOutgoingCSR,
                       updatedIncomingCSR,
                       deviceUpdatedIncomingCSR)) {
        return false;
    }

    std::vector<int> initialCandidates;
    if (!loadChangeVertices(insertPath,
                            deletePath,
                            updatedOutgoingCSR.n,
                            initialCandidates)) {
        freeDeviceCsr(deviceUpdatedIncomingCSR);
        return false;
    }

    std::vector<int> finalDistances;
    std::vector<int> finalParents;

    bool ok = runCudaIncrementalSOSPUpdateInternal(updatedOutgoingCSR,
                                                   updatedIncomingCSR,
                                                   deviceUpdatedIncomingCSR,
                                                   initialDistances,
                                                   initialParents,
                                                   initialCandidates,
                                                   deletePath,
                                                   sourceVertex,
                                                   finalDistances,
                                                   finalParents);

    freeDeviceCsr(deviceUpdatedIncomingCSR);

    if (!ok) {
        std::cerr << "Error: cudaParallelSOSPUpdate failed.\n";
        return false;
    }

    if (!writeVectorToFile(distancesOutputPath, finalDistances)) return false;
    if (!writeVectorToFile(treeOutputPath, finalParents)) return false;

    return true;
}
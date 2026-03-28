#include "../headers/cudaCombinedGraph.cuh"
#include "../headers/read.h"
#include "../headers/cuda_graph.cuh"
#include "../headers/cuda_sosp_update.cuh"

#include <climits>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

/**
 * @file cudaCombinedGraph.cu
 * @brief Implements hybrid CUDA combined-graph construction and solve.
 */

using namespace std;

namespace {

bool readParentFile(const string& path, vector<int>& parent, int numberOfNodes) {
    ifstream file(path);
    if (!file.is_open()) {
        cout << "Error: Could not open tree file: " << path << "\n";
        return false;
    }

    parent.assign(numberOfNodes, -1);

    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        istringstream ss(line);
        int v, p;
        if (!(ss >> v >> p)) continue;
        if (v < 0 || v >= numberOfNodes) {
            cout << "Error: Vertex out of range in tree file: " << path << "\n";
            return false;
        }
        parent[v] = p;
    }

    return true;
}

bool writeVectorToFile(const string& path, const vector<int>& values) {
    filesystem::path outPath(path);
    if (!outPath.parent_path().empty()) {
        filesystem::create_directories(outPath.parent_path());
    }

    ofstream out(path);
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

bool writeParentsToFile(const string& path, const vector<int>& parents) {
    filesystem::path outPath(path);
    if (!outPath.parent_path().empty()) {
        filesystem::create_directories(outPath.parent_path());
    }

    ofstream out(path);
    if (!out.is_open()) {
        return false;
    }

    for (size_t i = 0; i < parents.size(); ++i) {
        out << i << " " << parents[i] << "\n";
    }

    return true;
}

bool writeCombinedCSR(const string& prefix,
                      int numberOfNodes,
                      const map<pair<int, int>, int>& edgeMembership,
                      int K) {
    filesystem::create_directories(filesystem::path(prefix).parent_path());

    vector<vector<pair<int, int>>> adjacency(numberOfNodes);

    for (const auto& entry : edgeMembership) {
        int u = entry.first.first;
        int v = entry.first.second;
        int m = entry.second;
        if (m < 1) m = 1;
        if (m > K) m = K;
        int w = K + 1 - m;
        adjacency[u].push_back({v, w});
    }

    ofstream rowFile(prefix + "RowPtr.txt");
    ofstream colFile(prefix + "ColInd.txt");
    ofstream valFile(prefix + "Values.txt");

    if (!rowFile.is_open() || !colFile.is_open() || !valFile.is_open()) {
        return false;
    }

    int edgeCount = 0;
    rowFile << 0 << "\n";
    for (int u = 0; u < numberOfNodes; ++u) {
        edgeCount += static_cast<int>(adjacency[u].size());
        rowFile << edgeCount << "\n";
    }

    for (int u = 0; u < numberOfNodes; ++u) {
        for (const auto& edge : adjacency[u]) {
            colFile << edge.first << "\n";
            valFile << edge.second << "\n";
        }
    }

    return true;
}

bool prepareGraphs(const string& prefix,
                   int objectiveIndex,
                   HostCsrGraph& outgoingCSR,
                   HostCsrGraph& incomingCSR,
                   DeviceCsrGraph& deviceIncomingCSR) {
    Graph graph;
    int numberOfObjectives = 0;

    if (!readCSR(prefix, graph, numberOfObjectives)) {
        return false;
    }

    if (objectiveIndex < 0 || objectiveIndex >= numberOfObjectives) {
        return false;
    }

    if (!buildOutgoingCSR(graph, objectiveIndex, outgoingCSR)) {
        return false;
    }

    if (!buildIncomingCSR(graph, objectiveIndex, incomingCSR)) {
        return false;
    }

    if (!copyHostCsrToDevice(incomingCSR, deviceIncomingCSR)) {
        return false;
    }

    return true;
}

} // namespace

bool cudaCombinedGraph(const string& originalCsrPrefix,
                       const vector<string>& treeInputPaths,
                       int K,
                       int source,
                       const string& workDir,
                       const string& distancesOutputPath,
                       const string& treeOutputPath) {
    if (K <= 0 || static_cast<int>(treeInputPaths.size()) < K) {
        cout << "Error: invalid K or insufficient tree files.\n";
        return false;
    }

    Graph baseGraph;
    int numberOfObjectives = 0;
    if (!readCSR(originalCsrPrefix, baseGraph, numberOfObjectives)) {
        cout << "Error: could not read original CSR graph.\n";
        return false;
    }

    int numberOfNodes = static_cast<int>(baseGraph.size());
    baseGraph.clear();

    if (numberOfNodes == 0) {
        cout << "Error: graph has no vertices.\n";
        return false;
    }

    if (source < 0 || source >= numberOfNodes) {
        cout << "Error: source out of range.\n";
        return false;
    }

    vector<vector<int>> parents(K);
    for (int k = 0; k < K; ++k) {
        if (!readParentFile(treeInputPaths[k], parents[k], numberOfNodes)) {
            return false;
        }
    }

    map<pair<int, int>, int> edgeMembership;
    for (int k = 0; k < K; ++k) {
        for (int v = 0; v < numberOfNodes; ++v) {
            if (v == source) continue;
            int u = parents[k][v];
            if (u < 0 || u >= numberOfNodes) continue;
            edgeMembership[{u, v}] += 1;
        }
    }

    filesystem::create_directories(workDir);
    string combinedPrefix = workDir + "/combinedGraphCsr";

    if (!writeCombinedCSR(combinedPrefix, numberOfNodes, edgeMembership, K)) {
        cout << "Error: failed to write combined CSR.\n";
        return false;
    }

    HostCsrGraph outgoingCSR;
    HostCsrGraph incomingCSR;
    DeviceCsrGraph deviceIncomingCSR;

    if (!prepareGraphs(combinedPrefix, 0, outgoingCSR, incomingCSR, deviceIncomingCSR)) {
        cout << "Error: failed to prepare combined graph for CUDA.\n";
        return false;
    }

    vector<int> finalDistances;
    vector<int> finalParents;

    bool ok = runHybridSOSPUpdate(outgoingCSR,
                                  incomingCSR,
                                  deviceIncomingCSR,
                                  source,
                                  finalDistances,
                                  finalParents);

    freeDeviceCsr(deviceIncomingCSR);

    if (!ok) {
        cout << "Error: CUDA SOSP solve failed on combined graph.\n";
        return false;
    }

    if (!writeVectorToFile(distancesOutputPath, finalDistances)) {
        cout << "Error: failed to write combined distances.\n";
        return false;
    }

    if (!writeParentsToFile(treeOutputPath, finalParents)) {
        cout << "Error: failed to write combined parents.\n";
        return false;
    }

    cout << "cudaCombinedGraph completed.\n";
    cout << "Distances: " << distancesOutputPath << "\n";
    cout << "Parents:   " << treeOutputPath << "\n";

    return true;
}
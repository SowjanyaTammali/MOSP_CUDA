#include "../headers/cudaCombinedGraph.cuh"
#include "../headers/read.h"
#include "../headers/dijkstra.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};

bool readParentFile(const std::string& path, std::vector<int>& parents) {
    std::ifstream in(path);                                // open parent file
    if (!in.is_open()) {
        std::cerr << "Error: could not read parent file: " << path << "\n";
        return false;
    }

    parents.clear();                                       // reset output
    int vertex = 0;                                        // vertex id
    std::string token;                                     // parent token

    while (in >> vertex >> token) {
        if (vertex < 0) continue;                          // skip bad row
        if (vertex >= static_cast<int>(parents.size())) {
            parents.resize(vertex + 1, -1);                // grow if needed
        }

        if (token == "INF") parents[vertex] = -1;         // no parent
        else parents[vertex] = std::stoi(token);          // numeric parent
    }

    return true;
}

bool writeCombinedGraphCSR(const std::string& prefix,
                           int numVertices,
                           const std::vector<std::tuple<int, int, int>>& edges) {
    std::filesystem::path prefixPath(prefix);              // output prefix path
    if (!prefixPath.parent_path().empty()) {
        std::filesystem::create_directories(prefixPath.parent_path()); // make dir
    }

    std::vector<std::vector<std::pair<int, int>>> adj(numVertices); // u -> [(v,w)]

    for (const auto& e : edges) {
        int u = std::get<0>(e);                            // tail
        int v = std::get<1>(e);                            // head
        int w = std::get<2>(e);                            // combined weight
        if (u >= 0 && u < numVertices && v >= 0 && v < numVertices) {
            adj[u].push_back({v, w});                      // save edge
        }
    }

    std::ofstream rowPtrFile(prefix + "RowPtr.txt");       // repo-style CSR rowPtr
    std::ofstream colIndFile(prefix + "ColInd.txt");       // repo-style CSR colInd
    std::ofstream valuesFile(prefix + "Values.txt");       // repo-style CSR values

    if (!rowPtrFile.is_open() || !colIndFile.is_open() || !valuesFile.is_open()) {
        std::cerr << "Error: failed to write combined CSR files.\n";
        return false;
    }

    int edgeCounter = 0;                                   // running edge count
    rowPtrFile << 0 << "\n";                               // first rowPtr

    for (int u = 0; u < numVertices; ++u) {
        for (const auto& [v, w] : adj[u]) {
            colIndFile << v << "\n";                       // destination
            valuesFile << w << "\n";                       // single-objective weight
            ++edgeCounter;                                 // count edge
        }
        rowPtrFile << edgeCounter << "\n";                 // next rowPtr
    }

    return true;
}

} // namespace

bool cudaCombinedGraph(const std::string& originalCsrPrefix,
                       const std::vector<std::string>& objTreePaths,
                       int numberOfObjectives,
                       int sourceVertex,
                       const std::string& combinedGraphDir,
                       const std::string& distancesOutputPath,
                       const std::string& treeOutputPath) {
    Graph originalGraph;                                   // original graph
    int originalObjectives = 0;                            // old objective count

    if (!readCSR(originalCsrPrefix, originalGraph, originalObjectives)) {
        std::cerr << "Error: failed to read original CSR graph.\n";
        return false;
    }

    int numVertices = static_cast<int>(originalGraph.size()); // vertex count
    if (numVertices == 0) {
        std::cerr << "Error: original graph is empty.\n";
        return false;
    }

    std::unordered_map<std::pair<int, int>, int, PairHash> edgeFrequency; // edge counts

    for (const std::string& treePath : objTreePaths) {
        std::vector<int> parents;                          // one objective tree
        if (!readParentFile(treePath, parents)) {
            return false;
        }

        for (int v = 0; v < static_cast<int>(parents.size()); ++v) {
            int p = parents[v];                            // parent of v
            if (p >= 0) {
                edgeFrequency[{p, v}]++;                   // count tree usage
            }
        }
    }

    std::vector<std::tuple<int, int, int>> combinedEdges; // (u,v,w)
    combinedEdges.reserve(edgeFrequency.size());           // reserve once

    for (const auto& entry : edgeFrequency) {
        int u = entry.first.first;                         // tail
        int v = entry.first.second;                        // head
        int freq = entry.second;                           // number of trees
        int weight = numberOfObjectives + 1 - freq;       // OpenMP-style weight
        combinedEdges.push_back({u, v, weight});          // save edge
    }
        if (combinedEdges.empty()) {                           // no edges in combined graph
        std::filesystem::create_directories(combinedGraphDir);

        std::ofstream distOut(distancesOutputPath);        // write trivial distances
        std::ofstream treeOut(treeOutputPath);             // write trivial parents

        if (!distOut.is_open() || !treeOut.is_open()) {
            std::cerr << "Error: failed to write empty combined-graph outputs.\n";
            return false;
        }

        for (int v = 0; v < numVertices; ++v) {
            if (v == sourceVertex) distOut << v << " 0\n"; // source distance
            else distOut << v << " INF\n";                 // unreachable

            if (v == sourceVertex) treeOut << v << " -1\n"; // source parent
            else treeOut << v << " -1\n";                   // no parent
        }

        std::cout << "cudaCombinedGraph completed.\n";
        std::cout << "Distances: " << distancesOutputPath << "\n";
        std::cout << "Parents:   " << treeOutputPath << "\n";
        return true;
    }

    std::string combinedPrefix = combinedGraphDir + "/combinedGraphCsr"; // CSR prefix

    if (!writeCombinedGraphCSR(combinedPrefix, numVertices, combinedEdges)) {
        return false;
    }

    std::filesystem::path distOutPath(distancesOutputPath); // output dist path
    if (!distOutPath.parent_path().empty()) {
        std::filesystem::create_directories(distOutPath.parent_path()); // make dir
    }

    std::filesystem::path treeOutPath(treeOutputPath);      // output tree path
    if (!treeOutPath.parent_path().empty()) {
        std::filesystem::create_directories(treeOutPath.parent_path()); // make dir
    }

    if (!runDijkstraCSR(combinedPrefix,                     // solve combined graph
                        0,                                  // single objective
                        sourceVertex,
                        distancesOutputPath,
                        treeOutputPath)) {
        std::cerr << "Error: runDijkstraCSR failed on combined graph.\n";
        return false;
    }

    std::cout << "cudaCombinedGraph completed.\n";
    std::cout << "Distances: " << distancesOutputPath << "\n";
    std::cout << "Parents:   " << treeOutputPath << "\n";
    return true;
}
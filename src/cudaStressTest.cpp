#include "../headers/dijkstra.h"
#include "../headers/generateChangedEdges.h"
#include "../headers/generateGraphCSR.h"
#include "../headers/read.h"
#include "../headers/cuda_graph.cuh"
#include "../headers/cuda_sosp_update.cuh"
#include "../headers/updateGraphCSR.h"

#include <climits>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

/**
 * @file cudaStressTest.cpp
 * @brief Randomized stress test for the hybrid CUDA incremental SOSP update.
 */

namespace {

bool readDistanceFile(const string& path, vector<int>& distances) {
    ifstream in(path);
    if (!in.is_open()) return false;

    distances.clear();
    int vertex;
    string token;

    while (in >> vertex >> token) {
        if (vertex != static_cast<int>(distances.size())) {
            distances.resize(vertex + 1, INT_MAX);
        } else {
            distances.push_back(INT_MAX);
        }

        if (token == "INF") distances[vertex] = INT_MAX;
        else distances[vertex] = stoi(token);
    }

    return true;
}

bool readParentFile(const string& path, vector<int>& parents) {
    ifstream in(path);
    if (!in.is_open()) return false;

    parents.clear();
    int vertex, parent;

    while (in >> vertex >> parent) {
        if (vertex != static_cast<int>(parents.size())) {
            parents.resize(vertex + 1, -1);
        } else {
            parents.push_back(-1);
        }

        parents[vertex] = parent;
    }

    return true;
}

bool vectorsEqual(const vector<int>& a, const vector<int>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

bool loadInitialCandidatesFromChanges(const string& insertPath,
                                      const string& deletePath,
                                      vector<int>& initialCandidates) {
    vector<char> seen(100000, 0);
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
        ifstream in(insertPath);
        if (!in.is_open()) return false;

        string line;
        while (getline(in, line)) {
            if (line.empty()) continue;
            istringstream iss(line);
            int u, v;
            if (!(iss >> u >> v)) continue;
            addVertex(u);
            addVertex(v);
        }
    }

    {
        ifstream in(deletePath);
        if (!in.is_open()) return false;

        string line;
        while (getline(in, line)) {
            if (line.empty()) continue;
            istringstream iss(line);
            int u, v;
            if (!(iss >> u >> v)) continue;
            addVertex(u);
            addVertex(v);
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

void printVectorCompact(const string& name, const vector<int>& values) {
    cout << name << ": ";
    for (int x : values) {
        if (x == INT_MAX) cout << "INF ";
        else cout << x << " ";
    }
    cout << "\n";
}

} // namespace

int main() {
    const int totalRuns = 50;
    const string baseDir = "cudaStressTest";

    mt19937 rng(random_device{}());
    uniform_int_distribution<int> nodesDist(4, 30);
    uniform_int_distribution<int> objDist(1, 3);
    uniform_int_distribution<int> weightDist(1, 50);
    uniform_int_distribution<int> changeDist(1, 10);
    uniform_int_distribution<int> pctDist(0, 100);
    uniform_int_distribution<unsigned int> seedDist(1, 999999);

    int distancePassCount = 0;
    int fullPassCount = 0;
    int parentWarningCount = 0;
    int distanceFailCount = 0;
    int pipelineErrorCount = 0;

    for (int run = 0; run < totalRuns; ++run) {
        int numberOfNodes = nodesDist(rng);
        int minEdges = numberOfNodes - 1;
        int maxEdges = min(numberOfNodes * (numberOfNodes - 1), minEdges + 40);
        uniform_int_distribution<int> edgeDist(minEdges, maxEdges);
        int numberOfEdges = edgeDist(rng);

        int numberOfObjectives = objDist(rng);
        int objectiveEndRange = weightDist(rng);
        int objectiveIndex =
            uniform_int_distribution<int>(0, numberOfObjectives - 1)(rng);

        int numberOfChangedEdges = changeDist(rng);
        int insertPct = pctDist(rng);
        int deletePct = 100 - insertPct;

        unsigned int graphSeed = seedDist(rng);
        unsigned int changeSeed = seedDist(rng);

        string dir = baseDir + "/run" + to_string(run);
        string originalPrefix = dir + "/originalGraph/graphCsr";
        string updatedPrefix = dir + "/updatedGraph/updatedGraphCsr";
        string insertPath = dir + "/changedEdges/insert.txt";
        string deletePath = dir + "/changedEdges/delete.txt";
        string expectedDir = dir + "/expected";

        bool ok = true;

        ok = ok && generateGraphCSR(numberOfNodes, numberOfEdges, true,
                                    originalPrefix, numberOfObjectives,
                                    1, objectiveEndRange, graphSeed);

        ok = ok && generateChangedEdges(1, objectiveEndRange, numberOfObjectives,
                                        numberOfNodes, numberOfChangedEdges,
                                        insertPct, deletePct, true, true, true,
                                        false, originalPrefix, insertPath,
                                        deletePath, changeSeed);

        ok = ok && updateGraphCSR(originalPrefix, updatedPrefix,
                                  insertPath, deletePath, true);

        ok = ok && runDijkstraCSR(originalPrefix, objectiveIndex, 0,
                                  expectedDir + "/distancesOriginal.txt",
                                  expectedDir + "/SSSPTreeOriginal.txt");

        ok = ok && runDijkstraCSR(updatedPrefix, objectiveIndex, 0,
                                  expectedDir + "/distancesUpdated.txt",
                                  expectedDir + "/SSSPTreeUpdated.txt");

        if (!ok) {
            cout << "Run " << run << ": ERROR (pipeline setup failed)\n";
            ++pipelineErrorCount;
            continue;
        }

        vector<int> originalDistances;
        vector<int> originalParents;
        vector<int> expectedUpdatedDistances;
        vector<int> expectedUpdatedParents;

        ok = ok && readDistanceFile(expectedDir + "/distancesOriginal.txt", originalDistances);
        ok = ok && readParentFile(expectedDir + "/SSSPTreeOriginal.txt", originalParents);
        ok = ok && readDistanceFile(expectedDir + "/distancesUpdated.txt", expectedUpdatedDistances);
        ok = ok && readParentFile(expectedDir + "/SSSPTreeUpdated.txt", expectedUpdatedParents);

        if (!ok) {
            cout << "Run " << run << ": ERROR (failed to read baseline files)\n";
            ++pipelineErrorCount;
            continue;
        }

        HostCsrGraph updatedOutgoingCSR;
        HostCsrGraph updatedIncomingCSR;
        DeviceCsrGraph deviceUpdatedIncomingCSR;

        ok = ok && prepareGraphs(updatedPrefix, objectiveIndex,
                                 updatedOutgoingCSR, updatedIncomingCSR,
                                 deviceUpdatedIncomingCSR);

        if (!ok) {
            cout << "Run " << run << ": ERROR (failed to prepare CUDA graphs)\n";
            ++pipelineErrorCount;
            continue;
        }

        vector<int> initialCandidates;
        ok = ok && loadInitialCandidatesFromChanges(insertPath, deletePath, initialCandidates);

        if (!ok) {
            cout << "Run " << run << ": ERROR (failed to load initial candidates)\n";
            freeDeviceCsr(deviceUpdatedIncomingCSR);
            ++pipelineErrorCount;
            continue;
        }

        vector<int> cudaUpdatedDistances;
        vector<int> cudaUpdatedParents;

        ok = ok && runHybridIncrementalSOSPUpdate(updatedOutgoingCSR,
                                                  updatedIncomingCSR,
                                                  deviceUpdatedIncomingCSR,
                                                  originalDistances,
                                                  originalParents,
                                                  initialCandidates,
                                                  deletePath,
                                                  0,
                                                  cudaUpdatedDistances,
                                                  cudaUpdatedParents);

        freeDeviceCsr(deviceUpdatedIncomingCSR);

        if (!ok) {
            cout << "Run " << run << ": ERROR (CUDA incremental update failed)\n";
            ++pipelineErrorCount;
            continue;
        }

        bool distanceMatch = vectorsEqual(expectedUpdatedDistances, cudaUpdatedDistances);
        bool parentMatch = vectorsEqual(expectedUpdatedParents, cudaUpdatedParents);

        if (distanceMatch) {
            ++distancePassCount;
        }

        if (distanceMatch && parentMatch) {
            ++fullPassCount;
        } else if (distanceMatch && !parentMatch) {
            ++parentWarningCount;
            cout << "Run " << run << ": WARNING (parent mismatch only)"
                 << " (nodes=" << numberOfNodes
                 << " edges=" << numberOfEdges
                 << " objs=" << numberOfObjectives
                 << " objIdx=" << objectiveIndex
                 << " changes=" << numberOfChangedEdges
                 << " ins%=" << insertPct
                 << " graphSeed=" << graphSeed
                 << " changeSeed=" << changeSeed
                 << ")\n";
            printVectorCompact("  CPU Parent", expectedUpdatedParents);
            printVectorCompact("  GPU Parent", cudaUpdatedParents);
        } else {
            ++distanceFailCount;
            cout << "Run " << run << ": FAIL (distance mismatch)"
                 << " (nodes=" << numberOfNodes
                 << " edges=" << numberOfEdges
                 << " objs=" << numberOfObjectives
                 << " objIdx=" << objectiveIndex
                 << " changes=" << numberOfChangedEdges
                 << " ins%=" << insertPct
                 << " graphSeed=" << graphSeed
                 << " changeSeed=" << changeSeed
                 << ")\n";
            printVectorCompact("  CPU Dist", expectedUpdatedDistances);
            printVectorCompact("  GPU Dist", cudaUpdatedDistances);

            if (!parentMatch) {
                printVectorCompact("  CPU Parent", expectedUpdatedParents);
                printVectorCompact("  GPU Parent", cudaUpdatedParents);
            }
        }
    }

    cout << "\n=== CUDA Stress Test Summary ===\n";
    cout << "Distance matches      : " << distancePassCount << "/" << totalRuns << "\n";
    cout << "Full matches          : " << fullPassCount << "/" << totalRuns << "\n";
    cout << "Parent-only warnings  : " << parentWarningCount << "/" << totalRuns << "\n";
    cout << "Distance failures     : " << distanceFailCount << "/" << totalRuns << "\n";
    cout << "Pipeline errors       : " << pipelineErrorCount << "/" << totalRuns << "\n";

    return (distanceFailCount > 0 || pipelineErrorCount > 0) ? 1 : 0;
}
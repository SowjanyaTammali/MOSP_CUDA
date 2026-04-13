#include "../headers/dijkstra.h"
#include "../headers/generateChangedEdges.h"
#include "../headers/generateGraphCSR.h"
#include "../headers/cudaCombinedGraph.cuh"
#include "../headers/cudaParallelSOSPUpdate.cuh"
#include "../headers/updateGraphCSR.h"
#include "../headers/sequentialSOSPUpdate.h"

#include <algorithm>
#include <climits>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

namespace {

bool readDistanceFile(const string& path, vector<int>& distances) {
    ifstream in(path);                                      // open file
    if (!in.is_open()) return false;                       // fail if missing

    distances.clear();                                     // reset output
    int vertex;                                            // vertex id
    string token;                                          // distance token

    while (in >> vertex >> token) {
        if (vertex < 0) continue;                          // skip bad rows
        if (vertex >= static_cast<int>(distances.size())) {
            distances.resize(vertex + 1, INT_MAX);         // grow vector
        }

        if (token == "INF") distances[vertex] = INT_MAX;   // unreachable
        else distances[vertex] = stoi(token);              // normal value
    }

    return true;
}

bool readParentFile(const string& path, vector<int>& parents) {
    ifstream in(path);                                     // open file
    if (!in.is_open()) return false;                      // fail if missing

    parents.clear();                                       // reset output
    int vertex;                                            // vertex id
    int parent;                                            // parent id

    while (in >> vertex >> parent) {
        if (vertex < 0) continue;                          // skip bad rows
        if (vertex >= static_cast<int>(parents.size())) {
            parents.resize(vertex + 1, -1);                // grow vector
        }

        parents[vertex] = parent;                          // store parent
    }

    return true;
}

bool vectorsEqual(const vector<int>& a, const vector<int>& b) {
    if (a.size() != b.size()) return false;                // size mismatch
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return false;                    // value mismatch
    }
    return true;                                           // exact match
}

void printVectorCompact(const string& name, const vector<int>& values) {
    cout << name << ": ";                                  // label
    for (int x : values) {
        if (x == INT_MAX) cout << "INF ";
        else cout << x << " ";
    }
    cout << "\n";
}

bool compareDistanceAndParentFiles(const string& expectedDistPath,
                                   const string& expectedParentPath,
                                   const string& actualDistPath,
                                   const string& actualParentPath,
                                   bool& distanceMatch,
                                   bool& parentMatch,
                                   vector<int>& expectedDistances,
                                   vector<int>& expectedParents,
                                   vector<int>& actualDistances,
                                   vector<int>& actualParents) {
    if (!readDistanceFile(expectedDistPath, expectedDistances)) return false;
    if (!readParentFile(expectedParentPath, expectedParents)) return false;
    if (!readDistanceFile(actualDistPath, actualDistances)) return false;
    if (!readParentFile(actualParentPath, actualParents)) return false;

    distanceMatch = vectorsEqual(expectedDistances, actualDistances); // compare dist
    parentMatch = vectorsEqual(expectedParents, actualParents);       // compare tree
    return true;
}

} // namespace

int main() {
    const int totalRuns = 50;                               // number of random cases
    const string baseDir = "cudaStressTest";                // output root
    const int source = 0;                                   // fixed source
    const bool directed = true;                             // directed graph

    mt19937 rng(random_device{}());                         // random generator
    uniform_int_distribution<int> nodesDist(4, 30);        // graph size
    uniform_int_distribution<int> objDist(1, 3);           // objectives
    uniform_int_distribution<int> weightDist(1, 50);       // max weight
    uniform_int_distribution<int> changeDist(1, 10);       // number of changed edges
    uniform_int_distribution<int> pctDist(0, 100);         // insert %
    uniform_int_distribution<unsigned int> seedDist(1, 999999); // random seeds

    int objectiveDistancePassCount = 0;                     // per-objective distance passes
    int objectiveFullPassCount = 0;                         // per-objective full passes
    int objectiveParentWarningCount = 0;                    // parent-only warnings
    int objectiveDistanceFailCount = 0;                     // per-objective distance failures
    int combinedPassCount = 0;                              // combined graph passes
    int combinedFailCount = 0;                              // combined graph failures
    int pipelineErrorCount = 0;                             // setup/runtime errors
    int totalObjectiveChecks = 0;                           // total objective checks

    for (int run = 0; run < totalRuns; ++run) {
        int numberOfNodes = nodesDist(rng);                 // random graph size
        int minEdges = numberOfNodes - 1;                   // keep graph reasonable
        int maxEdges = min(numberOfNodes * (numberOfNodes - 1), minEdges + 40);
        uniform_int_distribution<int> edgeDist(minEdges, maxEdges);
        int numberOfEdges = edgeDist(rng);                  // random edge count

        int numberOfObjectives = objDist(rng);              // random objectives
        int objectiveEndRange = weightDist(rng);            // random max weight
        int numberOfChangedEdges = changeDist(rng);         // random changes
        int insertPct = pctDist(rng);                       // insertion %
        int deletePct = 100 - insertPct;                    // deletion %

        unsigned int graphSeed = seedDist(rng);             // graph seed
        unsigned int changeSeed = seedDist(rng);            // change seed

        string dir = baseDir + "/run" + to_string(run);     // per-run dir
        string originalPrefix = dir + "/originalGraph/graphCsr";
        string updatedPrefix = dir + "/updatedGraph/updatedGraphCsr";
        string insertPath = dir + "/changedEdges/insert.txt";
        string deletePath = dir + "/changedEdges/delete.txt";
        string expectedDir = dir + "/expected";
        string cudaDir = dir + "/cuda";
        string expectedCombinedDir = dir + "/expectedCombined";
        string cudaCombinedDir = dir + "/cudaCombined";

        filesystem::create_directories(dir + "/originalGraph");   // make dirs
        filesystem::create_directories(dir + "/updatedGraph");
        filesystem::create_directories(dir + "/changedEdges");
        filesystem::create_directories(expectedDir);
        filesystem::create_directories(cudaDir);
        filesystem::create_directories(expectedCombinedDir);
        filesystem::create_directories(cudaCombinedDir);

        bool ok = true;                                     // pipeline status

        ok = ok && generateGraphCSR(numberOfNodes, numberOfEdges, directed,
                                    originalPrefix, numberOfObjectives,
                                    1, objectiveEndRange, graphSeed);

        ok = ok && generateChangedEdges(1, objectiveEndRange, numberOfObjectives,
                                        numberOfNodes, numberOfChangedEdges,
                                        insertPct, deletePct, directed,
                                        true, true, false,
                                        originalPrefix, insertPath,
                                        deletePath, changeSeed);

        ok = ok && updateGraphCSR(originalPrefix, updatedPrefix,
                                  insertPath, deletePath, directed);

        if (!ok) {
            cout << "Run " << run << ": ERROR (pipeline setup failed)\n";
            ++pipelineErrorCount;
            continue;
        }

        vector<string> expectedTreePaths(numberOfObjectives); // expected updated trees
        vector<string> cudaTreePaths(numberOfObjectives);     // CUDA updated trees
        bool runHasObjectiveDistanceFail = false;             // track objective failure
        bool runHasPipelineError = false;                    // track per-run error

        for (int obj = 0; obj < numberOfObjectives; ++obj) {
            ++totalObjectiveChecks;                          // count this objective

            string expectedObjDir = expectedDir + "/obj" + to_string(obj);
            string cudaObjDir = cudaDir + "/obj" + to_string(obj);
            filesystem::create_directories(expectedObjDir);  // make dirs
            filesystem::create_directories(cudaObjDir);

            string originalDistPath = expectedObjDir + "/distancesOriginal.txt";
            string originalTreePath = expectedObjDir + "/SSSPTreeOriginal.txt";
            string expectedUpdatedDistPath = expectedObjDir + "/distancesUpdated.txt";
            string expectedUpdatedTreePath = expectedObjDir + "/SSSPTreeUpdated.txt";
            string cudaUpdatedDistPath = cudaObjDir + "/distancesUpdated.txt";
            string cudaUpdatedTreePath = cudaObjDir + "/SSSPTreeUpdated.txt";

            ok = true;                                      // reset for this objective

            ok = ok && runDijkstraCSR(originalPrefix, obj, source,  // original baseline
                                      originalDistPath,
                                      originalTreePath);

            ok = ok && sequentialSOSPUpdate(originalPrefix,          // CPU incremental baseline
                                originalDistPath,
                                originalTreePath,
                                insertPath,
                                deletePath,
                                obj,
                                source,
                                expectedUpdatedDistPath,
                                expectedUpdatedTreePath);

            ok = ok && cudaParallelSOSPUpdate(updatedPrefix,        // public CUDA path
                                              originalDistPath,
                                              originalTreePath,
                                              insertPath,
                                              deletePath,
                                              obj,
                                              source,
                                              cudaUpdatedDistPath,
                                              cudaUpdatedTreePath);

            if (!ok) {
                cout << "Run " << run << ", obj " << obj
                     << ": ERROR (objective pipeline failed)\n";
                ++pipelineErrorCount;
                runHasPipelineError = true;
                break;
            }

            expectedTreePaths[obj] = expectedUpdatedTreePath;       // keep expected tree
            cudaTreePaths[obj] = cudaUpdatedTreePath;               // keep CUDA tree

            bool distanceMatch = false;                             // compare results
            bool parentMatch = false;
            vector<int> expectedDistances;
            vector<int> expectedParents;
            vector<int> actualDistances;
            vector<int> actualParents;

            ok = compareDistanceAndParentFiles(expectedUpdatedDistPath,
                                               expectedUpdatedTreePath,
                                               cudaUpdatedDistPath,
                                               cudaUpdatedTreePath,
                                               distanceMatch,
                                               parentMatch,
                                               expectedDistances,
                                               expectedParents,
                                               actualDistances,
                                               actualParents);

            if (!ok) {
                cout << "Run " << run << ", obj " << obj
                     << ": ERROR (failed to read objective outputs)\n";
                ++pipelineErrorCount;
                runHasPipelineError = true;
                break;
            }

            if (distanceMatch) ++objectiveDistancePassCount;        // count distance pass

            if (distanceMatch && parentMatch) {
                ++objectiveFullPassCount;                           // exact objective match
            } else if (distanceMatch && !parentMatch) {
                ++objectiveParentWarningCount;                     // only parent differs
                cout << "Run " << run << ", obj " << obj
                     << ": WARNING (parent mismatch only)"
                     << " (nodes=" << numberOfNodes
                     << " edges=" << numberOfEdges
                     << " objs=" << numberOfObjectives
                     << " changes=" << numberOfChangedEdges
                     << " ins%=" << insertPct
                     << " graphSeed=" << graphSeed
                     << " changeSeed=" << changeSeed
                     << ")\n";
                printVectorCompact("  CPU Parent", expectedParents);
                printVectorCompact("  GPU Parent", actualParents);
            } else {
                ++objectiveDistanceFailCount;                      // real objective failure
                runHasObjectiveDistanceFail = true;
                cout << "Run " << run << ", obj " << obj
                     << ": FAIL (distance mismatch)"
                     << " (nodes=" << numberOfNodes
                     << " edges=" << numberOfEdges
                     << " objs=" << numberOfObjectives
                     << " changes=" << numberOfChangedEdges
                     << " ins%=" << insertPct
                     << " graphSeed=" << graphSeed
                     << " changeSeed=" << changeSeed
                     << ")\n";
                printVectorCompact("  CPU Dist", expectedDistances);
                printVectorCompact("  GPU Dist", actualDistances);

                if (!parentMatch) {
                    printVectorCompact("  CPU Parent", expectedParents);
                    printVectorCompact("  GPU Parent", actualParents);
                }
            }
        }

        if (runHasPipelineError) continue;                         // skip combined check
        if (runHasObjectiveDistanceFail) {
            ++combinedFailCount;                                   // combined run already not valid
            continue;
        }

        bool combinedOk = true;                                    // combined stage

        combinedOk = combinedOk && cudaCombinedGraph(originalPrefix,
                                                     expectedTreePaths,
                                                     numberOfObjectives,
                                                     source,
                                                     expectedCombinedDir,
                                                     expectedCombinedDir + "/distancesCsr.txt",
                                                     expectedCombinedDir + "/SSSPTreeCsr.txt");

        combinedOk = combinedOk && cudaCombinedGraph(originalPrefix,
                                                     cudaTreePaths,
                                                     numberOfObjectives,
                                                     source,
                                                     cudaCombinedDir,
                                                     cudaCombinedDir + "/distancesCsr.txt",
                                                     cudaCombinedDir + "/SSSPTreeCsr.txt");

        if (!combinedOk) {
            cout << "Run " << run << ": ERROR (combined graph stage failed)\n";
            ++pipelineErrorCount;
            continue;
        }

        bool combinedDistanceMatch = false;                        // compare combined outputs
        bool combinedParentMatch = false;
        vector<int> expectedCombinedDistances;
        vector<int> expectedCombinedParents;
        vector<int> actualCombinedDistances;
        vector<int> actualCombinedParents;

        combinedOk = compareDistanceAndParentFiles(expectedCombinedDir + "/distancesCsr.txt",
                                                   expectedCombinedDir + "/SSSPTreeCsr.txt",
                                                   cudaCombinedDir + "/distancesCsr.txt",
                                                   cudaCombinedDir + "/SSSPTreeCsr.txt",
                                                   combinedDistanceMatch,
                                                   combinedParentMatch,
                                                   expectedCombinedDistances,
                                                   expectedCombinedParents,
                                                   actualCombinedDistances,
                                                   actualCombinedParents);

        if (!combinedOk) {
            cout << "Run " << run << ": ERROR (failed to read combined outputs)\n";
            ++pipelineErrorCount;
            continue;
        }

        if (combinedDistanceMatch && combinedParentMatch) {
            ++combinedPassCount;                                   // final MOSP pass
        } else {
            ++combinedFailCount;                                   // final MOSP fail
            cout << "Run " << run << ": FAIL (combined graph mismatch)"
                 << " (nodes=" << numberOfNodes
                 << " edges=" << numberOfEdges
                 << " objs=" << numberOfObjectives
                 << " changes=" << numberOfChangedEdges
                 << " ins%=" << insertPct
                 << " graphSeed=" << graphSeed
                 << " changeSeed=" << changeSeed
                 << ")\n";
            printVectorCompact("  EXP Combined Dist", expectedCombinedDistances);
            printVectorCompact("  CUDA Combined Dist", actualCombinedDistances);

            if (!combinedParentMatch) {
                printVectorCompact("  EXP Combined Parent", expectedCombinedParents);
                printVectorCompact("  CUDA Combined Parent", actualCombinedParents);
            }
        }
    }

    cout << "\n=== CUDA Stress Test Summary ===\n";
    cout << "Objective distance matches : " << objectiveDistancePassCount
         << "/" << totalObjectiveChecks << "\n";
    cout << "Objective full matches     : " << objectiveFullPassCount
         << "/" << totalObjectiveChecks << "\n";
    cout << "Parent-only warnings       : " << objectiveParentWarningCount
         << "/" << totalObjectiveChecks << "\n";
    cout << "Objective distance fails   : " << objectiveDistanceFailCount
         << "/" << totalObjectiveChecks << "\n";
    cout << "Combined graph passes      : " << combinedPassCount
         << "/" << totalRuns << "\n";
    cout << "Combined graph fails       : " << combinedFailCount
         << "/" << totalRuns << "\n";
    cout << "Pipeline errors            : " << pipelineErrorCount
         << "/" << totalRuns << "\n";

    return (objectiveDistanceFailCount > 0 ||
            combinedFailCount > 0 ||
            pipelineErrorCount > 0) ? 1 : 0;
}
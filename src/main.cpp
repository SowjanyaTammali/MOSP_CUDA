#include <climits>      // INT_MAX
#include <filesystem>   // folders
#include <fstream>      // file io
#include <iostream>     // cout/cerr
#include <string>       // strings
#include <vector>       // vector

#include "../headers/read.h"                    // readCSR
#include "../headers/dijkstra.h"                // runDijkstraCSR
#include "../headers/generateChangedEdges.h"    // generate insert/delete files
#include "../headers/generateGraphCSR.h"        // build original CSR graph
#include "../headers/cudaCombinedGraph.cuh"     // combine objective trees
#include "../headers/cudaParallelSOSPUpdate.cuh"// CUDA incremental update
#include "../headers/sequentialSOSPUpdate.h"    // CPU incremental update
#include "../headers/updateGraphCSR.h"          // build updated CSR graph

using namespace std;

namespace {

bool readDistanceFile(const string& path, vector<int>& distances) {
    ifstream in(path);                                           // open file
    if (!in.is_open()) return false;                             // fail if missing

    distances.clear();                                           // reset output
    int vertex = 0;                                              // vertex id
    string token;                                                // distance token

    while (in >> vertex >> token) {
        if (vertex < 0) continue;                                // skip bad row
        if (vertex >= static_cast<int>(distances.size())) {
            distances.resize(vertex + 1, INT_MAX);               // grow vector
        }

        if (token == "INF") distances[vertex] = INT_MAX;         // unreachable
        else distances[vertex] = stoi(token);                    // normal value
    }

    return true;
}

bool distanceFilesMatch(const string& pathA, const string& pathB) {
    vector<int> a, b;                                            // two distance arrays
    if (!readDistanceFile(pathA, a)) return false;               // read A
    if (!readDistanceFile(pathB, b)) return false;               // read B
    return a == b;                                               // exact distance match
}

bool runPipelineFromFiles(const string& originalGraphPrefix,
                          const string& updatedGraphPrefix,
                          const string& insertPath,
                          const string& deletePath,
                          int source) {
    Graph graph;                                                 // read original graph
    int numberOfObjectives = 0;                                  // objective count
    if (!readCSR(originalGraphPrefix, graph, numberOfObjectives)) {
        cerr << "Error: failed to read original graph.\n";
        return false;
    }

    if (numberOfObjectives <= 0) {
        cerr << "Error: invalid number of objectives.\n";
        return false;
    }

    const string distancesTreesDir = "output/distancesTrees";                // original outputs
    const string updatedDistancesTreesDir = "output/updatedDistancesTrees";  // updated outputs
    const string sospUpdateDir = "output/sospUpdateDistancesTrees";          // CPU update outputs

    filesystem::create_directories("output");                                // root output
    filesystem::create_directories(distancesTreesDir);                       // original baseline dir
    filesystem::create_directories(updatedDistancesTreesDir);                // updated baseline dir
    filesystem::create_directories(sospUpdateDir);                           // CPU baseline dir
    filesystem::create_directories("output/combinedGraph");                  // combined graph dir

    const int objectiveNumber = 0;                                           // OpenMP-style baseline objective

    if (!runDijkstraCSR(originalGraphPrefix, objectiveNumber, source,        // original baseline
                        distancesTreesDir + "/distancesCsr.txt",
                        distancesTreesDir + "/SSSPTreeCsr.txt")) {
        return false;
    }

    if (!runDijkstraCSR(updatedGraphPrefix, objectiveNumber, source,         // updated baseline
                        updatedDistancesTreesDir + "/updatedDistancesCsr.txt",
                        updatedDistancesTreesDir + "/updatedSSSPTreeCsr.txt")) {
        return false;
    }

    if (!sequentialSOSPUpdate(originalGraphPrefix,                           // CPU SOSP baseline
                              distancesTreesDir + "/distancesCsr.txt",
                              distancesTreesDir + "/SSSPTreeCsr.txt",
                              insertPath,
                              deletePath,
                              objectiveNumber,
                              source,
                              sospUpdateDir + "/distancesCsr.txt",
                              sospUpdateDir + "/SSSPTreeCsr.txt")) {
        return false;
    }

    vector<string> objTreePaths(numberOfObjectives);                         // one tree per objective

    for (int obj = 0; obj < numberOfObjectives; ++obj) {
        const string objDir = "output/parallelSospObj" + to_string(obj);     // match OpenMP layout
        filesystem::create_directories(objDir);                              // make per-objective dir

        const string objDistOriginal = objDir + "/distancesOriginal.txt";    // original dist
        const string objTreeOriginal = objDir + "/SSSPTreeOriginal.txt";     // original tree
        const string objDistOut = objDir + "/distancesUpdated.txt";          // updated dist
        const string objTreeOut = objDir + "/SSSPTreeUpdated.txt";           // updated tree

        if (!runDijkstraCSR(originalGraphPrefix, obj, source,                // original result
                            objDistOriginal, objTreeOriginal)) {
            return false;
        }

        if (!cudaParallelSOSPUpdate(originalGraphPrefix,                     // OpenMP-style call
                                    objDistOriginal,
                                    objTreeOriginal,
                                    insertPath,
                                    deletePath,
                                    obj,
                                    source,
                                    objDistOut,
                                    objTreeOut)) {
            return false;
        }

        objTreePaths[obj] = objTreeOut;                                      // store updated tree
    }

    const string combinedGraphDir = "output/combinedGraph";                  // combined graph dir

    if (!cudaCombinedGraph(originalGraphPrefix,
                           objTreePaths,
                           numberOfObjectives,
                           source,
                           combinedGraphDir,
                           combinedGraphDir + "/distancesCsr.txt",
                           combinedGraphDir + "/SSSPTreeCsr.txt")) {
        return false;
    }

    return true;                                                             // success
}

bool generateDeterministicTestCases() {
    const int totalTests = 10;                                               // OpenMP-style count
    int passCount = 0;                                                       // passing tests

    for (int testId = 0; testId < totalTests; ++testId) {
        const int numberOfNodes = 5 + testId;                                // vary graph size
        const int maxEdges = numberOfNodes * (numberOfNodes - 1);            // directed max edges
        const int numberOfEdges = min(maxEdges, numberOfNodes + 4 + testId); // vary edge count
        const bool directed = true;                                          // directed graph
        const int numberOfObjectives = 1 + (testId % 3);                     // 1..3 objectives
        const int objectiveStartRange = 1;                                   // min edge weight
        const int objectiveEndRange = 9;                                     // max edge weight
        const int objectiveIndex = testId % numberOfObjectives;              // rotate objective
        const int source = 0;                                                // fixed source
        const int numberOfChangedEdges = 1 + (testId % 5);                   // vary changes
        const double insertionPercentage = 20.0 + (testId * 7) % 61;         // 20..80
        const double deletionPercentage = 100.0 - insertionPercentage;       // remaining delete %
        const unsigned int graphSeed = 1000 + testId;                        // deterministic graph seed
        const unsigned int changeSeed = 2000 + testId;                       // deterministic change seed

        const string testDir = "tests/testCase" + to_string(testId);         // test root
        const string originalPrefix = testDir + "/originalGraph/graphCsr";   // original CSR prefix
        const string updatedPrefix = testDir + "/updatedGraph/updatedGraphCsr"; // updated CSR prefix
        const string insertPath = testDir + "/changedEdges/insert.txt";      // insert file
        const string deletePath = testDir + "/changedEdges/delete.txt";      // delete file
        const string expectedDir = testDir + "/expected";                    // expected outputs
        const string cudaDir = testDir + "/cuda";                            // CUDA outputs

        filesystem::create_directories(testDir + "/originalGraph");          // make dirs
        filesystem::create_directories(testDir + "/updatedGraph");
        filesystem::create_directories(testDir + "/changedEdges");
        filesystem::create_directories(expectedDir);
        filesystem::create_directories(cudaDir);

        bool ok = true;                                                      // pipeline state

        ok = ok && generateGraphCSR(numberOfNodes, numberOfEdges, directed,
                                    originalPrefix, numberOfObjectives,
                                    objectiveStartRange, objectiveEndRange,
                                    graphSeed);

        ok = ok && generateChangedEdges(objectiveStartRange, objectiveEndRange,
                                        numberOfObjectives, numberOfNodes,
                                        numberOfChangedEdges, insertionPercentage,
                                        deletionPercentage, directed,
                                        true, true, false,
                                        originalPrefix, insertPath, deletePath,
                                        changeSeed);

        ok = ok && updateGraphCSR(originalPrefix, updatedPrefix,
                                  insertPath, deletePath, directed);

        ok = ok && runDijkstraCSR(originalPrefix, objectiveIndex, source,
                                  expectedDir + "/distancesOriginal.txt",
                                  expectedDir + "/SSSPTreeOriginal.txt");

        ok = ok && runDijkstraCSR(updatedPrefix, objectiveIndex, source,
                                  expectedDir + "/distancesUpdated.txt",
                                  expectedDir + "/SSSPTreeUpdated.txt");

        ok = ok && sequentialSOSPUpdate(originalPrefix,
                                        expectedDir + "/distancesOriginal.txt",
                                        expectedDir + "/SSSPTreeOriginal.txt",
                                        insertPath, deletePath,
                                        objectiveIndex, source,
                                        expectedDir + "/distancesSOSP.txt",
                                        expectedDir + "/SSSPTreeSOSP.txt");

        ok = ok && cudaParallelSOSPUpdate(originalPrefix,
                                          expectedDir + "/distancesOriginal.txt",
                                          expectedDir + "/SSSPTreeOriginal.txt",
                                          insertPath, deletePath,
                                          objectiveIndex, source,
                                          cudaDir + "/distancesUpdated.txt",
                                          cudaDir + "/SSSPTreeUpdated.txt");

        if (!ok) {
            cout << "Generating test case " << testId << "...\n";
            cout << "Test case " << testId << ": ERROR (pipeline failed)\n";
            continue;
        }

        const bool distanceMatch =
            distanceFilesMatch(expectedDir + "/distancesUpdated.txt",
                               cudaDir + "/distancesUpdated.txt");           // compare against ground truth

        cout << "Generating test case " << testId << "...\n";

        if (distanceMatch) {
            ++passCount;                                                     // count pass
            cout << "Test case " << testId
                 << ": PASS (distances match ground truth)\n";
        } else {
            cout << "Test case " << testId
                 << ": FAIL (distances do not match ground truth)\n";
        }
    }

    cout << "\n=== Test Summary: " << passCount << "/" << totalTests
         << " passed ===\n";

    return passCount == totalTests;                                          // final status
}

} // namespace

int main(int argc, char* argv[]) {
    // Optional shared-input mode:
    // ./bin/main workflow <originalPrefix> <updatedPrefix> <insertPath> <deletePath> <source>

    if (argc == 7 && string(argv[1]) == "workflow") {
        const string originalGraphPrefix = argv[2];                          // external original CSR prefix
        const string updatedGraphPrefix = argv[3];                           // external updated CSR prefix
        const string insertPath = argv[4];                                   // external insert file
        const string deletePath = argv[5];                                   // external delete file
        const int source = stoi(argv[6]);                                    // external source

        return runPipelineFromFiles(originalGraphPrefix,
                                    updatedGraphPrefix,
                                    insertPath,
                                    deletePath,
                                    source) ? 0 : 1;
    }

    if (argc != 1) {
        cerr << "Usage:\n";
        cerr << "  ./bin/main\n";
        cerr << "  ./bin/main workflow <originalPrefix> <updatedPrefix> "
                "<insertPath> <deletePath> <source>\n";
        return 1;
    }

    int numberOfNodes = 5;                                                   // small test graph
    int numberOfEdges = 6;                                                   // small test graph
    bool directed = true;                                                    // directed graph
    int numberOfObjectives = 3;                                              // K objectives
    int objectiveStartRange = 1;                                             // min edge weight
    int objectiveEndRange = 9;                                               // max edge weight
    int source = 0;                                                          // source node
    int numberOfChangedEdges = 4;                                            // changed edges
    double insertionPercentage = 50.0;                                       // insert %
    double deletionPercentage = 50.0;                                        // delete %

    const string originalGraphPrefix = "data/originalGraph/graphCsr";        // original CSR
    const string updatedGraphPrefix = "data/updatedGraph/updatedGraphCsr";   // updated CSR
    const string insertPath = "output/changedEdges/insert.txt";              // insert file
    const string deletePath = "output/changedEdges/delete.txt";              // delete file

    filesystem::create_directories("output");                                // root output dir
    filesystem::create_directories("output/changedEdges");                   // changed-edge dir

    if (!generateGraphCSR(numberOfNodes, numberOfEdges, directed,
                          originalGraphPrefix, numberOfObjectives,
                          objectiveStartRange, objectiveEndRange)) {
        return 1;
    }

    if (!generateChangedEdges(objectiveStartRange, objectiveEndRange,
                              numberOfObjectives, numberOfNodes,
                              numberOfChangedEdges, insertionPercentage,
                              deletionPercentage, directed, true, true, false,
                              originalGraphPrefix, insertPath, deletePath)) {
        return 1;
    }

    if (!updateGraphCSR(originalGraphPrefix, updatedGraphPrefix,
                        insertPath, deletePath, directed)) {
        return 1;
    }

    if (!runPipelineFromFiles(originalGraphPrefix,
                              updatedGraphPrefix,
                              insertPath,
                              deletePath,
                              source)) {
        return 1;
    }

    if (!generateDeterministicTestCases()) {                                // OpenMP-style test block
        return 1;
    }

    return 0;                                                               // success
}
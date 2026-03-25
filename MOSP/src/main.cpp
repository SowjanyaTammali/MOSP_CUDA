#include <string>

#include "csr.h"
#include "cuda_print.h"
#include "dijkstra.h"
#include "generateGraph.h"
#include "read.h"

using namespace std;

int main() {
    int numberOfNodes = 5;
    int numberOfEdges = 6;
    bool directed = true;
    string outputFile = "data/graph.mtx";
    int numberOfObjectives = 3;
    int objectiveStartRange = 1;
    int objectiveEndRange = 9;
    int objectiveNumber = 0;
    int source = 0;

    // Step 1: generate graph file
    if (!generateGraph(
            numberOfNodes,
            numberOfEdges,
            directed,
            outputFile,
            numberOfObjectives,
            objectiveStartRange,
            objectiveEndRange)) {
        return 1;
    }

    // Step 2: read graph into adjacency-list structure
    Graph graph;
    int detectedObjectives = 0;
    if (!readMtx(outputFile, graph, detectedObjectives)) {
        return 1;
    }

    // Step 3: convert to CSR
    CSRGraph csr;
    if (!buildCSR(graph, detectedObjectives, csr)) {
        return 1;
    }

    // Step 4: print CSR on CPU
    printCSR(csr);

    // Step 5: print CSR on GPU
    if (!printCSRFromGPU(csr)) {
        return 1;
    }

    // Step 6: still run CPU Dijkstra for now
    bool ok = runDijkstra(outputFile, objectiveNumber, source);
    return ok ? 0 : 1;
}
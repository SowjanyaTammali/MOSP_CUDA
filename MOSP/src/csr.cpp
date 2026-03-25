#include "csr.h"

#include <iostream>

using namespace std;

bool buildCSR(const Graph &graph, int numberOfObjectives, CSRGraph &csr) {
    if (numberOfObjectives <= 0) {
        cout << "Error: numberOfObjectives must be positive.\n";
        return false;
    }

    int numNodes = static_cast<int>(graph.size());
    int numEdges = 0;

    for (const auto &adjList : graph) {
        numEdges += static_cast<int>(adjList.size());
    }

    csr.numNodes = numNodes;
    csr.numEdges = numEdges;
    csr.numObjectives = numberOfObjectives;

    csr.row_ptr.assign(numNodes + 1, 0);
    csr.col_ind.assign(numEdges, 0);
    csr.weights.assign(numEdges * numberOfObjectives, 0);

    for (int u = 0; u < numNodes; ++u) {
        csr.row_ptr[u + 1] = csr.row_ptr[u] + static_cast<int>(graph[u].size());
    }

    int edgeIndex = 0;
    for (int u = 0; u < numNodes; ++u) {
        for (const auto &edge : graph[u]) {
            csr.col_ind[edgeIndex] = edge.to;

            if (static_cast<int>(edge.weights.size()) != numberOfObjectives) {
                cout << "Error: inconsistent objective count at node " << u << "\n";
                return false;
            }

            for (int k = 0; k < numberOfObjectives; ++k) {
                csr.weights[edgeIndex * numberOfObjectives + k] = edge.weights[k];
            }

            ++edgeIndex;
        }
    }

    return true;
}

void printCSR(const CSRGraph &csr) {
    cout << "CSR Graph:\n";

    cout << "row_ptr: ";
    for (int x : csr.row_ptr) {
        cout << x << " ";
    }
    cout << "\n";

    cout << "col_ind: ";
    for (int x : csr.col_ind) {
        cout << x << " ";
    }
    cout << "\n";

    cout << "weights (flattened): ";
    for (int x : csr.weights) {
        cout << x << " ";
    }
    cout << "\n\n";

    cout << "Graph from CSR:\n";
    for (int u = 0; u < csr.numNodes; ++u) {
        cout << "Node " << u << " -> ";
        for (int e = csr.row_ptr[u]; e < csr.row_ptr[u + 1]; ++e) {
            cout << "(" << csr.col_ind[e] << ", w=[";
            for (int k = 0; k < csr.numObjectives; ++k) {
                cout << csr.weights[e * csr.numObjectives + k];
                if (k + 1 < csr.numObjectives) {
                    cout << ", ";
                }
            }
            cout << "]) ";
        }
        cout << "\n";
    }
}
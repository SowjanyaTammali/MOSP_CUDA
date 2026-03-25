#ifndef CSR_H
#define CSR_H

#include "read.h"
#include <vector>

/**
 * @brief CSR graph structure for CUDA-friendly storage.
 *
 * weights is flattened:
 * weights[e * numObjectives + k]
 * = weight of edge e for objective k
 */
struct CSRGraph {
    int numNodes = 0;
    int numEdges = 0;
    int numObjectives = 0;

    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    std::vector<int> weights;
};

/**
 * @brief Convert adjacency-list graph to CSR.
 *
 * @param graph Input adjacency-list graph.
 * @param numberOfObjectives Number of objectives per edge.
 * @param csr Output CSR graph.
 * @return True on success, false otherwise.
 */
bool buildCSR(const Graph &graph, int numberOfObjectives, CSRGraph &csr);

/**
 * @brief Print CSR graph on CPU.
 *
 * @param csr CSR graph.
 */
void printCSR(const CSRGraph &csr);

#endif
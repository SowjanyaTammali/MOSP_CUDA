#ifndef CUDA_COMBINED_GRAPH_CUH
#define CUDA_COMBINED_GRAPH_CUH

#include <string>
#include <vector>

/**
 * @file cudaCombinedGraph.cuh
 * @brief Declares hybrid CUDA combined-graph construction and solve.
 */

/**
 * @brief Builds a combined graph from K SSSP trees and solves it with the CUDA SOSP wrapper.
 * @param originalCsrPrefix Prefix of the original CSR graph (used only for vertex count).
 * @param treeInputPaths Paths to the K parent/tree files.
 * @param K Number of objectives / trees.
 * @param source Source vertex.
 * @param workDir Directory for temporary combined-graph files.
 * @param distancesOutputPath Output distances path.
 * @param treeOutputPath Output parent path.
 * @return true if successful, false otherwise.
 */
bool cudaCombinedGraph(const std::string& originalCsrPrefix,
                       const std::vector<std::string>& treeInputPaths,
                       int K,
                       int source = 0,
                       const std::string& workDir = "output/cudaCombinedGraph",
                       const std::string& distancesOutputPath =
                           "output/cudaCombinedGraph/distancesCsr.txt",
                       const std::string& treeOutputPath =
                           "output/cudaCombinedGraph/SSSPTreeCsr.txt");

#endif
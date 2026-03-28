#ifndef CUDA_SOSP_UPDATE_CUH
#define CUDA_SOSP_UPDATE_CUH

#include "cuda_graph.cuh"
#include <string>
#include <vector>

/**
 * @file cuda_sosp_update.cuh
 * @brief Declares hybrid CUDA SOSP-style update functions.
 */

/**
 * @brief Runs a hybrid SOSP-style propagation loop using GPU candidate recomputation
 *        from scratch.
 * @param outgoingCSR Host outgoing CSR graph.
 * @param incomingCSR Host incoming CSR graph.
 * @param deviceIncomingCSR Device incoming CSR graph.
 * @param sourceVertex Source vertex.
 * @param finalDistances Output final distances.
 * @param finalParents Output final parents.
 * @return true if successful, false otherwise.
 */
bool runHybridSOSPUpdate(const HostCsrGraph& outgoingCSR,
                         const HostCsrGraph& incomingCSR,
                         const DeviceCsrGraph& deviceIncomingCSR,
                         int sourceVertex,
                         std::vector<int>& finalDistances,
                         std::vector<int>& finalParents);

/**
 * @brief Runs a hybrid incremental SOSP-style update using an existing solution,
 *        an initial set of affected vertices, and delete-edge information.
 * @param outgoingCSR Host outgoing CSR graph.
 * @param incomingCSR Host incoming CSR graph.
 * @param deviceIncomingCSR Device incoming CSR graph.
 * @param initialDistances Previous distance labels.
 * @param initialParents Previous parent labels.
 * @param initialCandidates Initially affected vertices to recompute.
 * @param deletePath Delete file path used to invalidate old-tree subtrees.
 * @param sourceVertex Source vertex.
 * @param finalDistances Output updated distances.
 * @param finalParents Output updated parents.
 * @return true if successful, false otherwise.
 */
bool runHybridIncrementalSOSPUpdate(const HostCsrGraph& outgoingCSR,
                                    const HostCsrGraph& incomingCSR,
                                    const DeviceCsrGraph& deviceIncomingCSR,
                                    const std::vector<int>& initialDistances,
                                    const std::vector<int>& initialParents,
                                    const std::vector<int>& initialCandidates,
                                    const std::string& deletePath,
                                    int sourceVertex,
                                    std::vector<int>& finalDistances,
                                    std::vector<int>& finalParents);

#endif
#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include "cuda_graph.cuh"
#include <vector>

bool computeOutDegreesOnDevice(const DeviceCsrGraph& deviceGraph,
                               std::vector<int>& degrees);

bool recomputeBestParentsOnDevice(const DeviceCsrGraph& incomingGraph,
                                  const std::vector<int>& currentDistances,
                                  int sourceVertex,
                                  std::vector<int>& newDistances,
                                  std::vector<int>& newParents);

bool recomputeCandidatesOnDevice(const DeviceCsrGraph& incomingGraph,
                                 const std::vector<int>& currentDistances,
                                 const std::vector<int>& candidateVertices,
                                 int sourceVertex,
                                 std::vector<int>& newDistances,
                                 std::vector<int>& newParents);

bool detectChangedCandidatesOnDevice(const std::vector<int>& oldDistances,
                                     const std::vector<int>& newDistances,
                                     const std::vector<int>& candidateVertices,
                                     std::vector<int>& changedVertices);

bool buildInitialCandidatesOnDevice(int numVertices,
                                    const std::vector<int>& insertU,
                                    const std::vector<int>& insertV,
                                    const std::vector<int>& deleteU,
                                    const std::vector<int>& deleteV,
                                    std::vector<int>& initialCandidates);

bool buildNextCandidatesOnDevice(const DeviceCsrGraph& outgoingGraph,
                                 const std::vector<int>& changedVertices,
                                 std::vector<int>& nextCandidates);

bool invalidateDeletedTreeSubtreesOnDevice(const std::vector<int>& initialParents,
                                           const std::vector<int>& deleteU,
                                           const std::vector<int>& deleteV,
                                           int sourceVertex,
                                           std::vector<int>& invalidFlags);

bool runFrontierUpdateLoopOnDevice(const DeviceCsrGraph& outgoingGraph,
                                   const DeviceCsrGraph& incomingGraph,
                                   const std::vector<int>& initialDistances,
                                   const std::vector<int>& initialParents,
                                   const std::vector<int>& initialCandidates,
                                   int sourceVertex,
                                   std::vector<int>& finalDistances,
                                   std::vector<int>& finalParents);

bool runIncrementalFrontierUpdateLoopOnDevice(const DeviceCsrGraph& outgoingGraph,
                                              const DeviceCsrGraph& incomingGraph,
                                              const std::vector<int>& initialDistances,
                                              const std::vector<int>& initialParents,
                                              const std::vector<int>& initialCandidates,
                                              const std::vector<int>& deleteU,
                                              const std::vector<int>& deleteV,
                                              int sourceVertex,
                                              std::vector<int>& finalDistances,
                                              std::vector<int>& finalParents);

#endif
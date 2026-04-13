#ifndef CUDA_PARALLEL_SOSP_UPDATE_CUH
#define CUDA_PARALLEL_SOSP_UPDATE_CUH

#include <string>
#include <vector>

#include "cuda_graph.cuh"

bool cudaParallelSOSPUpdate(
    const std::string& originalCsrPrefix,
    const std::string& distancesInputPath,
    const std::string& treeInputPath,
    const std::string& insertPath,
    const std::string& deletePath,
    int objectiveIndex,
    int sourceVertex,
    const std::string& distancesOutputPath,
    const std::string& treeOutputPath);

bool runCudaSOSPFromScratchInternal(const HostCsrGraph& outgoingCSR,
                                    const HostCsrGraph& incomingCSR,
                                    const DeviceCsrGraph& deviceIncomingCSR,
                                    int sourceVertex,
                                    std::vector<int>& finalDistances,
                                    std::vector<int>& finalParents);

bool runCudaIncrementalSOSPUpdateInternal(const HostCsrGraph& outgoingCSR,
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
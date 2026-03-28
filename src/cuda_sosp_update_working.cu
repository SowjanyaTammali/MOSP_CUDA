#include "../headers/cuda_sosp_update.cuh"
#include "../headers/cuda_kernels.cuh"

#include <climits>
#include <vector>

/**
 * @file cuda_sosp_update.cu
 * @brief Implements a hybrid CUDA SOSP-style propagation loop.
 */

static std::vector<int> collectChangedVertices(const std::vector<int>& oldDistances,
                                               const std::vector<int>& newDistances,
                                               const std::vector<int>& candidates) {
    std::vector<int> changed;
    for (int v : candidates) {
        if (v >= 0 && v < static_cast<int>(oldDistances.size()) &&
            oldDistances[v] != newDistances[v]) {
            changed.push_back(v);
        }
    }
    return changed;
}

static std::vector<int> collectNextCandidatesFromOutgoing(const HostCsrGraph& outgoingCSR,
                                                          const std::vector<int>& changedVertices) {
    std::vector<int> nextCandidates;
    std::vector<char> seen(outgoingCSR.n, 0);

    for (int u : changedVertices) {
        if (u < 0 || u >= outgoingCSR.n) continue;

        int start = outgoingCSR.rowPtr[u];
        int end = outgoingCSR.rowPtr[u + 1];

        for (int i = start; i < end; ++i) {
            int v = outgoingCSR.colInd[i];
            if (!seen[v]) {
                seen[v] = 1;
                nextCandidates.push_back(v);
            }
        }
    }

    return nextCandidates;
}

bool runHybridSOSPUpdate(const HostCsrGraph& outgoingCSR,
                         const HostCsrGraph& incomingCSR,
                         const DeviceCsrGraph& deviceIncomingCSR,
                         int sourceVertex,
                         std::vector<int>& finalDistances,
                         std::vector<int>& finalParents) {
    finalDistances.assign(outgoingCSR.n, INT_MAX);
    finalParents.assign(outgoingCSR.n, -1);
    finalDistances[sourceVertex] = 0;

    std::vector<int> candidates =
        collectNextCandidatesFromOutgoing(outgoingCSR, std::vector<int>{sourceVertex});

    int iteration = 0;
    int maxIterations = outgoingCSR.n * 2;

    while (!candidates.empty() && iteration < maxIterations) {
        ++iteration;

        std::vector<int> gpuNewDistances;
        std::vector<int> gpuNewParents;

        if (!recomputeCandidatesOnDevice(deviceIncomingCSR,
                                         finalDistances,
                                         candidates,
                                         sourceVertex,
                                         gpuNewDistances,
                                         gpuNewParents)) {
            return false;
        }

        std::vector<int> changedVertices =
            collectChangedVertices(finalDistances, gpuNewDistances, candidates);

        finalDistances = gpuNewDistances;

        for (int v : candidates) {
            if (v >= 0 && v < static_cast<int>(finalParents.size())) {
                finalParents[v] = gpuNewParents[v];
            }
        }

        candidates = collectNextCandidatesFromOutgoing(outgoingCSR, changedVertices);
    }

    return true;
}
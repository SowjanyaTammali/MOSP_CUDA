#include "../headers/cuda_sosp_update.cuh"
#include "../headers/cuda_kernels.cuh"

#include <climits>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

/**
 * @file cuda_sosp_update.cu
 * @brief Implements hybrid CUDA SOSP-style propagation loops.
 */

namespace {

std::vector<int> collectChangedVertices(const std::vector<int>& oldDistances,
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

std::vector<int> collectNextCandidatesFromOutgoing(const HostCsrGraph& outgoingCSR,
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

bool readDeletedEdges(const std::string& deletePath,
                      std::vector<std::pair<int, int>>& deletedEdges) {
    std::ifstream in(deletePath);
    if (!in.is_open()) {
        return false;
    }

    deletedEdges.clear();
    std::string line;

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        int u, v;
        if (!(iss >> u >> v)) continue;
        deletedEdges.push_back({u, v});
    }

    return true;
}

std::vector<std::vector<int>> buildChildrenFromParents(const std::vector<int>& parents) {
    std::vector<std::vector<int>> children(parents.size());

    for (int v = 0; v < static_cast<int>(parents.size()); ++v) {
        int p = parents[v];
        if (p >= 0 && p < static_cast<int>(parents.size())) {
            children[p].push_back(v);
        }
    }

    return children;
}

void markSubtreeAffected(int root,
                         const std::vector<std::vector<int>>& children,
                         std::vector<char>& affected) {
    if (root < 0 || root >= static_cast<int>(children.size())) return;
    if (affected[root]) return;

    std::vector<int> stack;
    stack.push_back(root);
    affected[root] = 1;

    while (!stack.empty()) {
        int u = stack.back();
        stack.pop_back();

        for (int child : children[u]) {
            if (!affected[child]) {
                affected[child] = 1;
                stack.push_back(child);
            }
        }
    }
}

void invalidateDeletedTreeSubtrees(const std::vector<int>& initialParents,
                                   const std::string& deletePath,
                                   std::vector<int>& workingDistances,
                                   std::vector<int>& workingParents,
                                   std::vector<int>& workingCandidates,
                                   int sourceVertex) {
    std::vector<std::pair<int, int>> deletedEdges;
    if (!readDeletedEdges(deletePath, deletedEdges)) {
        return;
    }

    std::vector<std::vector<int>> children = buildChildrenFromParents(initialParents);
    std::vector<char> affected(initialParents.size(), 0);

    for (const auto& edge : deletedEdges) {
        int u = edge.first;
        int v = edge.second;

        if (v >= 0 && v < static_cast<int>(initialParents.size()) &&
            initialParents[v] == u &&
            v != sourceVertex) {
            markSubtreeAffected(v, children, affected);
        }
    }

    std::vector<char> seen(initialParents.size(), 0);
    for (int v : workingCandidates) {
        if (v >= 0 && v < static_cast<int>(seen.size())) {
            seen[v] = 1;
        }
    }

    for (int v = 0; v < static_cast<int>(affected.size()); ++v) {
        if (!affected[v]) continue;

        workingDistances[v] = INT_MAX;
        workingParents[v] = -1;

        if (!seen[v]) {
            workingCandidates.push_back(v);
            seen[v] = 1;
        }
    }
}

} // namespace

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

bool runHybridIncrementalSOSPUpdate(const HostCsrGraph& outgoingCSR,
                                    const HostCsrGraph& incomingCSR,
                                    const DeviceCsrGraph& deviceIncomingCSR,
                                    const std::vector<int>& initialDistances,
                                    const std::vector<int>& initialParents,
                                    const std::vector<int>& initialCandidates,
                                    const std::string& deletePath,
                                    int sourceVertex,
                                    std::vector<int>& finalDistances,
                                    std::vector<int>& finalParents) {
    if (static_cast<int>(initialDistances.size()) != outgoingCSR.n ||
        static_cast<int>(initialParents.size()) != outgoingCSR.n) {
        return false;
    }

    finalDistances = initialDistances;
    finalParents = initialParents;

    std::vector<int> candidates = initialCandidates;

    invalidateDeletedTreeSubtrees(initialParents,
                                  deletePath,
                                  finalDistances,
                                  finalParents,
                                  candidates,
                                  sourceVertex);

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
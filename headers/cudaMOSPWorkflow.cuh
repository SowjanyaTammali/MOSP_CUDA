#ifndef CUDA_MOSP_WORKFLOW_CUH
#define CUDA_MOSP_WORKFLOW_CUH

#include <string>

/**
 * @file cudaMOSPWorkflow.cuh
 * @brief Declares an automated CUDA MOSP workflow wrapper.
 */

/**
 * @brief Runs the full CUDA MOSP-style workflow automatically using updated trees.
 *
 * Workflow:
 * 1. Reads the original CSR graph.
 * 2. Builds the updated CSR graph using insert/delete changes.
 * 3. Detects the number of objectives.
 * 4. Runs CUDA SOSP from scratch once per objective on the original graph.
 * 5. Runs CUDA incremental SOSP update once per objective on the updated graph.
 * 6. Saves one UPDATED tree/parent file per objective.
 * 7. Builds and solves the combined graph using those updated tree files.
 *
 * @param originalCsrPrefix Prefix of the original CSR graph.
 * @param updatedCsrPrefix Prefix where the updated CSR graph should exist/be written.
 * @param insertPath Insert file path.
 * @param deletePath Delete file path.
 * @param source Source vertex.
 * @param workDir Output directory for workflow files.
 * @return true if successful, false otherwise.
 */
bool runCudaMOSPWorkflow(const std::string& originalCsrPrefix,
                         const std::string& updatedCsrPrefix,
                         const std::string& insertPath,
                         const std::string& deletePath,
                         int source = 0,
                         const std::string& workDir = "output/cudaMOSPWorkflow");

#endif
#ifndef CUDA_PRINT_H
#define CUDA_PRINT_H

#include "csr.h"

/**
 * @brief Print CSR graph from GPU for debugging/verification.
 *
 * @param csr Host CSR graph.
 * @return True on success, false otherwise.
 */
bool printCSRFromGPU(const CSRGraph &csr);

#endif
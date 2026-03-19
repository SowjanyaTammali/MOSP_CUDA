#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cstring>
#include <cuda_runtime.h>

using namespace std;

// Simple CUDA error checker
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            cerr << "CUDA Error: " << cudaGetErrorString(err)                \
                 << " at line " << __LINE__ << endl;                         \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// GPU kernel: one thread handles one node and prints its outgoing edges
__global__ void printGraphCSRKernel(const int* row_ptr,
                                    const int* col_ind,
                                    const int* weights,
                                    int num_nodes) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int u = 0; u < num_nodes; u++) {
            printf("Node %d -> ", u + 1);
            for (int e = row_ptr[u]; e < row_ptr[u + 1]; e++) {
                printf("(%d, w=%d) ", col_ind[e] + 1, weights[e]);
            }
            printf("\n");
        }
    }
}

int main() {
    // -----------------------------------------
    // 1. Open the .mtx file on CPU
    // -----------------------------------------
    ifstream file("graph.mtx");
    if (!file.is_open()) {
        cout << "Error: Could not open graph.mtx\n";
        return 1;
    }

    string line;

    // Skip Matrix Market header/comments
    while (getline(file, line)) {
        if (!line.empty() && line[0] != '%') {
            break;
        }
    }

    // Read rows, cols, nnz
    stringstream ss(line);
    int rows, cols, nnz;
    ss >> rows >> cols >> nnz;

    // -----------------------------------------
    // 2. Temporary CPU storage for input edges
    //    (used only to help build CSR)
    // -----------------------------------------
    vector<int> from_list(nnz);
    vector<int> to_list(nnz);
    vector<int> weight_list(nnz);

    for (int i = 0; i < nnz; i++) {
        int r, c, w;
        file >> r >> c >> w;

        // Convert from 1-based to 0-based indexing
        r--;
        c--;

        from_list[i] = r;
        to_list[i] = c;
        weight_list[i] = w;
    }

    file.close();

    // -----------------------------------------
    // 3. Allocate CSR arrays using Unified Memory
    // -----------------------------------------
    int* row_ptr = nullptr;
    int* col_ind = nullptr;
    int* weights = nullptr;

    CUDA_CHECK(cudaMallocManaged(&row_ptr, (rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&col_ind, nnz * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&weights, nnz * sizeof(int)));

    // Initialize row_ptr to zero
    memset(row_ptr, 0, (rows + 1) * sizeof(int));

    // -----------------------------------------
    // 4. Build CSR: count outgoing edges
    // -----------------------------------------
    for (int i = 0; i < nnz; i++) {
        row_ptr[from_list[i] + 1]++;
    }

    // Prefix sum to get row_ptr
    for (int i = 1; i <= rows; i++) {
        row_ptr[i] += row_ptr[i - 1];
    }

    // Use a temporary CPU vector to track fill positions
    vector<int> current_pos(rows);
    for (int i = 0; i < rows; i++) {
        current_pos[i] = row_ptr[i];
    }

    // Fill col_ind and weights
    for (int i = 0; i < nnz; i++) {
        int u = from_list[i];
        int pos = current_pos[u]++;

        col_ind[pos] = to_list[i];
        weights[pos] = weight_list[i];
    }

    // -----------------------------------------
    // 5. Print graph on CPU using CSR
    // -----------------------------------------
    cout << "Graph printed from CPU using CSR:\n";
    for (int u = 0; u < rows; u++) {
        cout << "Node " << u + 1 << " -> ";
        for (int e = row_ptr[u]; e < row_ptr[u + 1]; e++) {
            cout << "(" << col_ind[e] + 1 << ", w=" << weights[e] << ") ";
        }
        cout << "\n";
    }

    // Make sure managed memory is ready before kernel launch
    CUDA_CHECK(cudaDeviceSynchronize());

    // -----------------------------------------
    // 6. Launch GPU kernel
    // -----------------------------------------
    cout << "\nGraph printed from GPU using CSR:\n";

    // int threadsPerBlock = 256;
    // int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;

  printGraphCSRKernel<<<1, 1>>>(row_ptr, col_ind, weights, rows);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // -----------------------------------------
    // 7. Free Unified Memory
    // -----------------------------------------
    CUDA_CHECK(cudaFree(row_ptr));
    CUDA_CHECK(cudaFree(col_ind));
    CUDA_CHECK(cudaFree(weights));

    return 0;
}
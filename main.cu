#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
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

// GPU kernel: one thread handles one edge
__global__ void printEdgesKernel(const int* from_list,
                                 const int* to_list,
                                 const int* weight_list,
                                 int nnz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nnz) {
        printf("Edge %d: %d -> %d  weight = %d\n",
               i,
               from_list[i] + 1,
               to_list[i] + 1,
               weight_list[i]);
    }
}

int main() {
    // -------------------------------
    // 1. Open the .mtx file on CPU
    // -------------------------------
    ifstream file("graph.mtx");
    if (!file.is_open()) {
        cout << "Error: Could not open graph.mtx\n";
        return 1;
    }

    string line;

    // Skip comment/header lines starting with %
    while (getline(file, line)) {
        if (!line.empty() && line[0] != '%') {
            break;
        }
    }

    // Read rows, cols, nnz
    stringstream ss(line);
    int rows, cols, nnz;
    ss >> rows >> cols >> nnz;

    // -------------------------------
    // 2. Store graph edges on CPU
    // -------------------------------
    vector<int> from_list;
    vector<int> to_list;
    vector<int> weight_list;

    from_list.reserve(nnz);
    to_list.reserve(nnz);
    weight_list.reserve(nnz);

    for (int i = 0; i < nnz; i++) {
        int r, c, w;
        file >> r >> c >> w;

        // Convert from 1-based to 0-based indexing
        r--;
        c--;

        from_list.push_back(r);
        to_list.push_back(c);
        weight_list.push_back(w);
    }

    file.close();

    // -------------------------------
    // 3. Print edges on CPU
    // -------------------------------
    cout << "Edges stored on CPU:\n";
    for (int i = 0; i < nnz; i++) {
        cout << "Edge " << i << ": "
             << from_list[i] + 1 << " -> "
             << to_list[i] + 1
             << "  weight = " << weight_list[i] << "\n";
    }

    // -------------------------------
    // 4. Allocate memory on GPU
    // -------------------------------
    int* d_from_list = nullptr;
    int* d_to_list = nullptr;
    int* d_weight_list = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_from_list, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_to_list, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_weight_list, nnz * sizeof(int)));

    // -------------------------------
    // 5. Copy CPU arrays to GPU
    // -------------------------------
    CUDA_CHECK(cudaMemcpy(d_from_list, from_list.data(),
                          nnz * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_to_list, to_list.data(),
                          nnz * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_weight_list, weight_list.data(),
                          nnz * sizeof(int), cudaMemcpyHostToDevice));

    // -------------------------------
    // 6. Launch GPU kernel
    // -------------------------------
    cout << "\nEdges printed by GPU:\n";

    int threadsPerBlock = 256;
    int blocksPerGrid = (nnz + threadsPerBlock - 1) / threadsPerBlock;

    printEdgesKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_from_list, d_to_list, d_weight_list, nnz
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // -------------------------------
    // 7. Free GPU memory
    // -------------------------------
    CUDA_CHECK(cudaFree(d_from_list));
    CUDA_CHECK(cudaFree(d_to_list));
    CUDA_CHECK(cudaFree(d_weight_list));

    return 0;
}
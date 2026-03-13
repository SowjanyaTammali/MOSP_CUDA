# MOSP_CUDA

This repository contains my first CUDA graph program.

## Files
- `main.cu` : CUDA program that reads a `.mtx` graph on the CPU, copies edge-list arrays to the GPU, and prints edges from both CPU and GPU.
- `graph.mtx` : Sample Matrix Market graph input file.

## Compilation
For the V100 GPU node on the cluster:

```bash
module load cuda-toolkit/12.9
nvcc -arch=sm_70 -o main main.cu

# Compiler settings
CXX = g++
NVCC = nvcc

# Compiler flags
CXXFLAGS = -std=c++17 -Wall -Iheaders
NVCCFLAGS = -std=c++17 -Iheaders
LDFLAGS = -lstdc++fs

# Executables
MAIN_TARGET = bin/mosp_cuda
STRESS_TARGET = bin/cuda_stress_test

# Main executable objects
MAIN_CPP_OBJECTS = build/main.o build/read.o build/updateGraphCSR.o build/sequentialSOSPUpdate.o
MAIN_CU_OBJECTS = build/cuda_graph.o build/cuda_kernels.o build/cuda_sosp_update.o build/cudaCombinedGraph.o build/cudaMOSPWorkflow.o
MAIN_OBJECTS = $(MAIN_CPP_OBJECTS) $(MAIN_CU_OBJECTS)

# Stress executable objects
STRESS_CPP_OBJECTS = build/cudaStressTest.o build/read.o build/updateGraphCSR.o build/sequentialSOSPUpdate.o build/Dijkstra.o build/generateGraphCSR.o build/generateChangedEdges.o
STRESS_CU_OBJECTS = build/cuda_graph.o build/cuda_kernels.o build/cuda_sosp_update.o
STRESS_OBJECTS = $(STRESS_CPP_OBJECTS) $(STRESS_CU_OBJECTS)

# Default target
all: $(MAIN_TARGET) $(STRESS_TARGET)

# Main executable
$(MAIN_TARGET): $(MAIN_OBJECTS)
	$(NVCC) $(MAIN_OBJECTS) -o $(MAIN_TARGET) $(LDFLAGS)

# Stress test executable
$(STRESS_TARGET): $(STRESS_OBJECTS)
	$(NVCC) $(STRESS_OBJECTS) -o $(STRESS_TARGET) $(LDFLAGS)

# Compile main driver
build/main.o: src/main.cpp headers/read.h headers/cuda_graph.cuh headers/cuda_sosp_update.cuh headers/updateGraphCSR.h headers/sequentialSOSPUpdate.h headers/cudaCombinedGraph.cuh headers/cudaMOSPWorkflow.cuh
	$(CXX) $(CXXFLAGS) -c src/main.cpp -o build/main.o

# Compile stress test driver
build/cudaStressTest.o: src/cudaStressTest.cpp headers/dijkstra.h headers/generateChangedEdges.h headers/generateGraphCSR.h headers/read.h headers/cuda_graph.cuh headers/cuda_sosp_update.cuh headers/updateGraphCSR.h
	$(CXX) $(CXXFLAGS) -c src/cudaStressTest.cpp -o build/cudaStressTest.o

# Common C++ files
build/read.o: src/read.cpp headers/read.h
	$(CXX) $(CXXFLAGS) -c src/read.cpp -o build/read.o

build/updateGraphCSR.o: src/updateGraphCSR.cpp headers/updateGraphCSR.h headers/read.h
	$(CXX) $(CXXFLAGS) -c src/updateGraphCSR.cpp -o build/updateGraphCSR.o

build/sequentialSOSPUpdate.o: src/sequentialSOSPUpdate.cpp headers/sequentialSOSPUpdate.h headers/read.h
	$(CXX) $(CXXFLAGS) -c src/sequentialSOSPUpdate.cpp -o build/sequentialSOSPUpdate.o

build/Dijkstra.o: src/Dijkstra.cpp headers/dijkstra.h headers/read.h
	$(CXX) $(CXXFLAGS) -c src/Dijkstra.cpp -o build/Dijkstra.o

build/generateGraphCSR.o: src/generateGraphCSR.cpp headers/generateGraphCSR.h
	$(CXX) $(CXXFLAGS) -c src/generateGraphCSR.cpp -o build/generateGraphCSR.o

build/generateChangedEdges.o: src/generateChangedEdges.cpp headers/generateChangedEdges.h headers/read.h
	$(CXX) $(CXXFLAGS) -c src/generateChangedEdges.cpp -o build/generateChangedEdges.o

# Common CUDA files
build/cuda_graph.o: src/cuda_graph.cu headers/cuda_graph.cuh headers/read.h
	$(NVCC) $(NVCCFLAGS) -c src/cuda_graph.cu -o build/cuda_graph.o

build/cuda_kernels.o: src/cuda_kernels.cu headers/cuda_kernels.cuh headers/cuda_graph.cuh
	$(NVCC) $(NVCCFLAGS) -c src/cuda_kernels.cu -o build/cuda_kernels.o

build/cuda_sosp_update.o: src/cuda_sosp_update.cu headers/cuda_sosp_update.cuh headers/cuda_kernels.cuh headers/cuda_graph.cuh
	$(NVCC) $(NVCCFLAGS) -c src/cuda_sosp_update.cu -o build/cuda_sosp_update.o

build/cudaCombinedGraph.o: src/cudaCombinedGraph.cu headers/cudaCombinedGraph.cuh headers/read.h headers/cuda_graph.cuh headers/cuda_sosp_update.cuh
	$(NVCC) $(NVCCFLAGS) -c src/cudaCombinedGraph.cu -o build/cudaCombinedGraph.o

build/cudaMOSPWorkflow.o: src/cudaMOSPWorkflow.cu headers/cudaMOSPWorkflow.cuh headers/read.h headers/cuda_graph.cuh headers/cuda_sosp_update.cuh headers/cudaCombinedGraph.cuh
	$(NVCC) $(NVCCFLAGS) -c src/cudaMOSPWorkflow.cu -o build/cudaMOSPWorkflow.o

# Clean
clean:
	rm -f build/*.o $(MAIN_TARGET) $(STRESS_TARGET)
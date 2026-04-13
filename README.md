# CUDA_MOSP

Build

From the project root:

```bash
make clean
make
```

Run

From the project root:

```bash
./bin/main
```

Workflow Mode

To run the CUDA pipeline on an already prepared case:

```bash
./bin/main workflow <originalPrefix> <updatedPrefix> <insertPath> <deletePath> <source>
```

Example:

```bash
./bin/main workflow data/originalGraph/graphCsr data/updatedGraph/updatedGraphCsr data/changes/insert.txt data/changes/delete.txt 0
```

This mode is useful when you want to compare CUDA output against another implementation on the exact same input files.

What the app does

The app performs the following steps:

1. Builds the original CSR graph
2. Generates edge insertions and deletions
3. Builds the updated CSR graph
4. Runs Dijkstra on the original graph
5. Runs Dijkstra on the updated graph
6. Runs the Sequential SOSP Update baseline
7. Runs CUDA-based incremental SOSP Update for each objective
8. Builds the combined graph from the updated objective trees
9. Generates deterministic test cases for validation

Output Files

The app writes the original graph to:

```text
data/originalGraph/graphCsrRowPtr.txt
data/originalGraph/graphCsrColInd.txt
data/originalGraph/graphCsrValues.txt
```

It also generates edge-change files:

```text
output/changedEdges/insert.txt
output/changedEdges/delete.txt
```

Then it writes the updated graph to:

```text
data/updatedGraph/updatedGraphCsrRowPtr.txt
data/updatedGraph/updatedGraphCsrColInd.txt
data/updatedGraph/updatedGraphCsrValues.txt
```

Then it runs Dijkstra on the original graph and writes:

```text
output/distancesTrees/distancesCsr.txt
output/distancesTrees/SSSPTreeCsr.txt
```

Then it runs Dijkstra on the updated graph and writes:

```text
output/updatedDistancesTrees/updatedDistancesCsr.txt
output/updatedDistancesTrees/updatedSSSPTreeCsr.txt
```

Then it runs the Sequential SOSP Update baseline and writes:

```text
output/sospUpdateDistancesTrees/distancesCsr.txt
output/sospUpdateDistancesTrees/SSSPTreeCsr.txt
```

Then it runs CUDA incremental update for each objective and writes:

```text
output/parallelSospObj0/distancesOriginal.txt
output/parallelSospObj0/SSSPTreeOriginal.txt
output/parallelSospObj0/distancesUpdated.txt
output/parallelSospObj0/SSSPTreeUpdated.txt

output/parallelSospObj1/distancesOriginal.txt
output/parallelSospObj1/SSSPTreeOriginal.txt
output/parallelSospObj1/distancesUpdated.txt
output/parallelSospObj1/SSSPTreeUpdated.txt

output/parallelSospObj2/distancesOriginal.txt
output/parallelSospObj2/SSSPTreeOriginal.txt
output/parallelSospObj2/distancesUpdated.txt
output/parallelSospObj2/SSSPTreeUpdated.txt
```

Each objective folder may also contain temporary updated CSR files used by the CUDA update step.

Then it builds the combined graph and writes:

```text
output/combinedGraph/combinedGraphCsrRowPtr.txt
output/combinedGraph/combinedGraphCsrColInd.txt
output/combinedGraph/combinedGraphCsrValues.txt
output/combinedGraph/distancesCsr.txt
output/combinedGraph/SSSPTreeCsr.txt
```

Test Cases

The app also generates 10 deterministic test cases under:

```text
tests/testCase0/
tests/testCase1/
...
tests/testCase9/
```

Each test case contains:

```text
originalGraph/   (CSR files)
changedEdges/    (insert.txt, delete.txt)
updatedGraph/    (CSR files after applying changes)
expected/        (ground truth distances and SSSP trees for original graph, updated graph, and sequential SOSP update)
cuda/            (CUDA incremental update output)
```

These test cases are seeded for reproducibility and vary across graph size, objective count, change ratio, and selected objective index.

CUDA Stress Test

The repo also includes a randomized CUDA stress test executable.

After building, run:

```bash
./bin/cuda_stress_test
```

This stress test validates:
- per-objective CUDA incremental update
- combined graph construction
- overall pipeline consistency across many generated cases

Project Structure

```text
headers/   header files
src/       source files
data/      generated CSR graph files
output/    generated run outputs
tests/     deterministic generated test cases
```
#ifndef GENERATE_GRAPH_H
#define GENERATE_GRAPH_H

#include <string>

bool generateGraph(
    int numberOfNodes,
    int numberOfEdges,
    bool directed,
    const std::string &outputFile,
    int numberOfObjectives,
    int objectiveStartRange,
    int objectiveEndRange
);

#endif
#ifndef READ_H
#define READ_H

#include <string>
#include <vector>

struct Edge {
    int to;
    std::vector<int> weights;
};

using Graph = std::vector<std::vector<Edge>>;

bool readMtx(const std::string &path, Graph &graph, int &numberOfObjectives);

#endif
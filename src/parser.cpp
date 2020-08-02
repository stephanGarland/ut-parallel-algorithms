// Parser - imports .gr files from DIMACS and outputs a Graph
// http://users.diag.uniroma1.it/challenge9/format.shtml#graph

#include <iostream>
#include <fstream>
#include <map>
#include <sstream>
#include <set>
#include <vector>
#include <unistd.h>

#include <bits/stdc++.h>

#include "../inc/bellmanford.h"

using std::cout;
using std::endl;

struct Graph *makeGraph(int V, int E)
{
    struct Graph *graph = new Graph;
    graph->V = V;
    graph->E = E;
    graph->edge = new Edge[E];
    return graph;
};

int main() {
    std::ifstream inp;
    std::stringstream buffer;
    int numLines = 0;
    std::vector<std::string> inpTokens;
    std::map<unsigned int, std::map<std::string, unsigned int>> edges;

    std::vector<unsigned int> src;
    std::vector<unsigned int> nodeIndex;
    std::vector<unsigned int> dst;
    std::vector<unsigned int> wgt;

    inp.open("./inp/example.gr");
    buffer << inp.rdbuf();

    // Thank you to Killzone Kid @ https://stackoverflow.com/a/49201823/4221094
    std::string const delims{" \n"};
    size_t beg, pos = 0;
    while ((beg = buffer.str().find_first_not_of(delims, pos)) != std::string::npos) {
        pos = buffer.str().find_first_of(delims, beg + 1);
        inpTokens.push_back(buffer.str().substr(beg, pos - beg));
    }

    for (unsigned int i = 0; i < inpTokens.size(); i++) {
        if (inpTokens[i] == "a") {
            edges[numLines]["src"] = stoi(inpTokens[i+1]);
            edges[numLines]["dst"] = stoi(inpTokens[i+2]);
            edges[numLines]["wgt"] = stoi(inpTokens[i+3]);
            numLines++;
        }
    }

    // Ideally these would be set to actual values, but since edges[i] has duplicate entries of dst for each src
    // can't just do edges.size() or anything of the sort
    // Instead, we're relying on a distance not being INT_MAX
    unsigned int V = 10000;
    unsigned int E = 10000;
    struct Graph* g = makeGraph(V, E);
    for (unsigned int i = 0; i < edges.size(); i++) {
        g->edge[i].src = edges[i]["src"];
        g->edge[i].dst = edges[i]["dst"];
        g->edge[i].wgt = edges[i]["wgt"];
    }

    BellmanFord(g, 1);

    return 0;
}
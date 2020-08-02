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

using std::cout;
using std::endl;

struct Edge {
    int src;
    int dst;
    int wgt;
};

struct Graph {
    int V;
    int E;
    struct Edge* edge;
};

struct Graph* makeGraph (int V, int E) {
    struct Graph* graph = new Graph;
    graph->V = V;
    graph->E = E;
    graph->edge = new Edge[E];
    return graph;
};

void printArr(int dist[], int n) {
    printf("Vertex   Distance from Source\n");
    for (int i = 0; i < n; ++i) {
        if (dist[i] != INT_MAX) {
            cout << i << "\t\t" << dist[i] << endl;
        }
    }
}
// For example purposes only; this is from https://www.geeksforgeeks.org/bellman-ford-algorithm-dp-23/
void BellmanFord(struct Graph *graph, int src) {
    int V = graph->V;
    int E = graph->E;
    int dist[V];

    // Step 1: Initialize distances from src to all other vertices
    // as INFINITE
    for (int i = 0; i < V; i++)
        dist[i] = INT_MAX;
    dist[src] = 0;

    // Step 2: Relax all edges |V| - 1 times. A simple shortest
    // path from src to any other vertex can have at-most |V| - 1
    // edges
    for (int i = 1; i <= V - 1; i++)
    {
        for (int j = 0; j < E; j++)
        {
            int u = graph->edge[j].src;
            int v = graph->edge[j].dst;
            int weight = graph->edge[j].wgt;
            if (dist[u] != INT_MAX && dist[u] + weight < dist[v])
                dist[v] = dist[u] + weight;
        }
    }

    // Step 3: check for negative-weight cycles.  The above step
    // guarantees shortest distances if graph doesn't contain
    // negative weight cycle.  If we get a shorter path, then there
    // is a cycle.
    for (int i = 0; i < E; i++)
    {
        int u = graph->edge[i].src;
        int v = graph->edge[i].dst;
        int weight = graph->edge[i].wgt;
        if (dist[u] != INT_MAX && dist[u] + weight < dist[v])
        {
            printf("Graph contains negative weight cycle");
            return; // If negative cycle is detected, simply return
        }
    }

    printArr(dist, V);

    return;
}

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
// Parser - imports .gr files from DIMACS and outputs a Graph
// http://users.diag.uniroma1.it/challenge9/format.shtml#graph

#include <iostream>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <unistd.h>

#include "./csr/CSRGraphV2.h"

using std::cout;
using std::endl;

int main() {
    std::ifstream inp;
    std::stringstream buffer;
    int numLines = 0;
    std::vector<std::string> inpTokens;
    std::map<int, std::map<std::string, int>> edges;

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

    unsigned int numEdgesEdges = edges.size() / 3;
    unsigned int numEdgesTotal = edges.size();
    CSRGraphV2 *g = new CSRGraphV2(numEdgesEdges, numEdgesTotal);
    for (unsigned int i = 0; i < numEdgesEdges; i++) {
        g->add_edge(edges[i]["src"], edges[i]["dst"], edges[i]["wgt"]);
    }
    g->finished();

    for (unsigned int i = 0; i < numEdgesEdges * 10; i++) {
        for (auto &j : g->get_neighbors((i % numEdgesEdges) + 1)) {
            cout << "src: " << j.first << "\tdst: " << j.second << endl;
        }
    }
    /*
    V         = [ 5 8 3 6 ]
    COL_INDEX = [ 0 1 2 1 ]
    ROW_INDEX = [ 0 0 2 3 4 ]
    */
   /*
    for (auto kv : edges) {
        cout << kv.first << ": ";
        for (auto kvv : kv.second) {
            cout << kvv.first << ": " << kvv.second << "\t";
        }
        cout << endl;
    }
    */
    return 0;
}
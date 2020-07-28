// Parser - imports .gr files from DIMACS and outputs a Graph
// http://users.diag.uniroma1.it/challenge9/format.shtml#graph

#include <iostream>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>

using std::cout;
using std::endl;

int main() {
    std::ifstream inp;
    std::stringstream buffer;
    int numLines = 0;
    std::vector<std::string> inpTokens;
    std::map<int, std::map<std::string, int>> edges;

    inp.open("./inp.gr");
    buffer << inp.rdbuf();
    std::string const delims{" \n"};
    size_t beg, pos = 0;
    while ((beg = buffer.str().find_first_not_of(delims, pos)) != std::string::npos) {
        pos = buffer.str().find_first_of(delims, beg + 1);
        inpTokens.push_back(buffer.str().substr(beg, pos - beg));
    }
    
    for (int i = 0; i < inpTokens.size(); i++) {
        if (inpTokens[i] == "a") {
            edges[numLines]["src"] = stoi(inpTokens[i+1]);
            edges[numLines]["dst"] = stoi(inpTokens[i+2]);
            edges[numLines]["wgt"] = stoi(inpTokens[i+3]);
            numLines++;
        }
    }

    for (auto kv : edges) {
        cout << kv.first << ": ";
        for (auto kvv : kv.second) {
            cout << kvv.first << ": " << kvv.second << "\t";
        }
        cout << endl;
    }
    return 0;
}
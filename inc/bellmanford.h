struct Edge
{
    int src;
    int dst;
    int wgt;
};

struct Graph
{
    int V;
    int E;
    struct Edge *edge;
};

void BellmanFord(struct Graph *graph, int src);

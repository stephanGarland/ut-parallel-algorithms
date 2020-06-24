package hw2

import (
	"github.com/gonum/graph"
)

// BellmanFord applies the Bellman-Ford algorihtm to Graph and return
// a shortest path tree.
//
// Note that this uses Shortest to make it easier for you,
// but you can use another struct if that makes more sense
// for the concurrency model you chose.
func BellmanFord(s graph.Node, g graph.Graph) Shortest {

	var weight Weighting
	if wg, ok := g.(graph.Weighter); ok {
		weight = wg.Weight
	} else {
		weight = UniformCost(g)
	}

	nodes := g.Nodes()

	path = newShortestFrom(s, nodes)
	path.dist[path.indexOf[s.ID()]] = 0

	for i := 1; i < len(nodes); i++ {
		for j := 0; j < g.Edge; j++ {

		}
	}

	return newShortestFrom(s, g.Nodes())
}

// DeltaStep applies the delta-stepping algorihtm to Graph and return
// a shortest path tree.
//
// Note that this uses Shortest to make it easier for you,
// but you can use another struct if that makes more sense
// for the concurrency model you chose.
func DeltaStep(s graph.Node, g graph.Graph) Shortest {
	// Your code goes here.
	return newShortestFrom(s, g.Nodes())
}

//  Dijkstra from gonum to make sure that the tests are correct.
func Dijkstra(s graph.Node, g graph.Graph) Shortest {
	return DijkstraFrom(s, g)
}

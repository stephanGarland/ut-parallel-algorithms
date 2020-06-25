package main

import (
	"fmt"
	"math"
)

// Edge defines an edge's structure
type Edge struct {
	src int
	dst int
	wgt int
}

// Graph defines a directed graph's structure
type Graph struct {
	V     int
	E     int
	edges []Edge
}

func printResult(dist []int, n int) {
	fmt.Println("Vertex\t Distance from Source")
	for i := int(0); i < n; i++ {
		fmt.Printf("%d \t %d\n", i, dist[i])
	}
}

// BellmanFord runs the algorithm of the same name
func BellmanFord(s int, g Graph) {
	var V = g.V
	var E = g.E
	var dist = make([]int, V)

	for i := 0; i < V; i++ {
		dist[i] = math.MaxInt64
	}
	dist[s] = 0

	for i := 0; i < V; i++ {
		for j := 0; j < E; j++ {
			var u = g.edges[j].src
			var v = g.edges[j].dst
			var w = g.edges[j].wgt
			if dist[u] != math.MaxInt64 && dist[u]+w < dist[v] {
				dist[v] = dist[u] + w
			}
		}
	}

	for i := 0; i < E; i++ {
		var u = g.edges[i].src
		var v = g.edges[i].dst
		var w = g.edges[i].wgt
		if dist[u] != math.MaxInt64 && dist[u]+w < dist[v] {
			fmt.Println("Negative cycle detected")
			return
		}
	}

	printResult(dist, V)
	return
}

func main() {
	var g Graph
	g.V = 5
	g.E = 8
	g.edges = make([]Edge, 0)

	e1 := Edge{
		src: 0,
		dst: 1,
		wgt: -1,
	}

	e2 := Edge{
		src: 0,
		dst: 2,
		wgt: 4,
	}

	e3 := Edge{
		src: 1,
		dst: 2,
		wgt: 3,
	}

	e4 := Edge{
		src: 1,
		dst: 3,
		wgt: 2,
	}

	e5 := Edge{
		src: 1,
		dst: 4,
		wgt: 2,
	}

	e6 := Edge{
		src: 3,
		dst: 2,
		wgt: 5,
	}

	e7 := Edge{
		src: 3,
		dst: 1,
		wgt: 1,
	}

	e8 := Edge{
		src: 4,
		dst: 3,
		wgt: -3,
	}

	g.edges = append(g.edges, e1)
	g.edges = append(g.edges, e2)
	g.edges = append(g.edges, e3)
	g.edges = append(g.edges, e4)
	g.edges = append(g.edges, e5)
	g.edges = append(g.edges, e6)
	g.edges = append(g.edges, e7)
	g.edges = append(g.edges, e8)

	BellmanFord(0, g)
}

/*
func BellmanFord(s graph.Node, g graph.Graph) Shortest {

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
*/

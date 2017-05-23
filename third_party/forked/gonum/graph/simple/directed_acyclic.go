package simple

import (
	"k8s.io/kubernetes/third_party/forked/gonum/graph"
)

// DirectedAcyclicGraph implements graph.Directed using UndirectedGraph,
// which only stores one edge for any node pair.
type DirectedAcyclicGraph struct {
	*UndirectedGraph
}

func NewDirectedAcyclicGraph(self, absent float64) *DirectedAcyclicGraph {
	return &DirectedAcyclicGraph{
		UndirectedGraph: NewUndirectedGraph(self, absent),
	}
}

func (g *DirectedAcyclicGraph) HasEdgeFromTo(u, v graph.Node) bool {
	edge := g.UndirectedGraph.EdgeBetween(u, v)
	if edge == nil {
		return false
	}
	return (edge.From().ID() == u.ID())
}

func (g *DirectedAcyclicGraph) From(n graph.Node) []graph.Node {
	if !g.Has(n) {
		return nil
	}

	fid := n.ID()
	nodes := make([]graph.Node, 0, len(g.UndirectedGraph.edges[n.ID()]))
	for _, edge := range g.UndirectedGraph.edges[n.ID()] {
		if edge.From().ID() == fid {
			nodes = append(nodes, g.UndirectedGraph.nodes[edge.To().ID()])
		}
	}
	return nodes
}

func (g *DirectedAcyclicGraph) To(n graph.Node) []graph.Node {
	if !g.Has(n) {
		return nil
	}

	tid := n.ID()
	nodes := make([]graph.Node, 0, len(g.UndirectedGraph.edges[n.ID()]))
	for _, edge := range g.UndirectedGraph.edges[n.ID()] {
		if edge.To().ID() == tid {
			nodes = append(nodes, g.UndirectedGraph.nodes[edge.From().ID()])
		}
	}
	return nodes
}

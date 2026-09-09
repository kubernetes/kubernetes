// Copyright ©2014 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package graph

// Node is a graph node. It returns a graph-unique integer ID.
type Node interface {
	ID() int
}

// Edge is a graph edge. In directed graphs, the direction of the
// edge is given from -> to, otherwise the edge is semantically
// unordered.
type Edge interface {
	From() Node
	To() Node
	Weight() float64
}

// Graph is a generalized graph.
type Graph interface {
	// Has returns whether the node exists within the graph.
	Has(Node) bool

	// Nodes returns all the nodes in the graph.
	Nodes() []Node

	// From returns all nodes that can be reached directly
	// from the given node.
	From(Node) []Node

	// HasEdgeBeteen returns whether an edge exists between
	// nodes x and y without considering direction.
	HasEdgeBetween(x, y Node) bool

	// Edge returns the edge from u to v if such an edge
	// exists and nil otherwise. The node v must be directly
	// reachable from u as defined by the From method.
	Edge(u, v Node) Edge
}

// Undirected is an undirected graph.
type Undirected interface {
	Graph

	// EdgeBetween returns the edge between nodes x and y.
	EdgeBetween(x, y Node) Edge
}

// Directed is a directed graph.
type Directed interface {
	Graph

	// HasEdgeFromTo returns whether an edge exists
	// in the graph from u to v.
	HasEdgeFromTo(u, v Node) bool

	// To returns all nodes that can reach directly
	// to the given node.
	To(Node) []Node
}

// Weighter defines graphs that can report edge weights.
type Weighter interface {
	// Weight returns the weight for the edge between
	// x and y if Edge(x, y) returns a non-nil Edge.
	// If x and y are the same node or there is no
	// joining edge between the two nodes the weight
	// value returned is implementation dependent.
	// Weight returns true if an edge exists between
	// x and y or if x and y have the same ID, false
	// otherwise.
	Weight(x, y Node) (w float64, ok bool)
}

// NodeAdder is an interface for adding arbitrary nodes to a graph.
type NodeAdder interface {
	// NewNodeID returns a new unique arbitrary ID.
	NewNodeID() int

	// Adds a node to the graph. AddNode panics if
	// the added node ID matches an existing node ID.
	AddNode(Node)
}

// NodeRemover is an interface for removing nodes from a graph.
type NodeRemover interface {
	// RemoveNode removes a node from the graph, as
	// well as any edges attached to it. If the node
	// is not in the graph it is a no-op.
	RemoveNode(Node)
}

// EdgeSetter is an interface for adding edges to a graph.
type EdgeSetter interface {
	// SetEdge adds an edge from one node to another.
	// If the graph supports node addition the nodes
	// will be added if they do not exist, otherwise
	// SetEdge will panic.
	// If the IDs returned by e.From and e.To are
	// equal, SetEdge will panic.
	SetEdge(e Edge)
}

// EdgeRemover is an interface for removing nodes from a graph.
type EdgeRemover interface {
	// RemoveEdge removes the given edge, leaving the
	// terminal nodes. If the edge does not exist it
	// is a no-op.
	RemoveEdge(Edge)
}

// Builder is a graph that can have nodes and edges added.
type Builder interface {
	NodeAdder
	EdgeSetter
}

// UndirectedBuilder is an undirected graph builder.
type UndirectedBuilder interface {
	Undirected
	Builder
}

// DirectedBuilder is a directed graph builder.
type DirectedBuilder interface {
	Directed
	Builder
}

// Copy copies nodes and edges as undirected edges from the source to the destination
// without first clearing the destination. Copy will panic if a node ID in the source
// graph matches a node ID in the destination.
//
// If the source is undirected and the destination is directed both directions will
// be present in the destination after the copy is complete.
//
// If the source is a directed graph, the destination is undirected, and a fundamental
// cycle exists with two nodes where the edge weights differ, the resulting destination
// graph's edge weight between those nodes is undefined. If there is a defined function
// to resolve such conflicts, an Undirect may be used to do this.
func Copy(dst Builder, src Graph) {
	nodes := src.Nodes()
	for _, n := range nodes {
		dst.AddNode(n)
	}
	for _, u := range nodes {
		for _, v := range src.From(u) {
			dst.SetEdge(src.Edge(u, v))
		}
	}
}

// Copyright ©2014 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package graph

// Node is a graph node. It returns a graph-unique integer ID.
type Node interface {
	ID() int64
}

// Edge is a graph edge. In directed graphs, the direction of the
// edge is given from -> to, otherwise the edge is semantically
// unordered.
type Edge interface {
	// From returns the from node of the edge.
	From() Node

	// To returns the to node of the edge.
	To() Node

	// ReversedEdge returns the edge reversal of the receiver
	// if a reversal is valid for the data type.
	// When a reversal is valid an edge of the same type as
	// the receiver with nodes of the receiver swapped should
	// be returned, otherwise the receiver should be returned
	// unaltered.
	ReversedEdge() Edge
}

// WeightedEdge is a weighted graph edge. In directed graphs, the direction
// of the edge is given from -> to, otherwise the edge is semantically
// unordered.
type WeightedEdge interface {
	Edge
	Weight() float64
}

// Graph is a generalized graph.
type Graph interface {
	// Node returns the node with the given ID if it exists
	// in the graph, and nil otherwise.
	Node(id int64) Node

	// Nodes returns all the nodes in the graph.
	//
	// Nodes must not return nil.
	Nodes() Nodes

	// From returns all nodes that can be reached directly
	// from the node with the given ID.
	//
	// From must not return nil.
	From(id int64) Nodes

	// HasEdgeBetween returns whether an edge exists between
	// nodes with IDs xid and yid without considering direction.
	HasEdgeBetween(xid, yid int64) bool

	// Edge returns the edge from u to v, with IDs uid and vid,
	// if such an edge exists and nil otherwise. The node v
	// must be directly reachable from u as defined by the
	// From method.
	Edge(uid, vid int64) Edge
}

// Weighted is a weighted graph.
type Weighted interface {
	Graph

	// WeightedEdge returns the weighted edge from u to v
	// with IDs uid and vid if such an edge exists and
	// nil otherwise. The node v must be directly
	// reachable from u as defined by the From method.
	WeightedEdge(uid, vid int64) WeightedEdge

	// Weight returns the weight for the edge between
	// x and y with IDs xid and yid if Edge(xid, yid)
	// returns a non-nil Edge.
	// If x and y are the same node or there is no
	// joining edge between the two nodes the weight
	// value returned is implementation dependent.
	// Weight returns true if an edge exists between
	// x and y or if x and y have the same ID, false
	// otherwise.
	Weight(xid, yid int64) (w float64, ok bool)
}

// Undirected is an undirected graph.
type Undirected interface {
	Graph

	// EdgeBetween returns the edge between nodes x and y
	// with IDs xid and yid.
	EdgeBetween(xid, yid int64) Edge
}

// WeightedUndirected is a weighted undirected graph.
type WeightedUndirected interface {
	Weighted

	// WeightedEdgeBetween returns the edge between nodes
	// x and y with IDs xid and yid.
	WeightedEdgeBetween(xid, yid int64) WeightedEdge
}

// Directed is a directed graph.
type Directed interface {
	Graph

	// HasEdgeFromTo returns whether an edge exists
	// in the graph from u to v with IDs uid and vid.
	HasEdgeFromTo(uid, vid int64) bool

	// To returns all nodes that can reach directly
	// to the node with the given ID.
	//
	// To must not return nil.
	To(id int64) Nodes
}

// WeightedDirected is a weighted directed graph.
type WeightedDirected interface {
	Weighted

	// HasEdgeFromTo returns whether an edge exists
	// in the graph from u to v with the IDs uid and
	// vid.
	HasEdgeFromTo(uid, vid int64) bool

	// To returns all nodes that can reach directly
	// to the node with the given ID.
	//
	// To must not return nil.
	To(id int64) Nodes
}

// NodeAdder is an interface for adding arbitrary nodes to a graph.
type NodeAdder interface {
	// NewNode returns a new Node with a unique
	// arbitrary ID.
	NewNode() Node

	// AddNode adds a node to the graph. AddNode panics if
	// the added node ID matches an existing node ID.
	AddNode(Node)
}

// NodeRemover is an interface for removing nodes from a graph.
type NodeRemover interface {
	// RemoveNode removes the node with the given ID
	// from the graph, as well as any edges attached
	// to it. If the node is not in the graph it is
	// a no-op.
	RemoveNode(id int64)
}

// EdgeAdder is an interface for adding edges to a graph.
type EdgeAdder interface {
	// NewEdge returns a new Edge from the source to the destination node.
	NewEdge(from, to Node) Edge

	// SetEdge adds an edge from one node to another.
	// If the graph supports node addition the nodes
	// will be added if they do not exist, otherwise
	// SetEdge will panic.
	// The behavior of an EdgeAdder when the IDs
	// returned by e.From() and e.To() are equal is
	// implementation-dependent.
	// Whether e, e.From() and e.To() are stored
	// within the graph is implementation dependent.
	SetEdge(e Edge)
}

// WeightedEdgeAdder is an interface for adding edges to a graph.
type WeightedEdgeAdder interface {
	// NewWeightedEdge returns a new WeightedEdge from
	// the source to the destination node.
	NewWeightedEdge(from, to Node, weight float64) WeightedEdge

	// SetWeightedEdge adds an edge from one node to
	// another. If the graph supports node addition
	// the nodes will be added if they do not exist,
	// otherwise SetWeightedEdge will panic.
	// The behavior of a WeightedEdgeAdder when the IDs
	// returned by e.From() and e.To() are equal is
	// implementation-dependent.
	// Whether e, e.From() and e.To() are stored
	// within the graph is implementation dependent.
	SetWeightedEdge(e WeightedEdge)
}

// EdgeRemover is an interface for removing nodes from a graph.
type EdgeRemover interface {
	// RemoveEdge removes the edge with the given end
	// IDs, leaving the terminal nodes. If the edge
	// does not exist it is a no-op.
	RemoveEdge(fid, tid int64)
}

// Builder is a graph that can have nodes and edges added.
type Builder interface {
	NodeAdder
	EdgeAdder
}

// WeightedBuilder is a graph that can have nodes and weighted edges added.
type WeightedBuilder interface {
	NodeAdder
	WeightedEdgeAdder
}

// UndirectedBuilder is an undirected graph builder.
type UndirectedBuilder interface {
	Undirected
	Builder
}

// UndirectedWeightedBuilder is an undirected weighted graph builder.
type UndirectedWeightedBuilder interface {
	Undirected
	WeightedBuilder
}

// DirectedBuilder is a directed graph builder.
type DirectedBuilder interface {
	Directed
	Builder
}

// DirectedWeightedBuilder is a directed weighted graph builder.
type DirectedWeightedBuilder interface {
	Directed
	WeightedBuilder
}

// Copy copies nodes and edges as undirected edges from the source to the destination
// without first clearing the destination. Copy will panic if a node ID in the source
// graph matches a node ID in the destination.
//
// If the source is undirected and the destination is directed both directions will
// be present in the destination after the copy is complete.
func Copy(dst Builder, src Graph) {
	nodes := src.Nodes()
	for nodes.Next() {
		dst.AddNode(nodes.Node())
	}
	nodes.Reset()
	for nodes.Next() {
		u := nodes.Node()
		uid := u.ID()
		to := src.From(uid)
		for to.Next() {
			v := to.Node()
			dst.SetEdge(src.Edge(uid, v.ID()))
		}
	}
}

// CopyWeighted copies nodes and edges as undirected edges from the source to the destination
// without first clearing the destination. Copy will panic if a node ID in the source
// graph matches a node ID in the destination.
//
// If the source is undirected and the destination is directed both directions will
// be present in the destination after the copy is complete.
//
// If the source is a directed graph, the destination is undirected, and a fundamental
// cycle exists with two nodes where the edge weights differ, the resulting destination
// graph's edge weight between those nodes is undefined. If there is a defined function
// to resolve such conflicts, an UndirectWeighted may be used to do this.
func CopyWeighted(dst WeightedBuilder, src Weighted) {
	nodes := src.Nodes()
	for nodes.Next() {
		dst.AddNode(nodes.Node())
	}
	nodes.Reset()
	for nodes.Next() {
		u := nodes.Node()
		uid := u.ID()
		to := src.From(uid)
		for to.Next() {
			v := to.Node()
			dst.SetWeightedEdge(src.WeightedEdge(uid, v.ID()))
		}
	}
}

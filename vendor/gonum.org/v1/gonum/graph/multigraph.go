// Copyright ©2014 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package graph

// Line is an edge in a multigraph. A Line returns an ID that must
// distinguish Lines sharing Node end points.
type Line interface {
	// From returns the from node of the edge.
	From() Node

	// To returns the to node of the edge.
	To() Node

	// ReversedLine returns a line that has the
	// end points of the receiver swapped.
	ReversedLine() Line

	// ID returns the unique ID for the Line.
	ID() int64
}

// WeightedLine is a weighted multigraph edge.
type WeightedLine interface {
	Line
	Weight() float64
}

// Multigraph is a generalized multigraph.
type Multigraph interface {
	// Node returns the node with the given ID if it exists
	// in the multigraph, and nil otherwise.
	Node(id int64) Node

	// Nodes returns all the nodes in the multigraph.
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

	// Lines returns the lines from u to v, with IDs uid and
	// vid, if any such lines exist and nil otherwise. The
	// node v must be directly reachable from u as defined by
	// the From method.
	//
	// Lines must not return nil.
	Lines(uid, vid int64) Lines
}

// WeightedMultigraph is a weighted multigraph.
type WeightedMultigraph interface {
	Multigraph

	// WeightedLines returns the weighted lines from u to v
	// with IDs uid and vid if any such lines exist and nil
	// otherwise. The node v must be directly reachable
	// from u as defined by the From method.
	//
	// WeightedLines must not return nil.
	WeightedLines(uid, vid int64) WeightedLines
}

// UndirectedMultigraph is an undirected multigraph.
type UndirectedMultigraph interface {
	Multigraph

	// LinesBetween returns the lines between nodes x and y
	// with IDs xid and yid.
	//
	// LinesBetween must not return nil.
	LinesBetween(xid, yid int64) Lines
}

// WeightedUndirectedMultigraph is a weighted undirected multigraph.
type WeightedUndirectedMultigraph interface {
	WeightedMultigraph

	// WeightedLinesBetween returns the lines between nodes
	// x and y with IDs xid and yid.
	//
	// WeightedLinesBetween must not return nil.
	WeightedLinesBetween(xid, yid int64) WeightedLines
}

// DirectedMultigraph is a directed multigraph.
type DirectedMultigraph interface {
	Multigraph

	// HasEdgeFromTo returns whether an edge exists
	// in the multigraph from u to v with IDs uid
	// and vid.
	HasEdgeFromTo(uid, vid int64) bool

	// To returns all nodes that can reach directly
	// to the node with the given ID.
	//
	// To must not return nil.
	To(id int64) Nodes
}

// WeightedDirectedMultigraph is a weighted directed multigraph.
type WeightedDirectedMultigraph interface {
	WeightedMultigraph

	// HasEdgeFromTo returns whether an edge exists
	// in the multigraph from u to v with IDs uid
	// and vid.
	HasEdgeFromTo(uid, vid int64) bool

	// To returns all nodes that can reach directly
	// to the node with the given ID.
	//
	// To must not return nil.
	To(id int64) Nodes
}

// LineAdder is an interface for adding lines to a multigraph.
type LineAdder interface {
	// NewLine returns a new Line from the source to the destination node.
	NewLine(from, to Node) Line

	// SetLine adds a Line from one node to another.
	// If the multigraph supports node addition the nodes
	// will be added if they do not exist, otherwise
	// SetLine will panic.
	// Whether l, l.From() and l.To() are stored
	// within the graph is implementation dependent.
	SetLine(l Line)
}

// WeightedLineAdder is an interface for adding lines to a multigraph.
type WeightedLineAdder interface {
	// NewWeightedLine returns a new WeightedLine from
	// the source to the destination node.
	NewWeightedLine(from, to Node, weight float64) WeightedLine

	// SetWeightedLine adds a weighted line from one node
	// to another. If the multigraph supports node addition
	// the nodes will be added if they do not exist,
	// otherwise SetWeightedLine will panic.
	// Whether l, l.From() and l.To() are stored
	// within the graph is implementation dependent.
	SetWeightedLine(l WeightedLine)
}

// LineRemover is an interface for removing lines from a multigraph.
type LineRemover interface {
	// RemoveLine removes the line with the given end
	// and line IDs, leaving the terminal nodes. If
	// the line does not exist it is a no-op.
	RemoveLine(fid, tid, id int64)
}

// MultigraphBuilder is a multigraph that can have nodes and lines added.
type MultigraphBuilder interface {
	NodeAdder
	LineAdder
}

// WeightedMultigraphBuilder is a multigraph that can have nodes and weighted lines added.
type WeightedMultigraphBuilder interface {
	NodeAdder
	WeightedLineAdder
}

// UndirectedMultgraphBuilder is an undirected multigraph builder.
type UndirectedMultigraphBuilder interface {
	UndirectedMultigraph
	MultigraphBuilder
}

// UndirectedWeightedMultigraphBuilder is an undirected weighted multigraph builder.
type UndirectedWeightedMultigraphBuilder interface {
	UndirectedMultigraph
	WeightedMultigraphBuilder
}

// DirectedMultigraphBuilder is a directed multigraph builder.
type DirectedMultigraphBuilder interface {
	DirectedMultigraph
	MultigraphBuilder
}

// DirectedWeightedMultigraphBuilder is a directed weighted multigraph builder.
type DirectedWeightedMultigraphBuilder interface {
	DirectedMultigraph
	WeightedMultigraphBuilder
}

// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package graph

// Undirect converts a directed graph to an undirected graph.
type Undirect struct {
	G Directed
}

var _ Undirected = Undirect{}

// Node returns the node with the given ID if it exists in the graph,
// and nil otherwise.
func (g Undirect) Node(id int64) Node { return g.G.Node(id) }

// Nodes returns all the nodes in the graph.
func (g Undirect) Nodes() Nodes { return g.G.Nodes() }

// From returns all nodes in g that can be reached directly from u.
func (g Undirect) From(uid int64) Nodes {
	if g.G.Node(uid) == nil {
		return Empty
	}
	return newNodeIteratorPair(g.G.From(uid), g.G.To(uid))
}

// HasEdgeBetween returns whether an edge exists between nodes x and y.
func (g Undirect) HasEdgeBetween(xid, yid int64) bool { return g.G.HasEdgeBetween(xid, yid) }

// Edge returns the edge from u to v if such an edge exists and nil otherwise.
// The node v must be directly reachable from u as defined by the From method.
// If an edge exists, the Edge returned is an EdgePair. The weight of
// the edge is determined by applying the Merge func to the weights of the
// edges between u and v.
func (g Undirect) Edge(uid, vid int64) Edge { return g.EdgeBetween(uid, vid) }

// EdgeBetween returns the edge between nodes x and y. If an edge exists, the
// Edge returned is an EdgePair. The weight of the edge is determined by
// applying the Merge func to the weights of edges between x and y.
func (g Undirect) EdgeBetween(xid, yid int64) Edge {
	fe := g.G.Edge(xid, yid)
	re := g.G.Edge(yid, xid)
	if fe == nil && re == nil {
		return nil
	}

	return EdgePair{fe, re}
}

// UndirectWeighted converts a directed weighted graph to an undirected weighted graph,
// resolving edge weight conflicts.
type UndirectWeighted struct {
	G WeightedDirected

	// Absent is the value used to
	// represent absent edge weights
	// passed to Merge if the reverse
	// edge is present.
	Absent float64

	// Merge defines how discordant edge
	// weights in G are resolved. A merge
	// is performed if at least one edge
	// exists between the nodes being
	// considered. The edges corresponding
	// to the two weights are also passed,
	// in the same order.
	// The order of weight parameters
	// passed to Merge is not defined, so
	// the function should be commutative.
	// If Merge is nil, the arithmetic
	// mean is used to merge weights.
	Merge func(x, y float64, xe, ye Edge) float64
}

var (
	_ Undirected         = UndirectWeighted{}
	_ WeightedUndirected = UndirectWeighted{}
)

// Node returns the node with the given ID if it exists in the graph,
// and nil otherwise.
func (g UndirectWeighted) Node(id int64) Node { return g.G.Node(id) }

// Nodes returns all the nodes in the graph.
func (g UndirectWeighted) Nodes() Nodes { return g.G.Nodes() }

// From returns all nodes in g that can be reached directly from u.
func (g UndirectWeighted) From(uid int64) Nodes {
	if g.G.Node(uid) == nil {
		return Empty
	}
	return newNodeIteratorPair(g.G.From(uid), g.G.To(uid))
}

// HasEdgeBetween returns whether an edge exists between nodes x and y.
func (g UndirectWeighted) HasEdgeBetween(xid, yid int64) bool { return g.G.HasEdgeBetween(xid, yid) }

// Edge returns the edge from u to v if such an edge exists and nil otherwise.
// The node v must be directly reachable from u as defined by the From method.
// If an edge exists, the Edge returned is an EdgePair. The weight of
// the edge is determined by applying the Merge func to the weights of the
// edges between u and v.
func (g UndirectWeighted) Edge(uid, vid int64) Edge { return g.WeightedEdgeBetween(uid, vid) }

// WeightedEdge returns the weighted edge from u to v if such an edge exists and nil otherwise.
// The node v must be directly reachable from u as defined by the From method.
// If an edge exists, the Edge returned is an EdgePair. The weight of
// the edge is determined by applying the Merge func to the weights of the
// edges between u and v.
func (g UndirectWeighted) WeightedEdge(uid, vid int64) WeightedEdge {
	return g.WeightedEdgeBetween(uid, vid)
}

// EdgeBetween returns the edge between nodes x and y. If an edge exists, the
// Edge returned is an EdgePair. The weight of the edge is determined by
// applying the Merge func to the weights of edges between x and y.
func (g UndirectWeighted) EdgeBetween(xid, yid int64) Edge {
	return g.WeightedEdgeBetween(xid, yid)
}

// WeightedEdgeBetween returns the weighted edge between nodes x and y. If an edge exists, the
// Edge returned is an EdgePair. The weight of the edge is determined by
// applying the Merge func to the weights of edges between x and y.
func (g UndirectWeighted) WeightedEdgeBetween(xid, yid int64) WeightedEdge {
	fe := g.G.Edge(xid, yid)
	re := g.G.Edge(yid, xid)
	if fe == nil && re == nil {
		return nil
	}

	f, ok := g.G.Weight(xid, yid)
	if !ok {
		f = g.Absent
	}
	r, ok := g.G.Weight(yid, xid)
	if !ok {
		r = g.Absent
	}

	var w float64
	if g.Merge == nil {
		w = (f + r) / 2
	} else {
		w = g.Merge(f, r, fe, re)
	}
	return WeightedEdgePair{EdgePair: [2]Edge{fe, re}, W: w}
}

// Weight returns the weight for the edge between x and y if Edge(x, y) returns a non-nil Edge.
// If x and y are the same node the internal node weight is returned. If there is no joining
// edge between the two nodes the weight value returned is zero. Weight returns true if an edge
// exists between x and y or if x and y have the same ID, false otherwise.
func (g UndirectWeighted) Weight(xid, yid int64) (w float64, ok bool) {
	fe := g.G.Edge(xid, yid)
	re := g.G.Edge(yid, xid)

	f, fOk := g.G.Weight(xid, yid)
	if !fOk {
		f = g.Absent
	}
	r, rOK := g.G.Weight(yid, xid)
	if !rOK {
		r = g.Absent
	}
	ok = fOk || rOK

	if g.Merge == nil {
		return (f + r) / 2, ok
	}
	return g.Merge(f, r, fe, re), ok
}

// EdgePair is an opposed pair of directed edges.
type EdgePair [2]Edge

// From returns the from node of the first non-nil edge, or nil.
func (e EdgePair) From() Node {
	if e[0] != nil {
		return e[0].From()
	} else if e[1] != nil {
		return e[1].From()
	}
	return nil
}

// To returns the to node of the first non-nil edge, or nil.
func (e EdgePair) To() Node {
	if e[0] != nil {
		return e[0].To()
	} else if e[1] != nil {
		return e[1].To()
	}
	return nil
}

// ReversedEdge returns a new Edge with the end point of the
// edges in the pair swapped.
func (e EdgePair) ReversedEdge() Edge {
	if e[0] != nil {
		e[0] = e[0].ReversedEdge()
	}
	if e[1] != nil {
		e[1] = e[1].ReversedEdge()
	}
	return e
}

// WeightedEdgePair is an opposed pair of directed edges.
type WeightedEdgePair struct {
	EdgePair
	W float64
}

// ReversedEdge returns a new Edge with the end point of the
// edges in the pair swapped.
func (e WeightedEdgePair) ReversedEdge() Edge {
	e.EdgePair = e.EdgePair.ReversedEdge().(EdgePair)
	return e
}

// Weight returns the merged edge weights of the two edges.
func (e WeightedEdgePair) Weight() float64 { return e.W }

// nodeIteratorPair combines two Nodes to produce a single stream of
// unique nodes.
type nodeIteratorPair struct {
	a, b Nodes

	curr Node

	idx, cnt int

	// unique indicates the node in b with the key ID is unique.
	unique map[int64]bool
}

func newNodeIteratorPair(a, b Nodes) *nodeIteratorPair {
	n := nodeIteratorPair{a: a, b: b, unique: make(map[int64]bool)}
	for n.b.Next() {
		n.unique[n.b.Node().ID()] = true
		n.cnt++
	}
	n.b.Reset()
	for n.a.Next() {
		if _, ok := n.unique[n.a.Node().ID()]; !ok {
			n.cnt++
		}
		n.unique[n.a.Node().ID()] = false
	}
	n.a.Reset()
	return &n
}

func (n *nodeIteratorPair) Len() int {
	return n.cnt - n.idx
}

func (n *nodeIteratorPair) Next() bool {
	if n.a.Next() {
		n.idx++
		n.curr = n.a.Node()
		return true
	}
	for n.b.Next() {
		if n.unique[n.b.Node().ID()] {
			n.idx++
			n.curr = n.b.Node()
			return true
		}
	}
	n.curr = nil
	return false
}

func (n *nodeIteratorPair) Node() Node {
	return n.curr
}

func (n *nodeIteratorPair) Reset() {
	n.idx = 0
	n.curr = nil
	n.a.Reset()
	n.b.Reset()
}

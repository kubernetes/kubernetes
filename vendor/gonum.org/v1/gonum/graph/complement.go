// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package graph

// Complement provides the complement of a graph. The complement will not include
// self-edges, and edges within the complement will not hold any information other
// than the nodes in the original graph and the connection topology. Nodes returned
// by the Complement directly or via queries to returned Edges will be those stored
// in the original graph.
type Complement struct {
	Graph
}

// Edge returns the edge from u to v if such an edge exists and nil otherwise.
// The node v must be directly reachable from u as defined by the From method.
func (g Complement) Edge(uid, vid int64) Edge {
	if g.Graph.Edge(uid, vid) != nil || uid == vid {
		return nil
	}
	u := g.Node(uid)
	v := g.Node(vid)
	if u == nil || v == nil {
		return nil
	}
	return shadow{F: u, T: v}
}

// From returns all nodes in g that can be reached directly from u in
// the complement.
func (g Complement) From(uid int64) Nodes {
	if g.Node(uid) == nil {
		return Empty
	}
	// At this point, we guarantee that g.Graph.From(uid) returns a set of
	// nodes in g.Nodes(), and that uid corresponds to a node in g.Nodes().
	return newNodeFilterIterator(g.Nodes(), g.Graph.From(uid), uid)
}

// HasEdgeBetween returns whether an edge exists between nodes x and y.
func (g Complement) HasEdgeBetween(xid, yid int64) bool {
	return xid != yid &&
		g.Node(xid) != nil && g.Node(yid) != nil &&
		!g.Graph.HasEdgeBetween(xid, yid)
}

// shadow is an edge that is not exposed to the user.
type shadow struct{ F, T Node }

func (e shadow) From() Node         { return e.F }
func (e shadow) To() Node           { return e.T }
func (e shadow) ReversedEdge() Edge { return shadow{F: e.T, T: e.F} }

// nodeFilterIterator combines Nodes to produce a single stream of
// filtered nodes.
type nodeFilterIterator struct {
	src Nodes

	// filter indicates the node in n with the key ID should be filtered out.
	filter map[int64]bool
}

// newNodeFilterIterator returns a new nodeFilterIterator. The nodes in filter and
// the nodes corresponding the root node ID must be in the src set of nodes. This
// invariant is not checked.
func newNodeFilterIterator(src, filter Nodes, root int64) *nodeFilterIterator {
	n := nodeFilterIterator{src: src, filter: map[int64]bool{root: true}}
	for filter.Next() {
		n.filter[filter.Node().ID()] = true
	}
	filter.Reset()
	n.src.Reset()
	return &n
}

func (n *nodeFilterIterator) Len() int {
	return n.src.Len() - len(n.filter)
}

func (n *nodeFilterIterator) Next() bool {
	for n.src.Next() {
		if !n.filter[n.src.Node().ID()] {
			return true
		}
	}
	return false
}

func (n *nodeFilterIterator) Node() Node {
	return n.src.Node()
}

func (n *nodeFilterIterator) Reset() {
	n.src.Reset()
}

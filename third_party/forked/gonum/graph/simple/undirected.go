// Copyright ©2014 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package simple

import (
	"fmt"

	"golang.org/x/tools/container/intsets"

	"k8s.io/kubernetes/third_party/forked/gonum/graph"
)

// UndirectedGraph implements a generalized undirected graph.
type UndirectedGraph struct {
	nodes map[int]graph.Node
	edges map[int]edgeHolder

	self, absent float64

	freeIDs intsets.Sparse
	usedIDs intsets.Sparse
}

// NewUndirectedGraph returns an UndirectedGraph with the specified self and absent
// edge weight values.
func NewUndirectedGraph(self, absent float64) *UndirectedGraph {
	return &UndirectedGraph{
		nodes: make(map[int]graph.Node),
		edges: make(map[int]edgeHolder),

		self:   self,
		absent: absent,
	}
}

// NewNodeID returns a new unique ID for a node to be added to g. The returned ID does
// not become a valid ID in g until it is added to g.
func (g *UndirectedGraph) NewNodeID() int {
	if len(g.nodes) == 0 {
		return 0
	}
	if len(g.nodes) == maxInt {
		panic(fmt.Sprintf("simple: cannot allocate node: no slot"))
	}

	var id int
	if g.freeIDs.Len() != 0 && g.freeIDs.TakeMin(&id) {
		return id
	}
	if id = g.usedIDs.Max(); id < maxInt {
		return id + 1
	}
	for id = 0; id < maxInt; id++ {
		if !g.usedIDs.Has(id) {
			return id
		}
	}
	panic("unreachable")
}

// AddNode adds n to the graph. It panics if the added node ID matches an existing node ID.
func (g *UndirectedGraph) AddNode(n graph.Node) {
	if _, exists := g.nodes[n.ID()]; exists {
		panic(fmt.Sprintf("simple: node ID collision: %d", n.ID()))
	}
	g.nodes[n.ID()] = n
	g.edges[n.ID()] = &sliceEdgeHolder{self: n.ID()}

	g.freeIDs.Remove(n.ID())
	g.usedIDs.Insert(n.ID())
}

// RemoveNode removes n from the graph, as well as any edges attached to it. If the node
// is not in the graph it is a no-op.
func (g *UndirectedGraph) RemoveNode(n graph.Node) {
	if _, ok := g.nodes[n.ID()]; !ok {
		return
	}
	delete(g.nodes, n.ID())

	g.edges[n.ID()].Visit(func(neighbor int, edge graph.Edge) {
		g.edges[neighbor] = g.edges[neighbor].Delete(n.ID())
	})
	delete(g.edges, n.ID())

	g.freeIDs.Insert(n.ID())
	g.usedIDs.Remove(n.ID())

}

// SetEdge adds e, an edge from one node to another. If the nodes do not exist, they are added.
// It will panic if the IDs of the e.From and e.To are equal.
func (g *UndirectedGraph) SetEdge(e graph.Edge) {
	var (
		from = e.From()
		fid  = from.ID()
		to   = e.To()
		tid  = to.ID()
	)

	if fid == tid {
		panic("simple: adding self edge")
	}

	if !g.Has(from) {
		g.AddNode(from)
	}
	if !g.Has(to) {
		g.AddNode(to)
	}

	g.edges[fid] = g.edges[fid].Set(tid, e)
	g.edges[tid] = g.edges[tid].Set(fid, e)
}

// RemoveEdge removes e from the graph, leaving the terminal nodes. If the edge does not exist
// it is a no-op.
func (g *UndirectedGraph) RemoveEdge(e graph.Edge) {
	from, to := e.From(), e.To()
	if _, ok := g.nodes[from.ID()]; !ok {
		return
	}
	if _, ok := g.nodes[to.ID()]; !ok {
		return
	}

	g.edges[from.ID()] = g.edges[from.ID()].Delete(to.ID())
	g.edges[to.ID()] = g.edges[to.ID()].Delete(from.ID())
}

// Node returns the node in the graph with the given ID.
func (g *UndirectedGraph) Node(id int) graph.Node {
	return g.nodes[id]
}

// Has returns whether the node exists within the graph.
func (g *UndirectedGraph) Has(n graph.Node) bool {
	_, ok := g.nodes[n.ID()]
	return ok
}

// Nodes returns all the nodes in the graph.
func (g *UndirectedGraph) Nodes() []graph.Node {
	nodes := make([]graph.Node, len(g.nodes))
	i := 0
	for _, n := range g.nodes {
		nodes[i] = n
		i++
	}

	return nodes
}

// Edges returns all the edges in the graph.
func (g *UndirectedGraph) Edges() []graph.Edge {
	var edges []graph.Edge

	seen := make(map[[2]int]struct{})
	for _, u := range g.edges {
		u.Visit(func(neighbor int, e graph.Edge) {
			uid := e.From().ID()
			vid := e.To().ID()
			if _, ok := seen[[2]int{uid, vid}]; ok {
				return
			}
			seen[[2]int{uid, vid}] = struct{}{}
			seen[[2]int{vid, uid}] = struct{}{}
			edges = append(edges, e)
		})
	}

	return edges
}

// From returns all nodes in g that can be reached directly from n.
func (g *UndirectedGraph) From(n graph.Node) []graph.Node {
	if !g.Has(n) {
		return nil
	}

	nodes := make([]graph.Node, g.edges[n.ID()].Len())
	i := 0
	g.edges[n.ID()].Visit(func(neighbor int, edge graph.Edge) {
		nodes[i] = g.nodes[neighbor]
		i++
	})

	return nodes
}

// HasEdgeBetween returns whether an edge exists between nodes x and y.
func (g *UndirectedGraph) HasEdgeBetween(x, y graph.Node) bool {
	_, ok := g.edges[x.ID()].Get(y.ID())
	return ok
}

// Edge returns the edge from u to v if such an edge exists and nil otherwise.
// The node v must be directly reachable from u as defined by the From method.
func (g *UndirectedGraph) Edge(u, v graph.Node) graph.Edge {
	return g.EdgeBetween(u, v)
}

// EdgeBetween returns the edge between nodes x and y.
func (g *UndirectedGraph) EdgeBetween(x, y graph.Node) graph.Edge {
	// We don't need to check if neigh exists because
	// it's implicit in the edges access.
	if !g.Has(x) {
		return nil
	}

	edge, _ := g.edges[x.ID()].Get(y.ID())
	return edge
}

// Weight returns the weight for the edge between x and y if Edge(x, y) returns a non-nil Edge.
// If x and y are the same node or there is no joining edge between the two nodes the weight
// value returned is either the graph's absent or self value. Weight returns true if an edge
// exists between x and y or if x and y have the same ID, false otherwise.
func (g *UndirectedGraph) Weight(x, y graph.Node) (w float64, ok bool) {
	xid := x.ID()
	yid := y.ID()
	if xid == yid {
		return g.self, true
	}
	if n, ok := g.edges[xid]; ok {
		if e, ok := n.Get(yid); ok {
			return e.Weight(), true
		}
	}
	return g.absent, false
}

// Degree returns the degree of n in g.
func (g *UndirectedGraph) Degree(n graph.Node) int {
	if _, ok := g.nodes[n.ID()]; !ok {
		return 0
	}

	return g.edges[n.ID()].Len()
}

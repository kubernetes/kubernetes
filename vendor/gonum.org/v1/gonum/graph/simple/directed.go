// Copyright ©2014 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package simple

import (
	"fmt"

	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/iterator"
	"gonum.org/v1/gonum/graph/set/uid"
)

var (
	dg *DirectedGraph

	_ graph.Graph       = dg
	_ graph.Directed    = dg
	_ graph.NodeAdder   = dg
	_ graph.NodeRemover = dg
	_ graph.EdgeAdder   = dg
	_ graph.EdgeRemover = dg
)

// DirectedGraph implements a generalized directed graph.
type DirectedGraph struct {
	nodes map[int64]graph.Node
	from  map[int64]map[int64]graph.Edge
	to    map[int64]map[int64]graph.Edge

	nodeIDs *uid.Set
}

// NewDirectedGraph returns a DirectedGraph.
func NewDirectedGraph() *DirectedGraph {
	return &DirectedGraph{
		nodes: make(map[int64]graph.Node),
		from:  make(map[int64]map[int64]graph.Edge),
		to:    make(map[int64]map[int64]graph.Edge),

		nodeIDs: uid.NewSet(),
	}
}

// AddNode adds n to the graph. It panics if the added node ID matches an existing node ID.
func (g *DirectedGraph) AddNode(n graph.Node) {
	if _, exists := g.nodes[n.ID()]; exists {
		panic(fmt.Sprintf("simple: node ID collision: %d", n.ID()))
	}
	g.nodes[n.ID()] = n
	g.nodeIDs.Use(n.ID())
}

// Edge returns the edge from u to v if such an edge exists and nil otherwise.
// The node v must be directly reachable from u as defined by the From method.
func (g *DirectedGraph) Edge(uid, vid int64) graph.Edge {
	edge, ok := g.from[uid][vid]
	if !ok {
		return nil
	}
	return edge
}

// Edges returns all the edges in the graph.
func (g *DirectedGraph) Edges() graph.Edges {
	var edges []graph.Edge
	for _, u := range g.nodes {
		for _, e := range g.from[u.ID()] {
			edges = append(edges, e)
		}
	}
	if len(edges) == 0 {
		return graph.Empty
	}
	return iterator.NewOrderedEdges(edges)
}

// From returns all nodes in g that can be reached directly from n.
//
// The returned graph.Nodes is only valid until the next mutation of
// the receiver.
func (g *DirectedGraph) From(id int64) graph.Nodes {
	if len(g.from[id]) == 0 {
		return graph.Empty
	}
	return iterator.NewNodesByEdge(g.nodes, g.from[id])
}

// HasEdgeBetween returns whether an edge exists between nodes x and y without
// considering direction.
func (g *DirectedGraph) HasEdgeBetween(xid, yid int64) bool {
	if _, ok := g.from[xid][yid]; ok {
		return true
	}
	_, ok := g.from[yid][xid]
	return ok
}

// HasEdgeFromTo returns whether an edge exists in the graph from u to v.
func (g *DirectedGraph) HasEdgeFromTo(uid, vid int64) bool {
	if _, ok := g.from[uid][vid]; !ok {
		return false
	}
	return true
}

// NewEdge returns a new Edge from the source to the destination node.
func (g *DirectedGraph) NewEdge(from, to graph.Node) graph.Edge {
	return Edge{F: from, T: to}
}

// NewNode returns a new unique Node to be added to g. The Node's ID does
// not become valid in g until the Node is added to g.
func (g *DirectedGraph) NewNode() graph.Node {
	if len(g.nodes) == 0 {
		return Node(0)
	}
	if int64(len(g.nodes)) == uid.Max {
		panic("simple: cannot allocate node: no slot")
	}
	return Node(g.nodeIDs.NewID())
}

// Node returns the node with the given ID if it exists in the graph,
// and nil otherwise.
func (g *DirectedGraph) Node(id int64) graph.Node {
	return g.nodes[id]
}

// Nodes returns all the nodes in the graph.
//
// The returned graph.Nodes is only valid until the next mutation of
// the receiver.
func (g *DirectedGraph) Nodes() graph.Nodes {
	if len(g.nodes) == 0 {
		return graph.Empty
	}
	return iterator.NewNodes(g.nodes)
}

// RemoveEdge removes the edge with the given end point IDs from the graph, leaving the terminal
// nodes. If the edge does not exist it is a no-op.
func (g *DirectedGraph) RemoveEdge(fid, tid int64) {
	if _, ok := g.nodes[fid]; !ok {
		return
	}
	if _, ok := g.nodes[tid]; !ok {
		return
	}

	delete(g.from[fid], tid)
	delete(g.to[tid], fid)
}

// RemoveNode removes the node with the given ID from the graph, as well as any edges attached
// to it. If the node is not in the graph it is a no-op.
func (g *DirectedGraph) RemoveNode(id int64) {
	if _, ok := g.nodes[id]; !ok {
		return
	}
	delete(g.nodes, id)

	for from := range g.from[id] {
		delete(g.to[from], id)
	}
	delete(g.from, id)

	for to := range g.to[id] {
		delete(g.from[to], id)
	}
	delete(g.to, id)

	g.nodeIDs.Release(id)
}

// SetEdge adds e, an edge from one node to another. If the nodes do not exist, they are added
// and are set to the nodes of the edge otherwise.
// It will panic if the IDs of the e.From and e.To are equal.
func (g *DirectedGraph) SetEdge(e graph.Edge) {
	var (
		from = e.From()
		fid  = from.ID()
		to   = e.To()
		tid  = to.ID()
	)

	if fid == tid {
		panic("simple: adding self edge")
	}

	if _, ok := g.nodes[fid]; !ok {
		g.AddNode(from)
	} else {
		g.nodes[fid] = from
	}
	if _, ok := g.nodes[tid]; !ok {
		g.AddNode(to)
	} else {
		g.nodes[tid] = to
	}

	if fm, ok := g.from[fid]; ok {
		fm[tid] = e
	} else {
		g.from[fid] = map[int64]graph.Edge{tid: e}
	}
	if tm, ok := g.to[tid]; ok {
		tm[fid] = e
	} else {
		g.to[tid] = map[int64]graph.Edge{fid: e}
	}
}

// To returns all nodes in g that can reach directly to n.
//
// The returned graph.Nodes is only valid until the next mutation of
// the receiver.
func (g *DirectedGraph) To(id int64) graph.Nodes {
	if len(g.to[id]) == 0 {
		return graph.Empty
	}
	return iterator.NewNodesByEdge(g.nodes, g.to[id])
}

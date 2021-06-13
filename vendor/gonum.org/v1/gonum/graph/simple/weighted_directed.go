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
	wdg *WeightedDirectedGraph

	_ graph.Graph             = wdg
	_ graph.Weighted          = wdg
	_ graph.Directed          = wdg
	_ graph.WeightedDirected  = wdg
	_ graph.NodeAdder         = wdg
	_ graph.NodeRemover       = wdg
	_ graph.WeightedEdgeAdder = wdg
	_ graph.EdgeRemover       = wdg
)

// WeightedDirectedGraph implements a generalized weighted directed graph.
type WeightedDirectedGraph struct {
	nodes map[int64]graph.Node
	from  map[int64]map[int64]graph.WeightedEdge
	to    map[int64]map[int64]graph.WeightedEdge

	self, absent float64

	nodeIDs *uid.Set
}

// NewWeightedDirectedGraph returns a WeightedDirectedGraph with the specified self and absent
// edge weight values.
func NewWeightedDirectedGraph(self, absent float64) *WeightedDirectedGraph {
	return &WeightedDirectedGraph{
		nodes: make(map[int64]graph.Node),
		from:  make(map[int64]map[int64]graph.WeightedEdge),
		to:    make(map[int64]map[int64]graph.WeightedEdge),

		self:   self,
		absent: absent,

		nodeIDs: uid.NewSet(),
	}
}

// AddNode adds n to the graph. It panics if the added node ID matches an existing node ID.
func (g *WeightedDirectedGraph) AddNode(n graph.Node) {
	if _, exists := g.nodes[n.ID()]; exists {
		panic(fmt.Sprintf("simple: node ID collision: %d", n.ID()))
	}
	g.nodes[n.ID()] = n
	g.nodeIDs.Use(n.ID())
}

// Edge returns the edge from u to v if such an edge exists and nil otherwise.
// The node v must be directly reachable from u as defined by the From method.
func (g *WeightedDirectedGraph) Edge(uid, vid int64) graph.Edge {
	return g.WeightedEdge(uid, vid)
}

// Edges returns all the edges in the graph.
func (g *WeightedDirectedGraph) Edges() graph.Edges {
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
func (g *WeightedDirectedGraph) From(id int64) graph.Nodes {
	if len(g.from[id]) == 0 {
		return graph.Empty
	}
	return iterator.NewNodesByWeightedEdge(g.nodes, g.from[id])
}

// HasEdgeBetween returns whether an edge exists between nodes x and y without
// considering direction.
func (g *WeightedDirectedGraph) HasEdgeBetween(xid, yid int64) bool {
	if _, ok := g.from[xid][yid]; ok {
		return true
	}
	_, ok := g.from[yid][xid]
	return ok
}

// HasEdgeFromTo returns whether an edge exists in the graph from u to v.
func (g *WeightedDirectedGraph) HasEdgeFromTo(uid, vid int64) bool {
	if _, ok := g.from[uid][vid]; !ok {
		return false
	}
	return true
}

// NewNode returns a new unique Node to be added to g. The Node's ID does
// not become valid in g until the Node is added to g.
func (g *WeightedDirectedGraph) NewNode() graph.Node {
	if len(g.nodes) == 0 {
		return Node(0)
	}
	if int64(len(g.nodes)) == uid.Max {
		panic("simple: cannot allocate node: no slot")
	}
	return Node(g.nodeIDs.NewID())
}

// NewWeightedEdge returns a new weighted edge from the source to the destination node.
func (g *WeightedDirectedGraph) NewWeightedEdge(from, to graph.Node, weight float64) graph.WeightedEdge {
	return WeightedEdge{F: from, T: to, W: weight}
}

// Node returns the node with the given ID if it exists in the graph,
// and nil otherwise.
func (g *WeightedDirectedGraph) Node(id int64) graph.Node {
	return g.nodes[id]
}

// Nodes returns all the nodes in the graph.
//
// The returned graph.Nodes is only valid until the next mutation of
// the receiver.
func (g *WeightedDirectedGraph) Nodes() graph.Nodes {
	if len(g.nodes) == 0 {
		return graph.Empty
	}
	return iterator.NewNodes(g.nodes)
}

// RemoveEdge removes the edge with the given end point IDs from the graph, leaving the terminal
// nodes. If the edge does not exist it is a no-op.
func (g *WeightedDirectedGraph) RemoveEdge(fid, tid int64) {
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
func (g *WeightedDirectedGraph) RemoveNode(id int64) {
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

// SetWeightedEdge adds a weighted edge from one node to another. If the nodes do not exist, they are added
// and are set to the nodes of the edge otherwise.
// It will panic if the IDs of the e.From and e.To are equal.
func (g *WeightedDirectedGraph) SetWeightedEdge(e graph.WeightedEdge) {
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
		g.from[fid] = map[int64]graph.WeightedEdge{tid: e}
	}
	if tm, ok := g.to[tid]; ok {
		tm[fid] = e
	} else {
		g.to[tid] = map[int64]graph.WeightedEdge{fid: e}
	}
}

// To returns all nodes in g that can reach directly to n.
//
// The returned graph.Nodes is only valid until the next mutation of
// the receiver.
func (g *WeightedDirectedGraph) To(id int64) graph.Nodes {
	if len(g.to[id]) == 0 {
		return graph.Empty
	}
	return iterator.NewNodesByWeightedEdge(g.nodes, g.to[id])
}

// Weight returns the weight for the edge between x and y if Edge(x, y) returns a non-nil Edge.
// If x and y are the same node or there is no joining edge between the two nodes the weight
// value returned is either the graph's absent or self value. Weight returns true if an edge
// exists between x and y or if x and y have the same ID, false otherwise.
func (g *WeightedDirectedGraph) Weight(xid, yid int64) (w float64, ok bool) {
	if xid == yid {
		return g.self, true
	}
	if to, ok := g.from[xid]; ok {
		if e, ok := to[yid]; ok {
			return e.Weight(), true
		}
	}
	return g.absent, false
}

// WeightedEdge returns the weighted edge from u to v if such an edge exists and nil otherwise.
// The node v must be directly reachable from u as defined by the From method.
func (g *WeightedDirectedGraph) WeightedEdge(uid, vid int64) graph.WeightedEdge {
	edge, ok := g.from[uid][vid]
	if !ok {
		return nil
	}
	return edge
}

// WeightedEdges returns all the weighted edges in the graph.
func (g *WeightedDirectedGraph) WeightedEdges() graph.WeightedEdges {
	var edges []graph.WeightedEdge
	for _, u := range g.nodes {
		for _, e := range g.from[u.ID()] {
			edges = append(edges, e)
		}
	}
	if len(edges) == 0 {
		return graph.Empty
	}
	return iterator.NewOrderedWeightedEdges(edges)
}

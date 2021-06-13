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
	wug *WeightedUndirectedGraph

	_ graph.Graph              = wug
	_ graph.Weighted           = wug
	_ graph.Undirected         = wug
	_ graph.WeightedUndirected = wug
	_ graph.NodeAdder          = wug
	_ graph.NodeRemover        = wug
	_ graph.WeightedEdgeAdder  = wug
	_ graph.EdgeRemover        = wug
)

// WeightedUndirectedGraph implements a generalized weighted undirected graph.
type WeightedUndirectedGraph struct {
	nodes map[int64]graph.Node
	edges map[int64]map[int64]graph.WeightedEdge

	self, absent float64

	nodeIDs *uid.Set
}

// NewWeightedUndirectedGraph returns an WeightedUndirectedGraph with the specified self and absent
// edge weight values.
func NewWeightedUndirectedGraph(self, absent float64) *WeightedUndirectedGraph {
	return &WeightedUndirectedGraph{
		nodes: make(map[int64]graph.Node),
		edges: make(map[int64]map[int64]graph.WeightedEdge),

		self:   self,
		absent: absent,

		nodeIDs: uid.NewSet(),
	}
}

// AddNode adds n to the graph. It panics if the added node ID matches an existing node ID.
func (g *WeightedUndirectedGraph) AddNode(n graph.Node) {
	if _, exists := g.nodes[n.ID()]; exists {
		panic(fmt.Sprintf("simple: node ID collision: %d", n.ID()))
	}
	g.nodes[n.ID()] = n
	g.nodeIDs.Use(n.ID())
}

// Edge returns the edge from u to v if such an edge exists and nil otherwise.
// The node v must be directly reachable from u as defined by the From method.
func (g *WeightedUndirectedGraph) Edge(uid, vid int64) graph.Edge {
	return g.WeightedEdgeBetween(uid, vid)
}

// EdgeBetween returns the edge between nodes x and y.
func (g *WeightedUndirectedGraph) EdgeBetween(xid, yid int64) graph.Edge {
	return g.WeightedEdgeBetween(xid, yid)
}

// Edges returns all the edges in the graph.
func (g *WeightedUndirectedGraph) Edges() graph.Edges {
	if len(g.edges) == 0 {
		return graph.Empty
	}
	var edges []graph.Edge
	seen := make(map[[2]int64]struct{})
	for _, u := range g.edges {
		for _, e := range u {
			uid := e.From().ID()
			vid := e.To().ID()
			if _, ok := seen[[2]int64{uid, vid}]; ok {
				continue
			}
			seen[[2]int64{uid, vid}] = struct{}{}
			seen[[2]int64{vid, uid}] = struct{}{}
			edges = append(edges, e)
		}
	}
	if len(edges) == 0 {
		return graph.Empty
	}
	return iterator.NewOrderedEdges(edges)
}

// From returns all nodes in g that can be reached directly from n.
func (g *WeightedUndirectedGraph) From(id int64) graph.Nodes {
	if len(g.edges[id]) == 0 {
		return graph.Empty
	}
	return iterator.NewNodesByWeightedEdge(g.nodes, g.edges[id])
}

// HasEdgeBetween returns whether an edge exists between nodes x and y.
func (g *WeightedUndirectedGraph) HasEdgeBetween(xid, yid int64) bool {
	_, ok := g.edges[xid][yid]
	return ok
}

// NewNode returns a new unique Node to be added to g. The Node's ID does
// not become valid in g until the Node is added to g.
func (g *WeightedUndirectedGraph) NewNode() graph.Node {
	if len(g.nodes) == 0 {
		return Node(0)
	}
	if int64(len(g.nodes)) == uid.Max {
		panic("simple: cannot allocate node: no slot")
	}
	return Node(g.nodeIDs.NewID())
}

// NewWeightedEdge returns a new weighted edge from the source to the destination node.
func (g *WeightedUndirectedGraph) NewWeightedEdge(from, to graph.Node, weight float64) graph.WeightedEdge {
	return WeightedEdge{F: from, T: to, W: weight}
}

// Node returns the node with the given ID if it exists in the graph,
// and nil otherwise.
func (g *WeightedUndirectedGraph) Node(id int64) graph.Node {
	return g.nodes[id]
}

// Nodes returns all the nodes in the graph.
//
// The returned graph.Nodes is only valid until the next mutation of
// the receiver.
func (g *WeightedUndirectedGraph) Nodes() graph.Nodes {
	if len(g.nodes) == 0 {
		return graph.Empty
	}
	return iterator.NewNodes(g.nodes)
}

// RemoveEdge removes the edge with the given end point IDs from the graph, leaving the terminal
// nodes. If the edge does not exist  it is a no-op.
func (g *WeightedUndirectedGraph) RemoveEdge(fid, tid int64) {
	if _, ok := g.nodes[fid]; !ok {
		return
	}
	if _, ok := g.nodes[tid]; !ok {
		return
	}

	delete(g.edges[fid], tid)
	delete(g.edges[tid], fid)
}

// RemoveNode removes the node with the given ID from the graph, as well as any edges attached
// to it. If the node is not in the graph it is a no-op.
func (g *WeightedUndirectedGraph) RemoveNode(id int64) {
	if _, ok := g.nodes[id]; !ok {
		return
	}
	delete(g.nodes, id)

	for from := range g.edges[id] {
		delete(g.edges[from], id)
	}
	delete(g.edges, id)

	g.nodeIDs.Release(id)
}

// SetWeightedEdge adds a weighted edge from one node to another. If the nodes do not exist, they are added
// and are set to the nodes of the edge otherwise.
// It will panic if the IDs of the e.From and e.To are equal.
func (g *WeightedUndirectedGraph) SetWeightedEdge(e graph.WeightedEdge) {
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

	if fm, ok := g.edges[fid]; ok {
		fm[tid] = e
	} else {
		g.edges[fid] = map[int64]graph.WeightedEdge{tid: e}
	}
	if tm, ok := g.edges[tid]; ok {
		tm[fid] = e
	} else {
		g.edges[tid] = map[int64]graph.WeightedEdge{fid: e}
	}
}

// Weight returns the weight for the edge between x and y if Edge(x, y) returns a non-nil Edge.
// If x and y are the same node or there is no joining edge between the two nodes the weight
// value returned is either the graph's absent or self value. Weight returns true if an edge
// exists between x and y or if x and y have the same ID, false otherwise.
func (g *WeightedUndirectedGraph) Weight(xid, yid int64) (w float64, ok bool) {
	if xid == yid {
		return g.self, true
	}
	if n, ok := g.edges[xid]; ok {
		if e, ok := n[yid]; ok {
			return e.Weight(), true
		}
	}
	return g.absent, false
}

// WeightedEdge returns the weighted edge from u to v if such an edge exists and nil otherwise.
// The node v must be directly reachable from u as defined by the From method.
func (g *WeightedUndirectedGraph) WeightedEdge(uid, vid int64) graph.WeightedEdge {
	return g.WeightedEdgeBetween(uid, vid)
}

// WeightedEdgeBetween returns the weighted edge between nodes x and y.
func (g *WeightedUndirectedGraph) WeightedEdgeBetween(xid, yid int64) graph.WeightedEdge {
	edge, ok := g.edges[xid][yid]
	if !ok {
		return nil
	}
	if edge.From().ID() == xid {
		return edge
	}
	return edge.ReversedEdge().(graph.WeightedEdge)
}

// WeightedEdges returns all the weighted edges in the graph.
func (g *WeightedUndirectedGraph) WeightedEdges() graph.WeightedEdges {
	var edges []graph.WeightedEdge
	seen := make(map[[2]int64]struct{})
	for _, u := range g.edges {
		for _, e := range u {
			uid := e.From().ID()
			vid := e.To().ID()
			if _, ok := seen[[2]int64{uid, vid}]; ok {
				continue
			}
			seen[[2]int64{uid, vid}] = struct{}{}
			seen[[2]int64{vid, uid}] = struct{}{}
			edges = append(edges, e)
		}
	}
	if len(edges) == 0 {
		return graph.Empty
	}
	return iterator.NewOrderedWeightedEdges(edges)
}

// Copyright ©2018 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build safe

package iterator

import (
	"reflect"

	"gonum.org/v1/gonum/graph"
)

// Nodes implements the graph.Nodes interfaces.
// The iteration order of Nodes is randomized.
type Nodes struct {
	nodes reflect.Value
	iter  *reflect.MapIter
	pos   int
	curr  graph.Node
}

// NewNodes returns a Nodes initialized with the provided nodes, a
// map of node IDs to graph.Nodes. No check is made that the keys
// match the graph.Node IDs, and the map keys are not used.
//
// Behavior of the Nodes is unspecified if nodes is mutated after
// the call the NewNodes.
func NewNodes(nodes map[int64]graph.Node) *Nodes {
	rv := reflect.ValueOf(nodes)
	return &Nodes{nodes: rv, iter: rv.MapRange()}
}

// Len returns the remaining number of nodes to be iterated over.
func (n *Nodes) Len() int {
	return n.nodes.Len() - n.pos
}

// Next returns whether the next call of Node will return a valid node.
func (n *Nodes) Next() bool {
	if n.pos >= n.nodes.Len() {
		return false
	}
	ok := n.iter.Next()
	if ok {
		n.pos++
		n.curr = n.iter.Value().Interface().(graph.Node)
	}
	return ok
}

// Node returns the current node of the iterator. Next must have been
// called prior to a call to Node.
func (n *Nodes) Node() graph.Node {
	return n.curr
}

// Reset returns the iterator to its initial state.
func (n *Nodes) Reset() {
	n.curr = nil
	n.pos = 0
	n.iter = n.nodes.MapRange()
}

// NodeSlice returns all the remaining nodes in the iterator and advances
// the iterator. The order of nodes within the returned slice is not
// specified.
func (n *Nodes) NodeSlice() []graph.Node {
	if n.Len() == 0 {
		return nil
	}
	nodes := make([]graph.Node, 0, n.Len())
	for n.iter.Next() {
		nodes = append(nodes, n.iter.Value().Interface().(graph.Node))
	}
	n.pos = n.nodes.Len()
	return nodes
}

// NodesByEdge implements the graph.Nodes interfaces.
// The iteration order of Nodes is randomized.
type NodesByEdge struct {
	nodes map[int64]graph.Node
	edges reflect.Value
	iter  *reflect.MapIter
	pos   int
	curr  graph.Node
}

// NewNodesByEdge returns a NodesByEdge initialized with the
// provided nodes, a map of node IDs to graph.Nodes, and the set
// of edges, a map of to-node IDs to graph.Edge, that can be
// traversed to reach the nodes that the NodesByEdge will iterate
// over. No check is made that the keys match the graph.Node IDs,
// and the map keys are not used.
//
// Behavior of the NodesByEdge is unspecified if nodes or edges
// is mutated after the call the NewNodes.
func NewNodesByEdge(nodes map[int64]graph.Node, edges map[int64]graph.Edge) *NodesByEdge {
	rv := reflect.ValueOf(edges)
	return &NodesByEdge{nodes: nodes, edges: rv, iter: rv.MapRange()}
}

// NewNodesByWeightedEdge returns a NodesByEdge initialized with the
// provided nodes, a map of node IDs to graph.Nodes, and the set
// of edges, a map of to-node IDs to graph.WeightedEdge, that can be
// traversed to reach the nodes that the NodesByEdge will iterate
// over. No check is made that the keys match the graph.Node IDs,
// and the map keys are not used.
//
// Behavior of the NodesByEdge is unspecified if nodes or edges
// is mutated after the call the NewNodes.
func NewNodesByWeightedEdge(nodes map[int64]graph.Node, edges map[int64]graph.WeightedEdge) *NodesByEdge {
	rv := reflect.ValueOf(edges)
	return &NodesByEdge{nodes: nodes, edges: rv, iter: rv.MapRange()}
}

// NewNodesByLines returns a NodesByEdge initialized with the
// provided nodes, a map of node IDs to graph.Nodes, and the set
// of lines, a map to-node IDs to map of graph.Line, that can be
// traversed to reach the nodes that the NodesByEdge will iterate
// over. No check is made that the keys match the graph.Node IDs,
// and the map keys are not used.
//
// Behavior of the NodesByEdge is unspecified if nodes or lines
// is mutated after the call the NewNodes.
func NewNodesByLines(nodes map[int64]graph.Node, lines map[int64]map[int64]graph.Line) *NodesByEdge {
	rv := reflect.ValueOf(lines)
	return &NodesByEdge{nodes: nodes, edges: rv, iter: rv.MapRange()}
}

// NewNodesByWeightedLines returns a NodesByEdge initialized with the
// provided nodes, a map of node IDs to graph.Nodes, and the set
// of lines, a map to-node IDs to map of graph.WeightedLine, that can be
// traversed to reach the nodes that the NodesByEdge will iterate
// over. No check is made that the keys match the graph.Node IDs,
// and the map keys are not used.
//
// Behavior of the NodesByEdge is unspecified if nodes or lines
// is mutated after the call the NewNodes.
func NewNodesByWeightedLines(nodes map[int64]graph.Node, lines map[int64]map[int64]graph.WeightedLine) *NodesByEdge {
	rv := reflect.ValueOf(lines)
	return &NodesByEdge{nodes: nodes, edges: rv, iter: rv.MapRange()}
}

// Len returns the remaining number of nodes to be iterated over.
func (n *NodesByEdge) Len() int {
	return n.edges.Len() - n.pos
}

// Next returns whether the next call of Node will return a valid node.
func (n *NodesByEdge) Next() bool {
	if n.pos >= n.edges.Len() {
		return false
	}
	ok := n.iter.Next()
	if ok {
		n.pos++
		n.curr = n.nodes[n.iter.Key().Int()]
	}
	return ok
}

// Node returns the current node of the iterator. Next must have been
// called prior to a call to Node.
func (n *NodesByEdge) Node() graph.Node {
	return n.curr
}

// Reset returns the iterator to its initial state.
func (n *NodesByEdge) Reset() {
	n.curr = nil
	n.pos = 0
	n.iter = n.edges.MapRange()
}

// NodeSlice returns all the remaining nodes in the iterator and advances
// the iterator. The order of nodes within the returned slice is not
// specified.
func (n *NodesByEdge) NodeSlice() []graph.Node {
	if n.Len() == 0 {
		return nil
	}
	nodes := make([]graph.Node, 0, n.Len())
	for n.iter.Next() {
		nodes = append(nodes, n.nodes[n.iter.Key().Int()])
	}
	n.pos = n.edges.Len()
	return nodes
}

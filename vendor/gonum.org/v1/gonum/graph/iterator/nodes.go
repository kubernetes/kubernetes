// Copyright ©2018 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iterator

import "gonum.org/v1/gonum/graph"

// OrderedNodes implements the graph.Nodes and graph.NodeSlicer interfaces.
// The iteration order of OrderedNodes is the order of nodes passed to
// NewNodeIterator.
type OrderedNodes struct {
	idx   int
	nodes []graph.Node
}

// NewOrderedNodes returns a OrderedNodes initialized with the provided nodes.
func NewOrderedNodes(nodes []graph.Node) *OrderedNodes {
	return &OrderedNodes{idx: -1, nodes: nodes}
}

// Len returns the remaining number of nodes to be iterated over.
func (n *OrderedNodes) Len() int {
	if n.idx >= len(n.nodes) {
		return 0
	}
	if n.idx <= 0 {
		return len(n.nodes)
	}
	return len(n.nodes[n.idx:])
}

// Next returns whether the next call of Node will return a valid node.
func (n *OrderedNodes) Next() bool {
	if uint(n.idx)+1 < uint(len(n.nodes)) {
		n.idx++
		return true
	}
	n.idx = len(n.nodes)
	return false
}

// Node returns the current node of the iterator. Next must have been
// called prior to a call to Node.
func (n *OrderedNodes) Node() graph.Node {
	if n.idx >= len(n.nodes) || n.idx < 0 {
		return nil
	}
	return n.nodes[n.idx]
}

// NodeSlice returns all the remaining nodes in the iterator and advances
// the iterator.
func (n *OrderedNodes) NodeSlice() []graph.Node {
	if n.idx >= len(n.nodes) {
		return nil
	}
	idx := n.idx
	if idx == -1 {
		idx = 0
	}
	n.idx = len(n.nodes)
	return n.nodes[idx:]
}

// Reset returns the iterator to its initial state.
func (n *OrderedNodes) Reset() {
	n.idx = -1
}

// ImplicitNodes implements the graph.Nodes interface for a set of nodes over
// a contiguous ID range.
type ImplicitNodes struct {
	beg, end int
	curr     int
	newNode  func(id int) graph.Node
}

// NewImplicitNodes returns a new implicit node iterator spanning nodes in [beg,end).
// The provided new func maps the id to a graph.Node. NewImplicitNodes will panic
// if beg is greater than end.
func NewImplicitNodes(beg, end int, new func(id int) graph.Node) *ImplicitNodes {
	if beg > end {
		panic("iterator: invalid range")
	}
	return &ImplicitNodes{beg: beg, end: end, curr: beg - 1, newNode: new}
}

// Len returns the remaining number of nodes to be iterated over.
func (n *ImplicitNodes) Len() int {
	return n.end - n.curr - 1
}

// Next returns whether the next call of Node will return a valid node.
func (n *ImplicitNodes) Next() bool {
	if n.curr == n.end {
		return false
	}
	n.curr++
	return n.curr < n.end
}

// Node returns the current node of the iterator. Next must have been
// called prior to a call to Node.
func (n *ImplicitNodes) Node() graph.Node {
	if n.Len() == -1 || n.curr < n.beg {
		return nil
	}
	return n.newNode(n.curr)
}

// Reset returns the iterator to its initial state.
func (n *ImplicitNodes) Reset() {
	n.curr = n.beg - 1
}

// NodeSlice returns all the remaining nodes in the iterator and advances
// the iterator.
func (n *ImplicitNodes) NodeSlice() []graph.Node {
	nodes := make([]graph.Node, 0, n.Len())
	for n.curr++; n.curr < n.end; n.curr++ {
		nodes = append(nodes, n.newNode(n.curr))
	}
	return nodes
}

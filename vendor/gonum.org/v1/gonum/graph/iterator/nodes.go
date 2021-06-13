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
	return len(n.nodes[n.idx+1:])
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
	idx := n.idx + 1
	n.idx = len(n.nodes)
	return n.nodes[idx:]
}

// Reset returns the iterator to its initial state.
func (n *OrderedNodes) Reset() {
	n.idx = -1
}

// LazyOrderedNodes implements the graph.Nodes and graph.NodeSlicer interfaces.
// The iteration order of LazyOrderedNodes is not determined until the first
// call to Next or NodeSlice. After that, the iteration order is fixed.
type LazyOrderedNodes struct {
	iter  OrderedNodes
	nodes map[int64]graph.Node
}

// NewLazyOrderedNodes returns a LazyOrderedNodes initialized with the provided nodes.
func NewLazyOrderedNodes(nodes map[int64]graph.Node) *LazyOrderedNodes {
	return &LazyOrderedNodes{nodes: nodes}
}

// Len returns the remaining number of nodes to be iterated over.
func (n *LazyOrderedNodes) Len() int {
	if n.iter.nodes == nil {
		return len(n.nodes)
	}
	return n.iter.Len()
}

// Next returns whether the next call of Node will return a valid node.
func (n *LazyOrderedNodes) Next() bool {
	if n.iter.nodes == nil {
		n.fillSlice()
	}
	return n.iter.Next()
}

// Node returns the current node of the iterator. Next must have been
// called prior to a call to Node.
func (n *LazyOrderedNodes) Node() graph.Node {
	return n.iter.Node()
}

// NodeSlice returns all the remaining nodes in the iterator and advances
// the iterator.
func (n *LazyOrderedNodes) NodeSlice() []graph.Node {
	if n.iter.nodes == nil {
		n.fillSlice()
	}
	return n.iter.NodeSlice()
}

// Reset returns the iterator to its initial state.
func (n *LazyOrderedNodes) Reset() {
	n.iter.Reset()
}

func (n *LazyOrderedNodes) fillSlice() {
	n.iter = OrderedNodes{idx: -1, nodes: make([]graph.Node, len(n.nodes))}
	i := 0
	for _, u := range n.nodes {
		n.iter.nodes[i] = u
		i++
	}
	n.nodes = nil
}

// LazyOrderedNodesByEdge implements the graph.Nodes and graph.NodeSlicer interfaces.
// The iteration order of LazyOrderedNodesByEdge is not determined until the first
// call to Next or NodeSlice. After that, the iteration order is fixed.
type LazyOrderedNodesByEdge struct {
	iter  OrderedNodes
	nodes map[int64]graph.Node
	edges map[int64]graph.Edge
}

// NewLazyOrderedNodesByEdge returns a LazyOrderedNodesByEdge initialized with the
// provided nodes.
func NewLazyOrderedNodesByEdge(nodes map[int64]graph.Node, edges map[int64]graph.Edge) *LazyOrderedNodesByEdge {
	return &LazyOrderedNodesByEdge{nodes: nodes, edges: edges}
}

// Len returns the remaining number of nodes to be iterated over.
func (n *LazyOrderedNodesByEdge) Len() int {
	if n.iter.nodes == nil {
		return len(n.edges)
	}
	return n.iter.Len()
}

// Next returns whether the next call of Node will return a valid node.
func (n *LazyOrderedNodesByEdge) Next() bool {
	if n.iter.nodes == nil {
		n.fillSlice()
	}
	return n.iter.Next()
}

// Node returns the current node of the iterator. Next must have been
// called prior to a call to Node.
func (n *LazyOrderedNodesByEdge) Node() graph.Node {
	return n.iter.Node()
}

// NodeSlice returns all the remaining nodes in the iterator and advances
// the iterator.
func (n *LazyOrderedNodesByEdge) NodeSlice() []graph.Node {
	if n.iter.nodes == nil {
		n.fillSlice()
	}
	return n.iter.NodeSlice()
}

// Reset returns the iterator to its initial state.
func (n *LazyOrderedNodesByEdge) Reset() {
	n.iter.Reset()
}

func (n *LazyOrderedNodesByEdge) fillSlice() {
	n.iter = OrderedNodes{idx: -1, nodes: make([]graph.Node, len(n.edges))}
	i := 0
	for id := range n.edges {
		n.iter.nodes[i] = n.nodes[id]
		i++
	}
	n.nodes = nil
	n.edges = nil
}

// LazyOrderedNodesByWeightedEdge implements the graph.Nodes and graph.NodeSlicer interfaces.
// The iteration order of LazyOrderedNodesByEeightedEdge is not determined until the first
// call to Next or NodeSlice. After that, the iteration order is fixed.
type LazyOrderedNodesByWeightedEdge struct {
	iter  OrderedNodes
	nodes map[int64]graph.Node
	edges map[int64]graph.WeightedEdge
}

// NewLazyOrderedNodesByWeightedEdge returns a LazyOrderedNodesByEdge initialized with the
// provided nodes.
func NewLazyOrderedNodesByWeightedEdge(nodes map[int64]graph.Node, edges map[int64]graph.WeightedEdge) *LazyOrderedNodesByWeightedEdge {
	return &LazyOrderedNodesByWeightedEdge{nodes: nodes, edges: edges}
}

// Len returns the remaining number of nodes to be iterated over.
func (n *LazyOrderedNodesByWeightedEdge) Len() int {
	if n.iter.nodes == nil {
		return len(n.edges)
	}
	return n.iter.Len()
}

// Next returns whether the next call of Node will return a valid node.
func (n *LazyOrderedNodesByWeightedEdge) Next() bool {
	if n.iter.nodes == nil {
		n.fillSlice()
	}
	return n.iter.Next()
}

// Node returns the current node of the iterator. Next must have been
// called prior to a call to Node.
func (n *LazyOrderedNodesByWeightedEdge) Node() graph.Node {
	return n.iter.Node()
}

// NodeSlice returns all the remaining nodes in the iterator and advances
// the iterator.
func (n *LazyOrderedNodesByWeightedEdge) NodeSlice() []graph.Node {
	if n.iter.nodes == nil {
		n.fillSlice()
	}
	return n.iter.NodeSlice()
}

// Reset returns the iterator to its initial state.
func (n *LazyOrderedNodesByWeightedEdge) Reset() {
	n.iter.Reset()
}

func (n *LazyOrderedNodesByWeightedEdge) fillSlice() {
	n.iter = OrderedNodes{idx: -1, nodes: make([]graph.Node, len(n.edges))}
	i := 0
	for id := range n.edges {
		n.iter.nodes[i] = n.nodes[id]
		i++
	}
	n.nodes = nil
	n.edges = nil
}

// LazyOrderedNodesByLines implements the graph.Nodes and graph.NodeSlicer interfaces.
// The iteration order of LazyOrderedNodesByLines is not determined until the first
// call to Next or NodeSlice. After that, the iteration order is fixed.
type LazyOrderedNodesByLines struct {
	iter  OrderedNodes
	nodes map[int64]graph.Node
	edges map[int64]map[int64]graph.Line
}

// NewLazyOrderedNodesByLine returns a LazyOrderedNodesByLines initialized with the
// provided nodes.
func NewLazyOrderedNodesByLines(nodes map[int64]graph.Node, edges map[int64]map[int64]graph.Line) *LazyOrderedNodesByLines {
	return &LazyOrderedNodesByLines{nodes: nodes, edges: edges}
}

// Len returns the remaining number of nodes to be iterated over.
func (n *LazyOrderedNodesByLines) Len() int {
	if n.iter.nodes == nil {
		return len(n.edges)
	}
	return n.iter.Len()
}

// Next returns whether the next call of Node will return a valid node.
func (n *LazyOrderedNodesByLines) Next() bool {
	if n.iter.nodes == nil {
		n.fillSlice()
	}
	return n.iter.Next()
}

// Node returns the current node of the iterator. Next must have been
// called prior to a call to Node.
func (n *LazyOrderedNodesByLines) Node() graph.Node {
	return n.iter.Node()
}

// NodeSlice returns all the remaining nodes in the iterator and advances
// the iterator.
func (n *LazyOrderedNodesByLines) NodeSlice() []graph.Node {
	if n.iter.nodes == nil {
		n.fillSlice()
	}
	return n.iter.NodeSlice()
}

// Reset returns the iterator to its initial state.
func (n *LazyOrderedNodesByLines) Reset() {
	n.iter.Reset()
}

func (n *LazyOrderedNodesByLines) fillSlice() {
	n.iter = OrderedNodes{idx: -1, nodes: make([]graph.Node, len(n.edges))}
	i := 0
	for id := range n.edges {
		n.iter.nodes[i] = n.nodes[id]
		i++
	}
	n.nodes = nil
	n.edges = nil
}

// LazyOrderedNodesByWeightedLines implements the graph.Nodes and graph.NodeSlicer interfaces.
// The iteration order of LazyOrderedNodesByEeightedLine is not determined until the first
// call to Next or NodeSlice. After that, the iteration order is fixed.
type LazyOrderedNodesByWeightedLines struct {
	iter  OrderedNodes
	nodes map[int64]graph.Node
	edges map[int64]map[int64]graph.WeightedLine
}

// NewLazyOrderedNodesByWeightedLines returns a LazyOrderedNodesByLines initialized with the
// provided nodes.
func NewLazyOrderedNodesByWeightedLines(nodes map[int64]graph.Node, edges map[int64]map[int64]graph.WeightedLine) *LazyOrderedNodesByWeightedLines {
	return &LazyOrderedNodesByWeightedLines{nodes: nodes, edges: edges}
}

// Len returns the remaining number of nodes to be iterated over.
func (n *LazyOrderedNodesByWeightedLines) Len() int {
	if n.iter.nodes == nil {
		return len(n.edges)
	}
	return n.iter.Len()
}

// Next returns whether the next call of Node will return a valid node.
func (n *LazyOrderedNodesByWeightedLines) Next() bool {
	if n.iter.nodes == nil {
		n.fillSlice()
	}
	return n.iter.Next()
}

// Node returns the current node of the iterator. Next must have been
// called prior to a call to Node.
func (n *LazyOrderedNodesByWeightedLines) Node() graph.Node {
	return n.iter.Node()
}

// NodeSlice returns all the remaining nodes in the iterator and advances
// the iterator.
func (n *LazyOrderedNodesByWeightedLines) NodeSlice() []graph.Node {
	if n.iter.nodes == nil {
		n.fillSlice()
	}
	return n.iter.NodeSlice()
}

// Reset returns the iterator to its initial state.
func (n *LazyOrderedNodesByWeightedLines) Reset() {
	n.iter.Reset()
}

func (n *LazyOrderedNodesByWeightedLines) fillSlice() {
	n.iter = OrderedNodes{idx: -1, nodes: make([]graph.Node, len(n.edges))}
	i := 0
	for id := range n.edges {
		n.iter.nodes[i] = n.nodes[id]
		i++
	}
	n.nodes = nil
	n.edges = nil
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
	if n.Len() == 0 {
		return nil
	}
	nodes := make([]graph.Node, 0, n.Len())
	for n.curr++; n.curr < n.end; n.curr++ {
		nodes = append(nodes, n.newNode(n.curr))
	}
	return nodes
}

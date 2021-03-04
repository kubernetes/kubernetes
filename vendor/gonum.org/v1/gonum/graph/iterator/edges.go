// Copyright ©2018 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iterator

import "gonum.org/v1/gonum/graph"

// OrderedEdges implements the graph.Edges and graph.EdgeSlicer interfaces.
// The iteration order of OrderedEdges is the order of edges passed to
// NewEdgeIterator.
type OrderedEdges struct {
	idx   int
	edges []graph.Edge
}

// NewOrderedEdges returns an OrderedEdges initialized with the provided edges.
func NewOrderedEdges(edges []graph.Edge) *OrderedEdges {
	return &OrderedEdges{idx: -1, edges: edges}
}

// Len returns the remaining number of edges to be iterated over.
func (e *OrderedEdges) Len() int {
	if e.idx >= len(e.edges) {
		return 0
	}
	if e.idx <= 0 {
		return len(e.edges)
	}
	return len(e.edges[e.idx:])
}

// Next returns whether the next call of Edge will return a valid edge.
func (e *OrderedEdges) Next() bool {
	if uint(e.idx)+1 < uint(len(e.edges)) {
		e.idx++
		return true
	}
	e.idx = len(e.edges)
	return false
}

// Edge returns the current edge of the iterator. Next must have been
// called prior to a call to Edge.
func (e *OrderedEdges) Edge() graph.Edge {
	if e.idx >= len(e.edges) || e.idx < 0 {
		return nil
	}
	return e.edges[e.idx]
}

// EdgeSlice returns all the remaining edges in the iterator and advances
// the iterator.
func (e *OrderedEdges) EdgeSlice() []graph.Edge {
	if e.idx >= len(e.edges) {
		return nil
	}
	idx := e.idx
	if idx == -1 {
		idx = 0
	}
	e.idx = len(e.edges)
	return e.edges[idx:]
}

// Reset returns the iterator to its initial state.
func (e *OrderedEdges) Reset() {
	e.idx = -1
}

// OrderedWeightedEdges implements the graph.Edges and graph.EdgeSlicer interfaces.
// The iteration order of OrderedWeightedEdges is the order of edges passed to
// NewEdgeIterator.
type OrderedWeightedEdges struct {
	idx   int
	edges []graph.WeightedEdge
}

// NewOrderedWeightedEdges returns an OrderedWeightedEdges initialized with the provided edges.
func NewOrderedWeightedEdges(edges []graph.WeightedEdge) *OrderedWeightedEdges {
	return &OrderedWeightedEdges{idx: -1, edges: edges}
}

// Len returns the remaining number of edges to be iterated over.
func (e *OrderedWeightedEdges) Len() int {
	if e.idx >= len(e.edges) {
		return 0
	}
	if e.idx <= 0 {
		return len(e.edges)
	}
	return len(e.edges[e.idx:])
}

// Next returns whether the next call of WeightedEdge will return a valid edge.
func (e *OrderedWeightedEdges) Next() bool {
	if uint(e.idx)+1 < uint(len(e.edges)) {
		e.idx++
		return true
	}
	e.idx = len(e.edges)
	return false
}

// WeightedEdge returns the current edge of the iterator. Next must have been
// called prior to a call to WeightedEdge.
func (e *OrderedWeightedEdges) WeightedEdge() graph.WeightedEdge {
	if e.idx >= len(e.edges) || e.idx < 0 {
		return nil
	}
	return e.edges[e.idx]
}

// WeightedEdgeSlice returns all the remaining edges in the iterator and advances
// the iterator.
func (e *OrderedWeightedEdges) WeightedEdgeSlice() []graph.WeightedEdge {
	if e.idx >= len(e.edges) {
		return nil
	}
	idx := e.idx
	if idx == -1 {
		idx = 0
	}
	e.idx = len(e.edges)
	return e.edges[idx:]
}

// Reset returns the iterator to its initial state.
func (e *OrderedWeightedEdges) Reset() {
	e.idx = -1
}

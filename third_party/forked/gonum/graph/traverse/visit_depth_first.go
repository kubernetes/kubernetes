// Copyright ©2015 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package traverse provides basic graph traversal primitives.
package traverse

import (
	"k8s.io/kubernetes/third_party/forked/gonum/graph"
	"k8s.io/kubernetes/third_party/forked/gonum/graph/internal/linear"
)

// VisitableGraph
type VisitableGraph interface {
	graph.Graph

	// VisitFrom invokes visitor with all nodes that can be reached directly from the given node.
	// If visitor returns false, visiting is short-circuited.
	VisitFrom(from graph.Node, visitor func(graph.Node) (shouldContinue bool))

	// VisitTo invokes visitor with all nodes that can directly reach the given node.
	// If visitor returns false, visiting is short-circuited.
	VisitTo(to graph.Node, visitor func(graph.Node) (shouldContinue bool))
}

// VisitingDepthFirst implements stateful depth-first graph traversal on a visitable graph.
type VisitingDepthFirst struct {
	EdgeFilter func(graph.Edge) bool
	Visit      func(u, v graph.Node)
	stack      linear.NodeStack
	visited    visitedSet

	// visiting and found hold the state the walk's visitor works on. They are
	// fields rather than locals so that the visitor captures nothing but the
	// traverser itself and is allocated once per walk instead of once per node.
	visiting graph.Node
	found    graph.Node
}

// Walk performs a depth-first traversal of the graph g starting from the given node,
// depending on the EdgeFilter field and the until parameter if they are non-nil. The
// traversal follows edges for which EdgeFilter(edge) is true and returns the first node
// for which until(node) is true. During the traversal, if the Visit field is non-nil, it
// is called with the nodes joined by each followed edge.
func (d *VisitingDepthFirst) Walk(g VisitableGraph, from graph.Node, until func(graph.Node) bool) graph.Node {
	return d.walk(g, from, until, false)
}

// WalkTo performs the same traversal as Walk with the edges followed in the opposite
// direction: it starts at the given node and visits the nodes that can reach it.
func (d *VisitingDepthFirst) WalkTo(g VisitableGraph, to graph.Node, until func(graph.Node) bool) graph.Node {
	return d.walk(g, to, until, true)
}

func (d *VisitingDepthFirst) walk(g VisitableGraph, from graph.Node, until func(graph.Node) bool, reverse bool) graph.Node {
	d.stack.Push(from)
	d.visited.insert(from.ID())
	if until != nil && until(from) {
		return from
	}

	d.found = nil
	visitor := func(n graph.Node) (shouldContinue bool) {
		if d.EdgeFilter != nil && !d.EdgeFilter(g.Edge(d.visiting, n)) {
			return true
		}
		if d.visited.has(n.ID()) {
			return true
		}
		if d.Visit != nil {
			d.Visit(d.visiting, n)
		}
		d.visited.insert(n.ID())
		d.stack.Push(n)
		if until != nil && until(n) {
			d.found = n
			return false
		}
		return true
	}

	for d.stack.Len() > 0 {
		d.visiting = d.stack.Pop()
		if reverse {
			g.VisitTo(d.visiting, visitor)
		} else {
			g.VisitFrom(d.visiting, visitor)
		}
		if d.found != nil {
			return d.found
		}
	}
	return nil
}

// Visited returned whether the node n was visited during a traverse.
func (d *VisitingDepthFirst) Visited(n graph.Node) bool {
	return d.visited.has(n.ID())
}

// Reset resets the state of the traverser for reuse.
func (d *VisitingDepthFirst) Reset() {
	d.stack = d.stack[:0]
	d.visited.clear()
	d.visiting, d.found = nil, nil
}

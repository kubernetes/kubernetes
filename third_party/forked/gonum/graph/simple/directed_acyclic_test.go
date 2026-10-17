// Copyright ©2014 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package simple

import (
	"fmt"
	"math"
	"testing"

	"k8s.io/kubernetes/third_party/forked/gonum/graph"
)

var _ graph.Graph = &DirectedAcyclicGraph{}
var _ graph.Directed = &DirectedAcyclicGraph{}

// Tests Issue #27
func TestAcyclicEdgeOvercounting(t *testing.T) {
	g := generateDummyAcyclicGraph()

	if neigh := g.From(Node(Node(2))); len(neigh) != 2 {
		t.Errorf("Node 2 has incorrect number of neighbors got neighbors %v (count %d), expected 2 neighbors {0,1}", neigh, len(neigh))
	}
}

func generateDummyAcyclicGraph() *DirectedAcyclicGraph {
	nodes := [4]struct{ srcID, targetID int }{
		{2, 1},
		{1, 0},
		{0, 2},
		{2, 0},
	}

	g := NewDirectedAcyclicGraph(0, math.Inf(1))

	for _, n := range nodes {
		g.SetEdge(Edge{F: Node(n.srcID), T: Node(n.targetID), W: 1})
	}

	return g
}

// Test for issue #123 https://github.com/gonum/graph/issues/123
func TestAcyclicIssue123DirectedGraph(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("unexpected panic: %v", r)
		}
	}()
	g := NewDirectedAcyclicGraph(0, math.Inf(1))

	n0 := Node(g.NewNodeID())
	g.AddNode(n0)

	n1 := Node(g.NewNodeID())
	g.AddNode(n1)

	g.RemoveNode(n0)

	n2 := Node(g.NewNodeID())
	g.AddNode(n2)
}

func TestAcyclicVisitStopsEarly(t *testing.T) {
	// 6 edges on the hub promotes its edgeHolder to a map.
	g := NewDirectedAcyclicGraph(0, math.Inf(1))
	for i := 1; i <= 3; i++ {
		g.SetEdge(Edge{F: Node(0), T: Node(i), W: 1})
	}
	for i := 4; i <= 6; i++ {
		g.SetEdge(Edge{F: Node(i), T: Node(0), W: 1})
	}

	fromCalls := 0
	g.VisitFrom(Node(0), func(n graph.Node) bool {
		fromCalls++
		if n.ID() < 1 || n.ID() > 3 {
			t.Errorf("VisitFrom visited unexpected node %d", n.ID())
		}
		return false
	})
	if fromCalls != 1 {
		t.Errorf("expected VisitFrom visitor to be called once, got %d", fromCalls)
	}

	toCalls := 0
	g.VisitTo(Node(0), func(n graph.Node) bool {
		toCalls++
		if n.ID() < 4 || n.ID() > 6 {
			t.Errorf("VisitTo visited unexpected node %d", n.ID())
		}
		return false
	})
	if toCalls != 1 {
		t.Errorf("expected VisitTo visitor to be called once, got %d", toCalls)
	}

	// Returning true must still visit every neighbor in the right direction.
	fromAll, toAll := 0, 0
	g.VisitFrom(Node(0), func(graph.Node) bool { fromAll++; return true })
	g.VisitTo(Node(0), func(graph.Node) bool { toAll++; return true })
	if fromAll != 3 || toAll != 3 {
		t.Errorf("expected 3 from and 3 to neighbors, got %d and %d", fromAll, toAll)
	}
}

// BenchmarkAcyclicVisitStopsEarly measures a visitor that stops after the
// first neighbor, which is what VisitingDepthFirst.Walk does once it finds
// its target. Cost should not scale with the hub's degree.
func BenchmarkAcyclicVisitStopsEarly(b *testing.B) {
	for _, degree := range []int{2, 100, 10000} {
		// Hub 0 has degree outgoing edges and degree incoming edges.
		g := NewDirectedAcyclicGraph(0, math.Inf(1))
		for i := 1; i <= degree; i++ {
			g.SetEdge(Edge{F: Node(0), T: Node(i), W: 1})
			g.SetEdge(Edge{F: Node(degree + i), T: Node(0), W: 1})
		}
		stop := func(graph.Node) bool { return false }

		b.Run(fmt.Sprintf("VisitFrom/degree=%d", degree), func(b *testing.B) {
			for b.Loop() {
				g.VisitFrom(Node(0), stop)
			}
		})
		b.Run(fmt.Sprintf("VisitTo/degree=%d", degree), func(b *testing.B) {
			for b.Loop() {
				g.VisitTo(Node(0), stop)
			}
		})
	}
}

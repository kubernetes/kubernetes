// Copyright ©2014 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package simple

import (
	"math"
	"testing"

	"k8s.io/kubernetes/third_party/forked/gonum/graph"
)

var _ graph.Graph = (*UndirectedGraph)(nil)

func TestAssertMutableNotDirected(t *testing.T) {
	var g graph.UndirectedBuilder = NewUndirectedGraph(0, math.Inf(1))
	if _, ok := g.(graph.Directed); ok {
		t.Fatal("Graph is directed, but a MutableGraph cannot safely be directed!")
	}
}

func TestMaxID(t *testing.T) {
	g := NewUndirectedGraph(0, math.Inf(1))
	nodes := make(map[graph.Node]struct{})
	for i := Node(0); i < 3; i++ {
		g.AddNode(i)
		nodes[i] = struct{}{}
	}
	g.RemoveNode(Node(0))
	delete(nodes, Node(0))
	g.RemoveNode(Node(2))
	delete(nodes, Node(2))
	n := Node(g.NewNodeID())
	g.AddNode(n)
	if !g.Has(n) {
		t.Error("added node does not exist in graph")
	}
	if _, exists := nodes[n]; exists {
		t.Errorf("Created already existing node id: %v", n.ID())
	}
}

// Test for issue #123 https://github.com/gonum/graph/issues/123
func TestIssue123UndirectedGraph(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("unexpected panic: %v", r)
		}
	}()
	g := NewUndirectedGraph(0, math.Inf(1))

	n0 := Node(g.NewNodeID())
	g.AddNode(n0)

	n1 := Node(g.NewNodeID())
	g.AddNode(n1)

	g.RemoveNode(n0)

	n2 := Node(g.NewNodeID())
	g.AddNode(n2)
}

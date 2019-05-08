// Copyright ©2014 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package simple

import (
	"math"

	"gonum.org/v1/gonum/graph"
)

// Node is a simple graph node.
type Node int64

// ID returns the ID number of the node.
func (n Node) ID() int64 {
	return int64(n)
}

func newSimpleNode(id int) graph.Node {
	return Node(id)
}

// Edge is a simple graph edge.
type Edge struct {
	F, T graph.Node
}

// From returns the from-node of the edge.
func (e Edge) From() graph.Node { return e.F }

// To returns the to-node of the edge.
func (e Edge) To() graph.Node { return e.T }

// ReversedLine returns a new Edge with the F and T fields
// swapped.
func (e Edge) ReversedEdge() graph.Edge { return Edge{F: e.T, T: e.F} }

// WeightedEdge is a simple weighted graph edge.
type WeightedEdge struct {
	F, T graph.Node
	W    float64
}

// From returns the from-node of the edge.
func (e WeightedEdge) From() graph.Node { return e.F }

// To returns the to-node of the edge.
func (e WeightedEdge) To() graph.Node { return e.T }

// ReversedLine returns a new Edge with the F and T fields
// swapped. The weight of the new Edge is the same as
// the weight of the receiver.
func (e WeightedEdge) ReversedEdge() graph.Edge { return WeightedEdge{F: e.T, T: e.F, W: e.W} }

// Weight returns the weight of the edge.
func (e WeightedEdge) Weight() float64 { return e.W }

// isSame returns whether two float64 values are the same where NaN values
// are equalable.
func isSame(a, b float64) bool {
	return a == b || (math.IsNaN(a) && math.IsNaN(b))
}

type edgeSetter interface {
	SetEdge(e graph.Edge)
}

type weightedEdgeSetter interface {
	SetWeightedEdge(e graph.WeightedEdge)
}

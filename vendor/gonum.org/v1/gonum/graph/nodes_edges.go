// Copyright ©2018 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package graph

// Iterator is an item iterator.
type Iterator interface {
	// Next advances the iterator and returns whether
	// the next call to the item method will return a
	// non-nil item.
	//
	// Next should be called prior to any call to the
	// iterator's item retrieval method after the
	// iterator has been obtained or reset.
	//
	// The order of iteration is implementation
	// dependent.
	Next() bool

	// Len returns the number of items remaining in the
	// iterator.
	//
	// If the number of items in the iterator is unknown,
	// too large to materialize or too costly to calculate
	// then Len may return a negative value.
	// In this case the consuming function must be able
	// to operate on the items of the iterator directly
	// without materializing the items into a slice.
	// The magnitude of a negative length has
	// implementation-dependent semantics.
	Len() int

	// Reset returns the iterator to its start position.
	Reset()
}

// Nodes is a Node iterator.
type Nodes interface {
	Iterator

	// Node returns the current Node from the iterator.
	Node() Node
}

// NodeSlicer wraps the NodeSlice method.
type NodeSlicer interface {
	// NodeSlice returns the set of nodes remaining
	// to be iterated by a Nodes iterator.
	// The holder of the iterator may arbitrarily
	// change elements in the returned slice, but
	// those changes may be reflected to other
	// iterators.
	NodeSlice() []Node
}

// NodesOf returns it.Len() nodes from it. If it is a NodeSlicer, the NodeSlice method
// is used to obtain the nodes. It is safe to pass a nil Nodes to NodesOf.
//
// If the Nodes has an indeterminate length, NodesOf will panic.
func NodesOf(it Nodes) []Node {
	if it == nil {
		return nil
	}
	len := it.Len()
	switch {
	case len == 0:
		return nil
	case len < 0:
		panic("graph: called NodesOf on indeterminate iterator")
	}
	switch it := it.(type) {
	case NodeSlicer:
		return it.NodeSlice()
	}
	n := make([]Node, 0, len)
	for it.Next() {
		n = append(n, it.Node())
	}
	return n
}

// Edges is an Edge iterator.
type Edges interface {
	Iterator

	// Edge returns the current Edge from the iterator.
	Edge() Edge
}

// EdgeSlicer wraps the EdgeSlice method.
type EdgeSlicer interface {
	// EdgeSlice returns the set of edges remaining
	// to be iterated by an Edges iterator.
	// The holder of the iterator may arbitrarily
	// change elements in the returned slice, but
	// those changes may be reflected to other
	// iterators.
	EdgeSlice() []Edge
}

// EdgesOf returns it.Len() nodes from it. If it is an EdgeSlicer, the EdgeSlice method is used
// to obtain the edges. It is safe to pass a nil Edges to EdgesOf.
//
// If the Edges has an indeterminate length, EdgesOf will panic.
func EdgesOf(it Edges) []Edge {
	if it == nil {
		return nil
	}
	len := it.Len()
	switch {
	case len == 0:
		return nil
	case len < 0:
		panic("graph: called EdgesOf on indeterminate iterator")
	}
	switch it := it.(type) {
	case EdgeSlicer:
		return it.EdgeSlice()
	}
	e := make([]Edge, 0, len)
	for it.Next() {
		e = append(e, it.Edge())
	}
	return e
}

// WeightedEdges is a WeightedEdge iterator.
type WeightedEdges interface {
	Iterator

	// Edge returns the current Edge from the iterator.
	WeightedEdge() WeightedEdge
}

// WeightedEdgeSlicer wraps the WeightedEdgeSlice method.
type WeightedEdgeSlicer interface {
	// EdgeSlice returns the set of edges remaining
	// to be iterated by an Edges iterator.
	// The holder of the iterator may arbitrarily
	// change elements in the returned slice, but
	// those changes may be reflected to other
	// iterators.
	WeightedEdgeSlice() []WeightedEdge
}

// WeightedEdgesOf returns it.Len() weighted edge from it. If it is a WeightedEdgeSlicer, the
// WeightedEdgeSlice method is used to obtain the edges. It is safe to pass a nil WeightedEdges
// to WeightedEdgesOf.
//
// If the WeightedEdges has an indeterminate length, WeightedEdgesOf will panic.
func WeightedEdgesOf(it WeightedEdges) []WeightedEdge {
	if it == nil {
		return nil
	}
	len := it.Len()
	switch {
	case len == 0:
		return nil
	case len < 0:
		panic("graph: called WeightedEdgesOf on indeterminate iterator")
	}
	switch it := it.(type) {
	case WeightedEdgeSlicer:
		return it.WeightedEdgeSlice()
	}
	e := make([]WeightedEdge, 0, len)
	for it.Next() {
		e = append(e, it.WeightedEdge())
	}
	return e
}

// Lines is a Line iterator.
type Lines interface {
	Iterator

	// Line returns the current Line from the iterator.
	Line() Line
}

// LineSlicer wraps the LineSlice method.
type LineSlicer interface {
	// LineSlice returns the set of lines remaining
	// to be iterated by an Lines iterator.
	// The holder of the iterator may arbitrarily
	// change elements in the returned slice, but
	// those changes may be reflected to other
	// iterators.
	LineSlice() []Line
}

// LinesOf returns it.Len() nodes from it. If it is a LineSlicer, the LineSlice method is used
// to obtain the lines. It is safe to pass a nil Lines to LinesOf.
//
// If the Lines has an indeterminate length, LinesOf will panic.
func LinesOf(it Lines) []Line {
	if it == nil {
		return nil
	}
	len := it.Len()
	switch {
	case len == 0:
		return nil
	case len < 0:
		panic("graph: called LinesOf on indeterminate iterator")
	}
	switch it := it.(type) {
	case LineSlicer:
		return it.LineSlice()
	}
	l := make([]Line, 0, len)
	for it.Next() {
		l = append(l, it.Line())
	}
	return l
}

// WeightedLines is a WeightedLine iterator.
type WeightedLines interface {
	Iterator

	// Line returns the current Line from the iterator.
	WeightedLine() WeightedLine
}

// WeightedLineSlicer wraps the WeightedLineSlice method.
type WeightedLineSlicer interface {
	// LineSlice returns the set of lines remaining
	// to be iterated by an Lines iterator.
	// The holder of the iterator may arbitrarily
	// change elements in the returned slice, but
	// those changes may be reflected to other
	// iterators.
	WeightedLineSlice() []WeightedLine
}

// WeightedLinesOf returns it.Len() weighted line from it. If it is a WeightedLineSlicer, the
// WeightedLineSlice method is used to obtain the lines. It is safe to pass a nil WeightedLines
// to WeightedLinesOf.
//
// If the WeightedLines has an indeterminate length, WeightedLinesOf will panic.
func WeightedLinesOf(it WeightedLines) []WeightedLine {
	if it == nil {
		return nil
	}
	len := it.Len()
	switch {
	case len == 0:
		return nil
	case len < 0:
		panic("graph: called WeightedLinesOf on indeterminate iterator")
	}
	switch it := it.(type) {
	case WeightedLineSlicer:
		return it.WeightedLineSlice()
	}
	l := make([]WeightedLine, 0, len)
	for it.Next() {
		l = append(l, it.WeightedLine())
	}
	return l
}

// Empty is an empty set of nodes, edges or lines. It should be used when
// a graph returns a zero-length Iterator. Empty implements the slicer
// interfaces for nodes, edges and lines, returning nil for each of these.
const Empty = nothing

var (
	_ Iterator           = Empty
	_ Nodes              = Empty
	_ NodeSlicer         = Empty
	_ Edges              = Empty
	_ EdgeSlicer         = Empty
	_ WeightedEdges      = Empty
	_ WeightedEdgeSlicer = Empty
	_ Lines              = Empty
	_ LineSlicer         = Empty
	_ WeightedLines      = Empty
	_ WeightedLineSlicer = Empty
)

const nothing = empty(true)

type empty bool

func (empty) Next() bool                        { return false }
func (empty) Len() int                          { return 0 }
func (empty) Reset()                            {}
func (empty) Node() Node                        { return nil }
func (empty) NodeSlice() []Node                 { return nil }
func (empty) Edge() Edge                        { return nil }
func (empty) EdgeSlice() []Edge                 { return nil }
func (empty) WeightedEdge() WeightedEdge        { return nil }
func (empty) WeightedEdgeSlice() []WeightedEdge { return nil }
func (empty) Line() Line                        { return nil }
func (empty) LineSlice() []Line                 { return nil }
func (empty) WeightedLine() WeightedLine        { return nil }
func (empty) WeightedLineSlice() []WeightedLine { return nil }

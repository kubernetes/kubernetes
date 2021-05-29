// Copyright ©2018 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iterator

import "gonum.org/v1/gonum/graph"

// OrderedLines implements the graph.Lines and graph.LineSlicer interfaces.
// The iteration order of OrderedLines is the order of lines passed to
// NewLineIterator.
type OrderedLines struct {
	idx   int
	lines []graph.Line
}

// NewOrderedLines returns an OrderedLines initialized with the provided lines.
func NewOrderedLines(lines []graph.Line) *OrderedLines {
	return &OrderedLines{idx: -1, lines: lines}
}

// Len returns the remaining number of lines to be iterated over.
func (e *OrderedLines) Len() int {
	if e.idx >= len(e.lines) {
		return 0
	}
	if e.idx <= 0 {
		return len(e.lines)
	}
	return len(e.lines[e.idx:])
}

// Next returns whether the next call of Line will return a valid line.
func (e *OrderedLines) Next() bool {
	if uint(e.idx)+1 < uint(len(e.lines)) {
		e.idx++
		return true
	}
	e.idx = len(e.lines)
	return false
}

// Line returns the current line of the iterator. Next must have been
// called prior to a call to Line.
func (e *OrderedLines) Line() graph.Line {
	if e.idx >= len(e.lines) || e.idx < 0 {
		return nil
	}
	return e.lines[e.idx]
}

// LineSlice returns all the remaining lines in the iterator and advances
// the iterator.
func (e *OrderedLines) LineSlice() []graph.Line {
	if e.idx >= len(e.lines) {
		return nil
	}
	idx := e.idx
	if idx == -1 {
		idx = 0
	}
	e.idx = len(e.lines)
	return e.lines[idx:]
}

// Reset returns the iterator to its initial state.
func (e *OrderedLines) Reset() {
	e.idx = -1
}

// OrderedWeightedLines implements the graph.Lines and graph.LineSlicer interfaces.
// The iteration order of OrderedWeightedLines is the order of lines passed to
// NewLineIterator.
type OrderedWeightedLines struct {
	idx   int
	lines []graph.WeightedLine
}

// NewWeightedLineIterator returns an OrderedWeightedLines initialized with the provided lines.
func NewOrderedWeightedLines(lines []graph.WeightedLine) *OrderedWeightedLines {
	return &OrderedWeightedLines{idx: -1, lines: lines}
}

// Len returns the remaining number of lines to be iterated over.
func (e *OrderedWeightedLines) Len() int {
	if e.idx >= len(e.lines) {
		return 0
	}
	if e.idx <= 0 {
		return len(e.lines)
	}
	return len(e.lines[e.idx:])
}

// Next returns whether the next call of WeightedLine will return a valid line.
func (e *OrderedWeightedLines) Next() bool {
	if uint(e.idx)+1 < uint(len(e.lines)) {
		e.idx++
		return true
	}
	e.idx = len(e.lines)
	return false
}

// WeightedLine returns the current line of the iterator. Next must have been
// called prior to a call to WeightedLine.
func (e *OrderedWeightedLines) WeightedLine() graph.WeightedLine {
	if e.idx >= len(e.lines) || e.idx < 0 {
		return nil
	}
	return e.lines[e.idx]
}

// WeightedLineSlice returns all the remaining lines in the iterator and advances
// the iterator.
func (e *OrderedWeightedLines) WeightedLineSlice() []graph.WeightedLine {
	if e.idx >= len(e.lines) {
		return nil
	}
	idx := e.idx
	if idx == -1 {
		idx = 0
	}
	e.idx = len(e.lines)
	return e.lines[idx:]
}

// Reset returns the iterator to its initial state.
func (e *OrderedWeightedLines) Reset() {
	e.idx = -1
}

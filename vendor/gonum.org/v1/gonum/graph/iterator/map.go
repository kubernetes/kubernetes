// Copyright ©2020 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !safe

package iterator

import (
	"unsafe"

	"gonum.org/v1/gonum/graph"
)

// A mapIter is an iterator for ranging over a map.
type mapIter struct {
	m  *emptyInterface
	it unsafe.Pointer
}

type emptyInterface struct {
	typ, word unsafe.Pointer
}

// newMapIterNodes returns a range iterator for a map of nodes.
func newMapIterNodes(m map[int64]graph.Node) *mapIter {
	return &mapIter{m: eface(m)}
}

// newMapIterEdges returns a range iterator for a map of edges.
// The returned mapIter must not have its node method called.
func newMapIterEdges(m map[int64]graph.Edge) *mapIter {
	return &mapIter{m: eface(m)}
}

// newMapIterWeightedEdges returns a range iterator for a map of edges.
// The returned mapIter must not have its node method called.
func newMapIterWeightedEdges(m map[int64]graph.WeightedEdge) *mapIter {
	return &mapIter{m: eface(m)}
}

// newMapIterLines returns a range iterator for a map of edges.
// The returned mapIter must not have its node method called.
func newMapIterLines(m map[int64]map[int64]graph.Line) *mapIter {
	return &mapIter{m: eface(m)}
}

// newMapIterWeightedLines returns a range iterator for a map of edges.
// The returned mapIter must not have its node method called.
func newMapIterWeightedLines(m map[int64]map[int64]graph.WeightedLine) *mapIter {
	return &mapIter{m: eface(m)}
}

func eface(i interface{}) *emptyInterface {
	return (*emptyInterface)(unsafe.Pointer(&i))
}

// id returns the key of the iterator's current map entry.
func (it *mapIter) id() int64 {
	if it.it == nil {
		panic("mapIter.id called before Next")
	}
	if mapiterkey(it.it) == nil {
		panic("mapIter.id called on exhausted iterator")
	}
	return *(*int64)(mapiterkey(it.it))
}

// node returns the value of the iterator's current map entry.
func (it *mapIter) node() graph.Node {
	if it.it == nil {
		panic("mapIter.node called before next")
	}
	if mapiterkey(it.it) == nil {
		panic("mapIter.node called on exhausted iterator")
	}
	return *(*graph.Node)(mapiterelem(it.it))
}

// next advances the map iterator and reports whether there is another
// entry. It returns false when the iterator is exhausted; subsequent
// calls to Key, Value, or next will panic.
func (it *mapIter) next() bool {
	if it.it == nil {
		it.it = mapiterinit(it.m.typ, it.m.word)
	} else {
		if mapiterkey(it.it) == nil {
			panic("mapIter.next called on exhausted iterator")
		}
		mapiternext(it.it)
	}
	return mapiterkey(it.it) != nil
}

// m escapes into the return value, but the caller of mapiterinit
// doesn't let the return value escape.
//go:linkname mapiterinit reflect.mapiterinit
//go:noescape
func mapiterinit(t, m unsafe.Pointer) unsafe.Pointer

//go:linkname mapiterkey reflect.mapiterkey
//go:noescape
func mapiterkey(it unsafe.Pointer) (key unsafe.Pointer)

//go:linkname mapiterelem reflect.mapiterelem
//go:noescape
func mapiterelem(it unsafe.Pointer) (elem unsafe.Pointer)

//go:linkname mapiternext reflect.mapiternext
//go:noescape
func mapiternext(it unsafe.Pointer)

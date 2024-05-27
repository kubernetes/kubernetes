// Copyright 2024 The Bazel Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.23

package starlark

import (
	"fmt"
	"iter"
)

func (d *Dict) Entries() iter.Seq2[Value, Value] { return d.ht.entries }

// Elements returns a go1.23 iterator over the elements of the list.
//
// Example:
//
//	for elem := range list.Elements() { ... }
func (l *List) Elements() iter.Seq[Value] {
	return func(yield func(Value) bool) {
		if !l.frozen {
			l.itercount++
			defer func() { l.itercount-- }()
		}
		for _, x := range l.elems {
			if !yield(x) {
				break
			}
		}
	}
}

// Elements returns a go1.23 iterator over the elements of the tuple.
//
// (A Tuple is a slice, so it is of course directly iterable. This
// method exists to provide a fast path for the [Elements] standalone
// function.)
func (t Tuple) Elements() iter.Seq[Value] {
	return func(yield func(Value) bool) {
		for _, x := range t {
			if !yield(x) {
				break
			}
		}
	}
}

func (s *Set) Elements() iter.Seq[Value] {
	return func(yield func(k Value) bool) {
		s.ht.entries(func(k, _ Value) bool { return yield(k) })
	}
}

// Elements returns an iterator for the elements of the iterable value.
//
// Example of go1.23 iteration:
//
//	for elem := range Elements(iterable) { ... }
//
// Push iterators are provided as a convenience for Go client code. The
// core iteration behavior of Starlark for-loops is defined by the
// [Iterable] interface.
func Elements(iterable Iterable) iter.Seq[Value] {
	// Use specialized push iterator if available (*List, Tuple, *Set).
	type hasElements interface {
		Elements() iter.Seq[Value]
	}
	if iterable, ok := iterable.(hasElements); ok {
		return iterable.Elements()
	}

	iter := iterable.Iterate()
	return func(yield func(Value) bool) {
		defer iter.Done()
		var x Value
		for iter.Next(&x) && yield(x) {
		}
	}
}

// Entries returns an iterator over the entries (key/value pairs) of
// the iterable mapping.
//
// Example of go1.23 iteration:
//
//	for k, v := range Entries(mapping) { ... }
//
// Push iterators are provided as a convenience for Go client code. The
// core iteration behavior of Starlark for-loops is defined by the
// [Iterable] interface.
func Entries(mapping IterableMapping) iter.Seq2[Value, Value] {
	// If available (e.g. *Dict), use specialized push iterator,
	// as it gets k and v in one shot.
	type hasEntries interface {
		Entries() iter.Seq2[Value, Value]
	}
	if mapping, ok := mapping.(hasEntries); ok {
		return mapping.Entries()
	}

	iter := mapping.Iterate()
	return func(yield func(k, v Value) bool) {
		defer iter.Done()
		var k Value
		for iter.Next(&k) {
			v, found, err := mapping.Get(k)
			if err != nil || !found {
				panic(fmt.Sprintf("Iterate and Get are inconsistent (mapping=%v, key=%v)",
					mapping.Type(), k.Type()))
			}
			if !yield(k, v) {
				break
			}
		}
	}
}

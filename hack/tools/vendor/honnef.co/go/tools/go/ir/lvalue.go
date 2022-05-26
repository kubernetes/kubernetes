// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

// lvalues are the union of addressable expressions and map-index
// expressions.

import (
	"go/ast"
	"go/types"
)

// An lvalue represents an assignable location that may appear on the
// left-hand side of an assignment.  This is a generalization of a
// pointer to permit updates to elements of maps.
//
type lvalue interface {
	store(fn *Function, v Value, source ast.Node) // stores v into the location
	load(fn *Function, source ast.Node) Value     // loads the contents of the location
	address(fn *Function) Value                   // address of the location
	typ() types.Type                              // returns the type of the location
}

// An address is an lvalue represented by a true pointer.
type address struct {
	addr Value
	expr ast.Expr // source syntax of the value (not address) [debug mode]
}

func (a *address) load(fn *Function, source ast.Node) Value {
	return emitLoad(fn, a.addr, source)
}

func (a *address) store(fn *Function, v Value, source ast.Node) {
	store := emitStore(fn, a.addr, v, source)
	if a.expr != nil {
		// store.Val is v, converted for assignability.
		emitDebugRef(fn, a.expr, store.Val, false)
	}
}

func (a *address) address(fn *Function) Value {
	if a.expr != nil {
		emitDebugRef(fn, a.expr, a.addr, true)
	}
	return a.addr
}

func (a *address) typ() types.Type {
	return deref(a.addr.Type())
}

// An element is an lvalue represented by m[k], the location of an
// element of a map.  These locations are not addressable
// since pointers cannot be formed from them, but they do support
// load() and store().
//
type element struct {
	m, k Value      // map
	t    types.Type // map element type
}

func (e *element) load(fn *Function, source ast.Node) Value {
	l := &MapLookup{
		X:     e.m,
		Index: e.k,
	}
	l.setType(e.t)
	return fn.emit(l, source)
}

func (e *element) store(fn *Function, v Value, source ast.Node) {
	up := &MapUpdate{
		Map:   e.m,
		Key:   e.k,
		Value: emitConv(fn, v, e.t, source),
	}
	fn.emit(up, source)
}

func (e *element) address(fn *Function) Value {
	panic("map elements are not addressable")
}

func (e *element) typ() types.Type {
	return e.t
}

// A blank is a dummy variable whose name is "_".
// It is not reified: loads are illegal and stores are ignored.
//
type blank struct{}

func (bl blank) load(fn *Function, source ast.Node) Value {
	panic("blank.load is illegal")
}

func (bl blank) store(fn *Function, v Value, source ast.Node) {
	s := &BlankStore{
		Val: v,
	}
	fn.emit(s, source)
}

func (bl blank) address(fn *Function) Value {
	panic("blank var is not addressable")
}

func (bl blank) typ() types.Type {
	// This should be the type of the blank Ident; the typechecker
	// doesn't provide this yet, but fortunately, we don't need it
	// yet either.
	panic("blank.typ is unimplemented")
}

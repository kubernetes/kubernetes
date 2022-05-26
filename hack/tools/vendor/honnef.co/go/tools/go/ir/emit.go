// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

// Helpers for emitting IR instructions.

import (
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
)

// emitNew emits to f a new (heap Alloc) instruction allocating an
// object of type typ.  pos is the optional source location.
//
func emitNew(f *Function, typ types.Type, source ast.Node) *Alloc {
	v := &Alloc{Heap: true}
	v.setType(types.NewPointer(typ))
	f.emit(v, source)
	return v
}

// emitLoad emits to f an instruction to load the address addr into a
// new temporary, and returns the value so defined.
//
func emitLoad(f *Function, addr Value, source ast.Node) *Load {
	v := &Load{X: addr}
	v.setType(deref(addr.Type()))
	f.emit(v, source)
	return v
}

func emitRecv(f *Function, ch Value, commaOk bool, typ types.Type, source ast.Node) Value {
	recv := &Recv{
		Chan:    ch,
		CommaOk: commaOk,
	}
	recv.setType(typ)
	return f.emit(recv, source)
}

// emitDebugRef emits to f a DebugRef pseudo-instruction associating
// expression e with value v.
//
func emitDebugRef(f *Function, e ast.Expr, v Value, isAddr bool) {
	if !f.debugInfo() {
		return // debugging not enabled
	}
	if v == nil || e == nil {
		panic("nil")
	}
	var obj types.Object
	e = unparen(e)
	if id, ok := e.(*ast.Ident); ok {
		if isBlankIdent(id) {
			return
		}
		obj = f.Pkg.objectOf(id)
		switch obj.(type) {
		case *types.Nil, *types.Const, *types.Builtin:
			return
		}
	}
	f.emit(&DebugRef{
		X:      v,
		Expr:   e,
		IsAddr: isAddr,
		object: obj,
	}, nil)
}

// emitArith emits to f code to compute the binary operation op(x, y)
// where op is an eager shift, logical or arithmetic operation.
// (Use emitCompare() for comparisons and Builder.logicalBinop() for
// non-eager operations.)
//
func emitArith(f *Function, op token.Token, x, y Value, t types.Type, source ast.Node) Value {
	switch op {
	case token.SHL, token.SHR:
		x = emitConv(f, x, t, source)
		// y may be signed or an 'untyped' constant.
		// TODO(adonovan): whence signed values?
		if b, ok := y.Type().Underlying().(*types.Basic); ok && b.Info()&types.IsUnsigned == 0 {
			y = emitConv(f, y, types.Typ[types.Uint64], source)
		}

	case token.ADD, token.SUB, token.MUL, token.QUO, token.REM, token.AND, token.OR, token.XOR, token.AND_NOT:
		x = emitConv(f, x, t, source)
		y = emitConv(f, y, t, source)

	default:
		panic("illegal op in emitArith: " + op.String())

	}
	v := &BinOp{
		Op: op,
		X:  x,
		Y:  y,
	}
	v.setType(t)
	return f.emit(v, source)
}

// emitCompare emits to f code compute the boolean result of
// comparison comparison 'x op y'.
//
func emitCompare(f *Function, op token.Token, x, y Value, source ast.Node) Value {
	xt := x.Type().Underlying()
	yt := y.Type().Underlying()

	// Special case to optimise a tagless SwitchStmt so that
	// these are equivalent
	//   switch { case e: ...}
	//   switch true { case e: ... }
	//   if e==true { ... }
	// even in the case when e's type is an interface.
	// TODO(adonovan): opt: generalise to x==true, false!=y, etc.
	if x, ok := x.(*Const); ok && op == token.EQL && x.Value != nil && x.Value.Kind() == constant.Bool && constant.BoolVal(x.Value) {
		if yt, ok := yt.(*types.Basic); ok && yt.Info()&types.IsBoolean != 0 {
			return y
		}
	}

	if types.Identical(xt, yt) {
		// no conversion necessary
	} else if _, ok := xt.(*types.Interface); ok {
		y = emitConv(f, y, x.Type(), source)
	} else if _, ok := yt.(*types.Interface); ok {
		x = emitConv(f, x, y.Type(), source)
	} else if _, ok := x.(*Const); ok {
		x = emitConv(f, x, y.Type(), source)
	} else if _, ok := y.(*Const); ok {
		y = emitConv(f, y, x.Type(), source)
		//lint:ignore SA9003 no-op
	} else {
		// other cases, e.g. channels.  No-op.
	}

	v := &BinOp{
		Op: op,
		X:  x,
		Y:  y,
	}
	v.setType(tBool)
	return f.emit(v, source)
}

// isValuePreserving returns true if a conversion from ut_src to
// ut_dst is value-preserving, i.e. just a change of type.
// Precondition: neither argument is a named type.
//
func isValuePreserving(ut_src, ut_dst types.Type) bool {
	// Identical underlying types?
	if structTypesIdentical(ut_dst, ut_src) {
		return true
	}

	switch ut_dst.(type) {
	case *types.Chan:
		// Conversion between channel types?
		_, ok := ut_src.(*types.Chan)
		return ok

	case *types.Pointer:
		// Conversion between pointers with identical base types?
		_, ok := ut_src.(*types.Pointer)
		return ok
	}
	return false
}

// emitConv emits to f code to convert Value val to exactly type typ,
// and returns the converted value.  Implicit conversions are required
// by language assignability rules in assignments, parameter passing,
// etc.
//
func emitConv(f *Function, val Value, typ types.Type, source ast.Node) Value {
	t_src := val.Type()

	// Identical types?  Conversion is a no-op.
	if types.Identical(t_src, typ) {
		return val
	}

	ut_dst := typ.Underlying()
	ut_src := t_src.Underlying()

	// Just a change of type, but not value or representation?
	if isValuePreserving(ut_src, ut_dst) {
		c := &ChangeType{X: val}
		c.setType(typ)
		return f.emit(c, source)
	}

	// Conversion to, or construction of a value of, an interface type?
	if _, ok := ut_dst.(*types.Interface); ok {
		// Assignment from one interface type to another?
		if _, ok := ut_src.(*types.Interface); ok {
			c := &ChangeInterface{X: val}
			c.setType(typ)
			return f.emit(c, source)
		}

		// Untyped nil constant?  Return interface-typed nil constant.
		if ut_src == tUntypedNil {
			return emitConst(f, nilConst(typ))
		}

		// Convert (non-nil) "untyped" literals to their default type.
		if t, ok := ut_src.(*types.Basic); ok && t.Info()&types.IsUntyped != 0 {
			val = emitConv(f, val, types.Default(ut_src), source)
		}

		f.Pkg.Prog.needMethodsOf(val.Type())
		mi := &MakeInterface{X: val}
		mi.setType(typ)
		return f.emit(mi, source)
	}

	// Conversion of a compile-time constant value?
	if c, ok := val.(*Const); ok {
		if _, ok := ut_dst.(*types.Basic); ok || c.IsNil() {
			// Conversion of a compile-time constant to
			// another constant type results in a new
			// constant of the destination type and
			// (initially) the same abstract value.
			// We don't truncate the value yet.
			return emitConst(f, NewConst(c.Value, typ))
		}

		// We're converting from constant to non-constant type,
		// e.g. string -> []byte/[]rune.
	}

	// Conversion from slice to array pointer?
	if slice, ok := ut_src.(*types.Slice); ok {
		if ptr, ok := ut_dst.(*types.Pointer); ok {
			if arr, ok := ptr.Elem().Underlying().(*types.Array); ok && types.Identical(slice.Elem(), arr.Elem()) {
				c := &SliceToArrayPointer{X: val}
				c.setType(ut_dst)
				return f.emit(c, source)
			}
		}
	}

	// A representation-changing conversion?
	// At least one of {ut_src,ut_dst} must be *Basic.
	// (The other may be []byte or []rune.)
	_, ok1 := ut_src.(*types.Basic)
	_, ok2 := ut_dst.(*types.Basic)
	if ok1 || ok2 {
		c := &Convert{X: val}
		c.setType(typ)
		return f.emit(c, source)
	}

	panic(fmt.Sprintf("in %s: cannot convert %s (%s) to %s", f, val, val.Type(), typ))
}

// emitStore emits to f an instruction to store value val at location
// addr, applying implicit conversions as required by assignability rules.
//
func emitStore(f *Function, addr, val Value, source ast.Node) *Store {
	s := &Store{
		Addr: addr,
		Val:  emitConv(f, val, deref(addr.Type()), source),
	}
	// make sure we call getMem after the call to emitConv, which may
	// itself update the memory state
	f.emit(s, source)
	return s
}

// emitJump emits to f a jump to target, and updates the control-flow graph.
// Postcondition: f.currentBlock is nil.
//
func emitJump(f *Function, target *BasicBlock, source ast.Node) *Jump {
	b := f.currentBlock
	j := new(Jump)
	b.emit(j, source)
	addEdge(b, target)
	f.currentBlock = nil
	return j
}

// emitIf emits to f a conditional jump to tblock or fblock based on
// cond, and updates the control-flow graph.
// Postcondition: f.currentBlock is nil.
//
func emitIf(f *Function, cond Value, tblock, fblock *BasicBlock, source ast.Node) *If {
	b := f.currentBlock
	stmt := &If{Cond: cond}
	b.emit(stmt, source)
	addEdge(b, tblock)
	addEdge(b, fblock)
	f.currentBlock = nil
	return stmt
}

// emitExtract emits to f an instruction to extract the index'th
// component of tuple.  It returns the extracted value.
//
func emitExtract(f *Function, tuple Value, index int, source ast.Node) Value {
	e := &Extract{Tuple: tuple, Index: index}
	e.setType(tuple.Type().(*types.Tuple).At(index).Type())
	return f.emit(e, source)
}

// emitTypeAssert emits to f a type assertion value := x.(t) and
// returns the value.  x.Type() must be an interface.
//
func emitTypeAssert(f *Function, x Value, t types.Type, source ast.Node) Value {
	a := &TypeAssert{X: x, AssertedType: t}
	a.setType(t)
	return f.emit(a, source)
}

// emitTypeTest emits to f a type test value,ok := x.(t) and returns
// a (value, ok) tuple.  x.Type() must be an interface.
//
func emitTypeTest(f *Function, x Value, t types.Type, source ast.Node) Value {
	a := &TypeAssert{
		X:            x,
		AssertedType: t,
		CommaOk:      true,
	}
	a.setType(types.NewTuple(
		newVar("value", t),
		varOk,
	))
	return f.emit(a, source)
}

// emitTailCall emits to f a function call in tail position.  The
// caller is responsible for all fields of 'call' except its type.
// Intended for wrapper methods.
// Precondition: f does/will not use deferred procedure calls.
// Postcondition: f.currentBlock is nil.
//
func emitTailCall(f *Function, call *Call, source ast.Node) {
	tresults := f.Signature.Results()
	nr := tresults.Len()
	if nr == 1 {
		call.typ = tresults.At(0).Type()
	} else {
		call.typ = tresults
	}
	tuple := f.emit(call, source)
	var ret Return
	switch nr {
	case 0:
		// no-op
	case 1:
		ret.Results = []Value{tuple}
	default:
		for i := 0; i < nr; i++ {
			v := emitExtract(f, tuple, i, source)
			// TODO(adonovan): in principle, this is required:
			//   v = emitConv(f, o.Type, f.Signature.Results[i].Type)
			// but in practice emitTailCall is only used when
			// the types exactly match.
			ret.Results = append(ret.Results, v)
		}
	}

	f.Exit = f.newBasicBlock("exit")
	emitJump(f, f.Exit, source)
	f.currentBlock = f.Exit
	f.emit(&ret, source)
	f.currentBlock = nil
}

// emitImplicitSelections emits to f code to apply the sequence of
// implicit field selections specified by indices to base value v, and
// returns the selected value.
//
// If v is the address of a struct, the result will be the address of
// a field; if it is the value of a struct, the result will be the
// value of a field.
//
func emitImplicitSelections(f *Function, v Value, indices []int, source ast.Node) Value {
	for _, index := range indices {
		fld := deref(v.Type()).Underlying().(*types.Struct).Field(index)

		if isPointer(v.Type()) {
			instr := &FieldAddr{
				X:     v,
				Field: index,
			}
			instr.setType(types.NewPointer(fld.Type()))
			v = f.emit(instr, source)
			// Load the field's value iff indirectly embedded.
			if isPointer(fld.Type()) {
				v = emitLoad(f, v, source)
			}
		} else {
			instr := &Field{
				X:     v,
				Field: index,
			}
			instr.setType(fld.Type())
			v = f.emit(instr, source)
		}
	}
	return v
}

// emitFieldSelection emits to f code to select the index'th field of v.
//
// If wantAddr, the input must be a pointer-to-struct and the result
// will be the field's address; otherwise the result will be the
// field's value.
// Ident id is used for position and debug info.
//
func emitFieldSelection(f *Function, v Value, index int, wantAddr bool, id *ast.Ident) Value {
	fld := deref(v.Type()).Underlying().(*types.Struct).Field(index)
	if isPointer(v.Type()) {
		instr := &FieldAddr{
			X:     v,
			Field: index,
		}
		instr.setSource(id)
		instr.setType(types.NewPointer(fld.Type()))
		v = f.emit(instr, id)
		// Load the field's value iff we don't want its address.
		if !wantAddr {
			v = emitLoad(f, v, id)
		}
	} else {
		instr := &Field{
			X:     v,
			Field: index,
		}
		instr.setSource(id)
		instr.setType(fld.Type())
		v = f.emit(instr, id)
	}
	emitDebugRef(f, id, v, wantAddr)
	return v
}

// zeroValue emits to f code to produce a zero value of type t,
// and returns it.
//
func zeroValue(f *Function, t types.Type, source ast.Node) Value {
	switch t.Underlying().(type) {
	case *types.Struct, *types.Array:
		return emitLoad(f, f.addLocal(t, source), source)
	default:
		return emitConst(f, zeroConst(t))
	}
}

func emitConst(f *Function, c *Const) *Const {
	f.consts = append(f.consts, c)
	return c
}

// Copyright 2019 The Bazel Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package resolve

import "go.starlark.net/syntax"

// This file defines resolver data types saved in the syntax tree.
// We cannot guarantee API stability for these types
// as they are closely tied to the implementation.

// A Binding contains resolver information about an identifier.
// The resolver populates the Binding field of each syntax.Identifier.
// The Binding ties together all identifiers that denote the same variable.
type Binding struct {
	Scope Scope

	// Index records the index into the enclosing
	// - {DefStmt,File}.Locals, if Scope==Local
	// - DefStmt.FreeVars,      if Scope==Free
	// - File.Globals,          if Scope==Global.
	// It is zero if Scope is Predeclared, Universal, or Undefined.
	Index int

	First *syntax.Ident // first binding use (iff Scope==Local/Free/Global)
}

// The Scope of Binding indicates what kind of scope it has.
type Scope uint8

const (
	Undefined   Scope = iota // name is not defined
	Local                    // name is local to its function or file
	Cell                     // name is function-local but shared with a nested function
	Free                     // name is cell of some enclosing function
	Global                   // name is global to module
	Predeclared              // name is predeclared for this module (e.g. glob)
	Universal                // name is universal (e.g. len)
)

var scopeNames = [...]string{
	Undefined:   "undefined",
	Local:       "local",
	Cell:        "cell",
	Free:        "free",
	Global:      "global",
	Predeclared: "predeclared",
	Universal:   "universal",
}

func (scope Scope) String() string { return scopeNames[scope] }

// A Module contains resolver information about a file.
// The resolver populates the Module field of each syntax.File.
type Module struct {
	Locals  []*Binding // the file's (comprehension-)local variables
	Globals []*Binding // the file's global variables
}

// A Function contains resolver information about a named or anonymous function.
// The resolver populates the Function field of each syntax.DefStmt and syntax.LambdaExpr.
type Function struct {
	Pos    syntax.Position // of DEF or LAMBDA
	Name   string          // name of def, or "lambda"
	Params []syntax.Expr   // param = ident | ident=expr | * | *ident | **ident
	Body   []syntax.Stmt   // contains synthetic 'return expr' for lambda

	HasVarargs      bool       // whether params includes *args (convenience)
	HasKwargs       bool       // whether params includes **kwargs (convenience)
	NumKwonlyParams int        // number of keyword-only optional parameters
	Locals          []*Binding // this function's local/cell variables, parameters first
	FreeVars        []*Binding // enclosing cells to capture in closure
}

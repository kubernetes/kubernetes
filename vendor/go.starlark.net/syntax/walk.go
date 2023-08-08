// Copyright 2017 The Bazel Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// Walk traverses a syntax tree in depth-first order.
// It starts by calling f(n); n must not be nil.
// If f returns true, Walk calls itself
// recursively for each non-nil child of n.
// Walk then calls f(nil).
func Walk(n Node, f func(Node) bool) {
	if n == nil {
		panic("nil")
	}
	if !f(n) {
		return
	}

	// TODO(adonovan): opt: order cases using profile data.
	switch n := n.(type) {
	case *File:
		walkStmts(n.Stmts, f)

	case *ExprStmt:
		Walk(n.X, f)

	case *BranchStmt:
		// no-op

	case *IfStmt:
		Walk(n.Cond, f)
		walkStmts(n.True, f)
		walkStmts(n.False, f)

	case *AssignStmt:
		Walk(n.LHS, f)
		Walk(n.RHS, f)

	case *DefStmt:
		Walk(n.Name, f)
		for _, param := range n.Params {
			Walk(param, f)
		}
		walkStmts(n.Body, f)

	case *ForStmt:
		Walk(n.Vars, f)
		Walk(n.X, f)
		walkStmts(n.Body, f)

	case *ReturnStmt:
		if n.Result != nil {
			Walk(n.Result, f)
		}

	case *LoadStmt:
		Walk(n.Module, f)
		for _, from := range n.From {
			Walk(from, f)
		}
		for _, to := range n.To {
			Walk(to, f)
		}

	case *Ident, *Literal:
		// no-op

	case *ListExpr:
		for _, x := range n.List {
			Walk(x, f)
		}

	case *ParenExpr:
		Walk(n.X, f)

	case *CondExpr:
		Walk(n.Cond, f)
		Walk(n.True, f)
		Walk(n.False, f)

	case *IndexExpr:
		Walk(n.X, f)
		Walk(n.Y, f)

	case *DictEntry:
		Walk(n.Key, f)
		Walk(n.Value, f)

	case *SliceExpr:
		Walk(n.X, f)
		if n.Lo != nil {
			Walk(n.Lo, f)
		}
		if n.Hi != nil {
			Walk(n.Hi, f)
		}
		if n.Step != nil {
			Walk(n.Step, f)
		}

	case *Comprehension:
		Walk(n.Body, f)
		for _, clause := range n.Clauses {
			Walk(clause, f)
		}

	case *IfClause:
		Walk(n.Cond, f)

	case *ForClause:
		Walk(n.Vars, f)
		Walk(n.X, f)

	case *TupleExpr:
		for _, x := range n.List {
			Walk(x, f)
		}

	case *DictExpr:
		for _, entry := range n.List {
			Walk(entry, f)
		}

	case *UnaryExpr:
		if n.X != nil {
			Walk(n.X, f)
		}

	case *BinaryExpr:
		Walk(n.X, f)
		Walk(n.Y, f)

	case *DotExpr:
		Walk(n.X, f)
		Walk(n.Name, f)

	case *CallExpr:
		Walk(n.Fn, f)
		for _, arg := range n.Args {
			Walk(arg, f)
		}

	case *LambdaExpr:
		for _, param := range n.Params {
			Walk(param, f)
		}
		Walk(n.Body, f)

	default:
		panic(n)
	}

	f(nil)
}

func walkStmts(stmts []Stmt, f func(Node) bool) {
	for _, stmt := range stmts {
		Walk(stmt, f)
	}
}

// Copyright 2020 Frederik Zipp. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gocyclo calculates the cyclomatic complexities of functions and
// methods in Go source code.
package gocyclo

import (
	"go/ast"
	"go/token"
)

// Complexity calculates the cyclomatic complexity of a function.
// The 'fn' node is either a *ast.FuncDecl or a *ast.FuncLit.
func Complexity(fn ast.Node) int {
	v := complexityVisitor{
		complexity: 1,
	}
	ast.Walk(&v, fn)
	return v.complexity
}

type complexityVisitor struct {
	// complexity is the cyclomatic complexity
	complexity int
}

// Visit implements the ast.Visitor interface.
func (v *complexityVisitor) Visit(n ast.Node) ast.Visitor {
	switch n := n.(type) {
	case *ast.IfStmt, *ast.ForStmt, *ast.RangeStmt:
		v.complexity++
	case *ast.CaseClause:
		if n.List != nil { // ignore default case
			v.complexity++
		}
	case *ast.CommClause:
		if n.Comm != nil { // ignore default case
			v.complexity++
		}
	case *ast.BinaryExpr:
		if n.Op == token.LAND || n.Op == token.LOR {
			v.complexity++
		}
	}
	return v
}

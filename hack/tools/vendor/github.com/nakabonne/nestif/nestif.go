// Copyright 2020 Ryo Nakao <ryo@nakao.dev>.
//
// All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package nestif provides an API to detect complex nested if statements.
package nestif

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/printer"
	"go/token"
	"io"
)

// Issue represents an issue of root if statement that has nested ifs.
type Issue struct {
	Pos        token.Position
	Complexity int
	Message    string
}

// Checker represents a checker that finds nested if statements.
type Checker struct {
	// Minimum complexity to report.
	MinComplexity int

	// For debug mode.
	debugWriter io.Writer
	issues      []Issue
}

// Check inspects a single file and returns found issues.
func (c *Checker) Check(f *ast.File, fset *token.FileSet) []Issue {
	c.issues = []Issue{} // refresh
	ast.Inspect(f, func(n ast.Node) bool {
		fn, ok := n.(*ast.FuncDecl)
		if !ok || fn.Body == nil {
			return true
		}
		for _, stmt := range fn.Body.List {
			c.checkFunc(&stmt, fset)
		}
		return true
	})

	return c.issues
}

// checkFunc inspects a function and sets a list of issues if there are.
func (c *Checker) checkFunc(stmt *ast.Stmt, fset *token.FileSet) {
	ast.Inspect(*stmt, func(n ast.Node) bool {
		ifStmt, ok := n.(*ast.IfStmt)
		if !ok {
			return true
		}

		c.checkIf(ifStmt, fset)
		return false
	})
}

// checkIf inspects a if statement and sets an issue if there is.
func (c *Checker) checkIf(stmt *ast.IfStmt, fset *token.FileSet) {
	v := newVisitor()
	ast.Walk(v, stmt)
	if v.complexity < c.MinComplexity {
		return
	}
	pos := fset.Position(stmt.Pos())
	c.issues = append(c.issues, Issue{
		Pos:        pos,
		Complexity: v.complexity,
		Message:    c.makeMessage(v.complexity, stmt.Cond, fset),
	})
}

type visitor struct {
	complexity int
	nesting    int
	// To avoid adding complexity including nesting level to `else if`.
	elseifs map[*ast.IfStmt]bool
}

func newVisitor() *visitor {
	return &visitor{
		elseifs: make(map[*ast.IfStmt]bool),
	}
}

// Visit traverses an AST in depth-first order by calling itself
// recursively, and calculates the complexities of if statements.
func (v *visitor) Visit(n ast.Node) ast.Visitor {
	ifStmt, ok := n.(*ast.IfStmt)
	if !ok {
		return v
	}

	v.incComplexity(ifStmt)
	v.nesting++
	ast.Walk(v, ifStmt.Body)
	v.nesting--

	switch t := ifStmt.Else.(type) {
	case *ast.BlockStmt:
		v.complexity++
		v.nesting++
		ast.Walk(v, t)
		v.nesting--
	case *ast.IfStmt:
		v.elseifs[t] = true
		ast.Walk(v, t)
	}

	return nil
}

func (v *visitor) incComplexity(n *ast.IfStmt) {
	// In case of `else if`, increase by 1.
	if v.elseifs[n] {
		v.complexity++
	} else {
		v.complexity += v.nesting
	}
}

func (c *Checker) makeMessage(complexity int, cond ast.Expr, fset *token.FileSet) string {
	p := &printer.Config{}
	b := new(bytes.Buffer)
	if err := p.Fprint(b, fset, cond); err != nil {
		c.debug("failed to convert condition into string: %v", err)
	}
	return fmt.Sprintf("`if %s` has complex nested blocks (complexity: %d)", b.String(), complexity)
}

// DebugMode makes it possible to emit debug logs.
func (c *Checker) DebugMode(w io.Writer) {
	c.debugWriter = w
}

func (c *Checker) debug(format string, a ...interface{}) {
	if c.debugWriter != nil {
		fmt.Fprintf(c.debugWriter, format, a...)
	}
}

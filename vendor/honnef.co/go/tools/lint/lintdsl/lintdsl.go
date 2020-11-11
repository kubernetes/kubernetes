// Package lintdsl provides helpers for implementing static analysis
// checks. Dot-importing this package is encouraged.
package lintdsl

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"

	"golang.org/x/tools/go/analysis"
	"honnef.co/go/tools/pattern"
)

func Inspect(node ast.Node, fn func(node ast.Node) bool) {
	if node == nil {
		return
	}
	ast.Inspect(node, fn)
}

func Match(pass *analysis.Pass, q pattern.Pattern, node ast.Node) (*pattern.Matcher, bool) {
	// Note that we ignore q.Relevant – callers of Match usually use
	// AST inspectors that already filter on nodes we're interested
	// in.
	m := &pattern.Matcher{TypesInfo: pass.TypesInfo}
	ok := m.Match(q.Root, node)
	return m, ok
}

func MatchAndEdit(pass *analysis.Pass, before, after pattern.Pattern, node ast.Node) (*pattern.Matcher, []analysis.TextEdit, bool) {
	m, ok := Match(pass, before, node)
	if !ok {
		return m, nil, false
	}
	r := pattern.NodeToAST(after.Root, m.State)
	buf := &bytes.Buffer{}
	format.Node(buf, pass.Fset, r)
	edit := []analysis.TextEdit{{
		Pos:     node.Pos(),
		End:     node.End(),
		NewText: buf.Bytes(),
	}}
	return m, edit, true
}

func Selector(x, sel string) *ast.SelectorExpr {
	return &ast.SelectorExpr{
		X:   &ast.Ident{Name: x},
		Sel: &ast.Ident{Name: sel},
	}
}

// ExhaustiveTypeSwitch panics when called. It can be used to ensure
// that type switches are exhaustive.
func ExhaustiveTypeSwitch(v interface{}) {
	panic(fmt.Sprintf("internal error: unhandled case %T", v))
}

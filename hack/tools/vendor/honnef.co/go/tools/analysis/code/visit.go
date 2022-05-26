package code

import (
	"bytes"
	"go/ast"
	"go/format"

	"honnef.co/go/tools/pattern"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
)

func Preorder(pass *analysis.Pass, fn func(ast.Node), types ...ast.Node) {
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder(types, fn)
}

func PreorderStack(pass *analysis.Pass, fn func(ast.Node, []ast.Node), types ...ast.Node) {
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).WithStack(types, func(n ast.Node, push bool, stack []ast.Node) (proceed bool) {
		if push {
			fn(n, stack)
		}
		return true
	})
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

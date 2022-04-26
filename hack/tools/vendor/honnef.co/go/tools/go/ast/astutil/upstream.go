package astutil

import (
	"go/ast"
	"go/token"
	_ "unsafe"

	"golang.org/x/tools/go/ast/astutil"
)

type Cursor = astutil.Cursor
type ApplyFunc = astutil.ApplyFunc

func Apply(root ast.Node, pre, post ApplyFunc) (result ast.Node) {
	return astutil.Apply(root, pre, post)
}

func PathEnclosingInterval(root *ast.File, start, end token.Pos) (path []ast.Node, exact bool) {
	return astutil.PathEnclosingInterval(root, start, end)
}

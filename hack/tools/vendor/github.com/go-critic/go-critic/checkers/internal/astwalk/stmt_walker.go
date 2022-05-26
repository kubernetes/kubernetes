package astwalk

import (
	"go/ast"
)

type stmtWalker struct {
	visitor StmtVisitor
}

func (w *stmtWalker) WalkFile(f *ast.File) {
	if !w.visitor.EnterFile(f) {
		return
	}

	for _, decl := range f.Decls {
		decl, ok := decl.(*ast.FuncDecl)
		if !ok || !w.visitor.EnterFunc(decl) {
			continue
		}
		ast.Inspect(decl.Body, func(x ast.Node) bool {
			if x, ok := x.(ast.Stmt); ok {
				w.visitor.VisitStmt(x)
				return !w.visitor.skipChilds()
			}
			return true
		})
	}
}

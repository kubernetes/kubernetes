package astwalk

import (
	"go/ast"
)

type funcDeclWalker struct {
	visitor FuncDeclVisitor
}

func (w *funcDeclWalker) WalkFile(f *ast.File) {
	if !w.visitor.EnterFile(f) {
		return
	}

	for _, decl := range f.Decls {
		decl, ok := decl.(*ast.FuncDecl)
		if !ok || !w.visitor.EnterFunc(decl) {
			continue
		}
		w.visitor.VisitFuncDecl(decl)
	}
}

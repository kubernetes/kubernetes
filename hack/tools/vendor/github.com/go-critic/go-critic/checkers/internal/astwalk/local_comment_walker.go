package astwalk

import (
	"go/ast"
)

type localCommentWalker struct {
	visitor LocalCommentVisitor
}

func (w *localCommentWalker) WalkFile(f *ast.File) {
	if !w.visitor.EnterFile(f) {
		return
	}

	for _, decl := range f.Decls {
		decl, ok := decl.(*ast.FuncDecl)
		if !ok || !w.visitor.EnterFunc(decl) {
			continue
		}

		for _, cg := range f.Comments {
			// Not sure that decls/comments are sorted
			// by positions, so do a naive full scan for now.
			if cg.Pos() < decl.Pos() || cg.Pos() > decl.End() {
				continue
			}

			visitCommentGroups(cg, w.visitor.VisitLocalComment)
		}
	}
}

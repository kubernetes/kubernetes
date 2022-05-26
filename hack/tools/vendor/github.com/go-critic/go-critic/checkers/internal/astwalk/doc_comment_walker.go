package astwalk

import (
	"go/ast"
)

type docCommentWalker struct {
	visitor DocCommentVisitor
}

func (w *docCommentWalker) WalkFile(f *ast.File) {
	for _, decl := range f.Decls {
		switch decl := decl.(type) {
		case *ast.FuncDecl:
			if decl.Doc != nil {
				w.visitor.VisitDocComment(decl.Doc)
			}
		case *ast.GenDecl:
			if decl.Doc != nil {
				w.visitor.VisitDocComment(decl.Doc)
			}
			for _, spec := range decl.Specs {
				switch spec := spec.(type) {
				case *ast.ImportSpec:
					if spec.Doc != nil {
						w.visitor.VisitDocComment(spec.Doc)
					}
				case *ast.ValueSpec:
					if spec.Doc != nil {
						w.visitor.VisitDocComment(spec.Doc)
					}
				case *ast.TypeSpec:
					if spec.Doc != nil {
						w.visitor.VisitDocComment(spec.Doc)
					}
					ast.Inspect(spec.Type, func(n ast.Node) bool {
						if n, ok := n.(*ast.Field); ok {
							if n.Doc != nil {
								w.visitor.VisitDocComment(n.Doc)
							}
						}
						return true
					})
				}
			}
		}
	}
}

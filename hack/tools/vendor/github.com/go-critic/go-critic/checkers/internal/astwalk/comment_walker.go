package astwalk

import (
	"go/ast"
	"strings"
)

type commentWalker struct {
	visitor CommentVisitor
}

func (w *commentWalker) WalkFile(f *ast.File) {
	if !w.visitor.EnterFile(f) {
		return
	}

	for _, cg := range f.Comments {
		visitCommentGroups(cg, w.visitor.VisitComment)
	}
}

func visitCommentGroups(cg *ast.CommentGroup, visit func(*ast.CommentGroup)) {
	var group []*ast.Comment
	visitGroup := func(list []*ast.Comment) {
		if len(list) == 0 {
			return
		}
		cg := &ast.CommentGroup{List: list}
		visit(cg)
	}
	for _, comment := range cg.List {
		if strings.HasPrefix(comment.Text, "/*") {
			visitGroup(group)
			group = group[:0]
			visitGroup([]*ast.Comment{comment})
		} else {
			group = append(group, comment)
		}
	}
	visitGroup(group)
}

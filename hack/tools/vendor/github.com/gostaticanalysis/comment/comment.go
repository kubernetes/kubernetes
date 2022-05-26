package comment

import (
	"go/ast"
	"go/token"
	"strings"
)

// Maps is slice of ast.CommentMap.
type Maps []ast.CommentMap

// New creates new a CommentMap slice from specified files.
func New(fset *token.FileSet, files []*ast.File) Maps {
	maps := make(Maps, len(files))
	for i := range files {
		maps[i] = ast.NewCommentMap(fset, files[i], files[i].Comments)
	}
	return maps
}

// Comments returns correspond a CommentGroup slice to specified AST node.
func (maps Maps) Comments(n ast.Node) []*ast.CommentGroup {
	for i := range maps {
		if maps[i][n] != nil {
			return maps[i][n]
		}
	}
	return nil
}

// CommentsByPos returns correspond a CommentGroup slice to specified pos.
func (maps Maps) CommentsByPos(pos token.Pos) []*ast.CommentGroup {
	for i := range maps {
		for n, cgs := range maps[i] {
			if n.Pos() == pos {
				return cgs
			}
		}
	}
	return nil
}

// Annotated checks either specified AST node is annotated or not.
func (maps Maps) Annotated(n ast.Node, annotation string) bool {
	for _, cg := range maps.Comments(n) {
		if strings.HasPrefix(strings.TrimSpace(cg.Text()), annotation) {
			return true
		}
	}
	return false
}

// Ignore checks either specified AST node is ignored by the check.
// It follows staticcheck style as the below.
//   //lint:ignore Check1[,Check2,...,CheckN] reason
func (maps Maps) Ignore(n ast.Node, check string) bool {
	for _, cg := range maps.Comments(n) {
		if hasIgnoreCheck(cg, check) {
			return true
		}
	}
	return false
}

// IgnorePos checks either specified postion of AST node is ignored by the check.
// It follows staticcheck style as the below.
//   //lint:ignore Check1[,Check2,...,CheckN] reason
func (maps Maps) IgnorePos(pos token.Pos, check string) bool {
	for _, cg := range maps.CommentsByPos(pos) {
		if hasIgnoreCheck(cg, check) {
			return true
		}
	}
	return false
}

// Deprecated: This function does not work with multiple files.
// CommentsByPosLine can be used instead of CommentsByLine.
//
// CommentsByLine returns correspond a CommentGroup slice to specified line.
func (maps Maps) CommentsByLine(fset *token.FileSet, line int) []*ast.CommentGroup {
	for i := range maps {
		for n, cgs := range maps[i] {
			l := fset.File(n.Pos()).Line(n.Pos())
			if l == line {
				return cgs
			}
		}
	}
	return nil
}

// CommentsByPosLine returns correspond a CommentGroup slice to specified line.
func (maps Maps) CommentsByPosLine(fset *token.FileSet, pos token.Pos) []*ast.CommentGroup {
	f1 := fset.File(pos)
	for i := range maps {
		for n, cgs := range maps[i] {
			f2 := fset.File(n.Pos())
			if f1 != f2 {
				// different file
				continue
			}

			if f1.Line(pos) == f2.Line(n.Pos()) {
				return cgs
			}
		}
	}
	return nil
}

// IgnoreLine checks either specified lineof AST node is ignored by the check.
// It follows staticcheck style as the below.
//   //lint:ignore Check1[,Check2,...,CheckN] reason
func (maps Maps) IgnoreLine(fset *token.FileSet, line int, check string) bool {
	for _, cg := range maps.CommentsByLine(fset, line) {
		if hasIgnoreCheck(cg, check) {
			return true
		}
	}
	return false
}

// hasIgnoreCheck returns true if the provided CommentGroup starts with a comment
// of the form "//lint:ignore Check1[,Check2,...,CheckN] reason" and one of the
// checks matches the provided check.
//
// The *ast.CommentGroup is checked directly rather than using "cg.Text()" because,
// starting in Go 1.15, the "cg.Text()" call no longer returns directive-style
// comments (see https://github.com/golang/go/issues/37974).
func hasIgnoreCheck(cg *ast.CommentGroup, check string) bool {
	for _, list := range cg.List {
		if !strings.HasPrefix(list.Text, "//") {
			continue
		}

		s := strings.TrimSpace(list.Text[2:]) // list.Text[2:]: trim "//"
		txt := strings.Split(s, " ")
		if len(txt) < 3 || txt[0] != "lint:ignore" {
			continue
		}

		checks := strings.Split(txt[1], ",") // txt[1]: trim "lint:ignore"
		for i := range checks {
			if check == checks[i] {
				return true
			}
		}
	}

	return false
}

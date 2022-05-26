package rule

import (
	"go/ast"
	"go/token"
	"strings"

	"github.com/mgechev/revive/lint"
)

// IfReturnRule lints given else constructs.
type IfReturnRule struct{}

// Apply applies the rule to given file.
func (r *IfReturnRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	onFailure := func(failure lint.Failure) {
		failures = append(failures, failure)
	}

	astFile := file.AST
	w := &lintElseError{astFile, onFailure}
	ast.Walk(w, astFile)
	return failures
}

// Name returns the rule name.
func (r *IfReturnRule) Name() string {
	return "if-return"
}

type lintElseError struct {
	file      *ast.File
	onFailure func(lint.Failure)
}

func (w *lintElseError) Visit(node ast.Node) ast.Visitor {
	switch v := node.(type) {
	case *ast.BlockStmt:
		for i := 0; i < len(v.List)-1; i++ {
			// if var := whatever; var != nil { return var }
			s, ok := v.List[i].(*ast.IfStmt)
			if !ok || s.Body == nil || len(s.Body.List) != 1 || s.Else != nil {
				continue
			}
			assign, ok := s.Init.(*ast.AssignStmt)
			if !ok || len(assign.Lhs) != 1 || !(assign.Tok == token.DEFINE || assign.Tok == token.ASSIGN) {
				continue
			}
			id, ok := assign.Lhs[0].(*ast.Ident)
			if !ok {
				continue
			}
			expr, ok := s.Cond.(*ast.BinaryExpr)
			if !ok || expr.Op != token.NEQ {
				continue
			}
			if lhs, ok := expr.X.(*ast.Ident); !ok || lhs.Name != id.Name {
				continue
			}
			if rhs, ok := expr.Y.(*ast.Ident); !ok || rhs.Name != "nil" {
				continue
			}
			r, ok := s.Body.List[0].(*ast.ReturnStmt)
			if !ok || len(r.Results) != 1 {
				continue
			}
			if r, ok := r.Results[0].(*ast.Ident); !ok || r.Name != id.Name {
				continue
			}

			// return nil
			r, ok = v.List[i+1].(*ast.ReturnStmt)
			if !ok || len(r.Results) != 1 {
				continue
			}
			if r, ok := r.Results[0].(*ast.Ident); !ok || r.Name != "nil" {
				continue
			}

			// check if there are any comments explaining the construct, don't emit an error if there are some.
			if containsComments(s.Pos(), r.Pos(), w.file) {
				continue
			}

			w.onFailure(lint.Failure{
				Confidence: .9,
				Node:       v.List[i],
				Failure:    "redundant if ...; err != nil check, just return error instead.",
			})
		}
	}
	return w
}

func containsComments(start, end token.Pos, f *ast.File) bool {
	for _, cgroup := range f.Comments {
		comments := cgroup.List
		if comments[0].Slash >= end {
			// All comments starting with this group are after end pos.
			return false
		}
		if comments[len(comments)-1].Slash < start {
			// Comments group ends before start pos.
			continue
		}
		for _, c := range comments {
			if start <= c.Slash && c.Slash < end && !strings.HasPrefix(c.Text, "// MATCH ") {
				return true
			}
		}
	}
	return false
}

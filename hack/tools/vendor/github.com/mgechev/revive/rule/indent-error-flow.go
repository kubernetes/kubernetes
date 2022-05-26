package rule

import (
	"go/ast"
	"go/token"

	"github.com/mgechev/revive/lint"
)

// IndentErrorFlowRule lints given else constructs.
type IndentErrorFlowRule struct{}

// Apply applies the rule to given file.
func (r *IndentErrorFlowRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	onFailure := func(failure lint.Failure) {
		failures = append(failures, failure)
	}

	w := lintElse{make(map[*ast.IfStmt]bool), onFailure}
	ast.Walk(w, file.AST)
	return failures
}

// Name returns the rule name.
func (r *IndentErrorFlowRule) Name() string {
	return "indent-error-flow"
}

type lintElse struct {
	ignore    map[*ast.IfStmt]bool
	onFailure func(lint.Failure)
}

func (w lintElse) Visit(node ast.Node) ast.Visitor {
	ifStmt, ok := node.(*ast.IfStmt)
	if !ok || ifStmt.Else == nil {
		return w
	}
	if w.ignore[ifStmt] {
		if elseif, ok := ifStmt.Else.(*ast.IfStmt); ok {
			w.ignore[elseif] = true
		}
		return w
	}
	if elseif, ok := ifStmt.Else.(*ast.IfStmt); ok {
		w.ignore[elseif] = true
		return w
	}
	if _, ok := ifStmt.Else.(*ast.BlockStmt); !ok {
		// only care about elses without conditions
		return w
	}
	if len(ifStmt.Body.List) == 0 {
		return w
	}
	shortDecl := false // does the if statement have a ":=" initialization statement?
	if ifStmt.Init != nil {
		if as, ok := ifStmt.Init.(*ast.AssignStmt); ok && as.Tok == token.DEFINE {
			shortDecl = true
		}
	}
	lastStmt := ifStmt.Body.List[len(ifStmt.Body.List)-1]
	if _, ok := lastStmt.(*ast.ReturnStmt); ok {
		extra := ""
		if shortDecl {
			extra = " (move short variable declaration to its own line if necessary)"
		}
		w.onFailure(lint.Failure{
			Confidence: 1,
			Node:       ifStmt.Else,
			Category:   "indent",
			Failure:    "if block ends with a return statement, so drop this else and outdent its block" + extra,
		})
	}
	return w
}

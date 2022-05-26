package rule

import (
	"go/ast"

	"github.com/mgechev/revive/lint"
)

// ErrorReturnRule lints given else constructs.
type ErrorReturnRule struct{}

// Apply applies the rule to given file.
func (r *ErrorReturnRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	fileAst := file.AST
	walker := lintErrorReturn{
		file:    file,
		fileAst: fileAst,
		onFailure: func(failure lint.Failure) {
			failures = append(failures, failure)
		},
	}

	ast.Walk(walker, fileAst)

	return failures
}

// Name returns the rule name.
func (r *ErrorReturnRule) Name() string {
	return "error-return"
}

type lintErrorReturn struct {
	file      *lint.File
	fileAst   *ast.File
	onFailure func(lint.Failure)
}

func (w lintErrorReturn) Visit(n ast.Node) ast.Visitor {
	fn, ok := n.(*ast.FuncDecl)
	if !ok || fn.Type.Results == nil {
		return w
	}
	ret := fn.Type.Results.List
	if len(ret) <= 1 {
		return w
	}
	if isIdent(ret[len(ret)-1].Type, "error") {
		return nil
	}
	// An error return parameter should be the last parameter.
	// Flag any error parameters found before the last.
	for _, r := range ret[:len(ret)-1] {
		if isIdent(r.Type, "error") {
			w.onFailure(lint.Failure{
				Category:   "arg-order",
				Confidence: 0.9,
				Node:       fn,
				Failure:    "error should be the last type when returning multiple items",
			})
			break // only flag one
		}
	}
	return w
}

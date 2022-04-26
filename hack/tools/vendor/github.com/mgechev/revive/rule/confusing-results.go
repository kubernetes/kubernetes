package rule

import (
	"go/ast"

	"github.com/mgechev/revive/lint"
)

// ConfusingResultsRule lints given function declarations
type ConfusingResultsRule struct{}

// Apply applies the rule to given file.
func (r *ConfusingResultsRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	fileAst := file.AST
	walker := lintConfusingResults{
		onFailure: func(failure lint.Failure) {
			failures = append(failures, failure)
		},
	}

	ast.Walk(walker, fileAst)

	return failures
}

// Name returns the rule name.
func (r *ConfusingResultsRule) Name() string {
	return "confusing-results"
}

type lintConfusingResults struct {
	onFailure func(lint.Failure)
}

func (w lintConfusingResults) Visit(n ast.Node) ast.Visitor {
	fn, ok := n.(*ast.FuncDecl)
	if !ok || fn.Type.Results == nil || len(fn.Type.Results.List) < 2 {
		return w
	}
	lastType := ""
	for _, result := range fn.Type.Results.List {
		if len(result.Names) > 0 {
			return w
		}

		t, ok := result.Type.(*ast.Ident)
		if !ok {
			return w
		}

		if t.Name == lastType {
			w.onFailure(lint.Failure{
				Node:       n,
				Confidence: 1,
				Category:   "naming",
				Failure:    "unnamed results of the same type may be confusing, consider using named results",
			})
			break
		}
		lastType = t.Name

	}

	return w
}

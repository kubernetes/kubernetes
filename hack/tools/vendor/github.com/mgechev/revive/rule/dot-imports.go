package rule

import (
	"go/ast"

	"github.com/mgechev/revive/lint"
)

// DotImportsRule lints given else constructs.
type DotImportsRule struct{}

// Apply applies the rule to given file.
func (r *DotImportsRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	fileAst := file.AST
	walker := lintImports{
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
func (r *DotImportsRule) Name() string {
	return "dot-imports"
}

type lintImports struct {
	file      *lint.File
	fileAst   *ast.File
	onFailure func(lint.Failure)
}

func (w lintImports) Visit(_ ast.Node) ast.Visitor {
	for i, is := range w.fileAst.Imports {
		_ = i
		if is.Name != nil && is.Name.Name == "." && !w.file.IsTest() {
			w.onFailure(lint.Failure{
				Confidence: 1,
				Failure:    "should not use dot imports",
				Node:       is,
				Category:   "imports",
			})
		}
	}
	return nil
}

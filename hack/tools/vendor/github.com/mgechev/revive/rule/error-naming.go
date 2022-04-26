package rule

import (
	"fmt"
	"go/ast"
	"go/token"
	"strings"

	"github.com/mgechev/revive/lint"
)

// ErrorNamingRule lints given else constructs.
type ErrorNamingRule struct{}

// Apply applies the rule to given file.
func (r *ErrorNamingRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	fileAst := file.AST
	walker := lintErrors{
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
func (r *ErrorNamingRule) Name() string {
	return "error-naming"
}

type lintErrors struct {
	file      *lint.File
	fileAst   *ast.File
	onFailure func(lint.Failure)
}

func (w lintErrors) Visit(_ ast.Node) ast.Visitor {
	for _, decl := range w.fileAst.Decls {
		gd, ok := decl.(*ast.GenDecl)
		if !ok || gd.Tok != token.VAR {
			continue
		}
		for _, spec := range gd.Specs {
			spec := spec.(*ast.ValueSpec)
			if len(spec.Names) != 1 || len(spec.Values) != 1 {
				continue
			}
			ce, ok := spec.Values[0].(*ast.CallExpr)
			if !ok {
				continue
			}
			if !isPkgDot(ce.Fun, "errors", "New") && !isPkgDot(ce.Fun, "fmt", "Errorf") {
				continue
			}

			id := spec.Names[0]
			prefix := "err"
			if id.IsExported() {
				prefix = "Err"
			}
			if !strings.HasPrefix(id.Name, prefix) {
				w.onFailure(lint.Failure{
					Node:       id,
					Confidence: 0.9,
					Category:   "naming",
					Failure:    fmt.Sprintf("error var %s should have name of the form %sFoo", id.Name, prefix),
				})
			}
		}
	}
	return nil
}

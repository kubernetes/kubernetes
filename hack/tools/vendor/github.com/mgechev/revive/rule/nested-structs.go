package rule

import (
	"go/ast"

	"github.com/mgechev/revive/lint"
)

// NestedStructs lints nested structs.
type NestedStructs struct{}

// Apply applies the rule to given file.
func (r *NestedStructs) Apply(file *lint.File, arguments lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	if len(arguments) > 0 {
		panic(r.Name() + " doesn't take any arguments")
	}

	walker := &lintNestedStructs{
		fileAST: file.AST,
		onFailure: func(failure lint.Failure) {
			failures = append(failures, failure)
		},
	}

	ast.Walk(walker, file.AST)

	return failures
}

// Name returns the rule name.
func (r *NestedStructs) Name() string {
	return "nested-structs"
}

type lintNestedStructs struct {
	fileAST   *ast.File
	onFailure func(lint.Failure)
}

func (l *lintNestedStructs) Visit(n ast.Node) ast.Visitor {
	switch v := n.(type) {
	case *ast.FuncDecl:
		if v.Body != nil {
			ast.Walk(l, v.Body)
		}
		return nil
	case *ast.Field:
		if _, ok := v.Type.(*ast.StructType); ok {
			l.onFailure(lint.Failure{
				Failure:    "no nested structs are allowed",
				Category:   "style",
				Node:       v,
				Confidence: 1,
			})
			break
		}
	}
	return l
}

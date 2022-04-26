package rule

import (
	"fmt"
	"go/ast"
	"go/types"

	"github.com/mgechev/revive/lint"
)

// ContextKeysType lints given else constructs.
type ContextKeysType struct{}

// Apply applies the rule to given file.
func (r *ContextKeysType) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	fileAst := file.AST
	walker := lintContextKeyTypes{
		file:    file,
		fileAst: fileAst,
		onFailure: func(failure lint.Failure) {
			failures = append(failures, failure)
		},
	}

	file.Pkg.TypeCheck()
	ast.Walk(walker, fileAst)

	return failures
}

// Name returns the rule name.
func (r *ContextKeysType) Name() string {
	return "context-keys-type"
}

type lintContextKeyTypes struct {
	file      *lint.File
	fileAst   *ast.File
	onFailure func(lint.Failure)
}

func (w lintContextKeyTypes) Visit(n ast.Node) ast.Visitor {
	switch n := n.(type) {
	case *ast.CallExpr:
		checkContextKeyType(w, n)
	}

	return w
}

func checkContextKeyType(w lintContextKeyTypes, x *ast.CallExpr) {
	f := w.file
	sel, ok := x.Fun.(*ast.SelectorExpr)
	if !ok {
		return
	}
	pkg, ok := sel.X.(*ast.Ident)
	if !ok || pkg.Name != "context" {
		return
	}
	if sel.Sel.Name != "WithValue" {
		return
	}

	// key is second argument to context.WithValue
	if len(x.Args) != 3 {
		return
	}
	key := f.Pkg.TypesInfo.Types[x.Args[1]]

	if ktyp, ok := key.Type.(*types.Basic); ok && ktyp.Kind() != types.Invalid {
		w.onFailure(lint.Failure{
			Confidence: 1,
			Node:       x,
			Category:   "content",
			Failure:    fmt.Sprintf("should not use basic type %s as key in context.WithValue", key.Type),
		})
	}
}

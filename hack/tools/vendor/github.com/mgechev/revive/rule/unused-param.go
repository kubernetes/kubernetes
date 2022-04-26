package rule

import (
	"fmt"
	"go/ast"

	"github.com/mgechev/revive/lint"
)

// UnusedParamRule lints unused params in functions.
type UnusedParamRule struct{}

// Apply applies the rule to given file.
func (r *UnusedParamRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	onFailure := func(failure lint.Failure) {
		failures = append(failures, failure)
	}

	w := lintUnusedParamRule{onFailure: onFailure}

	ast.Walk(w, file.AST)

	return failures
}

// Name returns the rule name.
func (r *UnusedParamRule) Name() string {
	return "unused-parameter"
}

type lintUnusedParamRule struct {
	onFailure func(lint.Failure)
}

func (w lintUnusedParamRule) Visit(node ast.Node) ast.Visitor {
	switch n := node.(type) {
	case *ast.FuncDecl:
		params := retrieveNamedParams(n.Type.Params)
		if len(params) < 1 {
			return nil // skip, func without parameters
		}

		if n.Body == nil {
			return nil // skip, is a function prototype
		}

		// inspect the func body looking for references to parameters
		fselect := func(n ast.Node) bool {
			ident, isAnID := n.(*ast.Ident)

			if !isAnID {
				return false
			}

			_, isAParam := params[ident.Obj]
			if isAParam {
				params[ident.Obj] = false // mark as used
			}

			return false
		}
		_ = pick(n.Body, fselect, nil)

		for _, p := range n.Type.Params.List {
			for _, n := range p.Names {
				if params[n.Obj] {
					w.onFailure(lint.Failure{
						Confidence: 1,
						Node:       n,
						Category:   "bad practice",
						Failure:    fmt.Sprintf("parameter '%s' seems to be unused, consider removing or renaming it as _", n.Name),
					})
				}
			}
		}

		return nil // full method body already inspected
	}

	return w
}

func retrieveNamedParams(params *ast.FieldList) map[*ast.Object]bool {
	result := map[*ast.Object]bool{}
	if params.List == nil {
		return result
	}

	for _, p := range params.List {
		for _, n := range p.Names {
			if n.Name == "_" {
				continue
			}

			result[n.Obj] = true
		}
	}

	return result
}

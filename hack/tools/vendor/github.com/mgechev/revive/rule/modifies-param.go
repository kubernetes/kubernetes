package rule

import (
	"fmt"
	"go/ast"

	"github.com/mgechev/revive/lint"
)

// ModifiesParamRule lints given else constructs.
type ModifiesParamRule struct{}

// Apply applies the rule to given file.
func (r *ModifiesParamRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	onFailure := func(failure lint.Failure) {
		failures = append(failures, failure)
	}

	w := lintModifiesParamRule{onFailure: onFailure}
	ast.Walk(w, file.AST)
	return failures
}

// Name returns the rule name.
func (r *ModifiesParamRule) Name() string {
	return "modifies-parameter"
}

type lintModifiesParamRule struct {
	params    map[string]bool
	onFailure func(lint.Failure)
}

func retrieveParamNames(pl []*ast.Field) map[string]bool {
	result := make(map[string]bool, len(pl))
	for _, p := range pl {
		for _, n := range p.Names {
			if n.Name == "_" {
				continue
			}

			result[n.Name] = true
		}
	}
	return result
}

func (w lintModifiesParamRule) Visit(node ast.Node) ast.Visitor {
	switch v := node.(type) {
	case *ast.FuncDecl:
		w.params = retrieveParamNames(v.Type.Params.List)
	case *ast.IncDecStmt:
		if id, ok := v.X.(*ast.Ident); ok {
			checkParam(id, &w)
		}
	case *ast.AssignStmt:
		lhs := v.Lhs
		for _, e := range lhs {
			id, ok := e.(*ast.Ident)
			if ok {
				checkParam(id, &w)
			}
		}
	}

	return w
}

func checkParam(id *ast.Ident, w *lintModifiesParamRule) {
	if w.params[id.Name] {
		w.onFailure(lint.Failure{
			Confidence: 0.5, // confidence is low because of shadow variables
			Node:       id,
			Category:   "bad practice",
			Failure:    fmt.Sprintf("parameter '%s' seems to be modified", id),
		})
	}
}

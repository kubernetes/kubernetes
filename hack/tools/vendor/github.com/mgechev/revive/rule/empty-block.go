package rule

import (
	"go/ast"

	"github.com/mgechev/revive/lint"
)

// EmptyBlockRule lints given else constructs.
type EmptyBlockRule struct{}

// Apply applies the rule to given file.
func (r *EmptyBlockRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	onFailure := func(failure lint.Failure) {
		failures = append(failures, failure)
	}

	w := lintEmptyBlock{make(map[*ast.BlockStmt]bool), onFailure}
	ast.Walk(w, file.AST)
	return failures
}

// Name returns the rule name.
func (r *EmptyBlockRule) Name() string {
	return "empty-block"
}

type lintEmptyBlock struct {
	ignore    map[*ast.BlockStmt]bool
	onFailure func(lint.Failure)
}

func (w lintEmptyBlock) Visit(node ast.Node) ast.Visitor {
	switch n := node.(type) {
	case *ast.FuncDecl:
		w.ignore[n.Body] = true
		return w
	case *ast.FuncLit:
		w.ignore[n.Body] = true
		return w
	case *ast.RangeStmt:
		if len(n.Body.List) == 0 {
			w.onFailure(lint.Failure{
				Confidence: 0.9,
				Node:       n,
				Category:   "logic",
				Failure:    "this block is empty, you can remove it",
			})
			return nil // skip visiting the range subtree (it will produce a duplicated failure)
		}
	case *ast.BlockStmt:
		if !w.ignore[n] && len(n.List) == 0 {
			w.onFailure(lint.Failure{
				Confidence: 1,
				Node:       n,
				Category:   "logic",
				Failure:    "this block is empty, you can remove it",
			})
		}
	}

	return w
}

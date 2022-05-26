package rule

import (
	"go/ast"

	"github.com/mgechev/revive/lint"
)

// EarlyReturnRule lints given else constructs.
type EarlyReturnRule struct{}

// Apply applies the rule to given file.
func (r *EarlyReturnRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	onFailure := func(failure lint.Failure) {
		failures = append(failures, failure)
	}

	w := lintEarlyReturnRule{onFailure: onFailure}
	ast.Walk(w, file.AST)
	return failures
}

// Name returns the rule name.
func (r *EarlyReturnRule) Name() string {
	return "early-return"
}

type lintEarlyReturnRule struct {
	onFailure func(lint.Failure)
}

func (w lintEarlyReturnRule) Visit(node ast.Node) ast.Visitor {
	switch n := node.(type) {
	case *ast.IfStmt:
		if n.Else == nil {
			// no else branch
			return w
		}

		elseBlock, ok := n.Else.(*ast.BlockStmt)
		if !ok {
			// is if-else-if
			return w
		}

		lenElseBlock := len(elseBlock.List)
		if lenElseBlock < 1 {
			// empty else block, continue (there is another rule that warns on empty blocks)
			return w
		}

		lenThenBlock := len(n.Body.List)
		if lenThenBlock < 1 {
			// then block is empty thus the stmt can be simplified
			w.onFailure(lint.Failure{
				Confidence: 1,
				Node:       n,
				Failure:    "if c { } else {... return} can be simplified to if !c { ... return }",
			})

			return w
		}

		_, lastThenStmtIsReturn := n.Body.List[lenThenBlock-1].(*ast.ReturnStmt)
		_, lastElseStmtIsReturn := elseBlock.List[lenElseBlock-1].(*ast.ReturnStmt)
		if lastElseStmtIsReturn && !lastThenStmtIsReturn {
			w.onFailure(lint.Failure{
				Confidence: 1,
				Node:       n,
				Failure:    "if c {...} else {... return } can be simplified to if !c { ... return } ...",
			})
		}
	}

	return w
}

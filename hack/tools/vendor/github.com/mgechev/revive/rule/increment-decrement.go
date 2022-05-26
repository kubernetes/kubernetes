package rule

import (
	"fmt"
	"go/ast"
	"go/token"

	"github.com/mgechev/revive/lint"
)

// IncrementDecrementRule lints given else constructs.
type IncrementDecrementRule struct{}

// Apply applies the rule to given file.
func (r *IncrementDecrementRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	fileAst := file.AST
	walker := lintIncrementDecrement{
		file: file,
		onFailure: func(failure lint.Failure) {
			failures = append(failures, failure)
		},
	}

	ast.Walk(walker, fileAst)

	return failures
}

// Name returns the rule name.
func (r *IncrementDecrementRule) Name() string {
	return "increment-decrement"
}

type lintIncrementDecrement struct {
	file      *lint.File
	fileAst   *ast.File
	onFailure func(lint.Failure)
}

func (w lintIncrementDecrement) Visit(n ast.Node) ast.Visitor {
	as, ok := n.(*ast.AssignStmt)
	if !ok {
		return w
	}
	if len(as.Lhs) != 1 {
		return w
	}
	if !isOne(as.Rhs[0]) {
		return w
	}
	var suffix string
	switch as.Tok {
	case token.ADD_ASSIGN:
		suffix = "++"
	case token.SUB_ASSIGN:
		suffix = "--"
	default:
		return w
	}
	w.onFailure(lint.Failure{
		Confidence: 0.8,
		Node:       as,
		Category:   "unary-op",
		Failure:    fmt.Sprintf("should replace %s with %s%s", w.file.Render(as), w.file.Render(as.Lhs[0]), suffix),
	})
	return w
}

func isOne(expr ast.Expr) bool {
	lit, ok := expr.(*ast.BasicLit)
	return ok && lit.Kind == token.INT && lit.Value == "1"
}

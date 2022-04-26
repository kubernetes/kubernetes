package rule

import (
	"fmt"
	"go/ast"
	"go/token"

	"github.com/mgechev/revive/lint"
)

// OptimizeOperandsOrderRule lints given else constructs.
type OptimizeOperandsOrderRule struct{}

// Apply applies the rule to given file.
func (r *OptimizeOperandsOrderRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	onFailure := func(failure lint.Failure) {
		failures = append(failures, failure)
	}
	w := lintOptimizeOperandsOrderlExpr{
		onFailure: onFailure,
	}
	ast.Walk(w, file.AST)
	return failures
}

// Name returns the rule name.
func (r *OptimizeOperandsOrderRule) Name() string {
	return "optimize-operands-order"
}

type lintOptimizeOperandsOrderlExpr struct {
	onFailure func(failure lint.Failure)
}

// Visit checks boolean AND and OR expressions to determine
// if swapping their operands may result in an execution speedup.
func (w lintOptimizeOperandsOrderlExpr) Visit(node ast.Node) ast.Visitor {
	binExpr, ok := node.(*ast.BinaryExpr)
	if !ok {
		return w
	}

	switch binExpr.Op {
	case token.LAND, token.LOR:
	default:
		return w
	}

	isCaller := func(n ast.Node) bool {
		_, ok := n.(*ast.CallExpr)
		return ok
	}

	// check if the left sub-expression contains a function call
	nodes := pick(binExpr.X, isCaller, nil)
	if len(nodes) < 1 {
		return w
	}

	// check if the right sub-expression does not contain a function call
	nodes = pick(binExpr.Y, isCaller, nil)
	if len(nodes) > 0 {
		return w
	}

	newExpr := ast.BinaryExpr{X: binExpr.Y, Y: binExpr.X, Op: binExpr.Op}
	w.onFailure(lint.Failure{
		Failure:    fmt.Sprintf("for better performance '%v' might be rewritten as '%v'", gofmt(binExpr), gofmt(&newExpr)),
		Node:       node,
		Category:   "optimization",
		Confidence: 0.3,
	})

	return w
}

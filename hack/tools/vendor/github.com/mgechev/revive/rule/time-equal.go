package rule

import (
	"fmt"
	"go/ast"
	"go/token"

	"github.com/mgechev/revive/lint"
)

// TimeEqualRule shows where "==" and "!=" used for equality check time.Time
type TimeEqualRule struct{}

// Apply applies the rule to given file.
func (*TimeEqualRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	onFailure := func(failure lint.Failure) {
		failures = append(failures, failure)
	}

	w := &lintTimeEqual{file, onFailure}
	if w.file.Pkg.TypeCheck() != nil {
		return nil
	}

	ast.Walk(w, file.AST)
	return failures
}

// Name returns the rule name.
func (*TimeEqualRule) Name() string {
	return "time-equal"
}

type lintTimeEqual struct {
	file      *lint.File
	onFailure func(lint.Failure)
}

func (l *lintTimeEqual) Visit(node ast.Node) ast.Visitor {
	expr, ok := node.(*ast.BinaryExpr)
	if !ok {
		return l
	}

	switch expr.Op {
	case token.EQL, token.NEQ:
	default:
		return l
	}

	xtyp := l.file.Pkg.TypeOf(expr.X)
	ytyp := l.file.Pkg.TypeOf(expr.Y)

	if !isNamedType(xtyp, "time", "Time") || !isNamedType(ytyp, "time", "Time") {
		return l
	}

	var failure string
	switch expr.Op {
	case token.EQL:
		failure = fmt.Sprintf("use %s.Equal(%s) instead of %q operator", expr.X, expr.Y, expr.Op)
	case token.NEQ:
		failure = fmt.Sprintf("use !%s.Equal(%s) instead of %q operator", expr.X, expr.Y, expr.Op)
	}

	l.onFailure(lint.Failure{
		Category:   "time",
		Confidence: 1,
		Node:       node,
		Failure:    failure,
	})

	return l
}

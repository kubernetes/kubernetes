package rule

import (
	"fmt"
	"go/ast"
	"go/token"
	"strings"

	"github.com/mgechev/revive/lint"
)

// RangeValAddress lints
type RangeValAddress struct{}

// Apply applies the rule to given file.
func (r *RangeValAddress) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	walker := rangeValAddress{
		file: file,
		onFailure: func(failure lint.Failure) {
			failures = append(failures, failure)
		},
	}

	file.Pkg.TypeCheck()
	ast.Walk(walker, file.AST)

	return failures
}

// Name returns the rule name.
func (r *RangeValAddress) Name() string {
	return "range-val-address"
}

type rangeValAddress struct {
	file      *lint.File
	onFailure func(lint.Failure)
}

func (w rangeValAddress) Visit(node ast.Node) ast.Visitor {
	n, ok := node.(*ast.RangeStmt)
	if !ok {
		return w
	}

	value, ok := n.Value.(*ast.Ident)
	if !ok {
		return w
	}

	valueIsStarExpr := false
	if t := w.file.Pkg.TypeOf(value); t != nil {
		valueIsStarExpr = strings.HasPrefix(t.String(), "*")
	}

	ast.Walk(rangeBodyVisitor{
		valueIsStarExpr: valueIsStarExpr,
		valueID:         value.Obj,
		onFailure:       w.onFailure,
	}, n.Body)

	return w
}

type rangeBodyVisitor struct {
	valueIsStarExpr bool
	valueID         *ast.Object
	onFailure       func(lint.Failure)
}

func (bw rangeBodyVisitor) Visit(node ast.Node) ast.Visitor {
	asgmt, ok := node.(*ast.AssignStmt)
	if !ok {
		return bw
	}

	for _, exp := range asgmt.Lhs {
		e, ok := exp.(*ast.IndexExpr)
		if !ok {
			continue
		}
		if bw.isAccessingRangeValueAddress(e.Index) { // e.g. a[&value]...
			bw.onFailure(bw.newFailure(e.Index))
		}
	}

	for _, exp := range asgmt.Rhs {
		switch e := exp.(type) {
		case *ast.UnaryExpr: // e.g. ...&value, ...&value.id
			if bw.isAccessingRangeValueAddress(e) {
				bw.onFailure(bw.newFailure(e))
			}
		case *ast.CallExpr:
			if fun, ok := e.Fun.(*ast.Ident); ok && fun.Name == "append" { // e.g. ...append(arr, &value)
				for _, v := range e.Args {
					if bw.isAccessingRangeValueAddress(v) {
						bw.onFailure(bw.newFailure(e))
					}
				}
			}
		}
	}
	return bw
}

func (bw rangeBodyVisitor) isAccessingRangeValueAddress(exp ast.Expr) bool {
	u, ok := exp.(*ast.UnaryExpr)
	if !ok {
		return false
	}

	if u.Op != token.AND {
		return false
	}

	v, ok := u.X.(*ast.Ident)
	if !ok {
		var s *ast.SelectorExpr
		s, ok = u.X.(*ast.SelectorExpr)
		if !ok {
			return false
		}
		v, ok = s.X.(*ast.Ident)
		if !ok {
			return false
		}

		if bw.valueIsStarExpr { // check type of value
			return false
		}
	}

	return ok && v.Obj == bw.valueID
}

func (bw rangeBodyVisitor) newFailure(node ast.Node) lint.Failure {
	return lint.Failure{
		Node:       node,
		Confidence: 1,
		Failure:    fmt.Sprintf("suspicious assignment of '%s'. range-loop variables always have the same address", bw.valueID.Name),
	}
}

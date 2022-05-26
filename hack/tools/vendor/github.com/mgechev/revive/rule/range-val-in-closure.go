package rule

import (
	"fmt"
	"go/ast"

	"github.com/mgechev/revive/lint"
)

// RangeValInClosureRule lints given else constructs.
type RangeValInClosureRule struct{}

// Apply applies the rule to given file.
func (r *RangeValInClosureRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	walker := rangeValInClosure{
		onFailure: func(failure lint.Failure) {
			failures = append(failures, failure)
		},
	}

	ast.Walk(walker, file.AST)

	return failures
}

// Name returns the rule name.
func (r *RangeValInClosureRule) Name() string {
	return "range-val-in-closure"
}

type rangeValInClosure struct {
	onFailure func(lint.Failure)
}

func (w rangeValInClosure) Visit(node ast.Node) ast.Visitor {

	// Find the variables updated by the loop statement.
	var vars []*ast.Ident
	addVar := func(expr ast.Expr) {
		if id, ok := expr.(*ast.Ident); ok {
			vars = append(vars, id)
		}
	}
	var body *ast.BlockStmt
	switch n := node.(type) {
	case *ast.RangeStmt:
		body = n.Body
		addVar(n.Key)
		addVar(n.Value)
	case *ast.ForStmt:
		body = n.Body
		switch post := n.Post.(type) {
		case *ast.AssignStmt:
			// e.g. for p = head; p != nil; p = p.next
			for _, lhs := range post.Lhs {
				addVar(lhs)
			}
		case *ast.IncDecStmt:
			// e.g. for i := 0; i < n; i++
			addVar(post.X)
		}
	}
	if vars == nil {
		return w
	}

	// Inspect a go or defer statement
	// if it's the last one in the loop body.
	// (We give up if there are following statements,
	// because it's hard to prove go isn't followed by wait,
	// or defer by return.)
	if len(body.List) == 0 {
		return w
	}
	var last *ast.CallExpr
	switch s := body.List[len(body.List)-1].(type) {
	case *ast.GoStmt:
		last = s.Call
	case *ast.DeferStmt:
		last = s.Call
	default:
		return w
	}
	lit, ok := last.Fun.(*ast.FuncLit)
	if !ok {
		return w
	}

	if lit.Type == nil {
		// Not referring to a variable (e.g. struct field name)
		return w
	}

	var inspector func(n ast.Node) bool
	inspector = func(n ast.Node) bool {
		kv, ok := n.(*ast.KeyValueExpr)
		if ok {
			// do not check identifiers acting as key in key-value expressions (see issue #637)
			ast.Inspect(kv.Value, inspector)
			return false
		}
		id, ok := n.(*ast.Ident)
		if !ok || id.Obj == nil {
			return true
		}

		for _, v := range vars {
			if v.Obj == id.Obj {
				w.onFailure(lint.Failure{
					Confidence: 1,
					Failure:    fmt.Sprintf("loop variable %v captured by func literal", id.Name),
					Node:       n,
				})
			}
		}
		return true
	}
	ast.Inspect(lit.Body, inspector)
	return w
}

package rule

import (
	"fmt"
	"go/ast"

	"github.com/mgechev/revive/lint"
)

// DeferRule lints unused params in functions.
type DeferRule struct {
	allow map[string]bool
}

// Apply applies the rule to given file.
func (r *DeferRule) Apply(file *lint.File, arguments lint.Arguments) []lint.Failure {
	if r.allow == nil {
		r.allow = r.allowFromArgs(arguments)
	}
	var failures []lint.Failure
	onFailure := func(failure lint.Failure) {
		failures = append(failures, failure)
	}

	w := lintDeferRule{onFailure: onFailure, allow: r.allow}

	ast.Walk(w, file.AST)

	return failures
}

// Name returns the rule name.
func (r *DeferRule) Name() string {
	return "defer"
}

func (r *DeferRule) allowFromArgs(args lint.Arguments) map[string]bool {
	if len(args) < 1 {
		allow := map[string]bool{
			"loop":        true,
			"call-chain":  true,
			"method-call": true,
			"return":      true,
			"recover":     true,
		}

		return allow
	}

	aa, ok := args[0].([]interface{})
	if !ok {
		panic(fmt.Sprintf("Invalid argument '%v' for 'defer' rule. Expecting []string, got %T", args[0], args[0]))
	}

	allow := make(map[string]bool, len(aa))
	for _, subcase := range aa {
		sc, ok := subcase.(string)
		if !ok {
			panic(fmt.Sprintf("Invalid argument '%v' for 'defer' rule. Expecting string, got %T", subcase, subcase))
		}
		allow[sc] = true
	}

	return allow
}

type lintDeferRule struct {
	onFailure  func(lint.Failure)
	inALoop    bool
	inADefer   bool
	inAFuncLit bool
	allow      map[string]bool
}

func (w lintDeferRule) Visit(node ast.Node) ast.Visitor {
	switch n := node.(type) {
	case *ast.ForStmt:
		w.visitSubtree(n.Body, w.inADefer, true, w.inAFuncLit)
		return nil
	case *ast.RangeStmt:
		w.visitSubtree(n.Body, w.inADefer, true, w.inAFuncLit)
		return nil
	case *ast.FuncLit:
		w.visitSubtree(n.Body, w.inADefer, false, true)
		return nil
	case *ast.ReturnStmt:
		if len(n.Results) != 0 && w.inADefer && w.inAFuncLit {
			w.newFailure("return in a defer function has no effect", n, 1.0, "logic", "return")
		}
	case *ast.CallExpr:
		if !w.inADefer && isIdent(n.Fun, "recover") {
			// confidence is not 1 because recover can be in a function that is deferred elsewhere
			w.newFailure("recover must be called inside a deferred function", n, 0.8, "logic", "recover")
		}
	case *ast.DeferStmt:
		w.visitSubtree(n.Call.Fun, true, false, false)

		if w.inALoop {
			w.newFailure("prefer not to defer inside loops", n, 1.0, "bad practice", "loop")
		}

		switch fn := n.Call.Fun.(type) {
		case *ast.CallExpr:
			w.newFailure("prefer not to defer chains of function calls", fn, 1.0, "bad practice", "call-chain")
		case *ast.SelectorExpr:
			if id, ok := fn.X.(*ast.Ident); ok {
				isMethodCall := id != nil && id.Obj != nil && id.Obj.Kind == ast.Typ
				if isMethodCall {
					w.newFailure("be careful when deferring calls to methods without pointer receiver", fn, 0.8, "bad practice", "method-call")
				}
			}
		}
		return nil
	}

	return w
}

func (w lintDeferRule) visitSubtree(n ast.Node, inADefer, inALoop, inAFuncLit bool) {
	nw := &lintDeferRule{
		onFailure:  w.onFailure,
		inADefer:   inADefer,
		inALoop:    inALoop,
		inAFuncLit: inAFuncLit,
		allow:      w.allow}
	ast.Walk(nw, n)
}

func (w lintDeferRule) newFailure(msg string, node ast.Node, confidence float64, cat string, subcase string) {
	if !w.allow[subcase] {
		return
	}

	w.onFailure(lint.Failure{
		Confidence: confidence,
		Node:       node,
		Category:   cat,
		Failure:    msg,
	})
}

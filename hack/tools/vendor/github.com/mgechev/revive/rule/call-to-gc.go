package rule

import (
	"go/ast"

	"github.com/mgechev/revive/lint"
)

// CallToGCRule lints calls to the garbage collector.
type CallToGCRule struct{}

// Apply applies the rule to given file.
func (r *CallToGCRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure
	onFailure := func(failure lint.Failure) {
		failures = append(failures, failure)
	}

	var gcTriggeringFunctions = map[string]map[string]bool{
		"runtime": {"GC": true},
	}

	w := lintCallToGC{onFailure, gcTriggeringFunctions}
	ast.Walk(w, file.AST)

	return failures
}

// Name returns the rule name.
func (r *CallToGCRule) Name() string {
	return "call-to-gc"
}

type lintCallToGC struct {
	onFailure             func(lint.Failure)
	gcTriggeringFunctions map[string]map[string]bool
}

func (w lintCallToGC) Visit(node ast.Node) ast.Visitor {
	ce, ok := node.(*ast.CallExpr)
	if !ok {
		return w // nothing to do, the node is not a call
	}

	fc, ok := ce.Fun.(*ast.SelectorExpr)
	if !ok {
		return nil // nothing to do, the call is not of the form pkg.func(...)
	}

	id, ok := fc.X.(*ast.Ident)

	if !ok {
		return nil // in case X is not an id (it should be!)
	}

	fn := fc.Sel.Name
	pkg := id.Name
	if !w.gcTriggeringFunctions[pkg][fn] {
		return nil // it isn't a call to a GC triggering function
	}

	w.onFailure(lint.Failure{
		Confidence: 1,
		Node:       node,
		Category:   "bad practice",
		Failure:    "explicit call to the garbage collector",
	})

	return w
}

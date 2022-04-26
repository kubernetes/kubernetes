package rule

import (
	"fmt"
	"go/ast"

	"github.com/mgechev/revive/lint"
)

// UnusedReceiverRule lints unused params in functions.
type UnusedReceiverRule struct{}

// Apply applies the rule to given file.
func (*UnusedReceiverRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	onFailure := func(failure lint.Failure) {
		failures = append(failures, failure)
	}

	w := lintUnusedReceiverRule{onFailure: onFailure}

	ast.Walk(w, file.AST)

	return failures
}

// Name returns the rule name.
func (*UnusedReceiverRule) Name() string {
	return "unused-receiver"
}

type lintUnusedReceiverRule struct {
	onFailure func(lint.Failure)
}

func (w lintUnusedReceiverRule) Visit(node ast.Node) ast.Visitor {
	switch n := node.(type) {
	case *ast.FuncDecl:
		if n.Recv == nil {
			return nil // skip this func decl, not a method
		}

		rec := n.Recv.List[0] // safe to access only the first (unique) element of the list
		if len(rec.Names) < 1 {
			return nil // the receiver is anonymous: func (aType) Foo(...) ...
		}

		recID := rec.Names[0]
		if recID.Name == "_" {
			return nil // the receiver is already named _
		}

		// inspect the func body looking for references to the receiver id
		fselect := func(n ast.Node) bool {
			ident, isAnID := n.(*ast.Ident)

			return isAnID && ident.Obj == recID.Obj
		}
		refs2recID := pick(n.Body, fselect, nil)

		if len(refs2recID) > 0 {
			return nil // the receiver is referenced in the func body
		}

		w.onFailure(lint.Failure{
			Confidence: 1,
			Node:       recID,
			Category:   "bad practice",
			Failure:    fmt.Sprintf("method receiver '%s' is not referenced in method's body, consider removing or renaming it as _", recID.Name),
		})

		return nil // full method body already inspected
	}

	return w
}

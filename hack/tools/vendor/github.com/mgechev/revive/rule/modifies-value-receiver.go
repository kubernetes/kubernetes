package rule

import (
	"go/ast"
	"strings"

	"github.com/mgechev/revive/lint"
)

// ModifiesValRecRule lints assignments to value method-receivers.
type ModifiesValRecRule struct{}

// Apply applies the rule to given file.
func (r *ModifiesValRecRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure

	onFailure := func(failure lint.Failure) {
		failures = append(failures, failure)
	}

	w := lintModifiesValRecRule{file: file, onFailure: onFailure}
	file.Pkg.TypeCheck()
	ast.Walk(w, file.AST)

	return failures
}

// Name returns the rule name.
func (r *ModifiesValRecRule) Name() string {
	return "modifies-value-receiver"
}

type lintModifiesValRecRule struct {
	file      *lint.File
	onFailure func(lint.Failure)
}

func (w lintModifiesValRecRule) Visit(node ast.Node) ast.Visitor {
	switch n := node.(type) {
	case *ast.FuncDecl:
		if n.Recv == nil {
			return nil // skip, not a method
		}

		receiver := n.Recv.List[0]
		if _, ok := receiver.Type.(*ast.StarExpr); ok {
			return nil // skip, method with pointer receiver
		}

		if w.skipType(receiver.Type) {
			return nil // skip, receiver is a map or array
		}

		if len(receiver.Names) < 1 {
			return nil // skip, anonymous receiver
		}

		receiverName := receiver.Names[0].Name
		if receiverName == "_" {
			return nil // skip, anonymous receiver
		}

		fselect := func(n ast.Node) bool {
			// look for assignments with the receiver in the right hand
			asgmt, ok := n.(*ast.AssignStmt)
			if !ok {
				return false
			}

			for _, exp := range asgmt.Lhs {
				switch e := exp.(type) {
				case *ast.IndexExpr: // receiver...[] = ...
					continue
				case *ast.StarExpr: // *receiver = ...
					continue
				case *ast.SelectorExpr: // receiver.field = ...
					name := w.getNameFromExpr(e.X)
					if name == "" || name != receiverName {
						continue
					}

					if w.skipType(ast.Expr(e.Sel)) {
						continue
					}

				case *ast.Ident: // receiver := ...
					if e.Name != receiverName {
						continue
					}
				default:
					continue
				}

				return true
			}

			return false
		}

		assignmentsToReceiver := pick(n.Body, fselect, nil)

		for _, assignment := range assignmentsToReceiver {
			w.onFailure(lint.Failure{
				Node:       assignment,
				Confidence: 1,
				Failure:    "suspicious assignment to a by-value method receiver",
			})
		}
	}

	return w
}

func (w lintModifiesValRecRule) skipType(t ast.Expr) bool {
	rt := w.file.Pkg.TypeOf(t)
	if rt == nil {
		return false
	}

	rt = rt.Underlying()
	rtName := rt.String()

	// skip when receiver is a map or array
	return strings.HasPrefix(rtName, "[]") || strings.HasPrefix(rtName, "map[")
}

func (lintModifiesValRecRule) getNameFromExpr(ie ast.Expr) string {
	ident, ok := ie.(*ast.Ident)
	if !ok {
		return ""
	}

	return ident.Name
}

package rule

import (
	"go/ast"
	"go/token"
	"go/types"

	"github.com/mgechev/revive/lint"
)

// AtomicRule lints given else constructs.
type AtomicRule struct{}

// Apply applies the rule to given file.
func (r *AtomicRule) Apply(file *lint.File, _ lint.Arguments) []lint.Failure {
	var failures []lint.Failure
	walker := atomic{
		pkgTypesInfo: file.Pkg.TypesInfo,
		onFailure: func(failure lint.Failure) {
			failures = append(failures, failure)
		},
	}

	ast.Walk(walker, file.AST)

	return failures
}

// Name returns the rule name.
func (r *AtomicRule) Name() string {
	return "atomic"
}

type atomic struct {
	pkgTypesInfo *types.Info
	onFailure    func(lint.Failure)
}

func (w atomic) Visit(node ast.Node) ast.Visitor {
	n, ok := node.(*ast.AssignStmt)
	if !ok {
		return w
	}

	if len(n.Lhs) != len(n.Rhs) {
		return nil // skip assignment sub-tree
	}
	if len(n.Lhs) == 1 && n.Tok == token.DEFINE {
		return nil // skip assignment sub-tree
	}

	for i, right := range n.Rhs {
		call, ok := right.(*ast.CallExpr)
		if !ok {
			continue
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			continue
		}
		pkgIdent, _ := sel.X.(*ast.Ident)
		if w.pkgTypesInfo != nil {
			pkgName, ok := w.pkgTypesInfo.Uses[pkgIdent].(*types.PkgName)
			if !ok || pkgName.Imported().Path() != "sync/atomic" {
				continue
			}
		}

		switch sel.Sel.Name {
		case "AddInt32", "AddInt64", "AddUint32", "AddUint64", "AddUintptr":
			left := n.Lhs[i]
			if len(call.Args) != 2 {
				continue
			}
			arg := call.Args[0]
			broken := false

			if uarg, ok := arg.(*ast.UnaryExpr); ok && uarg.Op == token.AND {
				broken = gofmt(left) == gofmt(uarg.X)
			} else if star, ok := left.(*ast.StarExpr); ok {
				broken = gofmt(star.X) == gofmt(arg)
			}

			if broken {
				w.onFailure(lint.Failure{
					Confidence: 1,
					Failure:    "direct assignment to atomic value",
					Node:       n,
				})
			}
		}
	}
	return w
}

package checkers

import (
	"go/ast"
	"go/token"
	"go/types"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/checkers/internal/lintutil"
	"github.com/go-critic/go-critic/framework/linter"
	"github.com/go-toolsmith/astcast"
	"github.com/go-toolsmith/astequal"
	"github.com/go-toolsmith/typep"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "evalOrder"
	info.Tags = []string{"diagnostic", "experimental"}
	info.Summary = "Detects unwanted dependencies on the evaluation order"
	info.Before = `return x, f(&x)`
	info.After = `
err := f(&x)
return x, err
`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		return astwalk.WalkerForStmt(&evalOrderChecker{ctx: ctx}), nil
	})
}

type evalOrderChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext
}

func (c *evalOrderChecker) VisitStmt(stmt ast.Stmt) {
	ret := astcast.ToReturnStmt(stmt)
	if len(ret.Results) < 2 {
		return
	}

	// TODO(quasilyte): handle selector expressions like o.val in addition
	// to bare identifiers.
	addrTake := &ast.UnaryExpr{Op: token.AND}
	for _, res := range ret.Results {
		id, ok := res.(*ast.Ident)
		if !ok {
			continue
		}
		addrTake.X = id // addrTake is &id now
		for _, res := range ret.Results {
			call, ok := res.(*ast.CallExpr)
			if !ok {
				continue
			}

			// 1. Check if there is a call in form of id.method() where
			// method takes id by a pointer.
			if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
				if astequal.Node(sel.X, id) && c.hasPtrRecv(sel.Sel) {
					c.warn(call)
				}
			}

			// 2. Check that there is no call that uses &id as an argument.
			dependency := lintutil.ContainsNode(call, func(n ast.Node) bool {
				return astequal.Node(addrTake, n)
			})
			if dependency {
				c.warn(call)
			}
		}
	}
}

func (c *evalOrderChecker) hasPtrRecv(fn *ast.Ident) bool {
	sig, ok := c.ctx.TypeOf(fn).(*types.Signature)
	if !ok {
		return false
	}
	return typep.IsPointer(sig.Recv().Type())
}

func (c *evalOrderChecker) warn(call *ast.CallExpr) {
	c.ctx.Warn(call, "may want to evaluate %s before the return statement", call)
}

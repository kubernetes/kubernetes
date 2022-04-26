package checkers

import (
	"go/ast"
	"go/types"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/framework/linter"
	"github.com/go-toolsmith/astcast"
	"github.com/go-toolsmith/astp"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "underef"
	info.Tags = []string{"style"}
	info.Params = linter.CheckerParams{
		"skipRecvDeref": {
			Value: true,
			Usage: "whether to skip (*x).method() calls where x is a pointer receiver",
		},
	}
	info.Summary = "Detects dereference expressions that can be omitted"
	info.Before = `
(*k).field = 5
v := (*a)[5] // only if a is array`
	info.After = `
k.field = 5
v := a[5]`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		c := &underefChecker{ctx: ctx}
		c.skipRecvDeref = info.Params.Bool("skipRecvDeref")
		return astwalk.WalkerForExpr(c), nil
	})
}

type underefChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext

	skipRecvDeref bool
}

func (c *underefChecker) VisitExpr(expr ast.Expr) {
	switch n := expr.(type) {
	case *ast.SelectorExpr:
		expr := astcast.ToParenExpr(n.X)
		if c.skipRecvDeref && c.isPtrRecvMethodCall(n.Sel) {
			return
		}

		if expr, ok := expr.X.(*ast.StarExpr); ok {
			if c.checkStarExpr(expr) {
				c.warnSelect(n)
			}
		}
	case *ast.IndexExpr:
		expr := astcast.ToParenExpr(n.X)
		if expr, ok := expr.X.(*ast.StarExpr); ok {
			if !c.checkStarExpr(expr) {
				return
			}
			if c.checkArray(expr) {
				c.warnArray(n)
			}
		}
	}
}

func (c *underefChecker) isPtrRecvMethodCall(fn *ast.Ident) bool {
	typ, ok := c.ctx.TypeOf(fn).(*types.Signature)
	if ok && typ != nil && typ.Recv() != nil {
		_, ok := typ.Recv().Type().(*types.Pointer)
		return ok
	}
	return false
}

func (c *underefChecker) underef(x *ast.ParenExpr) ast.Expr {
	// If there is only 1 deref, can remove parenthesis,
	// otherwise can remove StarExpr only.
	dereferenced := x.X.(*ast.StarExpr).X
	if astp.IsStarExpr(dereferenced) {
		return &ast.ParenExpr{X: dereferenced}
	}
	return dereferenced
}

func (c *underefChecker) warnSelect(expr *ast.SelectorExpr) {
	// TODO: add () to function output.
	c.ctx.Warn(expr, "could simplify %s to %s.%s",
		expr,
		c.underef(expr.X.(*ast.ParenExpr)),
		expr.Sel.Name)
}

func (c *underefChecker) warnArray(expr *ast.IndexExpr) {
	c.ctx.Warn(expr, "could simplify %s to %s[%s]",
		expr,
		c.underef(expr.X.(*ast.ParenExpr)),
		expr.Index)
}

// checkStarExpr checks if ast.StarExpr could be simplified.
func (c *underefChecker) checkStarExpr(expr *ast.StarExpr) bool {
	typ, ok := c.ctx.TypeOf(expr.X).Underlying().(*types.Pointer)
	if !ok {
		return false
	}

	switch typ.Elem().Underlying().(type) {
	case *types.Pointer, *types.Interface:
		return false
	default:
		return true
	}
}

func (c *underefChecker) checkArray(expr *ast.StarExpr) bool {
	typ, ok := c.ctx.TypeOf(expr.X).(*types.Pointer)
	if !ok {
		return false
	}
	_, ok = typ.Elem().(*types.Array)
	return ok
}

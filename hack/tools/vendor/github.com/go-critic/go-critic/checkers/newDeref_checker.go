package checkers

import (
	"go/ast"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/checkers/internal/lintutil"
	"github.com/go-critic/go-critic/framework/linter"
	"github.com/go-toolsmith/astcast"
	"golang.org/x/tools/go/ast/astutil"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "newDeref"
	info.Tags = []string{"style"}
	info.Summary = "Detects immediate dereferencing of `new` expressions"
	info.Before = `x := *new(bool)`
	info.After = `x := false`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		return astwalk.WalkerForExpr(&newDerefChecker{ctx: ctx}), nil
	})
}

type newDerefChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext
}

func (c *newDerefChecker) VisitExpr(expr ast.Expr) {
	deref := astcast.ToStarExpr(expr)
	call := astcast.ToCallExpr(deref.X)
	if astcast.ToIdent(call.Fun).Name == "new" {
		typ := c.ctx.TypeOf(call.Args[0])
		zv := lintutil.ZeroValueOf(astutil.Unparen(call.Args[0]), typ)
		if zv != nil {
			c.warn(expr, zv)
		}
	}
}

func (c *newDerefChecker) warn(cause, suggestion ast.Expr) {
	c.ctx.Warn(cause, "replace `%s` with `%s`", cause, suggestion)
}

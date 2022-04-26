package checkers

import (
	"go/ast"
	"go/token"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/checkers/internal/lintutil"
	"github.com/go-critic/go-critic/framework/linter"
	"github.com/go-toolsmith/astcast"
	"github.com/go-toolsmith/astequal"
	"github.com/go-toolsmith/typep"
	"golang.org/x/tools/go/ast/astutil"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "weakCond"
	info.Tags = []string{"diagnostic", "experimental"}
	info.Summary = "Detects conditions that are unsafe due to not being exhaustive"
	info.Before = `xs != nil && xs[0] != nil`
	info.After = `len(xs) != 0 && xs[0] != nil`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		return astwalk.WalkerForExpr(&weakCondChecker{ctx: ctx}), nil
	})
}

type weakCondChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext
}

func (c *weakCondChecker) VisitExpr(expr ast.Expr) {
	// TODO(Quasilyte): more patterns.
	// TODO(Quasilyte): analyze and fix false positives.

	cond := astcast.ToBinaryExpr(expr)
	lhs := astcast.ToBinaryExpr(astutil.Unparen(cond.X))
	rhs := astutil.Unparen(cond.Y)

	// Pattern 1.
	// `x != nil && usageOf(x[i])`
	// Pattern 2.
	// `x == nil || usageOf(x[i])`

	// lhs is `x <op> nil`
	x := lhs.X
	if !typep.IsSlice(c.ctx.TypeOf(x)) {
		return
	}
	if astcast.ToIdent(lhs.Y).Name != "nil" {
		return
	}

	pat1prefix := cond.Op == token.LAND && lhs.Op == token.NEQ
	pat2prefix := cond.Op == token.LOR && lhs.Op == token.EQL
	if !pat1prefix && !pat2prefix {
		return
	}

	if c.isIndexed(rhs, x) {
		c.warn(expr, "nil check may not be enough, check for len")
	}
}

// isIndexed reports whether x is indexed inside given expr tree.
func (c *weakCondChecker) isIndexed(tree, x ast.Expr) bool {
	return lintutil.ContainsNode(tree, func(n ast.Node) bool {
		indexing := astcast.ToIndexExpr(n)
		return astequal.Expr(x, indexing.X)
	})
}

func (c *weakCondChecker) warn(cause ast.Node, suggest string) {
	c.ctx.Warn(cause, "suspicious `%s`; %s", cause, suggest)
}

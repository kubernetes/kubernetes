package checkers

import (
	"go/ast"
	"go/token"
	"go/types"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/framework/linter"
	"github.com/go-toolsmith/astequal"
	"github.com/go-toolsmith/typep"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "dupSubExpr"
	info.Tags = []string{"diagnostic"}
	info.Summary = "Detects suspicious duplicated sub-expressions"
	info.Before = `
sort.Slice(xs, func(i, j int) bool {
	return xs[i].v < xs[i].v // Duplicated index
})`
	info.After = `
sort.Slice(xs, func(i, j int) bool {
	return xs[i].v < xs[j].v
})`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		c := &dupSubExprChecker{ctx: ctx}

		ops := []struct {
			op    token.Token
			float bool // Whether float args require special care
		}{
			{op: token.LOR},     // x || x
			{op: token.LAND},    // x && x
			{op: token.OR},      // x | x
			{op: token.AND},     // x & x
			{op: token.XOR},     // x ^ x
			{op: token.LSS},     // x < x
			{op: token.GTR},     // x > x
			{op: token.AND_NOT}, // x &^ x
			{op: token.REM},     // x % x

			{op: token.EQL, float: true}, // x == x
			{op: token.NEQ, float: true}, // x != x
			{op: token.LEQ, float: true}, // x <= x
			{op: token.GEQ, float: true}, // x >= x
			{op: token.QUO, float: true}, // x / x
			{op: token.SUB, float: true}, // x - x
		}

		c.opSet = make(map[token.Token]bool)
		c.floatOpsSet = make(map[token.Token]bool)
		for _, opInfo := range ops {
			c.opSet[opInfo.op] = true
			if opInfo.float {
				c.floatOpsSet[opInfo.op] = true
			}
		}

		return astwalk.WalkerForExpr(c), nil
	})
}

type dupSubExprChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext

	// opSet is a set of binary operations that do not make
	// sense with duplicated (same) RHS and LHS.
	opSet map[token.Token]bool

	floatOpsSet map[token.Token]bool
}

func (c *dupSubExprChecker) VisitExpr(expr ast.Expr) {
	if expr, ok := expr.(*ast.BinaryExpr); ok {
		c.checkBinaryExpr(expr)
	}
}

func (c *dupSubExprChecker) checkBinaryExpr(expr *ast.BinaryExpr) {
	if !c.opSet[expr.Op] {
		return
	}
	if c.resultIsFloat(expr.X) && c.floatOpsSet[expr.Op] {
		return
	}
	if typep.SideEffectFree(c.ctx.TypesInfo, expr) && c.opSet[expr.Op] && astequal.Expr(expr.X, expr.Y) {
		c.warn(expr)
	}
}

func (c *dupSubExprChecker) resultIsFloat(expr ast.Expr) bool {
	typ, ok := c.ctx.TypeOf(expr).(*types.Basic)
	return ok && typ.Info()&types.IsFloat != 0
}

func (c *dupSubExprChecker) warn(cause *ast.BinaryExpr) {
	c.ctx.Warn(cause, "suspicious identical LHS and RHS for `%s` operator", cause.Op)
}

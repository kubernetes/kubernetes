package checkers

import (
	"fmt"
	"go/ast"
	"go/token"
	"strconv"

	"github.com/go-toolsmith/astcast"
	"github.com/go-toolsmith/astcopy"
	"github.com/go-toolsmith/astequal"
	"github.com/go-toolsmith/astp"
	"github.com/go-toolsmith/typep"
	"golang.org/x/tools/go/ast/astutil"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/checkers/internal/lintutil"
	"github.com/go-critic/go-critic/framework/linter"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "boolExprSimplify"
	info.Tags = []string{"style", "experimental"}
	info.Summary = "Detects bool expressions that can be simplified"
	info.Before = `
a := !(elapsed >= expectElapsedMin)
b := !(x) == !(y)`
	info.After = `
a := elapsed < expectElapsedMin
b := (x) == (y)`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		return astwalk.WalkerForExpr(&boolExprSimplifyChecker{ctx: ctx}), nil
	})
}

type boolExprSimplifyChecker struct {
	astwalk.WalkHandler
	ctx       *linter.CheckerContext
	hasFloats bool
}

func (c *boolExprSimplifyChecker) VisitExpr(x ast.Expr) {
	if !astp.IsBinaryExpr(x) && !astp.IsUnaryExpr(x) {
		return
	}

	// Throw away non-bool expressions and avoid redundant
	// AST copying below.
	if typ := c.ctx.TypeOf(x); typ == nil || !typep.HasBoolKind(typ.Underlying()) {
		return
	}

	// We'll loose all types info after a copy,
	// this is why we record valuable info before doing it.
	c.hasFloats = lintutil.ContainsNode(x, func(n ast.Node) bool {
		if x, ok := n.(*ast.BinaryExpr); ok {
			return typep.HasFloatProp(c.ctx.TypeOf(x.X).Underlying()) ||
				typep.HasFloatProp(c.ctx.TypeOf(x.Y).Underlying())
		}
		return false
	})

	y := c.simplifyBool(astcopy.Expr(x))
	if !astequal.Expr(x, y) {
		c.warn(x, y)
	}
}

func (c *boolExprSimplifyChecker) simplifyBool(x ast.Expr) ast.Expr {
	return astutil.Apply(x, nil, func(cur *astutil.Cursor) bool {
		return c.doubleNegation(cur) ||
			c.negatedEquals(cur) ||
			c.invertComparison(cur) ||
			c.combineChecks(cur) ||
			c.removeIncDec(cur) ||
			c.foldRanges(cur) ||
			true
	}).(ast.Expr)
}

func (c *boolExprSimplifyChecker) doubleNegation(cur *astutil.Cursor) bool {
	neg1 := astcast.ToUnaryExpr(cur.Node())
	neg2 := astcast.ToUnaryExpr(astutil.Unparen(neg1.X))
	if neg1.Op == token.NOT && neg2.Op == token.NOT {
		cur.Replace(astutil.Unparen(neg2.X))
		return true
	}
	return false
}

func (c *boolExprSimplifyChecker) negatedEquals(cur *astutil.Cursor) bool {
	x, ok := cur.Node().(*ast.BinaryExpr)
	if !ok || x.Op != token.EQL {
		return false
	}
	neg1 := astcast.ToUnaryExpr(x.X)
	neg2 := astcast.ToUnaryExpr(x.Y)
	if neg1.Op == token.NOT && neg2.Op == token.NOT {
		x.X = neg1.X
		x.Y = neg2.X
		return true
	}
	return false
}

func (c *boolExprSimplifyChecker) invertComparison(cur *astutil.Cursor) bool {
	if c.hasFloats { // See #673
		return false
	}

	neg := astcast.ToUnaryExpr(cur.Node())
	cmp := astcast.ToBinaryExpr(astutil.Unparen(neg.X))
	if neg.Op != token.NOT {
		return false
	}

	// Replace operator to its negated form.
	switch cmp.Op {
	case token.EQL:
		cmp.Op = token.NEQ
	case token.NEQ:
		cmp.Op = token.EQL
	case token.LSS:
		cmp.Op = token.GEQ
	case token.GTR:
		cmp.Op = token.LEQ
	case token.LEQ:
		cmp.Op = token.GTR
	case token.GEQ:
		cmp.Op = token.LSS

	default:
		return false
	}
	cur.Replace(cmp)
	return true
}

func (c *boolExprSimplifyChecker) isSafe(x ast.Expr) bool {
	return typep.SideEffectFree(c.ctx.TypesInfo, x)
}

func (c *boolExprSimplifyChecker) combineChecks(cur *astutil.Cursor) bool {
	or, ok := cur.Node().(*ast.BinaryExpr)
	if !ok || or.Op != token.LOR {
		return false
	}

	lhs := astcast.ToBinaryExpr(astutil.Unparen(or.X))
	rhs := astcast.ToBinaryExpr(astutil.Unparen(or.Y))

	if !astequal.Expr(lhs.X, rhs.X) || !astequal.Expr(lhs.Y, rhs.Y) {
		return false
	}
	if !c.isSafe(lhs.X) || !c.isSafe(lhs.Y) {
		return false
	}

	combTable := [...]struct {
		x      token.Token
		y      token.Token
		result token.Token
	}{
		{token.GTR, token.EQL, token.GEQ},
		{token.EQL, token.GTR, token.GEQ},
		{token.LSS, token.EQL, token.LEQ},
		{token.EQL, token.LSS, token.LEQ},
	}
	for _, comb := range &combTable {
		if comb.x == lhs.Op && comb.y == rhs.Op {
			lhs.Op = comb.result
			cur.Replace(lhs)
			return true
		}
	}
	return false
}

func (c *boolExprSimplifyChecker) removeIncDec(cur *astutil.Cursor) bool {
	cmp := astcast.ToBinaryExpr(cur.Node())

	matchOneWay := func(op token.Token, x, y *ast.BinaryExpr) bool {
		if x.Op != op || astcast.ToBasicLit(x.Y).Value != "1" {
			return false
		}
		if y.Op == op && astcast.ToBasicLit(y.Y).Value == "1" {
			return false
		}
		return true
	}
	replace := func(lhsOp, rhsOp, replacement token.Token) bool {
		lhs := astcast.ToBinaryExpr(cmp.X)
		rhs := astcast.ToBinaryExpr(cmp.Y)
		switch {
		case matchOneWay(lhsOp, lhs, rhs):
			cmp.X = lhs.X
			cmp.Op = replacement
			cur.Replace(cmp)
			return true
		case matchOneWay(rhsOp, rhs, lhs):
			cmp.Y = rhs.X
			cmp.Op = replacement
			cur.Replace(cmp)
			return true
		default:
			return false
		}
	}

	switch cmp.Op {
	case token.GTR:
		// `x > y-1` => `x >= y`
		// `x+1 > y` => `x >= y`
		return replace(token.ADD, token.SUB, token.GEQ)

	case token.GEQ:
		// `x >= y+1` => `x > y`
		// `x-1 >= y` => `x > y`
		return replace(token.SUB, token.ADD, token.GTR)

	case token.LSS:
		// `x < y+1` => `x <= y`
		// `x-1 < y` => `x <= y`
		return replace(token.SUB, token.ADD, token.LEQ)

	case token.LEQ:
		// `x <= y-1` => `x < y`
		// `x+1 <= y` => `x < y`
		return replace(token.ADD, token.SUB, token.LSS)

	default:
		return false
	}
}

func (c *boolExprSimplifyChecker) foldRanges(cur *astutil.Cursor) bool {
	if c.hasFloats { // See #848
		return false
	}

	e, ok := cur.Node().(*ast.BinaryExpr)
	if !ok {
		return false
	}
	lhs := astcast.ToBinaryExpr(e.X)
	rhs := astcast.ToBinaryExpr(e.Y)
	if !c.isSafe(lhs.X) || !c.isSafe(rhs.X) {
		return false
	}
	if !astequal.Expr(lhs.X, rhs.X) {
		return false
	}

	c1, ok := c.int64val(lhs.Y)
	if !ok {
		return false
	}
	c2, ok := c.int64val(rhs.Y)
	if !ok {
		return false
	}

	type combination struct {
		lhsOp    token.Token
		rhsOp    token.Token
		rhsDiff  int64
		resDelta int64
	}
	match := func(comb *combination) bool {
		if lhs.Op != comb.lhsOp || rhs.Op != comb.rhsOp {
			return false
		}
		if c2-c1 != comb.rhsDiff {
			return false
		}
		return true
	}

	switch e.Op {
	case token.LAND:
		combTable := [...]combination{
			// `x > c && x < c+2` => `x == c+1`
			{token.GTR, token.LSS, 2, 1},
			// `x >= c && x < c+1` => `x == c`
			{token.GEQ, token.LSS, 1, 0},
			// `x > c && x <= c+1` => `x == c+1`
			{token.GTR, token.LEQ, 1, 1},
			// `x >= c && x <= c` => `x == c`
			{token.GEQ, token.LEQ, 0, 0},
		}
		for i := range combTable {
			comb := combTable[i]
			if match(&comb) {
				lhs.Op = token.EQL
				v := c1 + comb.resDelta
				lhs.Y.(*ast.BasicLit).Value = fmt.Sprint(v)
				cur.Replace(lhs)
				return true
			}
		}

	case token.LOR:
		combTable := [...]combination{
			// `x < c || x > c` => `x != c`
			{token.LSS, token.GTR, 0, 0},
			// `x <= c || x > c+1` => `x != c+1`
			{token.LEQ, token.GTR, 1, 1},
			// `x < c || x >= c+1` => `x != c`
			{token.LSS, token.GEQ, 1, 0},
			// `x <= c || x >= c+2` => `x != c+1`
			{token.LEQ, token.GEQ, 2, 1},
		}
		for i := range combTable {
			comb := combTable[i]
			if match(&comb) {
				lhs.Op = token.NEQ
				v := c1 + comb.resDelta
				lhs.Y.(*ast.BasicLit).Value = fmt.Sprint(v)
				cur.Replace(lhs)
				return true
			}
		}
	}

	return false
}

func (c *boolExprSimplifyChecker) int64val(x ast.Expr) (int64, bool) {
	// TODO(quasilyte): if we had types info, we could use TypesInfo.Types[x].Value,
	// but since copying erases leaves us without it, only basic literals are handled.
	lit, ok := x.(*ast.BasicLit)
	if !ok {
		return 0, false
	}
	v, err := strconv.ParseInt(lit.Value, 10, 64)
	if err != nil {
		return 0, false
	}
	return v, true
}

func (c *boolExprSimplifyChecker) warn(cause, suggestion ast.Expr) {
	c.SkipChilds = true
	c.ctx.Warn(cause, "can simplify `%s` to `%s`", cause, suggestion)
}

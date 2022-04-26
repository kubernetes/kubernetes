package checkers

import (
	"go/ast"
	"go/constant"
	"go/token"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/checkers/internal/lintutil"
	"github.com/go-critic/go-critic/framework/linter"
	"github.com/go-toolsmith/astcast"
	"github.com/go-toolsmith/astcopy"
	"github.com/go-toolsmith/astequal"
	"github.com/go-toolsmith/typep"
	"golang.org/x/tools/go/ast/astutil"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "badCond"
	info.Tags = []string{"diagnostic"}
	info.Summary = "Detects suspicious condition expressions"
	info.Before = `
for i := 0; i > n; i++ {
	xs[i] = 0
}`
	info.After = `
for i := 0; i < n; i++ {
	xs[i] = 0
}`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		return astwalk.WalkerForFuncDecl(&badCondChecker{ctx: ctx}), nil
	})
}

type badCondChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext
}

func (c *badCondChecker) VisitFuncDecl(decl *ast.FuncDecl) {
	ast.Inspect(decl.Body, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.ForStmt:
			c.checkForStmt(n)
		case ast.Expr:
			c.checkExpr(n)
		}
		return true
	})
}

func (c *badCondChecker) checkExpr(expr ast.Expr) {
	// TODO(quasilyte): recognize more patterns.

	cond := astcast.ToBinaryExpr(expr)
	lhs := astcast.ToBinaryExpr(astutil.Unparen(cond.X))
	rhs := astcast.ToBinaryExpr(astutil.Unparen(cond.Y))

	if cond.Op != token.LAND {
		return
	}

	// Notes:
	// `x != a || x != b` handled by go vet.

	// Pattern 1.
	// `x < a && x > b`; Where `a` is less than `b`.
	if c.lessAndGreater(lhs, rhs) {
		c.warnCond(cond, "always false")
		return
	}

	// Pattern 2.
	// `x == a && x == b`
	//
	// Valid when `b == a` is intended, but still reported.
	// We can disable "just suspicious" warnings by default
	// is users are upset with the current behavior.
	if c.equalToBoth(lhs, rhs) {
		c.warnCond(cond, "suspicious")
		return
	}
}

func (c *badCondChecker) equalToBoth(lhs, rhs *ast.BinaryExpr) bool {
	return lhs.Op == token.EQL && rhs.Op == token.EQL &&
		astequal.Expr(lhs.X, rhs.X)
}

func (c *badCondChecker) lessAndGreater(lhs, rhs *ast.BinaryExpr) bool {
	if lhs.Op != token.LSS || rhs.Op != token.GTR {
		return false
	}
	if !astequal.Expr(lhs.X, rhs.X) {
		return false
	}
	a := c.ctx.TypesInfo.Types[lhs.Y].Value
	b := c.ctx.TypesInfo.Types[rhs.Y].Value
	return a != nil && b != nil && constant.Compare(a, token.LSS, b)
}

func (c *badCondChecker) checkForStmt(stmt *ast.ForStmt) {
	// TODO(quasilyte): handle other kinds of bad conditionals.

	init := astcast.ToAssignStmt(stmt.Init)
	if init.Tok != token.DEFINE || len(init.Lhs) != 1 || len(init.Rhs) != 1 {
		return
	}
	if astcast.ToBasicLit(init.Rhs[0]).Value != "0" {
		return
	}

	iter := astcast.ToIdent(init.Lhs[0])
	cond := astcast.ToBinaryExpr(stmt.Cond)
	if cond.Op != token.GTR || !astequal.Expr(iter, cond.X) {
		return
	}
	if !typep.SideEffectFree(c.ctx.TypesInfo, cond.Y) {
		return
	}

	post := astcast.ToIncDecStmt(stmt.Post)
	if post.Tok != token.INC || !astequal.Expr(iter, post.X) {
		return
	}

	mutated := lintutil.CouldBeMutated(c.ctx.TypesInfo, stmt.Body, cond.Y) ||
		lintutil.CouldBeMutated(c.ctx.TypesInfo, stmt.Body, iter)
	if mutated {
		return
	}

	c.warnForStmt(stmt, cond)
}

func (c *badCondChecker) warnForStmt(cause ast.Node, cond *ast.BinaryExpr) {
	suggest := astcopy.BinaryExpr(cond)
	suggest.Op = token.LSS
	c.ctx.Warn(cause, "`%s` in loop; probably meant `%s`?",
		cond, suggest)
}

func (c *badCondChecker) warnCond(cond *ast.BinaryExpr, tag string) {
	c.ctx.Warn(cond, "`%s` condition is %s", cond, tag)
}

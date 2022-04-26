package checkers

import (
	"go/ast"
	"go/token"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/checkers/internal/lintutil"
	"github.com/go-critic/go-critic/framework/linter"
	"github.com/go-toolsmith/astcast"
	"github.com/go-toolsmith/astequal"
	"github.com/go-toolsmith/astp"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "typeAssertChain"
	info.Tags = []string{"style", "experimental"}
	info.Summary = "Detects repeated type assertions and suggests to replace them with type switch statement"
	info.Before = `
if x, ok := v.(T1); ok {
	// Code A, uses x.
} else if x, ok := v.(T2); ok {
	// Code B, uses x.
} else if x, ok := v.(T3); ok {
	// Code C, uses x.
}`
	info.After = `
switch x := v.(T1) {
case cond1:
	// Code A, uses x.
case cond2:
	// Code B, uses x.
default:
	// Code C, uses x.
}`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		return astwalk.WalkerForStmt(&typeAssertChainChecker{ctx: ctx}), nil
	})
}

type typeAssertChainChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext

	cause   *ast.IfStmt
	visited map[*ast.IfStmt]bool
	typeSet lintutil.AstSet
}

func (c *typeAssertChainChecker) EnterFunc(fn *ast.FuncDecl) bool {
	if fn.Body == nil {
		return false
	}
	c.visited = make(map[*ast.IfStmt]bool)
	return true
}

func (c *typeAssertChainChecker) VisitStmt(stmt ast.Stmt) {
	ifstmt, ok := stmt.(*ast.IfStmt)
	if !ok || c.visited[ifstmt] || ifstmt.Init == nil {
		return
	}
	assertion := c.getTypeAssert(ifstmt)
	if assertion == nil {
		return
	}
	c.cause = ifstmt
	c.checkIfStmt(ifstmt, assertion)
}

func (c *typeAssertChainChecker) getTypeAssert(ifstmt *ast.IfStmt) *ast.TypeAssertExpr {
	assign := astcast.ToAssignStmt(ifstmt.Init)
	if len(assign.Lhs) != 2 || len(assign.Rhs) != 1 {
		return nil
	}
	if !astp.IsIdent(assign.Lhs[0]) || assign.Tok != token.DEFINE {
		return nil
	}
	if !astequal.Expr(assign.Lhs[1], ifstmt.Cond) {
		return nil
	}

	assertion, ok := assign.Rhs[0].(*ast.TypeAssertExpr)
	if !ok {
		return nil
	}
	return assertion
}

func (c *typeAssertChainChecker) checkIfStmt(stmt *ast.IfStmt, assertion *ast.TypeAssertExpr) {
	if c.countTypeAssertions(stmt, assertion) >= 2 {
		c.warn()
	}
}

func (c *typeAssertChainChecker) countTypeAssertions(stmt *ast.IfStmt, assertion *ast.TypeAssertExpr) int {
	c.typeSet.Clear()

	count := 1
	x := assertion.X
	c.typeSet.Insert(assertion.Type)
	for {
		e, ok := stmt.Else.(*ast.IfStmt)
		if !ok {
			return count
		}
		assertion = c.getTypeAssert(e)
		if assertion == nil {
			return count
		}
		if !c.typeSet.Insert(assertion.Type) {
			// Asserted type is duplicated.
			// Type switch does not permit duplicate cases,
			// so give up.
			return 0
		}
		if !astequal.Expr(x, assertion.X) {
			// Mixed type asserting chain.
			// Can't be easily translated to a type switch.
			return 0
		}
		stmt = e
		count++
		c.visited[e] = true
	}
}

func (c *typeAssertChainChecker) warn() {
	c.ctx.Warn(c.cause, "rewrite if-else to type switch statement")
}

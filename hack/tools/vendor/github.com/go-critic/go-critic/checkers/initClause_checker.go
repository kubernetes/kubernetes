package checkers

import (
	"go/ast"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/framework/linter"
	"github.com/go-toolsmith/astp"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "initClause"
	info.Tags = []string{"style", "opinionated", "experimental"}
	info.Summary = "Detects non-assignment statements inside if/switch init clause"
	info.Before = `if sideEffect(); cond {
}`
	info.After = `sideEffect()
if cond {
}`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		return astwalk.WalkerForStmt(&initClauseChecker{ctx: ctx}), nil
	})
}

type initClauseChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext
}

func (c *initClauseChecker) VisitStmt(stmt ast.Stmt) {
	initClause := c.getInitClause(stmt)
	if initClause != nil && !astp.IsAssignStmt(initClause) {
		c.warn(stmt, initClause)
	}
}

func (c *initClauseChecker) getInitClause(x ast.Stmt) ast.Stmt {
	switch x := x.(type) {
	case *ast.IfStmt:
		return x.Init
	case *ast.SwitchStmt:
		return x.Init
	default:
		return nil
	}
}

func (c *initClauseChecker) warn(stmt, clause ast.Stmt) {
	name := "if"
	if astp.IsSwitchStmt(stmt) {
		name = "switch"
	}
	c.ctx.Warn(stmt, "consider to move `%s` before %s", clause, name)
}

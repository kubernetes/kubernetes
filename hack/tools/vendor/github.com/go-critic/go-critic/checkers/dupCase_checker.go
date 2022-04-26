package checkers

import (
	"go/ast"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/checkers/internal/lintutil"
	"github.com/go-critic/go-critic/framework/linter"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "dupCase"
	info.Tags = []string{"diagnostic"}
	info.Summary = "Detects duplicated case clauses inside switch or select statements"
	info.Before = `
switch x {
case ys[0], ys[1], ys[2], ys[0], ys[4]:
}`
	info.After = `
switch x {
case ys[0], ys[1], ys[2], ys[3], ys[4]:
}`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		return astwalk.WalkerForStmt(&dupCaseChecker{ctx: ctx}), nil
	})
}

type dupCaseChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext

	astSet lintutil.AstSet
}

func (c *dupCaseChecker) VisitStmt(stmt ast.Stmt) {
	switch stmt := stmt.(type) {
	case *ast.SwitchStmt:
		c.checkSwitch(stmt)
	case *ast.SelectStmt:
		c.checkSelect(stmt)
	}
}

func (c *dupCaseChecker) checkSwitch(stmt *ast.SwitchStmt) {
	c.astSet.Clear()
	for i := range stmt.Body.List {
		cc := stmt.Body.List[i].(*ast.CaseClause)
		for _, x := range cc.List {
			if !c.astSet.Insert(x) {
				c.warn(x)
			}
		}
	}
}

func (c *dupCaseChecker) checkSelect(stmt *ast.SelectStmt) {
	c.astSet.Clear()
	for i := range stmt.Body.List {
		x := stmt.Body.List[i].(*ast.CommClause).Comm
		if !c.astSet.Insert(x) {
			c.warn(x)
		}
	}
}

func (c *dupCaseChecker) warn(cause ast.Node) {
	c.ctx.Warn(cause, "'case %s' is duplicated", cause)
}

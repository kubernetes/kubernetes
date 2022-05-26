package checkers

import (
	"go/ast"
	"go/token"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/framework/linter"
	"golang.org/x/tools/go/ast/astutil"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "singleCaseSwitch"
	info.Tags = []string{"style"}
	info.Summary = "Detects switch statements that could be better written as if statement"
	info.Before = `
switch x := x.(type) {
case int:
	body()
}`
	info.After = `
if x, ok := x.(int); ok {
	body()
}`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		return astwalk.WalkerForStmt(&singleCaseSwitchChecker{ctx: ctx}), nil
	})
}

type singleCaseSwitchChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext
}

func (c *singleCaseSwitchChecker) VisitStmt(stmt ast.Stmt) {
	switch stmt := stmt.(type) {
	case *ast.SwitchStmt:
		c.checkSwitchStmt(stmt, stmt.Body)
	case *ast.TypeSwitchStmt:
		c.checkSwitchStmt(stmt, stmt.Body)
	}
}

func (c *singleCaseSwitchChecker) checkSwitchStmt(stmt ast.Stmt, body *ast.BlockStmt) {
	if len(body.List) != 1 {
		return
	}
	cc := body.List[0].(*ast.CaseClause)
	if c.hasBreak(cc) {
		return
	}
	switch {
	case cc.List == nil:
		c.warnDefault(stmt)
	case len(cc.List) == 1:
		c.warn(stmt)
	}
}

func (c *singleCaseSwitchChecker) hasBreak(stmt ast.Stmt) bool {
	found := false
	astutil.Apply(stmt, func(cur *astutil.Cursor) bool {
		switch n := cur.Node().(type) {
		case *ast.BranchStmt:
			if n.Tok == token.BREAK {
				found = true
			}
		case *ast.ForStmt, *ast.RangeStmt, *ast.SelectStmt, *ast.SwitchStmt:
			return false
		}
		return true
	}, nil)
	return found
}

func (c *singleCaseSwitchChecker) warn(stmt ast.Stmt) {
	c.ctx.Warn(stmt, "should rewrite switch statement to if statement")
}

func (c *singleCaseSwitchChecker) warnDefault(stmt ast.Stmt) {
	c.ctx.Warn(stmt, "found switch with default case only")
}

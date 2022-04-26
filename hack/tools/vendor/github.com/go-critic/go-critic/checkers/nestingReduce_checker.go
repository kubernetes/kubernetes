package checkers

import (
	"go/ast"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/framework/linter"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "nestingReduce"
	info.Tags = []string{"style", "opinionated", "experimental"}
	info.Params = linter.CheckerParams{
		"bodyWidth": {
			Value: 5,
			Usage: "min number of statements inside a branch to trigger a warning",
		},
	}
	info.Summary = "Finds where nesting level could be reduced"
	info.Before = `
for _, v := range a {
	if v.Bool {
		body()
	}
}`
	info.After = `
for _, v := range a {
	if !v.Bool {
		continue
	}
	body()
}`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		c := &nestingReduceChecker{ctx: ctx}
		c.bodyWidth = info.Params.Int("bodyWidth")
		return astwalk.WalkerForStmt(c), nil
	})
}

type nestingReduceChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext

	bodyWidth int
}

func (c *nestingReduceChecker) VisitStmt(stmt ast.Stmt) {
	switch stmt := stmt.(type) {
	case *ast.ForStmt:
		c.checkLoopBody(stmt.Body.List)
	case *ast.RangeStmt:
		c.checkLoopBody(stmt.Body.List)
	}
}

func (c *nestingReduceChecker) checkLoopBody(body []ast.Stmt) {
	if len(body) != 1 {
		return
	}
	stmt, ok := body[0].(*ast.IfStmt)
	if !ok {
		return
	}
	if len(stmt.Body.List) >= c.bodyWidth && stmt.Else == nil {
		c.warnLoop(stmt)
	}
}

func (c *nestingReduceChecker) warnLoop(cause ast.Node) {
	c.ctx.Warn(cause, "invert if cond, replace body with `continue`, move old body after the statement")
}

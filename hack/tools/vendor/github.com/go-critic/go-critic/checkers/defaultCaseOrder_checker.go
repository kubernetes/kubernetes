package checkers

import (
	"go/ast"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/framework/linter"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "defaultCaseOrder"
	info.Tags = []string{"style"}
	info.Summary = "Detects when default case in switch isn't on 1st or last position"
	info.Before = `
switch {
case x > y:
	// ...
default: // <- not the best position
	// ...
case x == 10:
	// ...
}`
	info.After = `
switch {
case x > y:
	// ...
case x == 10:
	// ...
default: // <- last case (could also be the first one)
	// ...
}`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		return astwalk.WalkerForStmt(&defaultCaseOrderChecker{ctx: ctx}), nil
	})
}

type defaultCaseOrderChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext
}

func (c *defaultCaseOrderChecker) VisitStmt(stmt ast.Stmt) {
	swtch, ok := stmt.(*ast.SwitchStmt)
	if !ok {
		return
	}
	for i, stmt := range swtch.Body.List {
		caseStmt, ok := stmt.(*ast.CaseClause)
		if !ok {
			continue
		}
		// is `default` case
		if caseStmt.List == nil {
			if i != 0 && i != len(swtch.Body.List)-1 {
				c.warn(caseStmt)
			}
		}
	}
}

func (c *defaultCaseOrderChecker) warn(cause *ast.CaseClause) {
	c.ctx.Warn(cause, "consider to make `default` case as first or as last case")
}

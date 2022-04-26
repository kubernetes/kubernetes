package checkers

import (
	"go/ast"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/framework/linter"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "ifElseChain"
	info.Tags = []string{"style"}
	info.Summary = "Detects repeated if-else statements and suggests to replace them with switch statement"
	info.Before = `
if cond1 {
	// Code A.
} else if cond2 {
	// Code B.
} else {
	// Code C.
}`
	info.After = `
switch {
case cond1:
	// Code A.
case cond2:
	// Code B.
default:
	// Code C.
}`
	info.Note = `
Permits single else or else-if; repeated else-if or else + else-if
will trigger suggestion to use switch statement.
See [EffectiveGo#switch](https://golang.org/doc/effective_go.html#switch).`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		return astwalk.WalkerForStmt(&ifElseChainChecker{ctx: ctx}), nil
	})
}

type ifElseChainChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext

	cause   *ast.IfStmt
	visited map[*ast.IfStmt]bool
}

func (c *ifElseChainChecker) EnterFunc(fn *ast.FuncDecl) bool {
	if fn.Body == nil {
		return false
	}
	c.visited = make(map[*ast.IfStmt]bool)
	return true
}

func (c *ifElseChainChecker) VisitStmt(stmt ast.Stmt) {
	if stmt, ok := stmt.(*ast.IfStmt); ok {
		if c.visited[stmt] {
			return
		}
		c.cause = stmt
		c.checkIfStmt(stmt)
	}
}

func (c *ifElseChainChecker) checkIfStmt(stmt *ast.IfStmt) {
	const minThreshold = 2
	if c.countIfelseLen(stmt) >= minThreshold {
		c.warn()
	}
}

func (c *ifElseChainChecker) countIfelseLen(stmt *ast.IfStmt) int {
	count := 0
	for {
		switch e := stmt.Else.(type) {
		case *ast.IfStmt:
			if e.Init != nil {
				return 0 // Give up
			}
			// Else if.
			stmt = e
			count++
			c.visited[e] = true
		case *ast.BlockStmt:
			// Else branch.
			return count + 1
		default:
			// No else or else if.
			return count
		}
	}
}

func (c *ifElseChainChecker) warn() {
	c.ctx.Warn(c.cause, "rewrite if-else to switch statement")
}

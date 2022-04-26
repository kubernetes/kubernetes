package checkers

import (
	"go/ast"
	"go/token"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/framework/linter"
	"github.com/go-toolsmith/astp"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "unnecessaryBlock"
	info.Tags = []string{"style", "opinionated", "experimental"}
	info.Summary = "Detects unnecessary braced statement blocks"
	info.Before = `
x := 1
{
	print(x)
}`
	info.After = `
x := 1
print(x)`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		return astwalk.WalkerForStmtList(&unnecessaryBlockChecker{ctx: ctx}), nil
	})
}

type unnecessaryBlockChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext
}

func (c *unnecessaryBlockChecker) VisitStmtList(x ast.Node, statements []ast.Stmt) {
	// Using StmtListVisitor instead of StmtVisitor makes it easier to avoid
	// false positives on IfStmt, RangeStmt, ForStmt and alike.
	// We only inspect BlockStmt inside statement lists, so this method is not
	// called for IfStmt itself, for example.

	if (astp.IsCaseClause(x) || astp.IsCommClause(x)) && len(statements) == 1 {
		if _, ok := statements[0].(*ast.BlockStmt); ok {
			c.ctx.Warn(statements[0], "case statement doesn't require a block statement")
			return
		}
	}

	for _, stmt := range statements {
		stmt, ok := stmt.(*ast.BlockStmt)
		if ok && !c.hasDefinitions(stmt) {
			c.warn(stmt)
		}
	}
}

func (c *unnecessaryBlockChecker) hasDefinitions(stmt *ast.BlockStmt) bool {
	for _, bs := range stmt.List {
		switch stmt := bs.(type) {
		case *ast.AssignStmt:
			if stmt.Tok == token.DEFINE {
				return true
			}
		case *ast.DeclStmt:
			decl := stmt.Decl.(*ast.GenDecl)
			if len(decl.Specs) != 0 {
				return true
			}
		}
	}

	return false
}

func (c *unnecessaryBlockChecker) warn(expr ast.Stmt) {
	c.ctx.Warn(expr, "block doesn't have definitions, can be simply deleted")
}

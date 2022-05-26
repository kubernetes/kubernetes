package checkers

import (
	"go/ast"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/framework/linter"
	"github.com/go-toolsmith/astfmt"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "unnecessaryDefer"
	info.Tags = []string{"diagnostic", "experimental"}
	info.Summary = "Detects redundantly deferred calls"
	info.Before = `
func() {
	defer os.Remove(filename)
}`
	info.After = `
func() {
	os.Remove(filename)
}`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		return astwalk.WalkerForFuncDecl(&unnecessaryDeferChecker{ctx: ctx}), nil
	})
}

type unnecessaryDeferChecker struct {
	astwalk.WalkHandler
	ctx    *linter.CheckerContext
	isFunc bool
}

// Visit implements the ast.Visitor. This visitor keeps track of the block
// statement belongs to a function or any other block. If the block is not a
// function and ends with a defer statement that should be OK since it's
// defering the outer function.
func (c *unnecessaryDeferChecker) Visit(node ast.Node) ast.Visitor {
	switch n := node.(type) {
	case *ast.FuncDecl, *ast.FuncLit:
		c.isFunc = true
	case *ast.BlockStmt:
		c.checkDeferBeforeReturn(n)
	default:
		c.isFunc = false
	}

	return c
}

func (c *unnecessaryDeferChecker) VisitFuncDecl(funcDecl *ast.FuncDecl) {
	// We always start as a function (*ast.FuncDecl.Body passed)
	c.isFunc = true

	ast.Walk(c, funcDecl.Body)
}

func (c *unnecessaryDeferChecker) checkDeferBeforeReturn(funcDecl *ast.BlockStmt) {
	// Check if we have an explicit return or if it's just the end of the scope.
	explicitReturn := false
	retIndex := len(funcDecl.List)
	for i, stmt := range funcDecl.List {
		retStmt, ok := stmt.(*ast.ReturnStmt)
		if !ok {
			continue
		}
		explicitReturn = true
		if !c.isTrivialReturn(retStmt) {
			continue
		}
		retIndex = i
		break
	}
	if retIndex == 0 {
		return
	}

	if deferStmt, ok := funcDecl.List[retIndex-1].(*ast.DeferStmt); ok {
		// If the block is a function and ending with return or if we have an
		// explicit return in any other block we should warn about
		// unnecessary defer.
		if c.isFunc || explicitReturn {
			c.warn(deferStmt)
		}
	}
}

func (c *unnecessaryDeferChecker) isTrivialReturn(ret *ast.ReturnStmt) bool {
	for _, e := range ret.Results {
		if !c.isConstExpr(e) {
			return false
		}
	}
	return true
}

func (c *unnecessaryDeferChecker) isConstExpr(e ast.Expr) bool {
	return c.ctx.TypesInfo.Types[e].Value != nil
}

func (c *unnecessaryDeferChecker) warn(deferStmt *ast.DeferStmt) {
	s := astfmt.Sprint(deferStmt)
	if fnlit, ok := deferStmt.Call.Fun.(*ast.FuncLit); ok {
		// To avoid long and multi-line warning messages,
		// collapse the function literals.
		s = "defer " + astfmt.Sprint(fnlit.Type) + "{...}(...)"
	}
	c.ctx.Warn(deferStmt, "%s is placed just before return", s)
}

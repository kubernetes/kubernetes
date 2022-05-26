package checkers

import (
	"go/ast"
	"go/token"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/checkers/internal/lintutil"
	"github.com/go-critic/go-critic/framework/linter"
	"github.com/go-toolsmith/astcast"
	"github.com/go-toolsmith/astequal"
	"github.com/go-toolsmith/typep"
	"golang.org/x/tools/go/ast/astutil"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "sortSlice"
	info.Tags = []string{"diagnostic", "experimental"}
	info.Summary = "Detects suspicious sort.Slice calls"
	info.Before = `sort.Slice(xs, func(i, j) bool { return keys[i] < keys[j] })`
	info.After = `sort.Slice(kv, func(i, j) bool { return kv[i].key < kv[j].key })`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		return astwalk.WalkerForExpr(&sortSliceChecker{ctx: ctx}), nil
	})
}

type sortSliceChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext
}

func (c *sortSliceChecker) VisitExpr(expr ast.Expr) {
	call := astcast.ToCallExpr(expr)
	if len(call.Args) != 2 {
		return
	}
	switch qualifiedName(call.Fun) {
	case "sort.Slice", "sort.SliceStable":
		// OK.
	default:
		return
	}

	slice := c.unwrapSlice(call.Args[0])
	lessFunc, ok := call.Args[1].(*ast.FuncLit)
	if !ok {
		return
	}
	if !typep.SideEffectFree(c.ctx.TypesInfo, slice) {
		return // Don't check unpredictable slice values
	}

	ivar, jvar := c.paramIdents(lessFunc.Type)
	if ivar == nil || jvar == nil {
		return
	}

	if len(lessFunc.Body.List) != 1 {
		return
	}
	ret, ok := lessFunc.Body.List[0].(*ast.ReturnStmt)
	if !ok {
		return
	}
	cmp := astcast.ToBinaryExpr(astutil.Unparen(ret.Results[0]))
	if !typep.SideEffectFree(c.ctx.TypesInfo, cmp) {
		return
	}
	switch cmp.Op {
	case token.LSS, token.LEQ, token.GTR, token.GEQ:
		// Both cmp.X and cmp.Y are expected to be some expressions
		// over the `slice` expression. In the simplest case,
		// it's a `slice[i] <op> slice[j]`.
		if !c.containsSlice(cmp.X, slice) && !c.containsSlice(cmp.Y, slice) {
			c.warnSlice(cmp, slice)
		}

		// This one is more about the style, but can reveal potential issue
		// or misprint in sorting condition.
		// We give a warn if X contains indexing with `i` index and Y
		// contains indexing with `j`.
		if c.containsIndex(cmp.X, jvar) && c.containsIndex(cmp.Y, ivar) {
			c.warnIndex(cmp, ivar, jvar)
		}
	}
}

func (c *sortSliceChecker) paramIdents(e *ast.FuncType) (ivar, jvar *ast.Ident) {
	// Covers both `i, j int` and `i int, j int`.
	idents := make([]*ast.Ident, 0, 2)
	for _, field := range e.Params.List {
		idents = append(idents, field.Names...)
	}
	if len(idents) == 2 {
		return idents[0], idents[1]
	}
	return nil, nil
}

func (c *sortSliceChecker) unwrapSlice(e ast.Expr) ast.Expr {
	switch e := e.(type) {
	case *ast.ParenExpr:
		return c.unwrapSlice(e.X)
	case *ast.SliceExpr:
		return e.X
	default:
		return e
	}
}

func (c *sortSliceChecker) containsIndex(e, index ast.Expr) bool {
	return lintutil.ContainsNode(e, func(n ast.Node) bool {
		indexing, ok := n.(*ast.IndexExpr)
		if !ok {
			return false
		}
		return astequal.Expr(indexing.Index, index)
	})
}

func (c *sortSliceChecker) containsSlice(e, slice ast.Expr) bool {
	return lintutil.ContainsNode(e, func(n ast.Node) bool {
		return astequal.Node(n, slice)
	})
}

func (c *sortSliceChecker) warnSlice(cause ast.Node, slice ast.Expr) {
	c.ctx.Warn(cause, "cmp func must use %s slice in comparison", slice)
}

func (c *sortSliceChecker) warnIndex(cause ast.Node, ivar, jvar *ast.Ident) {
	c.ctx.Warn(cause, "unusual order of {%s,%s} params in comparison", ivar, jvar)
}

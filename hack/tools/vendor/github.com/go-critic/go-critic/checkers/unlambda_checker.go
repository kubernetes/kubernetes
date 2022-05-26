package checkers

import (
	"go/ast"
	"go/token"
	"go/types"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/checkers/internal/lintutil"
	"github.com/go-critic/go-critic/framework/linter"
	"github.com/go-toolsmith/astcast"
	"github.com/go-toolsmith/astequal"
	"github.com/go-toolsmith/typep"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "unlambda"
	info.Tags = []string{"style"}
	info.Summary = "Detects function literals that can be simplified"
	info.Before = `func(x int) int { return fn(x) }`
	info.After = `fn`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		return astwalk.WalkerForExpr(&unlambdaChecker{ctx: ctx}), nil
	})
}

type unlambdaChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext
}

func (c *unlambdaChecker) VisitExpr(x ast.Expr) {
	fn, ok := x.(*ast.FuncLit)
	if !ok || len(fn.Body.List) != 1 {
		return
	}

	ret, ok := fn.Body.List[0].(*ast.ReturnStmt)
	if !ok || len(ret.Results) != 1 {
		return
	}

	result := astcast.ToCallExpr(ret.Results[0])
	callable := qualifiedName(result.Fun)
	if callable == "" {
		return // Skip tricky cases; only handle simple calls
	}
	if isBuiltin(callable) {
		return // See #762
	}
	hasVars := lintutil.ContainsNode(result.Fun, func(n ast.Node) bool {
		id, ok := n.(*ast.Ident)
		if !ok {
			return false
		}
		obj, ok := c.ctx.TypesInfo.ObjectOf(id).(*types.Var)
		if !ok {
			return false
		}
		// Permit only non-pointer struct method values.
		return !typep.IsStruct(obj.Type().Underlying())
	})
	if hasVars {
		return // See #888 #1007
	}

	fnType := c.ctx.TypeOf(fn)
	resultType := c.ctx.TypeOf(result.Fun)
	if !types.Identical(fnType, resultType) {
		return
	}
	// Now check that all arguments match the parameters.
	n := 0
	for _, params := range fn.Type.Params.List {
		if _, ok := params.Type.(*ast.Ellipsis); ok {
			if result.Ellipsis == token.NoPos {
				return
			}
			n++
			continue
		}

		for _, id := range params.Names {
			if !astequal.Expr(id, result.Args[n]) {
				return
			}
			n++
		}
	}

	if len(result.Args) == n {
		c.warn(fn, callable)
	}
}

func (c *unlambdaChecker) warn(cause ast.Node, suggestion string) {
	c.ctx.Warn(cause, "replace `%s` with `%s`", cause, suggestion)
}

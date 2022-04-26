package checkers

import (
	"go/ast"
	"go/types"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/framework/linter"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "ptrToRefParam"
	info.Tags = []string{"style", "opinionated", "experimental"}
	info.Summary = "Detects input and output parameters that have a type of pointer to referential type"
	info.Before = `func f(m *map[string]int) (*chan *int)`
	info.After = `func f(m map[string]int) (chan *int)`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		return astwalk.WalkerForFuncDecl(&ptrToRefParamChecker{ctx: ctx}), nil
	})
}

type ptrToRefParamChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext
}

func (c *ptrToRefParamChecker) VisitFuncDecl(fn *ast.FuncDecl) {
	c.checkParams(fn.Type.Params.List)
	if fn.Type.Results != nil {
		c.checkParams(fn.Type.Results.List)
	}
}

func (c *ptrToRefParamChecker) checkParams(params []*ast.Field) {
	for _, param := range params {
		ptr, ok := c.ctx.TypeOf(param.Type).(*types.Pointer)
		if !ok {
			continue
		}

		if c.isRefType(ptr.Elem()) {
			if len(param.Names) == 0 {
				c.ctx.Warn(param, "consider to make non-pointer type for `%s`", param.Type)
			} else {
				for i := range param.Names {
					c.warn(param.Names[i])
				}
			}
		}
	}
}

func (c *ptrToRefParamChecker) isRefType(x types.Type) bool {
	switch typ := x.(type) {
	case *types.Map, *types.Chan, *types.Interface:
		return true
	case *types.Named:
		// Handle underlying type only for interfaces.
		if _, ok := typ.Underlying().(*types.Interface); ok {
			return true
		}
	}
	return false
}

func (c *ptrToRefParamChecker) warn(id *ast.Ident) {
	c.ctx.Warn(id, "consider `%s' to be of non-pointer type", id)
}

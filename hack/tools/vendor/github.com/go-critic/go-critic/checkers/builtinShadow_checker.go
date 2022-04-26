package checkers

import (
	"go/ast"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/framework/linter"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "builtinShadow"
	info.Tags = []string{"style", "opinionated"}
	info.Summary = "Detects when predeclared identifiers are shadowed in assignments"
	info.Before = `len := 10`
	info.After = `length := 10`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		return astwalk.WalkerForLocalDef(&builtinShadowChecker{ctx: ctx}, ctx.TypesInfo), nil
	})
}

type builtinShadowChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext
}

func (c *builtinShadowChecker) VisitLocalDef(name astwalk.Name, _ ast.Expr) {
	if isBuiltin(name.ID.Name) {
		c.warn(name.ID)
	}
}

func (c *builtinShadowChecker) warn(ident *ast.Ident) {
	c.ctx.Warn(ident, "shadowing of predeclared identifier: %s", ident)
}

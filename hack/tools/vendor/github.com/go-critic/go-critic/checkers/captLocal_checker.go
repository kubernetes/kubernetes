package checkers

import (
	"go/ast"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/framework/linter"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "captLocal"
	info.Tags = []string{"style"}
	info.Params = linter.CheckerParams{
		"paramsOnly": {
			Value: true,
			Usage: "whether to restrict checker to params only",
		},
	}
	info.Summary = "Detects capitalized names for local variables"
	info.Before = `func f(IN int, OUT *int) (ERR error) {}`
	info.After = `func f(in int, out *int) (err error) {}`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		c := &captLocalChecker{ctx: ctx}
		c.paramsOnly = info.Params.Bool("paramsOnly")
		return astwalk.WalkerForLocalDef(c, ctx.TypesInfo), nil
	})
}

type captLocalChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext

	paramsOnly bool
}

func (c *captLocalChecker) VisitLocalDef(def astwalk.Name, _ ast.Expr) {
	if c.paramsOnly && def.Kind != astwalk.NameParam {
		return
	}
	if ast.IsExported(def.ID.Name) {
		c.warn(def.ID)
	}
}

func (c *captLocalChecker) warn(id ast.Node) {
	c.ctx.Warn(id, "`%s' should not be capitalized", id)
}

package checkers

import (
	"go/ast"
	"go/types"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/framework/linter"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "tooManyResultsChecker"
	info.Tags = []string{"style", "opinionated", "experimental"}
	info.Params = linter.CheckerParams{
		"maxResults": {
			Value: 5,
			Usage: "maximum number of results",
		},
	}
	info.Summary = "Detects function with too many results"
	info.Before = `func fn() (a, b, c, d float32, _ int, _ bool)`
	info.After = `func fn() (resultStruct, bool)`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		c := astwalk.WalkerForFuncDecl(&tooManyResultsChecker{
			ctx:       ctx,
			maxParams: info.Params.Int("maxResults"),
		})
		return c, nil
	})
}

type tooManyResultsChecker struct {
	astwalk.WalkHandler
	ctx       *linter.CheckerContext
	maxParams int
}

func (c *tooManyResultsChecker) VisitFuncDecl(decl *ast.FuncDecl) {
	typ := c.ctx.TypeOf(decl.Name)
	sig, ok := typ.(*types.Signature)
	if !ok {
		return
	}

	if count := sig.Results().Len(); count > c.maxParams {
		c.warn(decl)
	}
}

func (c *tooManyResultsChecker) warn(n ast.Node) {
	c.ctx.Warn(n, "function has more than %d results, consider to simplify the function", c.maxParams)
}

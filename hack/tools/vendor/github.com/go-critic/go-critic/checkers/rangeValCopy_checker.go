package checkers

import (
	"go/ast"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/framework/linter"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "rangeValCopy"
	info.Tags = []string{"performance"}
	info.Params = linter.CheckerParams{
		"sizeThreshold": {
			Value: 128,
			Usage: "size in bytes that makes the warning trigger",
		},
		"skipTestFuncs": {
			Value: true,
			Usage: "whether to check test functions",
		},
	}
	info.Summary = "Detects loops that copy big objects during each iteration"
	info.Details = "Suggests to use index access or take address and make use pointer instead."
	info.Before = `
xs := make([][1024]byte, length)
for _, x := range xs {
	// Loop body.
}`
	info.After = `
xs := make([][1024]byte, length)
for i := range xs {
	x := &xs[i]
	// Loop body.
}`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		c := &rangeValCopyChecker{ctx: ctx}
		c.sizeThreshold = int64(info.Params.Int("sizeThreshold"))
		c.skipTestFuncs = info.Params.Bool("skipTestFuncs")
		return astwalk.WalkerForStmt(c), nil
	})
}

type rangeValCopyChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext

	sizeThreshold int64
	skipTestFuncs bool
}

func (c *rangeValCopyChecker) EnterFunc(fn *ast.FuncDecl) bool {
	return fn.Body != nil &&
		!(c.skipTestFuncs && isUnitTestFunc(c.ctx, fn))
}

func (c *rangeValCopyChecker) VisitStmt(stmt ast.Stmt) {
	rng, ok := stmt.(*ast.RangeStmt)
	if !ok || rng.Value == nil {
		return
	}
	typ := c.ctx.TypeOf(rng.Value)
	if typ == nil {
		return
	}
	if size := c.ctx.SizesInfo.Sizeof(typ); size >= c.sizeThreshold {
		c.warn(rng, size)
	}
}

func (c *rangeValCopyChecker) warn(n ast.Node, size int64) {
	c.ctx.Warn(n, "each iteration copies %d bytes (consider pointers or indexing)", size)
}

package checkers

import (
	"go/ast"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/framework/linter"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "deferInLoop"
	info.Tags = []string{"diagnostic", "experimental"}
	info.Summary = "Detects loops inside functions that use defer"
	info.Before = `
for _, filename := range []string{"foo", "bar"} {
	 f, err := os.Open(filename)
	
	defer f.Close()
}
`
	info.After = `
func process(filename string) {
	 f, err := os.Open(filename)
	
	defer f.Close()
}
/* ... */
for _, filename := range []string{"foo", "bar"} {
	process(filename)
}`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		return astwalk.WalkerForFuncDecl(&deferInLoopChecker{ctx: ctx}), nil
	})
}

type deferInLoopChecker struct {
	astwalk.WalkHandler
	ctx   *linter.CheckerContext
	inFor bool
}

func (c *deferInLoopChecker) VisitFuncDecl(fn *ast.FuncDecl) {
	ast.Inspect(fn.Body, c.traversalFunc)
}

func (c deferInLoopChecker) traversalFunc(cur ast.Node) bool {
	switch n := cur.(type) {
	case *ast.DeferStmt:
		if c.inFor {
			c.warn(n)
		}
	case *ast.RangeStmt, *ast.ForStmt:
		if !c.inFor {
			ast.Inspect(cur, deferInLoopChecker{ctx: c.ctx, inFor: true}.traversalFunc)
			return false
		}
	case *ast.FuncLit:
		ast.Inspect(n.Body, deferInLoopChecker{ctx: c.ctx, inFor: false}.traversalFunc)
		return false
	case nil:
		return false
	}
	return true
}

func (c *deferInLoopChecker) warn(cause *ast.DeferStmt) {
	c.ctx.Warn(cause, "Possible resource leak, 'defer' is called in the 'for' loop")
}

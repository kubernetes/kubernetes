package checkers

import (
	"go/ast"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/framework/linter"
	"github.com/go-toolsmith/astfmt"
	"github.com/go-toolsmith/astp"
	"golang.org/x/tools/go/ast/astutil"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "exitAfterDefer"
	info.Tags = []string{"diagnostic"}
	info.Summary = "Detects calls to exit/fatal inside functions that use defer"
	info.Before = `
defer os.Remove(filename)
if bad {
	log.Fatalf("something bad happened")
}`
	info.After = `
defer os.Remove(filename)
if bad {
	log.Printf("something bad happened")
	return
}`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		return astwalk.WalkerForFuncDecl(&exitAfterDeferChecker{ctx: ctx}), nil
	})
}

type exitAfterDeferChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext
}

func (c *exitAfterDeferChecker) VisitFuncDecl(fn *ast.FuncDecl) {
	// TODO(quasilyte): handle goto and other kinds of flow that break
	// the algorithm below that expects the latter statement to be
	// executed after the ones that come before it.

	var deferStmt *ast.DeferStmt
	pre := func(cur *astutil.Cursor) bool {
		// Don't recurse into local anonymous functions.
		return !astp.IsFuncLit(cur.Node())
	}
	post := func(cur *astutil.Cursor) bool {
		switch n := cur.Node().(type) {
		case *ast.DeferStmt:
			deferStmt = n
		case *ast.CallExpr:
			// See #995. We allow `defer os.Exit()` calls
			// as it's harder to determine whether they're going
			// to clutter anything without actually trying to
			// simulate the defer stack + understanding the control flow.
			// TODO: can we use CFG here?
			if _, ok := cur.Parent().(*ast.DeferStmt); ok {
				return true
			}
			if deferStmt != nil {
				switch qualifiedName(n.Fun) {
				case "log.Fatal", "log.Fatalf", "log.Fatalln", "os.Exit":
					c.warn(n, deferStmt)
					return false
				}
			}
		}
		return true
	}
	astutil.Apply(fn.Body, pre, post)
}

func (c *exitAfterDeferChecker) warn(cause *ast.CallExpr, deferStmt *ast.DeferStmt) {
	s := astfmt.Sprint(deferStmt)
	if fnlit, ok := deferStmt.Call.Fun.(*ast.FuncLit); ok {
		// To avoid long and multi-line warning messages,
		// collapse the function literals.
		s = "defer " + astfmt.Sprint(fnlit.Type) + "{...}(...)"
	}
	c.ctx.Warn(cause, "%s will exit, and `%s` will not run", cause.Fun, s)
}

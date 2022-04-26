package checkers

import (
	"fmt"
	"go/ast"
	"go/token"
	"regexp"
	"strings"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/framework/linter"
	"github.com/go-toolsmith/strparse"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "commentedOutCode"
	info.Tags = []string{"diagnostic", "experimental"}
	info.Summary = "Detects commented-out code inside function bodies"
	info.Before = `
// fmt.Println("Debugging hard")
foo(1, 2)`
	info.After = `foo(1, 2)`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		return astwalk.WalkerForLocalComment(&commentedOutCodeChecker{
			ctx:              ctx,
			notQuiteFuncCall: regexp.MustCompile(`\w+\s+\([^)]*\)\s*$`),
		}), nil
	})
}

type commentedOutCodeChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext
	fn  *ast.FuncDecl

	notQuiteFuncCall *regexp.Regexp
}

func (c *commentedOutCodeChecker) EnterFunc(fn *ast.FuncDecl) bool {
	c.fn = fn // Need to store current function inside checker context
	return fn.Body != nil
}

func (c *commentedOutCodeChecker) VisitLocalComment(cg *ast.CommentGroup) {
	s := cg.Text() // Collect text once

	// We do multiple heuristics to avoid false positives.
	// Many things can be improved here.

	markers := []string{
		"TODO", // TODO comments with code are permitted.

		// "http://" is interpreted as a label with comment.
		// There are other protocols we might want to include.
		"http://",
		"https://",

		"e.g. ", // Clearly not a "selector expr" (mostly due to extra space)
	}
	for _, m := range markers {
		if strings.Contains(s, m) {
			return
		}
	}

	// Some very short comment that can be skipped.
	// Usually triggering on these results in false positive.
	// Unless there is a very popular call like print/println.
	cond := len(s) < len("quite too short") &&
		!strings.Contains(s, "print") &&
		!strings.Contains(s, "fmt.") &&
		!strings.Contains(s, "log.")
	if cond {
		return
	}

	// Almost looks like a commented-out function call,
	// but there is a whitespace between function name and
	// parameters list. Skip these to avoid false positives.
	if c.notQuiteFuncCall.MatchString(s) {
		return
	}

	stmt := strparse.Stmt(s)

	if c.isPermittedStmt(stmt) {
		return
	}

	if stmt != strparse.BadStmt {
		c.warn(cg)
		return
	}

	// Don't try to parse one-liner as block statement
	if len(cg.List) == 1 && !strings.Contains(s, "\n") {
		return
	}

	// Some attempts to avoid false positives.
	if c.skipBlock(s) {
		return
	}

	// Add braces to make block statement from
	// multiple statements.
	stmt = strparse.Stmt(fmt.Sprintf("{ %s }", s))

	if stmt, ok := stmt.(*ast.BlockStmt); ok && len(stmt.List) != 0 {
		c.warn(cg)
	}
}

func (c *commentedOutCodeChecker) skipBlock(s string) bool {
	lines := strings.Split(s, "\n") // There is at least 1 line, that's invariant

	// Special example test block.
	if isExampleTestFunc(c.fn) && strings.Contains(lines[0], "Output:") {
		return true
	}

	return false
}

func (c *commentedOutCodeChecker) isPermittedStmt(stmt ast.Stmt) bool {
	switch stmt := stmt.(type) {
	case *ast.ExprStmt:
		return c.isPermittedExpr(stmt.X)
	case *ast.LabeledStmt:
		return c.isPermittedStmt(stmt.Stmt)
	case *ast.DeclStmt:
		decl := stmt.Decl.(*ast.GenDecl)
		return decl.Tok == token.TYPE
	default:
		return false
	}
}

func (c *commentedOutCodeChecker) isPermittedExpr(x ast.Expr) bool {
	// Permit anything except expressions that can be used
	// with complete result discarding.
	switch x := x.(type) {
	case *ast.CallExpr:
		return false
	case *ast.UnaryExpr:
		// "<-" channel receive is not permitted.
		return x.Op != token.ARROW
	default:
		return true
	}
}

func (c *commentedOutCodeChecker) warn(cause ast.Node) {
	c.ctx.Warn(cause, "may want to remove commented-out code")
}

package checkers

import (
	"go/ast"
	"go/token"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/checkers/internal/lintutil"
	"github.com/go-critic/go-critic/framework/linter"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "unlabelStmt"
	info.Tags = []string{"style", "experimental"}
	info.Summary = "Detects redundant statement labels"
	info.Before = `
derp:
for x := range xs {
	if x == 0 {
		break derp
	}
}`
	info.After = `
for x := range xs {
	if x == 0 {
		break
	}
}`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		return astwalk.WalkerForStmt(&unlabelStmtChecker{ctx: ctx}), nil
	})
}

type unlabelStmtChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext
}

func (c *unlabelStmtChecker) EnterFunc(fn *ast.FuncDecl) bool {
	if fn.Body == nil {
		return false
	}
	// TODO(quasilyte): should not do additional traversal here.
	// For now, skip all functions that contain goto statement.
	return !lintutil.ContainsNode(fn.Body, func(n ast.Node) bool {
		br, ok := n.(*ast.BranchStmt)
		return ok && br.Tok == token.GOTO
	})
}

func (c *unlabelStmtChecker) VisitStmt(stmt ast.Stmt) {
	labeled, ok := stmt.(*ast.LabeledStmt)
	if !ok || !c.canBreakFrom(labeled.Stmt) {
		return
	}

	// We have a labeled statement from that have labeled continue/break.
	// This is an invariant, since unused label is a compile-time error
	// and we're currently skipping functions containing goto.
	//
	// Also note that Go labels are function-scoped and there
	// can be no re-definitions. This means that we don't
	// need to care about label shadowing or things like that.
	//
	// The task is to find cases where labeled branch (continue/break)
	// is redundant and can be re-written, decreasing the label usages
	// and potentially leading to its redundancy,
	// or finding the redundant labels right away.

	name := labeled.Label.Name

	// Simplest case that can prove that label is redundant.
	//
	// If labeled branch is somewhere inside the statement block itself
	// and none of the nested break'able statements refer to that label,
	// the label can be removed.
	matchUsage := func(n ast.Node) bool {
		return c.canBreakFrom(n) && c.usesLabel(c.blockStmtOf(n), name)
	}
	if !lintutil.ContainsNode(c.blockStmtOf(labeled.Stmt), matchUsage) {
		c.warnRedundant(labeled)
		return
	}

	// Only for loops: if last stmt in list is a loop
	// that contains labeled "continue" to the outer loop label,
	// it can be refactored to use "break" instead.
	// Exceptions: select statements with a labeled "continue" are ignored.
	if c.isLoop(labeled.Stmt) {
		body := c.blockStmtOf(labeled.Stmt)
		if len(body.List) == 0 {
			return
		}
		last := body.List[len(body.List)-1]
		if !c.isLoop(last) {
			return
		}
		br := lintutil.FindNode(c.blockStmtOf(last),
			func(n ast.Node) bool {
				switch n.(type) {
				case *ast.SelectStmt:
					return false
				default:
					return true
				}
			},
			func(n ast.Node) bool {
				br, ok := n.(*ast.BranchStmt)
				return ok && br.Label != nil &&
					br.Label.Name == name && br.Tok == token.CONTINUE
			})

		if br != nil {
			c.warnLabeledContinue(br, name)
		}
	}
}

// isLoop reports whether n is a loop of some kind.
// In other words, it tells whether n body can contain "continue"
// associated with n.
func (c *unlabelStmtChecker) isLoop(n ast.Node) bool {
	switch n.(type) {
	case *ast.ForStmt, *ast.RangeStmt:
		return true
	default:
		return false
	}
}

// canBreakFrom reports whether it is possible to "break" or "continue" from n body.
func (c *unlabelStmtChecker) canBreakFrom(n ast.Node) bool {
	switch n.(type) {
	case *ast.RangeStmt, *ast.ForStmt, *ast.SwitchStmt, *ast.TypeSwitchStmt, *ast.SelectStmt:
		return true
	default:
		return false
	}
}

// blockStmtOf returns body of specified node.
//
// TODO(quasilyte): handle other statements and see if it can be useful
// in other checkers.
func (c *unlabelStmtChecker) blockStmtOf(n ast.Node) *ast.BlockStmt {
	switch n := n.(type) {
	case *ast.RangeStmt:
		return n.Body
	case *ast.ForStmt:
		return n.Body
	case *ast.SwitchStmt:
		return n.Body
	case *ast.TypeSwitchStmt:
		return n.Body
	case *ast.SelectStmt:
		return n.Body

	default:
		return nil
	}
}

// usesLabel reports whether n contains a usage of label.
func (c *unlabelStmtChecker) usesLabel(n *ast.BlockStmt, label string) bool {
	return lintutil.ContainsNode(n, func(n ast.Node) bool {
		branch, ok := n.(*ast.BranchStmt)
		return ok && branch.Label != nil &&
			branch.Label.Name == label &&
			(branch.Tok == token.CONTINUE || branch.Tok == token.BREAK)
	})
}

func (c *unlabelStmtChecker) warnRedundant(cause *ast.LabeledStmt) {
	c.ctx.Warn(cause, "label %s is redundant", cause.Label)
}

func (c *unlabelStmtChecker) warnLabeledContinue(cause ast.Node, label string) {
	c.ctx.Warn(cause, "change `continue %s` to `break`", label)
}

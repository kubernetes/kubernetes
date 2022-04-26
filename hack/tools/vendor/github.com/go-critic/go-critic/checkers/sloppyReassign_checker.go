package checkers

import (
	"go/ast"
	"go/token"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/framework/linter"
	"github.com/go-toolsmith/astcast"
	"github.com/go-toolsmith/astcopy"
	"github.com/go-toolsmith/astequal"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "sloppyReassign"
	info.Tags = []string{"diagnostic", "experimental"}
	info.Summary = "Detects suspicious/confusing re-assignments"
	info.Before = `if err = f(); err != nil { return err }`
	info.After = `if err := f(); err != nil { return err }`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		return astwalk.WalkerForStmt(&sloppyReassignChecker{ctx: ctx}), nil
	})
}

type sloppyReassignChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext
}

func (c *sloppyReassignChecker) VisitStmt(stmt ast.Stmt) {
	// Right now only check assignments in if statements init.
	ifStmt := astcast.ToIfStmt(stmt)
	assign := astcast.ToAssignStmt(ifStmt.Init)
	if assign.Tok != token.ASSIGN {
		return
	}

	// TODO(quasilyte): is handling of multi-value assignments worthwhile?
	if len(assign.Lhs) != 1 || len(assign.Rhs) != 1 {
		return
	}

	// TODO(quasilyte): handle not only the simplest, return-only case.
	body := ifStmt.Body.List
	if len(body) != 1 {
		return
	}

	// Variable that is being re-assigned.
	reAssigned := astcast.ToIdent(assign.Lhs[0])
	if reAssigned.Name == "" {
		return
	}

	// TODO(quasilyte): handle not only nil comparisons.
	eqToNil := &ast.BinaryExpr{
		Op: token.NEQ,
		X:  reAssigned,
		Y:  &ast.Ident{Name: "nil"},
	}
	if !astequal.Expr(ifStmt.Cond, eqToNil) {
		return
	}

	results := astcast.ToReturnStmt(body[0]).Results
	for _, res := range results {
		if astequal.Expr(reAssigned, res) {
			c.warnAssignToDefine(assign, reAssigned.Name)
			break
		}
	}
}

func (c *sloppyReassignChecker) warnAssignToDefine(assign *ast.AssignStmt, name string) {
	suggest := astcopy.AssignStmt(assign)
	suggest.Tok = token.DEFINE
	c.ctx.Warn(assign, "re-assignment to `%s` can be replaced with `%s`", name, suggest)
}

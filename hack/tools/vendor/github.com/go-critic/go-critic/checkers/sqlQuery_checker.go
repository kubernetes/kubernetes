package checkers

import (
	"go/ast"
	"go/types"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/framework/linter"
	"github.com/go-toolsmith/astcast"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "sqlQuery"
	info.Tags = []string{"diagnostic", "experimental"}
	info.Summary = "Detects issue in Query() and Exec() calls"
	info.Before = `_, err := db.Query("UPDATE ...")`
	info.After = `_, err := db.Exec("UPDATE ...")`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		return astwalk.WalkerForStmt(&sqlQueryChecker{ctx: ctx}), nil
	})
}

type sqlQueryChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext
}

func (c *sqlQueryChecker) VisitStmt(stmt ast.Stmt) {
	assign := astcast.ToAssignStmt(stmt)
	if len(assign.Lhs) != 2 { // Query() has 2 return values.
		return
	}
	if len(assign.Rhs) != 1 {
		return
	}

	// If Query() is called, but first return value is ignored,
	// there is no way to close/read the returned rows.
	// This can cause a connection leak.
	if id, ok := assign.Lhs[0].(*ast.Ident); ok && id.Name != "_" {
		return
	}

	call := astcast.ToCallExpr(assign.Rhs[0])
	funcExpr := astcast.ToSelectorExpr(call.Fun)
	if !c.funcIsQuery(funcExpr) {
		return
	}

	if c.typeHasExecMethod(c.ctx.TypeOf(funcExpr.X)) {
		c.warnAndSuggestExec(funcExpr)
	} else {
		c.warnRowsIgnored(funcExpr)
	}
}

func (c *sqlQueryChecker) funcIsQuery(funcExpr *ast.SelectorExpr) bool {
	if funcExpr.Sel == nil {
		return false
	}
	switch funcExpr.Sel.Name {
	case "Query", "QueryContext":
		// Stdlib and friends.
	case "Queryx", "QueryxContext":
		// sqlx.
	default:
		return false
	}

	// To avoid false positives (unrelated types can have Query method)
	// check that the 1st returned type has Row-like name.
	typ, ok := c.ctx.TypeOf(funcExpr).Underlying().(*types.Signature)
	if !ok || typ.Results() == nil || typ.Results().Len() != 2 {
		return false
	}
	if !c.typeIsRowsLike(typ.Results().At(0).Type()) {
		return false
	}

	return true
}

func (c *sqlQueryChecker) typeIsRowsLike(typ types.Type) bool {
	switch typ := typ.(type) {
	case *types.Pointer:
		return c.typeIsRowsLike(typ.Elem())
	case *types.Named:
		return typ.Obj().Name() == "Rows"
	default:
		return false
	}
}

func (c *sqlQueryChecker) funcIsExec(fn *types.Func) bool {
	if fn.Name() != "Exec" {
		return false
	}

	// Expect exactly 2 results.
	sig := fn.Type().(*types.Signature)
	if sig.Results() == nil || sig.Results().Len() != 2 {
		return false
	}

	// Expect at least 1 param and it should be a string (query).
	params := sig.Params()
	if params == nil || params.Len() == 0 {
		return false
	}
	if typ, ok := params.At(0).Type().(*types.Basic); !ok || typ.Kind() != types.String {
		return false
	}

	return true
}

func (c *sqlQueryChecker) typeHasExecMethod(typ types.Type) bool {
	switch typ := typ.(type) {
	case *types.Struct:
		for i := 0; i < typ.NumFields(); i++ {
			if c.typeHasExecMethod(typ.Field(i).Type()) {
				return true
			}
		}
	case *types.Interface:
		for i := 0; i < typ.NumMethods(); i++ {
			if c.funcIsExec(typ.Method(i)) {
				return true
			}
		}
	case *types.Pointer:
		return c.typeHasExecMethod(typ.Elem())
	case *types.Named:
		for i := 0; i < typ.NumMethods(); i++ {
			if c.funcIsExec(typ.Method(i)) {
				return true
			}
		}
		switch ut := typ.Underlying().(type) {
		case *types.Interface:
			return c.typeHasExecMethod(ut)
		case *types.Struct:
			// Check embedded types.
			for i := 0; i < ut.NumFields(); i++ {
				field := ut.Field(i)
				if !field.Embedded() {
					continue
				}
				if c.typeHasExecMethod(field.Type()) {
					return true
				}
			}
		}
	}

	return false
}

func (c *sqlQueryChecker) warnAndSuggestExec(funcExpr *ast.SelectorExpr) {
	c.ctx.Warn(funcExpr, "use %s.Exec() if returned result is not needed", funcExpr.X)
}

func (c *sqlQueryChecker) warnRowsIgnored(funcExpr *ast.SelectorExpr) {
	c.ctx.Warn(funcExpr, "ignoring Query() rows result may lead to a connection leak")
}

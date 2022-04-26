package checkers

import (
	"go/ast"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/framework/linter"
	"github.com/go-toolsmith/astcopy"
	"github.com/go-toolsmith/astequal"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "typeUnparen"
	info.Tags = []string{"style", "opinionated"}
	info.Summary = "Detects unneded parenthesis inside type expressions and suggests to remove them"
	info.Before = `type foo [](func([](func())))`
	info.After = `type foo []func([]func())`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		return astwalk.WalkerForTypeExpr(&typeUnparenChecker{ctx: ctx}, ctx.TypesInfo), nil
	})
}

type typeUnparenChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext
}

func (c *typeUnparenChecker) VisitTypeExpr(e ast.Expr) {
	switch e := e.(type) {
	case *ast.ParenExpr:
		switch e.X.(type) {
		case *ast.StructType:
			c.ctx.Warn(e, "could simplify (struct{...}) to struct{...}")
		case *ast.InterfaceType:
			c.ctx.Warn(e, "could simplify (interface{...}) to interface{...}")
		default:
			c.checkType(e)
		}
	case *ast.StructType, *ast.InterfaceType:
		// Only nested fields are to be reported.
	default:
		c.checkType(e)
	}
}

func (c *typeUnparenChecker) checkType(e ast.Expr) {
	noParens := c.removeRedundantParens(astcopy.Expr(e))
	if !astequal.Expr(e, noParens) {
		c.warn(e, noParens)
	}
	c.SkipChilds = true
}

func (c *typeUnparenChecker) removeRedundantParens(e ast.Expr) ast.Expr {
	switch e := e.(type) {
	case *ast.ParenExpr:
		return c.removeRedundantParens(e.X)
	case *ast.ArrayType:
		e.Elt = c.removeRedundantParens(e.Elt)
	case *ast.StarExpr:
		e.X = c.removeRedundantParens(e.X)
	case *ast.TypeAssertExpr:
		e.Type = c.removeRedundantParens(e.Type)
	case *ast.FuncType:
		for _, field := range e.Params.List {
			field.Type = c.removeRedundantParens(field.Type)
		}
		if e.Results != nil {
			for _, field := range e.Results.List {
				field.Type = c.removeRedundantParens(field.Type)
			}
		}
	case *ast.MapType:
		e.Key = c.removeRedundantParens(e.Key)
		e.Value = c.removeRedundantParens(e.Value)
	case *ast.ChanType:
		if valueWithParens, ok := e.Value.(*ast.ParenExpr); ok {
			if nestedChan, ok := valueWithParens.X.(*ast.ChanType); ok {
				const anyDir = ast.SEND | ast.RECV
				if nestedChan.Dir != anyDir || e.Dir != anyDir {
					valueWithParens.X = c.removeRedundantParens(valueWithParens.X)
					return e
				}
			}
		}
		e.Value = c.removeRedundantParens(e.Value)
	}
	return e
}

func (c *typeUnparenChecker) warn(cause, noParens ast.Expr) {
	c.ctx.Warn(cause, "could simplify %s to %s", cause, noParens)
}

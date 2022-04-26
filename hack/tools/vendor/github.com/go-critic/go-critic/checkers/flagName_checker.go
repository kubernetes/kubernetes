package checkers

import (
	"go/ast"
	"go/constant"
	"go/types"
	"strings"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/framework/linter"
	"github.com/go-toolsmith/astcast"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "flagName"
	info.Tags = []string{"diagnostic"}
	info.Summary = "Detects suspicious flag names"
	info.Before = `b := flag.Bool(" foo ", false, "description")`
	info.After = `b := flag.Bool("foo", false, "description")`
	info.Note = "https://github.com/golang/go/issues/41792"

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		return astwalk.WalkerForExpr(&flagNameChecker{ctx: ctx}), nil
	})
}

type flagNameChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext
}

func (c *flagNameChecker) VisitExpr(expr ast.Expr) {
	call := astcast.ToCallExpr(expr)
	calledExpr := astcast.ToSelectorExpr(call.Fun)
	obj, ok := c.ctx.TypesInfo.ObjectOf(astcast.ToIdent(calledExpr.X)).(*types.PkgName)
	if !ok {
		return
	}
	sym := calledExpr.Sel
	pkg := obj.Imported()
	if pkg.Path() != "flag" {
		return
	}

	switch sym.Name {
	case "Bool", "Duration", "Float64", "String",
		"Int", "Int64", "Uint", "Uint64":
		c.checkFlagName(call, call.Args[0])
	case "BoolVar", "DurationVar", "Float64Var", "StringVar",
		"IntVar", "Int64Var", "UintVar", "Uint64Var":
		c.checkFlagName(call, call.Args[1])
	}
}

func (c *flagNameChecker) checkFlagName(call *ast.CallExpr, arg ast.Expr) {
	cv := c.ctx.TypesInfo.Types[arg].Value
	if cv == nil {
		return // Non-constant name
	}
	name := constant.StringVal(cv)
	switch {
	case name == "":
		c.warnEmpty(call)
	case strings.HasPrefix(name, "-"):
		c.warnHypenPrefix(call, name)
	case strings.Contains(name, "="):
		c.warnEq(call, name)
	case strings.Contains(name, " "):
		c.warnWhitespace(call, name)
	}
}

func (c *flagNameChecker) warnEmpty(cause ast.Node) {
	c.ctx.Warn(cause, "empty flag name")
}

func (c *flagNameChecker) warnHypenPrefix(cause ast.Node, name string) {
	c.ctx.Warn(cause, "flag name %q should not start with a hypen", name)
}

func (c *flagNameChecker) warnEq(cause ast.Node, name string) {
	c.ctx.Warn(cause, "flag name %q should not contain '='", name)
}

func (c *flagNameChecker) warnWhitespace(cause ast.Node, name string) {
	c.ctx.Warn(cause, "flag name %q contains whitespace", name)
}

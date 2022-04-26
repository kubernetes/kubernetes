package checkers

import (
	"go/ast"
	"go/token"
	"strings"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/framework/linter"
	"github.com/go-toolsmith/astcast"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "hexLiteral"
	info.Tags = []string{"style", "experimental"}
	info.Summary = "Detects hex literals that have mixed case letter digits"
	info.Before = `
x := 0X12
y := 0xfF`
	info.After = `
x := 0x12
// (A)
y := 0xff
// (B)
y := 0xFF`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		return astwalk.WalkerForExpr(&hexLiteralChecker{ctx: ctx}), nil
	})
}

type hexLiteralChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext
}

func (c *hexLiteralChecker) warn0X(lit *ast.BasicLit) {
	suggest := "0x" + lit.Value[len("0X"):]
	c.ctx.Warn(lit, "prefer 0x over 0X, s/%s/%s/", lit.Value, suggest)
}

func (c *hexLiteralChecker) warnMixedDigits(lit *ast.BasicLit) {
	c.ctx.Warn(lit, "don't mix hex literal letter digits casing")
}

func (c *hexLiteralChecker) VisitExpr(expr ast.Expr) {
	lit := astcast.ToBasicLit(expr)
	if lit.Kind != token.INT || len(lit.Value) < 3 {
		return
	}
	if strings.HasPrefix(lit.Value, "0X") {
		c.warn0X(lit)
		return
	}
	digits := lit.Value[len("0x"):]
	if strings.ToLower(digits) != digits && strings.ToUpper(digits) != digits {
		c.warnMixedDigits(lit)
	}
}

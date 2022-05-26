package checkers

import (
	"go/ast"
	"go/types"
	"strings"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/checkers/internal/lintutil"
	"github.com/go-critic/go-critic/framework/linter"
	"github.com/go-toolsmith/astcast"
	"github.com/go-toolsmith/astp"
	"github.com/go-toolsmith/typep"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "mapKey"
	info.Tags = []string{"diagnostic"}
	info.Summary = "Detects suspicious map literal keys"
	info.Before = `
_ = map[string]int{
	"foo": 1,
	"bar ": 2,
}`
	info.After = `
_ = map[string]int{
	"foo": 1,
	"bar": 2,
}`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		return astwalk.WalkerForExpr(&mapKeyChecker{ctx: ctx}), nil
	})
}

type mapKeyChecker struct {
	astwalk.WalkHandler
	ctx *linter.CheckerContext

	astSet lintutil.AstSet
}

func (c *mapKeyChecker) VisitExpr(expr ast.Expr) {
	lit := astcast.ToCompositeLit(expr)
	if len(lit.Elts) < 2 {
		return
	}

	typ, ok := c.ctx.TypeOf(lit).Underlying().(*types.Map)
	if !ok {
		return
	}
	if !typep.HasStringKind(typ.Key().Underlying()) {
		return
	}

	c.checkWhitespace(lit)
	c.checkDuplicates(lit)
}

func (c *mapKeyChecker) checkDuplicates(lit *ast.CompositeLit) {
	c.astSet.Clear()

	for _, elt := range lit.Elts {
		kv := astcast.ToKeyValueExpr(elt)
		if astp.IsBasicLit(kv.Key) {
			// Basic lits are handled by the compiler.
			continue
		}
		if !typep.SideEffectFree(c.ctx.TypesInfo, kv.Key) {
			continue
		}
		if !c.astSet.Insert(kv.Key) {
			c.warnDupKey(kv.Key)
		}
	}
}

func (c *mapKeyChecker) checkWhitespace(lit *ast.CompositeLit) {
	var whitespaceKey ast.Node
	for _, elt := range lit.Elts {
		key := astcast.ToBasicLit(astcast.ToKeyValueExpr(elt).Key)
		if len(key.Value) < len(`" "`) {
			continue
		}
		// s is unquoted string literal value.
		s := key.Value[len(`"`) : len(key.Value)-len(`"`)]
		if !strings.Contains(s, " ") {
			continue
		}
		if whitespaceKey != nil {
			// Already seen something with a whitespace.
			// More than one entry => not suspicious.
			return
		}
		if s == " " {
			// If space is used as a key, maybe this map
			// has something to do with spaces. Give up.
			return
		}
		// Check if it has exactly 1 space prefix or suffix.
		bad := strings.HasPrefix(s, " ") && !strings.HasPrefix(s, "  ") ||
			strings.HasSuffix(s, " ") && !strings.HasSuffix(s, "  ")
		if !bad {
			// These spaces can be a padding,
			// or a legitimate part of a key. Give up.
			return
		}
		whitespaceKey = key
	}

	if whitespaceKey != nil {
		c.warnWhitespace(whitespaceKey)
	}
}

func (c *mapKeyChecker) warnWhitespace(key ast.Node) {
	c.ctx.Warn(key, "suspucious whitespace in %s key", key)
}

func (c *mapKeyChecker) warnDupKey(key ast.Node) {
	c.ctx.Warn(key, "suspicious duplicate %s key", key)
}

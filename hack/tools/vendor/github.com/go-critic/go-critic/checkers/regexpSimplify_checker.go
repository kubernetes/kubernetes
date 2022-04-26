package checkers

import (
	"fmt"
	"go/ast"
	"go/constant"
	"log"
	"strings"
	"unicode/utf8"

	"github.com/go-critic/go-critic/checkers/internal/astwalk"
	"github.com/go-critic/go-critic/framework/linter"
	"github.com/quasilyte/regex/syntax"
)

func init() {
	var info linter.CheckerInfo
	info.Name = "regexpSimplify"
	info.Tags = []string{"style", "experimental", "opinionated"}
	info.Summary = "Detects regexp patterns that can be simplified"
	info.Before = "regexp.MustCompile(`(?:a|b|c)   [a-z][a-z]*`)"
	info.After = "regexp.MustCompile(`[abc] {3}[a-z]+`)"

	// TODO(quasilyte): add params to control most opinionated replacements
	// like `[0-9] -> \d`
	//      `[[:digit:]] -> \d`
	//      `[A-Za-z0-9_]` -> `\w`

	collection.AddChecker(&info, func(ctx *linter.CheckerContext) (linter.FileWalker, error) {
		opts := &syntax.ParserOptions{
			NoLiterals: true,
		}
		c := &regexpSimplifyChecker{
			ctx:    ctx,
			parser: syntax.NewParser(opts),
			out:    &strings.Builder{},
		}
		return astwalk.WalkerForExpr(c), nil
	})
}

type regexpSimplifyChecker struct {
	astwalk.WalkHandler
	ctx    *linter.CheckerContext
	parser *syntax.Parser

	// out is a tmp buffer where we build a simplified regexp pattern.
	out *strings.Builder
	// score is a number of applied simplifications
	score int
}

func (c *regexpSimplifyChecker) VisitExpr(x ast.Expr) {
	call, ok := x.(*ast.CallExpr)
	if !ok {
		return
	}

	switch qualifiedName(call.Fun) {
	case "regexp.Compile", "regexp.MustCompile":
		cv := c.ctx.TypesInfo.Types[call.Args[0]].Value
		if cv == nil || cv.Kind() != constant.String {
			return
		}
		pat := constant.StringVal(cv)
		if len(pat) > 60 {
			// Skip scary regexp patterns for now.
			break
		}

		// Only do 2 passes.
		simplified := pat
		for pass := 0; pass < 2; pass++ {
			candidate := c.simplify(pass, simplified)
			if candidate == "" {
				break
			}
			simplified = candidate
		}
		if simplified != "" && simplified != pat {
			c.warn(call.Args[0], pat, simplified)
		}
	}
}

func (c *regexpSimplifyChecker) simplify(pass int, pat string) string {
	re, err := c.parser.Parse(pat)
	if err != nil {
		return ""
	}

	c.score = 0
	c.out.Reset()

	// TODO(quasilyte): suggest char ranges for things like [012345689]?
	// TODO(quasilyte): evaluate char range to suggest better replacements.
	// TODO(quasilyte): (?:ab|ac) -> a[bc]
	// TODO(quasilyte): suggest "s" and "." flag if things like [\w\W] are used.
	// TODO(quasilyte): x{n}x? -> x{n,n+1}

	c.walk(re.Expr)

	if debug() {
		// This happens only in one of two cases:
		// 1. Parser has a bug and we got invalid AST for the given pattern.
		// 2. Simplifier incorrectly built a replacement string from the AST.
		if c.score == 0 && c.out.String() != pat {
			log.Printf("pass %d: unexpected pattern diff:\n\thave: %q\n\twant: %q",
				pass, c.out.String(), pat)
		}
	}

	if c.score > 0 {
		return c.out.String()
	}
	return ""
}

func (c *regexpSimplifyChecker) walk(e syntax.Expr) {
	out := c.out

	switch e.Op {
	case syntax.OpConcat:
		c.walkConcat(e)

	case syntax.OpAlt:
		c.walkAlt(e)

	case syntax.OpCharRange:
		s := c.simplifyCharRange(e)
		if s != "" {
			out.WriteString(s)
			c.score++
		} else {
			out.WriteString(e.Value)
		}

	case syntax.OpGroupWithFlags:
		out.WriteString("(")
		out.WriteString(e.Args[1].Value)
		out.WriteString(":")
		c.walk(e.Args[0])
		out.WriteString(")")
	case syntax.OpGroup:
		c.walkGroup(e)
	case syntax.OpCapture:
		out.WriteString("(")
		c.walk(e.Args[0])
		out.WriteString(")")
	case syntax.OpNamedCapture:
		out.WriteString("(?P<")
		out.WriteString(e.Args[1].Value)
		out.WriteString(">")
		c.walk(e.Args[0])
		out.WriteString(")")

	case syntax.OpRepeat:
		// TODO(quasilyte): is it worth it to analyze repeat argument
		// more closely and handle `{n,n} -> {n}` cases?
		rep := e.Args[1].Value
		switch rep {
		case "{0,1}":
			c.walk(e.Args[0])
			out.WriteString("?")
			c.score++
		case "{1,}":
			c.walk(e.Args[0])
			out.WriteString("+")
			c.score++
		case "{0,}":
			c.walk(e.Args[0])
			out.WriteString("*")
			c.score++
		case "{0}":
			// Maybe {0} should be reported by another check, regexpLint?
			c.score++
		case "{1}":
			c.walk(e.Args[0])
			c.score++
		default:
			c.walk(e.Args[0])
			out.WriteString(rep)
		}

	case syntax.OpPosixClass:
		out.WriteString(e.Value)

	case syntax.OpNegCharClass:
		s := c.simplifyNegCharClass(e)
		if s != "" {
			c.out.WriteString(s)
			c.score++
		} else {
			out.WriteString("[^")
			for _, e := range e.Args {
				c.walk(e)
			}
			out.WriteString("]")
		}

	case syntax.OpCharClass:
		s := c.simplifyCharClass(e)
		if s != "" {
			c.out.WriteString(s)
			c.score++
		} else {
			out.WriteString("[")
			for _, e := range e.Args {
				c.walk(e)
			}
			out.WriteString("]")
		}

	case syntax.OpEscapeChar:
		switch e.Value {
		case `\&`, `\#`, `\!`, `\@`, `\%`, `\<`, `\>`, `\:`, `\;`, `\/`, `\,`, `\=`, `\.`:
			c.score++
			out.WriteString(e.Value[len(`\`):])
		default:
			out.WriteString(e.Value)
		}

	case syntax.OpQuestion, syntax.OpNonGreedy:
		c.walk(e.Args[0])
		out.WriteString("?")
	case syntax.OpStar:
		c.walk(e.Args[0])
		out.WriteString("*")
	case syntax.OpPlus:
		c.walk(e.Args[0])
		out.WriteString("+")

	default:
		out.WriteString(e.Value)
	}
}

func (c *regexpSimplifyChecker) walkGroup(g syntax.Expr) {
	switch g.Args[0].Op {
	case syntax.OpChar, syntax.OpEscapeChar, syntax.OpEscapeMeta, syntax.OpCharClass:
		c.walk(g.Args[0])
		c.score++
		return
	}

	c.out.WriteString("(?:")
	c.walk(g.Args[0])
	c.out.WriteString(")")
}

func (c *regexpSimplifyChecker) simplifyNegCharClass(e syntax.Expr) string {
	switch e.Value {
	case `[^0-9]`:
		return `\D`
	case `[^\s]`:
		return `\S`
	case `[^\S]`:
		return `\s`
	case `[^\w]`:
		return `\W`
	case `[^\W]`:
		return `\w`
	case `[^\d]`:
		return `\D`
	case `[^\D]`:
		return `\d`
	case `[^[:^space:]]`:
		return `\s`
	case `[^[:space:]]`:
		return `\S`
	case `[^[:^word:]]`:
		return `\w`
	case `[^[:word:]]`:
		return `\W`
	case `[^[:^digit:]]`:
		return `\d`
	case `[^[:digit:]]`:
		return `\D`
	}

	return ""
}

func (c *regexpSimplifyChecker) simplifyCharClass(e syntax.Expr) string {
	switch e.Value {
	case `[0-9]`:
		return `\d`
	case `[[:word:]]`:
		return `\w`
	case `[[:^word:]]`:
		return `\W`
	case `[[:digit:]]`:
		return `\d`
	case `[[:^digit:]]`:
		return `\D`
	case `[[:space:]]`:
		return `\s`
	case `[[:^space:]]`:
		return `\S`
	case `[][]`:
		return `\]\[`
	case `[]]`:
		return `\]`
	}

	if len(e.Args) == 1 {
		switch e.Args[0].Op {
		case syntax.OpChar:
			switch v := e.Args[0].Value; v {
			case "|", "*", "+", "?", ".", "[", "^", "$", "(", ")":
				// Can't take outside of the char group without escaping.
			default:
				return v
			}
		case syntax.OpEscapeChar:
			return e.Args[0].Value
		}
	}

	return ""
}

func (c *regexpSimplifyChecker) canMerge(x, y syntax.Expr) bool {
	if x.Op != y.Op {
		return false
	}
	switch x.Op {
	case syntax.OpChar, syntax.OpCharClass, syntax.OpEscapeMeta, syntax.OpEscapeChar, syntax.OpNegCharClass, syntax.OpGroup:
		return x.Value == y.Value
	default:
		return false
	}
}

func (c *regexpSimplifyChecker) canCombine(x, y syntax.Expr) (threshold int, ok bool) {
	if x.Op != y.Op {
		return 0, false
	}

	switch x.Op {
	case syntax.OpDot:
		return 3, true

	case syntax.OpChar:
		if x.Value != y.Value {
			return 0, false
		}
		if x.Value == " " {
			return 1, true
		}
		return 4, true

	case syntax.OpEscapeMeta, syntax.OpEscapeChar:
		if x.Value == y.Value {
			return 2, true
		}

	case syntax.OpCharClass, syntax.OpNegCharClass, syntax.OpGroup:
		if x.Value == y.Value {
			return 1, true
		}
	}

	return 0, false
}

func (c *regexpSimplifyChecker) concatLiteral(e syntax.Expr) string {
	if e.Op == syntax.OpConcat && c.allChars(e) {
		return e.Value
	}
	return ""
}

func (c *regexpSimplifyChecker) allChars(e syntax.Expr) bool {
	for _, a := range e.Args {
		if a.Op != syntax.OpChar {
			return false
		}
	}
	return true
}

func (c *regexpSimplifyChecker) factorPrefixSuffix(alt syntax.Expr) bool {
	// TODO: more forms of prefixes/suffixes?
	//
	// A more generalized algorithm could handle `fo|fo1|fo2` -> `fo[12]?`.
	// but it's an open question whether the latter form universally better.
	//
	// Right now it handles only the simplest cases:
	// `http|https` -> `https?`
	// `xfoo|foo` -> `x?foo`
	if len(alt.Args) != 2 {
		return false
	}
	x := c.concatLiteral(alt.Args[0])
	y := c.concatLiteral(alt.Args[1])
	if x == y {
		return false // Reject non-literals and identical strings early
	}

	// Let x be a shorter string.
	if len(x) > len(y) {
		x, y = y, x
	}
	// Do we have a common prefix?
	tail := strings.TrimPrefix(y, x)
	if len(tail) <= utf8.UTFMax && utf8.RuneCountInString(tail) == 1 {
		c.out.WriteString(x + tail + "?")
		c.score++
		return true
	}
	// Do we have a common suffix?
	head := strings.TrimSuffix(y, x)
	if len(head) <= utf8.UTFMax && utf8.RuneCountInString(head) == 1 {
		c.out.WriteString(head + "?" + x)
		c.score++
		return true
	}
	return false
}

func (c *regexpSimplifyChecker) walkAlt(alt syntax.Expr) {
	// `x|y|z` -> `[xyz]`.
	if c.allChars(alt) {
		c.score++
		c.out.WriteString("[")
		for _, e := range alt.Args {
			c.out.WriteString(e.Value)
		}
		c.out.WriteString("]")
		return
	}

	if c.factorPrefixSuffix(alt) {
		return
	}

	for i, e := range alt.Args {
		c.walk(e)
		if i != len(alt.Args)-1 {
			c.out.WriteString("|")
		}
	}
}

func (c *regexpSimplifyChecker) walkConcat(concat syntax.Expr) {
	i := 0
	for i < len(concat.Args) {
		x := concat.Args[i]
		c.walk(x)
		i++

		if i >= len(concat.Args) {
			break
		}

		// Try merging `xy*` into `x+` where x=y.
		if concat.Args[i].Op == syntax.OpStar {
			if c.canMerge(x, concat.Args[i].Args[0]) {
				c.out.WriteString("+")
				c.score++
				i++
				continue
			}
		}

		// Try combining `xy` into `x{2}` where x=y.
		threshold, ok := c.canCombine(x, concat.Args[i])
		if !ok {
			continue
		}
		n := 1 // Can combine at least 1 pair.
		for j := i + 1; j < len(concat.Args); j++ {
			_, ok := c.canCombine(x, concat.Args[j])
			if !ok {
				break
			}
			n++
		}
		if n >= threshold {
			fmt.Fprintf(c.out, "{%d}", n+1)
			c.score++
			i += n
		}
	}
}

func (c *regexpSimplifyChecker) simplifyCharRange(rng syntax.Expr) string {
	if rng.Args[0].Op != syntax.OpChar || rng.Args[1].Op != syntax.OpChar {
		return ""
	}

	lo := rng.Args[0].Value
	hi := rng.Args[1].Value
	if len(lo) == 1 && len(hi) == 1 {
		switch hi[0] - lo[0] {
		case 0:
			return lo
		case 1:
			return fmt.Sprintf("%s%s", lo, hi)
		case 2:
			return fmt.Sprintf("%s%s%s", lo, string(lo[0]+1), hi)
		}
	}

	return ""
}

func (c *regexpSimplifyChecker) warn(cause ast.Expr, orig, suggest string) {
	c.ctx.Warn(cause, "can re-write `%s` as `%s`", orig, suggest)
}

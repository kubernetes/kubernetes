package syntax

import (
	"fmt"
	"strings"
)

type Regexp struct {
	Pattern string
	Expr    Expr
}

type RegexpPCRE struct {
	Pattern string
	Expr    Expr

	Source    string
	Modifiers string
	Delim     [2]byte
}

func (re *RegexpPCRE) HasModifier(mod byte) bool {
	return strings.IndexByte(re.Modifiers, mod) >= 0
}

type Expr struct {
	// The operations that this expression performs. See `operation.go`.
	Op Operation

	Form Form

	_ [2]byte // Reserved

	// Pos describes a source location inside regexp pattern.
	Pos Position

	// Args is a list of sub-expressions of this expression.
	//
	// See Operation constants documentation to learn how to
	// interpret the particular expression args.
	Args []Expr

	// Value holds expression textual value.
	//
	// Usually, that value is identical to src[Begin():End()],
	// but this is not true for programmatically generated objects.
	Value string
}

// Begin returns expression leftmost offset.
func (e Expr) Begin() uint16 { return e.Pos.Begin }

// End returns expression rightmost offset.
func (e Expr) End() uint16 { return e.Pos.End }

// LastArg returns expression last argument.
//
// Should not be called on expressions that may have 0 arguments.
func (e Expr) LastArg() Expr {
	return e.Args[len(e.Args)-1]
}

type Operation byte

type Form byte

func FormatSyntax(re *Regexp) string {
	return formatExprSyntax(re, re.Expr)
}

func formatExprSyntax(re *Regexp, e Expr) string {
	switch e.Op {
	case OpChar, OpLiteral:
		switch e.Value {
		case "{":
			return "'{'"
		case "}":
			return "'}'"
		default:
			return e.Value
		}
	case OpString, OpEscapeChar, OpEscapeMeta, OpEscapeOctal, OpEscapeUni, OpEscapeHex, OpPosixClass:
		return e.Value
	case OpRepeat:
		return fmt.Sprintf("(repeat %s %s)", formatExprSyntax(re, e.Args[0]), e.Args[1].Value)
	case OpCaret:
		return "^"
	case OpDollar:
		return "$"
	case OpDot:
		return "."
	case OpQuote:
		return fmt.Sprintf("(q %s)", e.Value)
	case OpCharRange:
		return fmt.Sprintf("%s-%s", formatExprSyntax(re, e.Args[0]), formatExprSyntax(re, e.Args[1]))
	case OpCharClass:
		return fmt.Sprintf("[%s]", formatArgsSyntax(re, e.Args))
	case OpNegCharClass:
		return fmt.Sprintf("[^%s]", formatArgsSyntax(re, e.Args))
	case OpConcat:
		return fmt.Sprintf("{%s}", formatArgsSyntax(re, e.Args))
	case OpAlt:
		return fmt.Sprintf("(or %s)", formatArgsSyntax(re, e.Args))
	case OpCapture:
		return fmt.Sprintf("(capture %s)", formatExprSyntax(re, e.Args[0]))
	case OpNamedCapture:
		return fmt.Sprintf("(capture %s %s)", formatExprSyntax(re, e.Args[0]), e.Args[1].Value)
	case OpGroup:
		return fmt.Sprintf("(group %s)", formatExprSyntax(re, e.Args[0]))
	case OpAtomicGroup:
		return fmt.Sprintf("(atomic %s)", formatExprSyntax(re, e.Args[0]))
	case OpGroupWithFlags:
		return fmt.Sprintf("(group %s ?%s)", formatExprSyntax(re, e.Args[0]), e.Args[1].Value)
	case OpFlagOnlyGroup:
		return fmt.Sprintf("(flags ?%s)", formatExprSyntax(re, e.Args[0]))
	case OpPositiveLookahead:
		return fmt.Sprintf("(?= %s)", formatExprSyntax(re, e.Args[0]))
	case OpNegativeLookahead:
		return fmt.Sprintf("(?! %s)", formatExprSyntax(re, e.Args[0]))
	case OpPositiveLookbehind:
		return fmt.Sprintf("(?<= %s)", formatExprSyntax(re, e.Args[0]))
	case OpNegativeLookbehind:
		return fmt.Sprintf("(?<! %s)", formatExprSyntax(re, e.Args[0]))
	case OpPlus:
		return fmt.Sprintf("(+ %s)", formatExprSyntax(re, e.Args[0]))
	case OpStar:
		return fmt.Sprintf("(* %s)", formatExprSyntax(re, e.Args[0]))
	case OpQuestion:
		return fmt.Sprintf("(? %s)", formatExprSyntax(re, e.Args[0]))
	case OpNonGreedy:
		return fmt.Sprintf("(non-greedy %s)", formatExprSyntax(re, e.Args[0]))
	case OpPossessive:
		return fmt.Sprintf("(possessive %s)", formatExprSyntax(re, e.Args[0]))
	case OpComment:
		return fmt.Sprintf("/*%s*/", e.Value)
	default:
		return fmt.Sprintf("<op=%d>", e.Op)
	}
}

func formatArgsSyntax(re *Regexp, args []Expr) string {
	parts := make([]string, len(args))
	for i, e := range args {
		parts[i] = formatExprSyntax(re, e)
	}
	return strings.Join(parts, " ")
}

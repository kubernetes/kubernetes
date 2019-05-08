//line build/parse.y:13
package build

import __yyfmt__ "fmt"

//line build/parse.y:13
//line build/parse.y:18
type yySymType struct {
	yys int
	// input tokens
	tok    string   // raw input syntax
	str    string   // decoding of quoted string
	pos    Position // position of token
	triple bool     // was string triple quoted?

	// partial syntax trees
	expr    Expr
	exprs   []Expr
	forc    *ForClause
	ifs     []*IfClause
	forifs  *ForClauseWithIfClausesOpt
	forsifs []*ForClauseWithIfClausesOpt
	string  *StringExpr
	strings []*StringExpr
	block   CodeBlock

	// supporting information
	comma    Position // position of trailing comma in list, if present
	lastRule Expr     // most recent rule, to attach line comments to
}

const _AUGM = 57346
const _AND = 57347
const _COMMENT = 57348
const _EOF = 57349
const _EQ = 57350
const _FOR = 57351
const _GE = 57352
const _IDENT = 57353
const _IF = 57354
const _ELSE = 57355
const _ELIF = 57356
const _IN = 57357
const _IS = 57358
const _LAMBDA = 57359
const _LOAD = 57360
const _LE = 57361
const _NE = 57362
const _NOT = 57363
const _OR = 57364
const _PYTHON = 57365
const _STRING = 57366
const _DEF = 57367
const _RETURN = 57368
const _INDENT = 57369
const _UNINDENT = 57370
const ShiftInstead = 57371
const _ASSERT = 57372
const _UNARY = 57373

var yyToknames = [...]string{
	"$end",
	"error",
	"$unk",
	"'%'",
	"'('",
	"')'",
	"'*'",
	"'+'",
	"','",
	"'-'",
	"'.'",
	"'/'",
	"':'",
	"'<'",
	"'='",
	"'>'",
	"'['",
	"']'",
	"'{'",
	"'}'",
	"_AUGM",
	"_AND",
	"_COMMENT",
	"_EOF",
	"_EQ",
	"_FOR",
	"_GE",
	"_IDENT",
	"_IF",
	"_ELSE",
	"_ELIF",
	"_IN",
	"_IS",
	"_LAMBDA",
	"_LOAD",
	"_LE",
	"_NE",
	"_NOT",
	"_OR",
	"_PYTHON",
	"_STRING",
	"_DEF",
	"_RETURN",
	"_INDENT",
	"_UNINDENT",
	"ShiftInstead",
	"'\\n'",
	"_ASSERT",
	"_UNARY",
	"';'",
}
var yyStatenames = [...]string{}

const yyEofCode = 1
const yyErrCode = 2
const yyInitialStackSize = 16

//line build/parse.y:738

// Go helper code.

// unary returns a unary expression with the given
// position, operator, and subexpression.
func unary(pos Position, op string, x Expr) Expr {
	return &UnaryExpr{
		OpStart: pos,
		Op:      op,
		X:       x,
	}
}

// binary returns a binary expression with the given
// operands, position, and operator.
func binary(x Expr, pos Position, op string, y Expr) Expr {
	_, xend := x.Span()
	ystart, _ := y.Span()
	return &BinaryExpr{
		X:         x,
		OpStart:   pos,
		Op:        op,
		LineBreak: xend.Line < ystart.Line,
		Y:         y,
	}
}

// isSimpleExpression returns whether an expression is simple and allowed to exist in
// compact forms of sequences.
// The formal criteria are the following: an expression is considered simple if it's
// a literal (variable, string or a number), a literal with a unary operator or an empty sequence.
func isSimpleExpression(expr *Expr) bool {
	switch x := (*expr).(type) {
	case *LiteralExpr, *StringExpr:
		return true
	case *UnaryExpr:
		_, ok := x.X.(*LiteralExpr)
		return ok
	case *ListExpr:
		return len(x.List) == 0
	case *TupleExpr:
		return len(x.List) == 0
	case *DictExpr:
		return len(x.List) == 0
	case *SetExpr:
		return len(x.List) == 0
	default:
		return false
	}
}

// forceCompact returns the setting for the ForceCompact field for a call or tuple.
//
// NOTE 1: The field is called ForceCompact, not ForceSingleLine,
// because it only affects the formatting associated with the call or tuple syntax,
// not the formatting of the arguments. For example:
//
//	call([
//		1,
//		2,
//		3,
//	])
//
// is still a compact call even though it runs on multiple lines.
//
// In contrast the multiline form puts a linebreak after the (.
//
//	call(
//		[
//			1,
//			2,
//			3,
//		],
//	)
//
// NOTE 2: Because of NOTE 1, we cannot use start and end on the
// same line as a signal for compact mode: the formatting of an
// embedded list might move the end to a different line, which would
// then look different on rereading and cause buildifier not to be
// idempotent. Instead, we have to look at properties guaranteed
// to be preserved by the reformatting, namely that the opening
// paren and the first expression are on the same line and that
// each subsequent expression begins on the same line as the last
// one ended (no line breaks after comma).
func forceCompact(start Position, list []Expr, end Position) bool {
	if len(list) <= 1 {
		// The call or tuple will probably be compact anyway; don't force it.
		return false
	}

	// If there are any named arguments or non-string, non-literal
	// arguments, cannot force compact mode.
	line := start.Line
	for _, x := range list {
		start, end := x.Span()
		if start.Line != line {
			return false
		}
		line = end.Line
		if !isSimpleExpression(&x) {
			return false
		}
	}
	return end.Line == line
}

// forceMultiLine returns the setting for the ForceMultiLine field.
func forceMultiLine(start Position, list []Expr, end Position) bool {
	if len(list) > 1 {
		// The call will be multiline anyway, because it has multiple elements. Don't force it.
		return false
	}

	if len(list) == 0 {
		// Empty list: use position of brackets.
		return start.Line != end.Line
	}

	// Single-element list.
	// Check whether opening bracket is on different line than beginning of
	// element, or closing bracket is on different line than end of element.
	elemStart, elemEnd := list[0].Span()
	return start.Line != elemStart.Line || end.Line != elemEnd.Line
}

//line yacctab:1
var yyExca = [...]int{
	-1, 1,
	1, -1,
	-2, 0,
}

const yyPrivate = 57344

const yyLast = 739

var yyAct = [...]int{

	13, 112, 136, 2, 138, 74, 17, 7, 118, 69,
	34, 9, 117, 81, 130, 58, 31, 59, 35, 64,
	65, 66, 161, 30, 102, 70, 72, 77, 37, 38,
	166, 84, 108, 33, 79, 73, 76, 85, 84, 120,
	88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
	98, 99, 100, 101, 167, 103, 104, 105, 106, 86,
	163, 83, 110, 111, 154, 149, 153, 129, 64, 127,
	120, 109, 29, 120, 178, 87, 25, 115, 20, 126,
	120, 27, 116, 64, 133, 123, 120, 125, 24, 172,
	26, 134, 132, 131, 171, 61, 68, 71, 114, 28,
	168, 60, 113, 139, 145, 18, 22, 62, 121, 19,
	141, 158, 30, 148, 146, 147, 63, 40, 142, 82,
	39, 124, 147, 143, 67, 41, 150, 35, 80, 155,
	157, 152, 150, 36, 150, 156, 40, 1, 160, 39,
	42, 162, 43, 23, 41, 78, 165, 164, 75, 32,
	12, 8, 150, 4, 151, 21, 119, 122, 25, 0,
	20, 0, 169, 27, 0, 170, 0, 173, 174, 0,
	24, 175, 26, 165, 177, 7, 6, 0, 25, 11,
	0, 28, 16, 27, 0, 0, 0, 18, 22, 0,
	24, 19, 26, 15, 30, 10, 14, 25, 176, 20,
	5, 28, 27, 0, 0, 0, 0, 0, 22, 24,
	0, 26, 0, 0, 30, 6, 3, 25, 11, 20,
	28, 16, 27, 0, 0, 0, 18, 22, 0, 24,
	19, 26, 15, 30, 10, 14, 0, 0, 0, 5,
	28, 0, 0, 0, 0, 0, 18, 22, 0, 0,
	19, 0, 15, 30, 40, 14, 0, 39, 42, 137,
	43, 0, 41, 128, 44, 50, 45, 0, 0, 0,
	0, 51, 55, 0, 0, 46, 0, 49, 0, 57,
	0, 0, 52, 56, 0, 0, 47, 48, 53, 54,
	40, 0, 0, 39, 42, 0, 43, 0, 41, 159,
	44, 50, 45, 0, 0, 0, 0, 51, 55, 0,
	0, 46, 0, 49, 0, 57, 0, 0, 52, 56,
	0, 0, 47, 48, 53, 54, 40, 0, 0, 39,
	42, 0, 43, 0, 41, 0, 44, 50, 45, 0,
	144, 0, 0, 51, 55, 0, 0, 46, 0, 49,
	0, 57, 0, 0, 52, 56, 0, 0, 47, 48,
	53, 54, 40, 0, 0, 39, 42, 0, 43, 0,
	41, 0, 44, 50, 45, 0, 0, 0, 0, 51,
	55, 0, 0, 46, 120, 49, 0, 57, 0, 0,
	52, 56, 0, 0, 47, 48, 53, 54, 40, 0,
	0, 39, 42, 0, 43, 0, 41, 0, 44, 50,
	45, 0, 0, 0, 0, 51, 55, 0, 0, 46,
	0, 49, 0, 57, 140, 0, 52, 56, 0, 0,
	47, 48, 53, 54, 40, 0, 0, 39, 42, 0,
	43, 0, 41, 135, 44, 50, 45, 0, 0, 0,
	0, 51, 55, 0, 0, 46, 0, 49, 0, 57,
	0, 0, 52, 56, 0, 0, 47, 48, 53, 54,
	40, 0, 0, 39, 42, 0, 43, 0, 41, 107,
	44, 50, 45, 0, 0, 0, 0, 51, 55, 0,
	0, 46, 0, 49, 0, 57, 0, 0, 52, 56,
	0, 0, 47, 48, 53, 54, 40, 0, 0, 39,
	42, 0, 43, 0, 41, 0, 44, 50, 45, 0,
	0, 0, 0, 51, 55, 0, 0, 46, 0, 49,
	0, 57, 0, 0, 52, 56, 0, 0, 47, 48,
	53, 54, 40, 0, 0, 39, 42, 0, 43, 0,
	41, 0, 44, 50, 45, 0, 0, 0, 0, 51,
	55, 0, 0, 46, 0, 49, 0, 0, 0, 0,
	52, 56, 0, 0, 47, 48, 53, 54, 40, 0,
	0, 39, 42, 0, 43, 0, 41, 0, 44, 0,
	45, 0, 0, 0, 0, 0, 55, 0, 0, 46,
	0, 49, 0, 57, 0, 0, 52, 56, 0, 0,
	47, 48, 53, 54, 40, 0, 0, 39, 42, 0,
	43, 0, 41, 0, 44, 0, 45, 0, 0, 0,
	0, 0, 55, 0, 0, 46, 0, 49, 0, 25,
	0, 20, 52, 56, 27, 0, 47, 48, 53, 54,
	0, 24, 0, 26, 0, 40, 0, 0, 39, 42,
	0, 43, 28, 41, 0, 44, 0, 45, 18, 22,
	0, 0, 19, 55, 15, 30, 46, 14, 49, 0,
	0, 0, 0, 0, 0, 0, 0, 47, 48, 40,
	54, 0, 39, 42, 0, 43, 0, 41, 0, 44,
	0, 45, 0, 0, 0, 40, 0, 55, 39, 42,
	46, 43, 49, 41, 0, 44, 0, 45, 0, 0,
	0, 47, 48, 0, 0, 0, 46, 0, 49, 0,
	0, 0, 0, 0, 0, 0, 0, 47, 48,
}
var yyPact = [...]int{

	-1000, -1000, 192, -1000, -1000, -1000, -31, -1000, -1000, -1000,
	5, 173, -2, 502, 71, -1000, 71, 90, 71, 71,
	71, -1000, 119, -18, 71, 71, 71, 173, -1000, -1000,
	-1000, -1000, -37, 114, 29, 90, 71, 46, -1000, 71,
	71, 71, 71, 71, 71, 71, 71, 71, 71, 71,
	71, 71, 71, -8, 71, 71, 71, 71, 502, 466,
	4, 71, 71, 89, 502, -1000, -1000, 71, -1000, 64,
	358, 99, 358, 115, 13, 59, 49, 250, 58, -1000,
	-33, 634, 71, 71, 173, 430, 212, -1000, -1000, -1000,
	-1000, 113, 113, 132, 132, 132, 132, 132, 132, 574,
	574, 651, 71, 685, 701, 651, 394, 212, -1000, 112,
	358, 322, 91, 71, 71, 107, -1000, 47, -1000, -1000,
	173, 71, -1000, 60, -1000, 44, -1000, -1000, 71, 71,
	-1000, -1000, 105, 286, 90, 212, -1000, -22, -1000, 651,
	71, -1000, -1000, 54, -1000, 71, 610, 502, -1000, -1000,
	-1000, 1, 22, -1000, -1000, 502, -1000, 250, 87, 212,
	-1000, -1000, 610, -1000, 76, 502, 71, 71, 212, -1000,
	153, -1000, 71, 538, 538, -1000, -1000, 56, -1000,
}
var yyPgo = [...]int{

	0, 157, 0, 1, 6, 97, 9, 10, 156, 8,
	12, 155, 154, 3, 153, 151, 150, 4, 11, 149,
	5, 148, 145, 72, 143, 2, 137, 133, 128,
}
var yyR1 = [...]int{

	0, 26, 25, 25, 13, 13, 13, 13, 14, 14,
	15, 15, 15, 16, 16, 16, 27, 27, 17, 19,
	19, 18, 18, 18, 18, 28, 28, 4, 4, 4,
	4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
	4, 4, 4, 4, 2, 2, 2, 2, 2, 2,
	2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
	2, 2, 2, 2, 2, 2, 2, 3, 3, 1,
	1, 20, 22, 22, 21, 21, 5, 5, 6, 6,
	7, 7, 23, 24, 24, 11, 8, 9, 10, 10,
	12, 12,
}
var yyR2 = [...]int{

	0, 2, 4, 1, 0, 2, 2, 3, 1, 1,
	7, 6, 1, 4, 5, 4, 2, 1, 4, 0,
	3, 1, 2, 1, 1, 0, 1, 1, 3, 4,
	4, 4, 6, 8, 5, 1, 3, 4, 4, 4,
	3, 3, 3, 2, 1, 4, 2, 2, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 4, 3, 3, 3, 5, 0, 1, 0,
	1, 3, 1, 3, 1, 2, 1, 3, 0, 2,
	1, 3, 1, 1, 2, 1, 4, 2, 1, 2,
	0, 3,
}
var yyChk = [...]int{

	-1000, -26, -13, 24, -14, 47, 23, -17, -15, -18,
	42, 26, -16, -2, 43, 40, 29, -4, 34, 38,
	7, -11, 35, -24, 17, 5, 19, 10, 28, -23,
	41, 47, -19, 28, -7, -4, -27, 30, 31, 7,
	4, 12, 8, 10, 14, 16, 25, 36, 37, 27,
	15, 21, 32, 38, 39, 22, 33, 29, -2, -2,
	11, 5, 17, -5, -2, -2, -2, 5, -23, -6,
	-2, -5, -2, -6, -20, -21, -6, -2, -22, -4,
	-28, 50, 5, 32, 9, -2, 13, 29, -2, -2,
	-2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
	-2, -2, 32, -2, -2, -2, -2, 13, 28, -6,
	-2, -2, -3, 13, 9, -6, 18, -10, -9, -8,
	26, 9, -1, -10, 6, -10, 20, 20, 13, 9,
	47, -18, -6, -2, -4, 13, -25, 47, -17, -2,
	30, -25, 6, -10, 18, 13, -2, -2, 6, 18,
	-9, -12, -7, 6, 20, -2, -20, -2, 6, 13,
	-25, 44, -2, 6, -3, -2, 29, 32, 13, -25,
	-13, 18, 13, -2, -2, -25, 45, -3, 18,
}
var yyDef = [...]int{

	4, -2, 0, 1, 5, 6, 0, 8, 9, 19,
	0, 0, 12, 21, 23, 24, 0, 44, 0, 0,
	0, 27, 0, 35, 78, 78, 78, 0, 85, 83,
	82, 7, 25, 0, 0, 80, 0, 0, 17, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 22, 0,
	0, 78, 67, 0, 76, 46, 47, 78, 84, 0,
	76, 69, 76, 0, 72, 0, 0, 76, 74, 43,
	0, 26, 78, 0, 0, 0, 0, 16, 48, 49,
	50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
	60, 61, 0, 63, 64, 65, 0, 0, 28, 0,
	76, 68, 0, 0, 0, 0, 36, 0, 88, 90,
	0, 70, 79, 0, 42, 0, 40, 41, 0, 75,
	18, 20, 0, 0, 81, 0, 15, 0, 3, 62,
	0, 13, 30, 0, 31, 67, 45, 77, 29, 37,
	89, 87, 0, 38, 39, 71, 73, 0, 0, 0,
	14, 4, 66, 34, 0, 68, 0, 0, 0, 11,
	0, 32, 67, 91, 86, 10, 2, 0, 33,
}
var yyTok1 = [...]int{

	1, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	47, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 4, 3, 3,
	5, 6, 7, 8, 9, 10, 11, 12, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 13, 50,
	14, 15, 16, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 17, 3, 18, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 19, 3, 20,
}
var yyTok2 = [...]int{

	2, 3, 21, 22, 23, 24, 25, 26, 27, 28,
	29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
	39, 40, 41, 42, 43, 44, 45, 46, 48, 49,
}
var yyTok3 = [...]int{
	0,
}

var yyErrorMessages = [...]struct {
	state int
	token int
	msg   string
}{}

//line yaccpar:1

/*	parser for yacc output	*/

var (
	yyDebug        = 0
	yyErrorVerbose = false
)

type yyLexer interface {
	Lex(lval *yySymType) int
	Error(s string)
}

type yyParser interface {
	Parse(yyLexer) int
	Lookahead() int
}

type yyParserImpl struct {
	lval  yySymType
	stack [yyInitialStackSize]yySymType
	char  int
}

func (p *yyParserImpl) Lookahead() int {
	return p.char
}

func yyNewParser() yyParser {
	return &yyParserImpl{}
}

const yyFlag = -1000

func yyTokname(c int) string {
	if c >= 1 && c-1 < len(yyToknames) {
		if yyToknames[c-1] != "" {
			return yyToknames[c-1]
		}
	}
	return __yyfmt__.Sprintf("tok-%v", c)
}

func yyStatname(s int) string {
	if s >= 0 && s < len(yyStatenames) {
		if yyStatenames[s] != "" {
			return yyStatenames[s]
		}
	}
	return __yyfmt__.Sprintf("state-%v", s)
}

func yyErrorMessage(state, lookAhead int) string {
	const TOKSTART = 4

	if !yyErrorVerbose {
		return "syntax error"
	}

	for _, e := range yyErrorMessages {
		if e.state == state && e.token == lookAhead {
			return "syntax error: " + e.msg
		}
	}

	res := "syntax error: unexpected " + yyTokname(lookAhead)

	// To match Bison, suggest at most four expected tokens.
	expected := make([]int, 0, 4)

	// Look for shiftable tokens.
	base := yyPact[state]
	for tok := TOKSTART; tok-1 < len(yyToknames); tok++ {
		if n := base + tok; n >= 0 && n < yyLast && yyChk[yyAct[n]] == tok {
			if len(expected) == cap(expected) {
				return res
			}
			expected = append(expected, tok)
		}
	}

	if yyDef[state] == -2 {
		i := 0
		for yyExca[i] != -1 || yyExca[i+1] != state {
			i += 2
		}

		// Look for tokens that we accept or reduce.
		for i += 2; yyExca[i] >= 0; i += 2 {
			tok := yyExca[i]
			if tok < TOKSTART || yyExca[i+1] == 0 {
				continue
			}
			if len(expected) == cap(expected) {
				return res
			}
			expected = append(expected, tok)
		}

		// If the default action is to accept or reduce, give up.
		if yyExca[i+1] != 0 {
			return res
		}
	}

	for i, tok := range expected {
		if i == 0 {
			res += ", expecting "
		} else {
			res += " or "
		}
		res += yyTokname(tok)
	}
	return res
}

func yylex1(lex yyLexer, lval *yySymType) (char, token int) {
	token = 0
	char = lex.Lex(lval)
	if char <= 0 {
		token = yyTok1[0]
		goto out
	}
	if char < len(yyTok1) {
		token = yyTok1[char]
		goto out
	}
	if char >= yyPrivate {
		if char < yyPrivate+len(yyTok2) {
			token = yyTok2[char-yyPrivate]
			goto out
		}
	}
	for i := 0; i < len(yyTok3); i += 2 {
		token = yyTok3[i+0]
		if token == char {
			token = yyTok3[i+1]
			goto out
		}
	}

out:
	if token == 0 {
		token = yyTok2[1] /* unknown char */
	}
	if yyDebug >= 3 {
		__yyfmt__.Printf("lex %s(%d)\n", yyTokname(token), uint(char))
	}
	return char, token
}

func yyParse(yylex yyLexer) int {
	return yyNewParser().Parse(yylex)
}

func (yyrcvr *yyParserImpl) Parse(yylex yyLexer) int {
	var yyn int
	var yyVAL yySymType
	var yyDollar []yySymType
	_ = yyDollar // silence set and not used
	yyS := yyrcvr.stack[:]

	Nerrs := 0   /* number of errors */
	Errflag := 0 /* error recovery flag */
	yystate := 0
	yyrcvr.char = -1
	yytoken := -1 // yyrcvr.char translated into internal numbering
	defer func() {
		// Make sure we report no lookahead when not parsing.
		yystate = -1
		yyrcvr.char = -1
		yytoken = -1
	}()
	yyp := -1
	goto yystack

ret0:
	return 0

ret1:
	return 1

yystack:
	/* put a state and value onto the stack */
	if yyDebug >= 4 {
		__yyfmt__.Printf("char %v in %v\n", yyTokname(yytoken), yyStatname(yystate))
	}

	yyp++
	if yyp >= len(yyS) {
		nyys := make([]yySymType, len(yyS)*2)
		copy(nyys, yyS)
		yyS = nyys
	}
	yyS[yyp] = yyVAL
	yyS[yyp].yys = yystate

yynewstate:
	yyn = yyPact[yystate]
	if yyn <= yyFlag {
		goto yydefault /* simple state */
	}
	if yyrcvr.char < 0 {
		yyrcvr.char, yytoken = yylex1(yylex, &yyrcvr.lval)
	}
	yyn += yytoken
	if yyn < 0 || yyn >= yyLast {
		goto yydefault
	}
	yyn = yyAct[yyn]
	if yyChk[yyn] == yytoken { /* valid shift */
		yyrcvr.char = -1
		yytoken = -1
		yyVAL = yyrcvr.lval
		yystate = yyn
		if Errflag > 0 {
			Errflag--
		}
		goto yystack
	}

yydefault:
	/* default state action */
	yyn = yyDef[yystate]
	if yyn == -2 {
		if yyrcvr.char < 0 {
			yyrcvr.char, yytoken = yylex1(yylex, &yyrcvr.lval)
		}

		/* look through exception table */
		xi := 0
		for {
			if yyExca[xi+0] == -1 && yyExca[xi+1] == yystate {
				break
			}
			xi += 2
		}
		for xi += 2; ; xi += 2 {
			yyn = yyExca[xi+0]
			if yyn < 0 || yyn == yytoken {
				break
			}
		}
		yyn = yyExca[xi+1]
		if yyn < 0 {
			goto ret0
		}
	}
	if yyn == 0 {
		/* error ... attempt to resume parsing */
		switch Errflag {
		case 0: /* brand new error */
			yylex.Error(yyErrorMessage(yystate, yytoken))
			Nerrs++
			if yyDebug >= 1 {
				__yyfmt__.Printf("%s", yyStatname(yystate))
				__yyfmt__.Printf(" saw %s\n", yyTokname(yytoken))
			}
			fallthrough

		case 1, 2: /* incompletely recovered error ... try again */
			Errflag = 3

			/* find a state where "error" is a legal shift action */
			for yyp >= 0 {
				yyn = yyPact[yyS[yyp].yys] + yyErrCode
				if yyn >= 0 && yyn < yyLast {
					yystate = yyAct[yyn] /* simulate a shift of "error" */
					if yyChk[yystate] == yyErrCode {
						goto yystack
					}
				}

				/* the current p has no shift on "error", pop stack */
				if yyDebug >= 2 {
					__yyfmt__.Printf("error recovery pops state %d\n", yyS[yyp].yys)
				}
				yyp--
			}
			/* there is no state on the stack with an error shift ... abort */
			goto ret1

		case 3: /* no shift yet; clobber input char */
			if yyDebug >= 2 {
				__yyfmt__.Printf("error recovery discards %s\n", yyTokname(yytoken))
			}
			if yytoken == yyEofCode {
				goto ret1
			}
			yyrcvr.char = -1
			yytoken = -1
			goto yynewstate /* try again in the same state */
		}
	}

	/* reduction by production yyn */
	if yyDebug >= 2 {
		__yyfmt__.Printf("reduce %v in:\n\t%v\n", yyn, yyStatname(yystate))
	}

	yynt := yyn
	yypt := yyp
	_ = yypt // guard against "declared and not used"

	yyp -= yyR2[yyn]
	// yyp is now the index of $0. Perform the default action. Iff the
	// reduced production is ε, $1 is possibly out of range.
	if yyp+1 >= len(yyS) {
		nyys := make([]yySymType, len(yyS)*2)
		copy(nyys, yyS)
		yyS = nyys
	}
	yyVAL = yyS[yyp+1]

	/* consult goto table to find next state */
	yyn = yyR1[yyn]
	yyg := yyPgo[yyn]
	yyj := yyg + yyS[yyp].yys + 1

	if yyj >= yyLast {
		yystate = yyAct[yyg]
	} else {
		yystate = yyAct[yyj]
		if yyChk[yystate] != -yyn {
			yystate = yyAct[yyg]
		}
	}
	// dummy call; replaced with literal code
	switch yynt {

	case 1:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line build/parse.y:167
		{
			yylex.(*input).file = &File{Stmt: yyDollar[1].exprs}
			return 0
		}
	case 2:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line build/parse.y:174
		{
			yyVAL.block = CodeBlock{
				Start:      yyDollar[2].pos,
				Statements: yyDollar[3].exprs,
				End:        End{Pos: yyDollar[4].pos},
			}
		}
	case 3:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line build/parse.y:182
		{
			// simple_stmt is never empty
			start, _ := yyDollar[1].exprs[0].Span()
			_, end := yyDollar[1].exprs[len(yyDollar[1].exprs)-1].Span()
			yyVAL.block = CodeBlock{
				Start:      start,
				Statements: yyDollar[1].exprs,
				End:        End{Pos: end},
			}
		}
	case 4:
		yyDollar = yyS[yypt-0 : yypt+1]
		//line build/parse.y:194
		{
			yyVAL.exprs = nil
			yyVAL.lastRule = nil
		}
	case 5:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line build/parse.y:199
		{
			// If this statement follows a comment block,
			// attach the comments to the statement.
			if cb, ok := yyDollar[1].lastRule.(*CommentBlock); ok {
				yyVAL.exprs = append(yyDollar[1].exprs[:len(yyDollar[1].exprs)-1], yyDollar[2].exprs...)
				yyDollar[2].exprs[0].Comment().Before = cb.After
				yyVAL.lastRule = yyDollar[2].exprs[len(yyDollar[2].exprs)-1]
				break
			}

			// Otherwise add to list.
			yyVAL.exprs = append(yyDollar[1].exprs, yyDollar[2].exprs...)
			yyVAL.lastRule = yyDollar[2].exprs[len(yyDollar[2].exprs)-1]

			// Consider this input:
			//
			//	foo()
			//	# bar
			//	baz()
			//
			// If we've just parsed baz(), the # bar is attached to
			// foo() as an After comment. Make it a Before comment
			// for baz() instead.
			if x := yyDollar[1].lastRule; x != nil {
				com := x.Comment()
				// stmt is never empty
				yyDollar[2].exprs[0].Comment().Before = com.After
				com.After = nil
			}
		}
	case 6:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line build/parse.y:230
		{
			// Blank line; sever last rule from future comments.
			yyVAL.exprs = yyDollar[1].exprs
			yyVAL.lastRule = nil
		}
	case 7:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line build/parse.y:236
		{
			yyVAL.exprs = yyDollar[1].exprs
			yyVAL.lastRule = yyDollar[1].lastRule
			if yyVAL.lastRule == nil {
				cb := &CommentBlock{Start: yyDollar[2].pos}
				yyVAL.exprs = append(yyVAL.exprs, cb)
				yyVAL.lastRule = cb
			}
			com := yyVAL.lastRule.Comment()
			com.After = append(com.After, Comment{Start: yyDollar[2].pos, Token: yyDollar[2].tok})
		}
	case 8:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line build/parse.y:250
		{
			yyVAL.exprs = yyDollar[1].exprs
		}
	case 9:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line build/parse.y:254
		{
			yyVAL.exprs = []Expr{yyDollar[1].expr}
		}
	case 10:
		yyDollar = yyS[yypt-7 : yypt+1]
		//line build/parse.y:260
		{
			yyVAL.expr = &FuncDef{
				Start:          yyDollar[1].pos,
				Name:           yyDollar[2].tok,
				ListStart:      yyDollar[3].pos,
				Args:           yyDollar[4].exprs,
				Body:           yyDollar[7].block,
				End:            yyDollar[7].block.End,
				ForceCompact:   forceCompact(yyDollar[3].pos, yyDollar[4].exprs, yyDollar[5].pos),
				ForceMultiLine: forceMultiLine(yyDollar[3].pos, yyDollar[4].exprs, yyDollar[5].pos),
			}
		}
	case 11:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line build/parse.y:273
		{
			yyVAL.expr = &ForLoop{
				Start:    yyDollar[1].pos,
				LoopVars: yyDollar[2].exprs,
				Iterable: yyDollar[4].expr,
				Body:     yyDollar[6].block,
				End:      yyDollar[6].block.End,
			}
		}
	case 12:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line build/parse.y:283
		{
			yyVAL.expr = yyDollar[1].expr
		}
	case 13:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line build/parse.y:289
		{
			yyVAL.expr = &IfElse{
				Start: yyDollar[1].pos,
				Conditions: []Condition{
					Condition{
						If:   yyDollar[2].expr,
						Then: yyDollar[4].block,
					},
				},
				End: yyDollar[4].block.End,
			}
		}
	case 14:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line build/parse.y:302
		{
			block := yyDollar[1].expr.(*IfElse)
			block.Conditions = append(block.Conditions, Condition{
				If:   yyDollar[3].expr,
				Then: yyDollar[5].block,
			})
			block.End = yyDollar[5].block.End
			yyVAL.expr = block
		}
	case 15:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line build/parse.y:312
		{
			block := yyDollar[1].expr.(*IfElse)
			block.Conditions = append(block.Conditions, Condition{
				Then: yyDollar[4].block,
			})
			block.End = yyDollar[4].block.End
			yyVAL.expr = block
		}
	case 18:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line build/parse.y:327
		{
			yyVAL.exprs = append([]Expr{yyDollar[1].expr}, yyDollar[2].exprs...)
			yyVAL.lastRule = yyVAL.exprs[len(yyVAL.exprs)-1]
		}
	case 19:
		yyDollar = yyS[yypt-0 : yypt+1]
		//line build/parse.y:333
		{
			yyVAL.exprs = []Expr{}
		}
	case 20:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line build/parse.y:337
		{
			yyVAL.exprs = append(yyDollar[1].exprs, yyDollar[3].expr)
		}
	case 22:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line build/parse.y:344
		{
			_, end := yyDollar[2].expr.Span()
			yyVAL.expr = &ReturnExpr{
				X:   yyDollar[2].expr,
				End: end,
			}
		}
	case 23:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line build/parse.y:352
		{
			yyVAL.expr = &ReturnExpr{End: yyDollar[1].pos}
		}
	case 24:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line build/parse.y:356
		{
			yyVAL.expr = &PythonBlock{Start: yyDollar[1].pos, Token: yyDollar[1].tok}
		}
	case 28:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line build/parse.y:366
		{
			yyVAL.expr = &DotExpr{
				X:       yyDollar[1].expr,
				Dot:     yyDollar[2].pos,
				NamePos: yyDollar[3].pos,
				Name:    yyDollar[3].tok,
			}
		}
	case 29:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line build/parse.y:375
		{
			yyVAL.expr = &CallExpr{
				X:              &LiteralExpr{Start: yyDollar[1].pos, Token: "load"},
				ListStart:      yyDollar[2].pos,
				List:           yyDollar[3].exprs,
				End:            End{Pos: yyDollar[4].pos},
				ForceCompact:   forceCompact(yyDollar[2].pos, yyDollar[3].exprs, yyDollar[4].pos),
				ForceMultiLine: forceMultiLine(yyDollar[2].pos, yyDollar[3].exprs, yyDollar[4].pos),
			}
		}
	case 30:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line build/parse.y:386
		{
			yyVAL.expr = &CallExpr{
				X:              yyDollar[1].expr,
				ListStart:      yyDollar[2].pos,
				List:           yyDollar[3].exprs,
				End:            End{Pos: yyDollar[4].pos},
				ForceCompact:   forceCompact(yyDollar[2].pos, yyDollar[3].exprs, yyDollar[4].pos),
				ForceMultiLine: forceMultiLine(yyDollar[2].pos, yyDollar[3].exprs, yyDollar[4].pos),
			}
		}
	case 31:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line build/parse.y:397
		{
			yyVAL.expr = &IndexExpr{
				X:          yyDollar[1].expr,
				IndexStart: yyDollar[2].pos,
				Y:          yyDollar[3].expr,
				End:        yyDollar[4].pos,
			}
		}
	case 32:
		yyDollar = yyS[yypt-6 : yypt+1]
		//line build/parse.y:406
		{
			yyVAL.expr = &SliceExpr{
				X:          yyDollar[1].expr,
				SliceStart: yyDollar[2].pos,
				From:       yyDollar[3].expr,
				FirstColon: yyDollar[4].pos,
				To:         yyDollar[5].expr,
				End:        yyDollar[6].pos,
			}
		}
	case 33:
		yyDollar = yyS[yypt-8 : yypt+1]
		//line build/parse.y:417
		{
			yyVAL.expr = &SliceExpr{
				X:           yyDollar[1].expr,
				SliceStart:  yyDollar[2].pos,
				From:        yyDollar[3].expr,
				FirstColon:  yyDollar[4].pos,
				To:          yyDollar[5].expr,
				SecondColon: yyDollar[6].pos,
				Step:        yyDollar[7].expr,
				End:         yyDollar[8].pos,
			}
		}
	case 34:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line build/parse.y:430
		{
			yyVAL.expr = &CallExpr{
				X:         yyDollar[1].expr,
				ListStart: yyDollar[2].pos,
				List: []Expr{
					&ListForExpr{
						Brack: "",
						Start: yyDollar[2].pos,
						X:     yyDollar[3].expr,
						For:   yyDollar[4].forsifs,
						End:   End{Pos: yyDollar[5].pos},
					},
				},
				End: End{Pos: yyDollar[5].pos},
			}
		}
	case 35:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line build/parse.y:447
		{
			if len(yyDollar[1].strings) == 1 {
				yyVAL.expr = yyDollar[1].strings[0]
				break
			}
			yyVAL.expr = yyDollar[1].strings[0]
			for _, x := range yyDollar[1].strings[1:] {
				_, end := yyVAL.expr.Span()
				yyVAL.expr = binary(yyVAL.expr, end, "+", x)
			}
		}
	case 36:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line build/parse.y:459
		{
			yyVAL.expr = &ListExpr{
				Start:          yyDollar[1].pos,
				List:           yyDollar[2].exprs,
				Comma:          yyDollar[2].comma,
				End:            End{Pos: yyDollar[3].pos},
				ForceMultiLine: forceMultiLine(yyDollar[1].pos, yyDollar[2].exprs, yyDollar[3].pos),
			}
		}
	case 37:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line build/parse.y:469
		{
			exprStart, _ := yyDollar[2].expr.Span()
			yyVAL.expr = &ListForExpr{
				Brack:          "[]",
				Start:          yyDollar[1].pos,
				X:              yyDollar[2].expr,
				For:            yyDollar[3].forsifs,
				End:            End{Pos: yyDollar[4].pos},
				ForceMultiLine: yyDollar[1].pos.Line != exprStart.Line,
			}
		}
	case 38:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line build/parse.y:481
		{
			exprStart, _ := yyDollar[2].expr.Span()
			yyVAL.expr = &ListForExpr{
				Brack:          "()",
				Start:          yyDollar[1].pos,
				X:              yyDollar[2].expr,
				For:            yyDollar[3].forsifs,
				End:            End{Pos: yyDollar[4].pos},
				ForceMultiLine: yyDollar[1].pos.Line != exprStart.Line,
			}
		}
	case 39:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line build/parse.y:493
		{
			exprStart, _ := yyDollar[2].expr.Span()
			yyVAL.expr = &ListForExpr{
				Brack:          "{}",
				Start:          yyDollar[1].pos,
				X:              yyDollar[2].expr,
				For:            yyDollar[3].forsifs,
				End:            End{Pos: yyDollar[4].pos},
				ForceMultiLine: yyDollar[1].pos.Line != exprStart.Line,
			}
		}
	case 40:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line build/parse.y:505
		{
			yyVAL.expr = &DictExpr{
				Start:          yyDollar[1].pos,
				List:           yyDollar[2].exprs,
				Comma:          yyDollar[2].comma,
				End:            End{Pos: yyDollar[3].pos},
				ForceMultiLine: forceMultiLine(yyDollar[1].pos, yyDollar[2].exprs, yyDollar[3].pos),
			}
		}
	case 41:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line build/parse.y:515
		{
			yyVAL.expr = &SetExpr{
				Start:          yyDollar[1].pos,
				List:           yyDollar[2].exprs,
				Comma:          yyDollar[2].comma,
				End:            End{Pos: yyDollar[3].pos},
				ForceMultiLine: forceMultiLine(yyDollar[1].pos, yyDollar[2].exprs, yyDollar[3].pos),
			}
		}
	case 42:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line build/parse.y:525
		{
			if len(yyDollar[2].exprs) == 1 && yyDollar[2].comma.Line == 0 {
				// Just a parenthesized expression, not a tuple.
				yyVAL.expr = &ParenExpr{
					Start:          yyDollar[1].pos,
					X:              yyDollar[2].exprs[0],
					End:            End{Pos: yyDollar[3].pos},
					ForceMultiLine: forceMultiLine(yyDollar[1].pos, yyDollar[2].exprs, yyDollar[3].pos),
				}
			} else {
				yyVAL.expr = &TupleExpr{
					Start:          yyDollar[1].pos,
					List:           yyDollar[2].exprs,
					Comma:          yyDollar[2].comma,
					End:            End{Pos: yyDollar[3].pos},
					ForceCompact:   forceCompact(yyDollar[1].pos, yyDollar[2].exprs, yyDollar[3].pos),
					ForceMultiLine: forceMultiLine(yyDollar[1].pos, yyDollar[2].exprs, yyDollar[3].pos),
				}
			}
		}
	case 43:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line build/parse.y:545
		{
			yyVAL.expr = unary(yyDollar[1].pos, yyDollar[1].tok, yyDollar[2].expr)
		}
	case 45:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line build/parse.y:550
		{
			yyVAL.expr = &LambdaExpr{
				Lambda: yyDollar[1].pos,
				Var:    yyDollar[2].exprs,
				Colon:  yyDollar[3].pos,
				Expr:   yyDollar[4].expr,
			}
		}
	case 46:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line build/parse.y:558
		{
			yyVAL.expr = unary(yyDollar[1].pos, yyDollar[1].tok, yyDollar[2].expr)
		}
	case 47:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line build/parse.y:559
		{
			yyVAL.expr = unary(yyDollar[1].pos, yyDollar[1].tok, yyDollar[2].expr)
		}
	case 48:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line build/parse.y:560
		{
			yyVAL.expr = binary(yyDollar[1].expr, yyDollar[2].pos, yyDollar[2].tok, yyDollar[3].expr)
		}
	case 49:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line build/parse.y:561
		{
			yyVAL.expr = binary(yyDollar[1].expr, yyDollar[2].pos, yyDollar[2].tok, yyDollar[3].expr)
		}
	case 50:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line build/parse.y:562
		{
			yyVAL.expr = binary(yyDollar[1].expr, yyDollar[2].pos, yyDollar[2].tok, yyDollar[3].expr)
		}
	case 51:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line build/parse.y:563
		{
			yyVAL.expr = binary(yyDollar[1].expr, yyDollar[2].pos, yyDollar[2].tok, yyDollar[3].expr)
		}
	case 52:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line build/parse.y:564
		{
			yyVAL.expr = binary(yyDollar[1].expr, yyDollar[2].pos, yyDollar[2].tok, yyDollar[3].expr)
		}
	case 53:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line build/parse.y:565
		{
			yyVAL.expr = binary(yyDollar[1].expr, yyDollar[2].pos, yyDollar[2].tok, yyDollar[3].expr)
		}
	case 54:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line build/parse.y:566
		{
			yyVAL.expr = binary(yyDollar[1].expr, yyDollar[2].pos, yyDollar[2].tok, yyDollar[3].expr)
		}
	case 55:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line build/parse.y:567
		{
			yyVAL.expr = binary(yyDollar[1].expr, yyDollar[2].pos, yyDollar[2].tok, yyDollar[3].expr)
		}
	case 56:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line build/parse.y:568
		{
			yyVAL.expr = binary(yyDollar[1].expr, yyDollar[2].pos, yyDollar[2].tok, yyDollar[3].expr)
		}
	case 57:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line build/parse.y:569
		{
			yyVAL.expr = binary(yyDollar[1].expr, yyDollar[2].pos, yyDollar[2].tok, yyDollar[3].expr)
		}
	case 58:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line build/parse.y:570
		{
			yyVAL.expr = binary(yyDollar[1].expr, yyDollar[2].pos, yyDollar[2].tok, yyDollar[3].expr)
		}
	case 59:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line build/parse.y:571
		{
			yyVAL.expr = binary(yyDollar[1].expr, yyDollar[2].pos, yyDollar[2].tok, yyDollar[3].expr)
		}
	case 60:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line build/parse.y:572
		{
			yyVAL.expr = binary(yyDollar[1].expr, yyDollar[2].pos, yyDollar[2].tok, yyDollar[3].expr)
		}
	case 61:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line build/parse.y:573
		{
			yyVAL.expr = binary(yyDollar[1].expr, yyDollar[2].pos, yyDollar[2].tok, yyDollar[3].expr)
		}
	case 62:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line build/parse.y:574
		{
			yyVAL.expr = binary(yyDollar[1].expr, yyDollar[2].pos, "not in", yyDollar[4].expr)
		}
	case 63:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line build/parse.y:575
		{
			yyVAL.expr = binary(yyDollar[1].expr, yyDollar[2].pos, yyDollar[2].tok, yyDollar[3].expr)
		}
	case 64:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line build/parse.y:576
		{
			yyVAL.expr = binary(yyDollar[1].expr, yyDollar[2].pos, yyDollar[2].tok, yyDollar[3].expr)
		}
	case 65:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line build/parse.y:578
		{
			if b, ok := yyDollar[3].expr.(*UnaryExpr); ok && b.Op == "not" {
				yyVAL.expr = binary(yyDollar[1].expr, yyDollar[2].pos, "is not", b.X)
			} else {
				yyVAL.expr = binary(yyDollar[1].expr, yyDollar[2].pos, yyDollar[2].tok, yyDollar[3].expr)
			}
		}
	case 66:
		yyDollar = yyS[yypt-5 : yypt+1]
		//line build/parse.y:586
		{
			yyVAL.expr = &ConditionalExpr{
				Then:      yyDollar[1].expr,
				IfStart:   yyDollar[2].pos,
				Test:      yyDollar[3].expr,
				ElseStart: yyDollar[4].pos,
				Else:      yyDollar[5].expr,
			}
		}
	case 67:
		yyDollar = yyS[yypt-0 : yypt+1]
		//line build/parse.y:597
		{
			yyVAL.expr = nil
		}
	case 69:
		yyDollar = yyS[yypt-0 : yypt+1]
		//line build/parse.y:607
		{
			yyVAL.pos = Position{}
		}
	case 71:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line build/parse.y:613
		{
			yyVAL.expr = &KeyValueExpr{
				Key:   yyDollar[1].expr,
				Colon: yyDollar[2].pos,
				Value: yyDollar[3].expr,
			}
		}
	case 72:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line build/parse.y:623
		{
			yyVAL.exprs = []Expr{yyDollar[1].expr}
		}
	case 73:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line build/parse.y:627
		{
			yyVAL.exprs = append(yyDollar[1].exprs, yyDollar[3].expr)
		}
	case 74:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line build/parse.y:633
		{
			yyVAL.exprs = yyDollar[1].exprs
		}
	case 75:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line build/parse.y:637
		{
			yyVAL.exprs = yyDollar[1].exprs
		}
	case 76:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line build/parse.y:643
		{
			yyVAL.exprs = []Expr{yyDollar[1].expr}
		}
	case 77:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line build/parse.y:647
		{
			yyVAL.exprs = append(yyDollar[1].exprs, yyDollar[3].expr)
		}
	case 78:
		yyDollar = yyS[yypt-0 : yypt+1]
		//line build/parse.y:652
		{
			yyVAL.exprs, yyVAL.comma = nil, Position{}
		}
	case 79:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line build/parse.y:656
		{
			yyVAL.exprs, yyVAL.comma = yyDollar[1].exprs, yyDollar[2].pos
		}
	case 80:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line build/parse.y:662
		{
			yyVAL.exprs = []Expr{yyDollar[1].expr}
		}
	case 81:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line build/parse.y:666
		{
			yyVAL.exprs = append(yyDollar[1].exprs, yyDollar[3].expr)
		}
	case 82:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line build/parse.y:672
		{
			yyVAL.string = &StringExpr{
				Start:       yyDollar[1].pos,
				Value:       yyDollar[1].str,
				TripleQuote: yyDollar[1].triple,
				End:         yyDollar[1].pos.add(yyDollar[1].tok),
				Token:       yyDollar[1].tok,
			}
		}
	case 83:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line build/parse.y:684
		{
			yyVAL.strings = []*StringExpr{yyDollar[1].string}
		}
	case 84:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line build/parse.y:688
		{
			yyVAL.strings = append(yyDollar[1].strings, yyDollar[2].string)
		}
	case 85:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line build/parse.y:694
		{
			yyVAL.expr = &LiteralExpr{Start: yyDollar[1].pos, Token: yyDollar[1].tok}
		}
	case 86:
		yyDollar = yyS[yypt-4 : yypt+1]
		//line build/parse.y:700
		{
			yyVAL.forc = &ForClause{
				For:  yyDollar[1].pos,
				Var:  yyDollar[2].exprs,
				In:   yyDollar[3].pos,
				Expr: yyDollar[4].expr,
			}
		}
	case 87:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line build/parse.y:710
		{
			yyVAL.forifs = &ForClauseWithIfClausesOpt{
				For: yyDollar[1].forc,
				Ifs: yyDollar[2].ifs,
			}
		}
	case 88:
		yyDollar = yyS[yypt-1 : yypt+1]
		//line build/parse.y:719
		{
			yyVAL.forsifs = []*ForClauseWithIfClausesOpt{yyDollar[1].forifs}
		}
	case 89:
		yyDollar = yyS[yypt-2 : yypt+1]
		//line build/parse.y:722
		{
			yyVAL.forsifs = append(yyDollar[1].forsifs, yyDollar[2].forifs)
		}
	case 90:
		yyDollar = yyS[yypt-0 : yypt+1]
		//line build/parse.y:727
		{
			yyVAL.ifs = nil
		}
	case 91:
		yyDollar = yyS[yypt-3 : yypt+1]
		//line build/parse.y:731
		{
			yyVAL.ifs = append(yyDollar[1].ifs, &IfClause{
				If:   yyDollar[2].pos,
				Cond: yyDollar[3].expr,
			})
		}
	}
	goto yystack /* stack new state and value */
}

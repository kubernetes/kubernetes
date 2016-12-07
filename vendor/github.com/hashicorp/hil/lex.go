package hil

import (
	"bytes"
	"fmt"
	"strconv"
	"unicode"
	"unicode/utf8"

	"github.com/hashicorp/hil/ast"
)

//go:generate go tool yacc -p parser lang.y

// The parser expects the lexer to return 0 on EOF.
const lexEOF = 0

// The parser uses the type <prefix>Lex as a lexer.  It must provide
// the methods Lex(*<prefix>SymType) int and Error(string).
type parserLex struct {
	Err   error
	Input string

	mode               parserMode
	interpolationDepth int
	pos                int
	width              int
	col, line          int
	lastLine           int
	astPos             *ast.Pos
}

// parserToken is the token yielded to the parser. The value can be
// determined within the parser type based on the enum value returned
// from Lex.
type parserToken struct {
	Value interface{}
	Pos   ast.Pos
}

// parserMode keeps track of what mode we're in for the parser. We have
// two modes: literal and interpolation. Literal mode is when strings
// don't have to be quoted, and interpolations are defined as ${foo}.
// Interpolation mode means that strings have to be quoted and unquoted
// things are identifiers, such as foo("bar").
type parserMode uint8

const (
	parserModeInvalid parserMode = 0
	parserModeLiteral            = 1 << iota
	parserModeInterpolation
)

// The parser calls this method to get each new token.
func (x *parserLex) Lex(yylval *parserSymType) int {
	// We always start in literal mode, since programs don't start
	// in an interpolation. ex. "foo ${bar}" vs "bar" (and assuming interp.)
	if x.mode == parserModeInvalid {
		x.mode = parserModeLiteral
	}

	// Defer an update to set the proper column/line we read the next token.
	defer func() {
		if yylval.token != nil && yylval.token.Pos.Column == 0 {
			yylval.token.Pos = *x.astPos
		}
	}()

	x.astPos = nil
	return x.lex(yylval)
}

func (x *parserLex) lex(yylval *parserSymType) int {
	switch x.mode {
	case parserModeLiteral:
		return x.lexModeLiteral(yylval)
	case parserModeInterpolation:
		return x.lexModeInterpolation(yylval)
	default:
		x.Error(fmt.Sprintf("Unknown parse mode: %d", x.mode))
		return lexEOF
	}
}

func (x *parserLex) lexModeLiteral(yylval *parserSymType) int {
	for {
		c := x.next()
		if c == lexEOF {
			return lexEOF
		}

		// Are we starting an interpolation?
		if c == '$' && x.peek() == '{' {
			x.next()
			x.interpolationDepth++
			x.mode = parserModeInterpolation
			return PROGRAM_BRACKET_LEFT
		}

		// We're just a normal string that isn't part of any interpolation yet.
		x.backup()
		result, terminated := x.lexString(yylval, x.interpolationDepth > 0)

		// If the string terminated and we're within an interpolation already
		// then that means that we finished a nested string, so pop
		// back out to interpolation mode.
		if terminated && x.interpolationDepth > 0 {
			x.mode = parserModeInterpolation

			// If the string is empty, just skip it. We're still in
			// an interpolation so we do this to avoid empty nodes.
			if yylval.token.Value.(string) == "" {
				return x.lex(yylval)
			}
		}

		return result
	}
}

func (x *parserLex) lexModeInterpolation(yylval *parserSymType) int {
	for {
		c := x.next()
		if c == lexEOF {
			return lexEOF
		}

		// Ignore all whitespace
		if unicode.IsSpace(c) {
			continue
		}

		// If we see a double quote then we're lexing a string since
		// we're in interpolation mode.
		if c == '"' {
			result, terminated := x.lexString(yylval, true)
			if !terminated {
				// The string didn't end, which means that we're in the
				// middle of starting another interpolation.
				x.mode = parserModeLiteral

				// If the string is empty and we're starting an interpolation,
				// then just skip it to avoid empty string AST nodes
				if yylval.token.Value.(string) == "" {
					return x.lex(yylval)
				}
			}

			return result
		}

		// If we are seeing a number, it is the start of a number. Lex it.
		if c >= '0' && c <= '9' {
			x.backup()
			return x.lexNumber(yylval)
		}

		switch c {
		case '}':
			// '}' means we ended the interpolation. Pop back into
			// literal mode and reduce our interpolation depth.
			x.interpolationDepth--
			x.mode = parserModeLiteral
			return PROGRAM_BRACKET_RIGHT
		case '(':
			return PAREN_LEFT
		case ')':
			return PAREN_RIGHT
		case '[':
			return SQUARE_BRACKET_LEFT
		case ']':
			return SQUARE_BRACKET_RIGHT
		case ',':
			return COMMA
		case '+':
			yylval.token = &parserToken{Value: ast.ArithmeticOpAdd}
			return ARITH_OP
		case '-':
			yylval.token = &parserToken{Value: ast.ArithmeticOpSub}
			return ARITH_OP
		case '*':
			yylval.token = &parserToken{Value: ast.ArithmeticOpMul}
			return ARITH_OP
		case '/':
			yylval.token = &parserToken{Value: ast.ArithmeticOpDiv}
			return ARITH_OP
		case '%':
			yylval.token = &parserToken{Value: ast.ArithmeticOpMod}
			return ARITH_OP
		default:
			x.backup()
			return x.lexId(yylval)
		}
	}
}

func (x *parserLex) lexId(yylval *parserSymType) int {
	var b bytes.Buffer
	var last rune
	for {
		c := x.next()
		if c == lexEOF {
			break
		}

		// We only allow * after a '.' for resource splast: type.name.*.id
		// Otherwise, its probably multiplication.
		if c == '*' && last != '.' {
			x.backup()
			break
		}

		// If this isn't a character we want in an ID, return out.
		// One day we should make this a regexp.
		if c != '_' &&
			c != '-' &&
			c != '.' &&
			c != '*' &&
			!unicode.IsLetter(c) &&
			!unicode.IsNumber(c) {
			x.backup()
			break
		}

		if _, err := b.WriteRune(c); err != nil {
			x.Error(err.Error())
			return lexEOF
		}

		last = c
	}

	yylval.token = &parserToken{Value: b.String()}
	return IDENTIFIER
}

// lexNumber lexes out a number: an integer or a float.
func (x *parserLex) lexNumber(yylval *parserSymType) int {
	var b bytes.Buffer
	gotPeriod := false
	for {
		c := x.next()
		if c == lexEOF {
			break
		}

		// If we see a period, we might be getting a float..
		if c == '.' {
			// If we've already seen a period, then ignore it, and
			// exit. This will probably result in a syntax error later.
			if gotPeriod {
				x.backup()
				break
			}

			gotPeriod = true
		} else if c < '0' || c > '9' {
			// If we're not seeing a number, then also exit.
			x.backup()
			break
		}

		if _, err := b.WriteRune(c); err != nil {
			x.Error(fmt.Sprintf("internal error: %s", err))
			return lexEOF
		}
	}

	// If we didn't see a period, it is an int
	if !gotPeriod {
		v, err := strconv.ParseInt(b.String(), 0, 0)
		if err != nil {
			x.Error(fmt.Sprintf("expected number: %s", err))
			return lexEOF
		}

		yylval.token = &parserToken{Value: int(v)}
		return INTEGER
	}

	// If we did see a period, it is a float
	f, err := strconv.ParseFloat(b.String(), 64)
	if err != nil {
		x.Error(fmt.Sprintf("expected float: %s", err))
		return lexEOF
	}

	yylval.token = &parserToken{Value: f}
	return FLOAT
}

func (x *parserLex) lexString(yylval *parserSymType, quoted bool) (int, bool) {
	var b bytes.Buffer
	terminated := false
	for {
		c := x.next()
		if c == lexEOF {
			if quoted {
				x.Error("unterminated string")
			}

			break
		}

		// Behavior is a bit different if we're lexing within a quoted string.
		if quoted {
			// If its a double quote, we've reached the end of the string
			if c == '"' {
				terminated = true
				break
			}

			// Let's check to see if we're escaping anything.
			if c == '\\' {
				switch n := x.next(); n {
				case '\\', '"':
					c = n
				case 'n':
					c = '\n'
				default:
					x.backup()
				}
			}
		}

		// If we hit a dollar sign, then check if we're starting
		// another interpolation. If so, then we're done.
		if c == '$' {
			n := x.peek()

			// If it is '{', then we're starting another interpolation
			if n == '{' {
				x.backup()
				break
			}

			// If it is '$', then we're escaping a dollar sign
			if n == '$' {
				x.next()
			}
		}

		if _, err := b.WriteRune(c); err != nil {
			x.Error(err.Error())
			return lexEOF, false
		}
	}

	yylval.token = &parserToken{Value: b.String()}
	return STRING, terminated
}

// Return the next rune for the lexer.
func (x *parserLex) next() rune {
	if int(x.pos) >= len(x.Input) {
		x.width = 0
		return lexEOF
	}

	r, w := utf8.DecodeRuneInString(x.Input[x.pos:])
	x.width = w
	x.pos += x.width

	if x.line == 0 {
		x.line = 1
		x.col = 1
	} else {
		x.col += 1
	}

	if r == '\n' {
		x.lastLine = x.col
		x.line += 1
		x.col = 1
	}

	if x.astPos == nil {
		x.astPos = &ast.Pos{Column: x.col, Line: x.line}
	}

	return r
}

// peek returns but does not consume the next rune in the input
func (x *parserLex) peek() rune {
	r := x.next()
	x.backup()
	return r
}

// backup steps back one rune. Can only be called once per next.
func (x *parserLex) backup() {
	x.pos -= x.width
	x.col -= 1

	// If we are at column 0, we're backing up across a line boundary
	// so we need to be careful to get the proper value.
	if x.col == 0 {
		x.col = x.lastLine
		x.line -= 1
	}
}

// The parser calls this method on a parse error.
func (x *parserLex) Error(s string) {
	x.Err = fmt.Errorf("parse error: %s", s)
}

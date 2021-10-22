package jmespath

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"unicode/utf8"
)

type token struct {
	tokenType tokType
	value     string
	position  int
	length    int
}

type tokType int

const eof = -1

// Lexer contains information about the expression being tokenized.
type Lexer struct {
	expression string       // The expression provided by the user.
	currentPos int          // The current position in the string.
	lastWidth  int          // The width of the current rune.  This
	buf        bytes.Buffer // Internal buffer used for building up values.
}

// SyntaxError is the main error used whenever a lexing or parsing error occurs.
type SyntaxError struct {
	msg        string // Error message displayed to user
	Expression string // Expression that generated a SyntaxError
	Offset     int    // The location in the string where the error occurred
}

func (e SyntaxError) Error() string {
	// In the future, it would be good to underline the specific
	// location where the error occurred.
	return "SyntaxError: " + e.msg
}

// HighlightLocation will show where the syntax error occurred.
// It will place a "^" character on a line below the expression
// at the point where the syntax error occurred.
func (e SyntaxError) HighlightLocation() string {
	return e.Expression + "\n" + strings.Repeat(" ", e.Offset) + "^"
}

//go:generate stringer -type=tokType
const (
	tUnknown tokType = iota
	tStar
	tDot
	tFilter
	tFlatten
	tLparen
	tRparen
	tLbracket
	tRbracket
	tLbrace
	tRbrace
	tOr
	tPipe
	tNumber
	tUnquotedIdentifier
	tQuotedIdentifier
	tComma
	tColon
	tLT
	tLTE
	tGT
	tGTE
	tEQ
	tNE
	tJSONLiteral
	tStringLiteral
	tCurrent
	tExpref
	tAnd
	tNot
	tEOF
)

var basicTokens = map[rune]tokType{
	'.': tDot,
	'*': tStar,
	',': tComma,
	':': tColon,
	'{': tLbrace,
	'}': tRbrace,
	']': tRbracket, // tLbracket not included because it could be "[]"
	'(': tLparen,
	')': tRparen,
	'@': tCurrent,
}

// Bit mask for [a-zA-Z_] shifted down 64 bits to fit in a single uint64.
// When using this bitmask just be sure to shift the rune down 64 bits
// before checking against identifierStartBits.
const identifierStartBits uint64 = 576460745995190270

// Bit mask for [a-zA-Z0-9], 128 bits -> 2 uint64s.
var identifierTrailingBits = [2]uint64{287948901175001088, 576460745995190270}

var whiteSpace = map[rune]bool{
	' ': true, '\t': true, '\n': true, '\r': true,
}

func (t token) String() string {
	return fmt.Sprintf("Token{%+v, %s, %d, %d}",
		t.tokenType, t.value, t.position, t.length)
}

// NewLexer creates a new JMESPath lexer.
func NewLexer() *Lexer {
	lexer := Lexer{}
	return &lexer
}

func (lexer *Lexer) next() rune {
	if lexer.currentPos >= len(lexer.expression) {
		lexer.lastWidth = 0
		return eof
	}
	r, w := utf8.DecodeRuneInString(lexer.expression[lexer.currentPos:])
	lexer.lastWidth = w
	lexer.currentPos += w
	return r
}

func (lexer *Lexer) back() {
	lexer.currentPos -= lexer.lastWidth
}

func (lexer *Lexer) peek() rune {
	t := lexer.next()
	lexer.back()
	return t
}

// tokenize takes an expression and returns corresponding tokens.
func (lexer *Lexer) tokenize(expression string) ([]token, error) {
	var tokens []token
	lexer.expression = expression
	lexer.currentPos = 0
	lexer.lastWidth = 0
loop:
	for {
		r := lexer.next()
		if identifierStartBits&(1<<(uint64(r)-64)) > 0 {
			t := lexer.consumeUnquotedIdentifier()
			tokens = append(tokens, t)
		} else if val, ok := basicTokens[r]; ok {
			// Basic single char token.
			t := token{
				tokenType: val,
				value:     string(r),
				position:  lexer.currentPos - lexer.lastWidth,
				length:    1,
			}
			tokens = append(tokens, t)
		} else if r == '-' || (r >= '0' && r <= '9') {
			t := lexer.consumeNumber()
			tokens = append(tokens, t)
		} else if r == '[' {
			t := lexer.consumeLBracket()
			tokens = append(tokens, t)
		} else if r == '"' {
			t, err := lexer.consumeQuotedIdentifier()
			if err != nil {
				return tokens, err
			}
			tokens = append(tokens, t)
		} else if r == '\'' {
			t, err := lexer.consumeRawStringLiteral()
			if err != nil {
				return tokens, err
			}
			tokens = append(tokens, t)
		} else if r == '`' {
			t, err := lexer.consumeLiteral()
			if err != nil {
				return tokens, err
			}
			tokens = append(tokens, t)
		} else if r == '|' {
			t := lexer.matchOrElse(r, '|', tOr, tPipe)
			tokens = append(tokens, t)
		} else if r == '<' {
			t := lexer.matchOrElse(r, '=', tLTE, tLT)
			tokens = append(tokens, t)
		} else if r == '>' {
			t := lexer.matchOrElse(r, '=', tGTE, tGT)
			tokens = append(tokens, t)
		} else if r == '!' {
			t := lexer.matchOrElse(r, '=', tNE, tNot)
			tokens = append(tokens, t)
		} else if r == '=' {
			t := lexer.matchOrElse(r, '=', tEQ, tUnknown)
			tokens = append(tokens, t)
		} else if r == '&' {
			t := lexer.matchOrElse(r, '&', tAnd, tExpref)
			tokens = append(tokens, t)
		} else if r == eof {
			break loop
		} else if _, ok := whiteSpace[r]; ok {
			// Ignore whitespace
		} else {
			return tokens, lexer.syntaxError(fmt.Sprintf("Unknown char: %s", strconv.QuoteRuneToASCII(r)))
		}
	}
	tokens = append(tokens, token{tEOF, "", len(lexer.expression), 0})
	return tokens, nil
}

// Consume characters until the ending rune "r" is reached.
// If the end of the expression is reached before seeing the
// terminating rune "r", then an error is returned.
// If no error occurs then the matching substring is returned.
// The returned string will not include the ending rune.
func (lexer *Lexer) consumeUntil(end rune) (string, error) {
	start := lexer.currentPos
	current := lexer.next()
	for current != end && current != eof {
		if current == '\\' && lexer.peek() != eof {
			lexer.next()
		}
		current = lexer.next()
	}
	if lexer.lastWidth == 0 {
		// Then we hit an EOF so we never reached the closing
		// delimiter.
		return "", SyntaxError{
			msg:        "Unclosed delimiter: " + string(end),
			Expression: lexer.expression,
			Offset:     len(lexer.expression),
		}
	}
	return lexer.expression[start : lexer.currentPos-lexer.lastWidth], nil
}

func (lexer *Lexer) consumeLiteral() (token, error) {
	start := lexer.currentPos
	value, err := lexer.consumeUntil('`')
	if err != nil {
		return token{}, err
	}
	value = strings.Replace(value, "\\`", "`", -1)
	return token{
		tokenType: tJSONLiteral,
		value:     value,
		position:  start,
		length:    len(value),
	}, nil
}

func (lexer *Lexer) consumeRawStringLiteral() (token, error) {
	start := lexer.currentPos
	currentIndex := start
	current := lexer.next()
	for current != '\'' && lexer.peek() != eof {
		if current == '\\' && lexer.peek() == '\'' {
			chunk := lexer.expression[currentIndex : lexer.currentPos-1]
			lexer.buf.WriteString(chunk)
			lexer.buf.WriteString("'")
			lexer.next()
			currentIndex = lexer.currentPos
		}
		current = lexer.next()
	}
	if lexer.lastWidth == 0 {
		// Then we hit an EOF so we never reached the closing
		// delimiter.
		return token{}, SyntaxError{
			msg:        "Unclosed delimiter: '",
			Expression: lexer.expression,
			Offset:     len(lexer.expression),
		}
	}
	if currentIndex < lexer.currentPos {
		lexer.buf.WriteString(lexer.expression[currentIndex : lexer.currentPos-1])
	}
	value := lexer.buf.String()
	// Reset the buffer so it can reused again.
	lexer.buf.Reset()
	return token{
		tokenType: tStringLiteral,
		value:     value,
		position:  start,
		length:    len(value),
	}, nil
}

func (lexer *Lexer) syntaxError(msg string) SyntaxError {
	return SyntaxError{
		msg:        msg,
		Expression: lexer.expression,
		Offset:     lexer.currentPos - 1,
	}
}

// Checks for a two char token, otherwise matches a single character
// token. This is used whenever a two char token overlaps a single
// char token, e.g. "||" -> tPipe, "|" -> tOr.
func (lexer *Lexer) matchOrElse(first rune, second rune, matchedType tokType, singleCharType tokType) token {
	start := lexer.currentPos - lexer.lastWidth
	nextRune := lexer.next()
	var t token
	if nextRune == second {
		t = token{
			tokenType: matchedType,
			value:     string(first) + string(second),
			position:  start,
			length:    2,
		}
	} else {
		lexer.back()
		t = token{
			tokenType: singleCharType,
			value:     string(first),
			position:  start,
			length:    1,
		}
	}
	return t
}

func (lexer *Lexer) consumeLBracket() token {
	// There's three options here:
	// 1. A filter expression "[?"
	// 2. A flatten operator "[]"
	// 3. A bare rbracket "["
	start := lexer.currentPos - lexer.lastWidth
	nextRune := lexer.next()
	var t token
	if nextRune == '?' {
		t = token{
			tokenType: tFilter,
			value:     "[?",
			position:  start,
			length:    2,
		}
	} else if nextRune == ']' {
		t = token{
			tokenType: tFlatten,
			value:     "[]",
			position:  start,
			length:    2,
		}
	} else {
		t = token{
			tokenType: tLbracket,
			value:     "[",
			position:  start,
			length:    1,
		}
		lexer.back()
	}
	return t
}

func (lexer *Lexer) consumeQuotedIdentifier() (token, error) {
	start := lexer.currentPos
	value, err := lexer.consumeUntil('"')
	if err != nil {
		return token{}, err
	}
	var decoded string
	asJSON := []byte("\"" + value + "\"")
	if err := json.Unmarshal([]byte(asJSON), &decoded); err != nil {
		return token{}, err
	}
	return token{
		tokenType: tQuotedIdentifier,
		value:     decoded,
		position:  start - 1,
		length:    len(decoded),
	}, nil
}

func (lexer *Lexer) consumeUnquotedIdentifier() token {
	// Consume runes until we reach the end of an unquoted
	// identifier.
	start := lexer.currentPos - lexer.lastWidth
	for {
		r := lexer.next()
		if r < 0 || r > 128 || identifierTrailingBits[uint64(r)/64]&(1<<(uint64(r)%64)) == 0 {
			lexer.back()
			break
		}
	}
	value := lexer.expression[start:lexer.currentPos]
	return token{
		tokenType: tUnquotedIdentifier,
		value:     value,
		position:  start,
		length:    lexer.currentPos - start,
	}
}

func (lexer *Lexer) consumeNumber() token {
	// Consume runes until we reach something that's not a number.
	start := lexer.currentPos - lexer.lastWidth
	for {
		r := lexer.next()
		if r < '0' || r > '9' {
			lexer.back()
			break
		}
	}
	value := lexer.expression[start:lexer.currentPos]
	return token{
		tokenType: tNumber,
		value:     value,
		position:  start,
		length:    lexer.currentPos - start,
	}
}

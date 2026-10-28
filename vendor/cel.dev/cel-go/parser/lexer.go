// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package parser

import (
	"cel.dev/cel-go/common/runes"
)

type tokenKind int

const (
	tokError tokenKind = iota
	tokEnd
	tokWhitespace
	tokComment

	// Keywords
	tokNull
	tokFalse
	tokTrue
	tokIn
	tokReservedWord

	// Literals
	tokInt
	tokUint
	tokFloat
	tokString
	tokBytes

	// Identifiers
	tokIdent

	// Delimiters
	tokLeftBracket  // [
	tokRightBracket // ]
	tokLeftBrace    // {
	tokRightBrace   // }
	tokLeftParen    // (
	tokRightParen   // )

	// Operators
	tokDot              // .
	tokComma            // ,
	tokMinus            // -
	tokPlus             // +
	tokAsterisk         // *
	tokSlash            // /
	tokPercent          // %
	tokQuestion         // ?
	tokColon            // :
	tokExclamation      // !
	tokEqual            // =
	tokEqualEqual       // ==
	tokExclamationEqual // !=
	tokLess             // <
	tokLessEqual        // <=
	tokGreater          // >
	tokGreaterEqual     // >=
	tokLogicalAnd       // &&
	tokLogicalOr        // ||
)

func (t tokenKind) String() string {
	switch t {
	case tokError:
		return "error"
	case tokEnd:
		return "end"
	case tokWhitespace:
		return "whitespace"
	case tokComment:
		return "comment"
	case tokNull:
		return "null"
	case tokFalse:
		return "false"
	case tokTrue:
		return "true"
	case tokIn:
		return "in"
	case tokReservedWord:
		return "reserved_word"
	case tokInt:
		return "int"
	case tokUint:
		return "uint"
	case tokFloat:
		return "float"
	case tokString:
		return "string"
	case tokBytes:
		return "bytes"
	case tokIdent:
		return "ident"
	case tokLeftBracket:
		return "["
	case tokRightBracket:
		return "]"
	case tokLeftBrace:
		return "{"
	case tokRightBrace:
		return "}"
	case tokLeftParen:
		return "("
	case tokRightParen:
		return ")"
	case tokDot:
		return "."
	case tokComma:
		return ","
	case tokMinus:
		return "-"
	case tokPlus:
		return "+"
	case tokAsterisk:
		return "*"
	case tokSlash:
		return "/"
	case tokPercent:
		return "%"
	case tokQuestion:
		return "?"
	case tokColon:
		return ":"
	case tokExclamation:
		return "!"
	case tokEqual:
		return "="
	case tokEqualEqual:
		return "=="
	case tokExclamationEqual:
		return "!="
	case tokLess:
		return "<"
	case tokLessEqual:
		return "<="
	case tokGreater:
		return ">"
	case tokGreaterEqual:
		return ">="
	case tokLogicalAnd:
		return "&&"
	case tokLogicalOr:
		return "||"
	default:
		return "<unknown>"
	}
}

type token struct {
	kind  tokenKind
	start int32
	end   int32
}

type lexerError struct {
	start   int32
	end     int32
	message string
}

var keywords = map[string]tokenKind{
	"false":     tokFalse,
	"true":      tokTrue,
	"null":      tokNull,
	"in":        tokIn,
	"as":        tokReservedWord,
	"break":     tokReservedWord,
	"const":     tokReservedWord,
	"continue":  tokReservedWord,
	"else":      tokReservedWord,
	"for":       tokReservedWord,
	"function":  tokReservedWord,
	"if":        tokReservedWord,
	"import":    tokReservedWord,
	"let":       tokReservedWord,
	"loop":      tokReservedWord,
	"package":   tokReservedWord,
	"namespace": tokReservedWord,
	"return":    tokReservedWord,
	"var":       tokReservedWord,
	"void":      tokReservedWord,
	"while":     tokReservedWord,
}

func isIdentTrailing(r rune) bool {
	return r <= 0x7f && ((r >= '0' && r <= '9') || (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || r == '_')
}

func isDigit(r rune) bool {
	return r >= '0' && r <= '9'
}

func isHexDigit(r rune) bool {
	return (r >= '0' && r <= '9') || (r >= 'a' && r <= 'f') || (r >= 'A' && r <= 'F')
}

func isAlpha(r rune) bool {
	return (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z')
}

func isPlusOrMinus(r rune) bool {
	return r == '+' || r == '-'
}

// lexer performs fast tokenization of CEL expression source code.
type lexer struct {
	content runes.Buffer
	length  int32
	pos     int32
	err     lexerError
}

func newLexer(content runes.Buffer) *lexer {
	return &lexer{
		content: content,
		length:  int32(content.Len()),
		pos:     0,
	}
}

func (l *lexer) SavePosition() int32 {
	return l.pos
}

func (l *lexer) RestorePosition(pos int32) {
	l.pos = pos
	l.err = lexerError{}
}

func (l *lexer) GetError() lexerError {
	return l.err
}

func (l *lexer) GetPosition() int32 {
	return l.pos
}

func (l *lexer) makeToken(kind tokenKind, start, end int32) token {
	return token{kind: kind, start: start, end: end}
}

func (l *lexer) setError(start, end int32, msg string) token {
	l.err = lexerError{start: start, end: end, message: msg}
	return token{kind: tokError, start: start, end: end}
}

func (l *lexer) advance(n int32) {
	l.pos += n
}

func (l *lexer) match(c rune) bool {
	return l.pos < l.length && l.content.Get(int(l.pos)) == c
}

func (l *lexer) matchIgnoreCase(c rune) bool {
	if l.pos >= l.length {
		return false
	}
	cp := l.content.Get(int(l.pos))
	if cp <= 0x7f && c <= 0x7f {
		cpLower := cp
		if cp >= 'A' && cp <= 'Z' {
			cpLower += 'a' - 'A'
		}
		cLower := c
		if c >= 'A' && c <= 'Z' {
			cLower += 'a' - 'A'
		}
		return cpLower == cLower
	}
	return cp == c
}

func (l *lexer) consume(c rune) bool {
	if l.match(c) {
		l.advance(1)
		return true
	}
	return false
}

func (l *lexer) consumeIgnoreCase(c rune) bool {
	if l.matchIgnoreCase(c) {
		l.advance(1)
		return true
	}
	return false
}

func (l *lexer) consumeIf(predicate func(rune) bool) bool {
	if l.pos < l.length && predicate(l.content.Get(int(l.pos))) {
		l.advance(1)
		return true
	}
	return false
}

func (l *lexer) consumeLine() {
	for l.pos < l.length {
		if l.content.Get(int(l.pos)) == '\n' {
			l.advance(1)
			return
		}
		l.advance(1)
	}
}

func (l *lexer) consumeWhitespace() {
	for l.pos < l.length {
		c := l.content.Get(int(l.pos))
		switch c {
		case '\f', '\n', ' ', '\r', '\t':
			l.advance(1)
		default:
			return
		}
	}
}

func (l *lexer) consumeDigits() bool {
	advanced := false
	for l.pos < l.length {
		c := l.content.Get(int(l.pos))
		if !isDigit(c) {
			break
		}
		l.advance(1)
		advanced = true
	}
	return advanced
}

func (l *lexer) consumeHexDigits() bool {
	advanced := false
	for l.pos < l.length {
		c := l.content.Get(int(l.pos))
		if !isHexDigit(c) {
			break
		}
		l.advance(1)
		advanced = true
	}
	return advanced
}

func (l *lexer) consumeIntegralSuffix() tokenKind {
	if l.consumeIgnoreCase('u') {
		return tokUint
	}
	return tokInt
}

func (l *lexer) consumeUntilAfter(c rune, isRaw bool) bool {
	pos := l.pos
	escaped := false
	for pos < l.length {
		cc := l.content.Get(int(pos))
		if cc == '\n' || cc == '\r' {
			l.pos = pos
			return false
		}
		if !isRaw && cc == '\\' {
			escaped = !escaped
		} else {
			if cc == c && !escaped {
				l.pos = pos + 1
				return true
			}
			escaped = false
		}
		pos++
	}
	l.pos = l.length
	return false
}

func (l *lexer) consumeUntilAfterTriple(quote rune, isRaw bool) bool {
	pos := l.pos
	escaped := false
	for pos < l.length {
		cc := l.content.Get(int(pos))
		if !isRaw && cc == '\\' {
			escaped = !escaped
		} else {
			if !escaped && pos+3 <= l.length {
				if l.content.Get(int(pos)) == quote &&
					l.content.Get(int(pos+1)) == quote &&
					l.content.Get(int(pos+2)) == quote {
					l.pos = pos + 3
					return true
				}
			}
			escaped = false
		}
		pos++
	}
	l.pos = l.length
	return false
}

func (l *lexer) consumeQuotedIdent() token {
	start := l.pos
	l.advance(1)
	if !l.consumeUntilAfter('`', true) {
		return l.setError(start, l.pos, "unterminated quoted identifier")
	}
	return l.makeToken(tokIdent, start, l.pos)
}

func (l *lexer) consumeStringLiteral(start int32, quote rune, isBytes, isRaw bool) token {
	l.advance(1)
	if l.pos+2 <= l.length && l.content.Get(int(l.pos)) == quote && l.content.Get(int(l.pos+1)) == quote {
		l.advance(2)
		if !l.consumeUntilAfterTriple(quote, isRaw) {
			msg := "unterminated string literal"
			if isBytes {
				msg = "unterminated bytes literal"
			}
			return l.setError(start, l.pos, msg)
		}
		kind := tokString
		if isBytes {
			kind = tokBytes
		}
		return l.makeToken(kind, start, l.pos)
	}
	if !l.consumeUntilAfter(quote, isRaw) {
		msg := "unterminated string literal"
		if isBytes {
			msg = "unterminated bytes literal"
		}
		return l.setError(start, l.pos, msg)
	}
	kind := tokString
	if isBytes {
		kind = tokBytes
	}
	return l.makeToken(kind, start, l.pos)
}

func (l *lexer) consumePrefixedStringLiteral() (token, bool) {
	start := l.pos
	if l.pos >= l.length {
		return token{}, false
	}
	c := l.content.Get(int(l.pos))
	isBytes := (c == 'b' || c == 'B')
	isRaw := (c == 'r' || c == 'R')
	lookahead := int32(1)
	if l.pos+1 < l.length {
		c2 := l.content.Get(int(l.pos + 1))
		if (isBytes && (c2 == 'r' || c2 == 'R')) || (!isBytes && (c2 == 'b' || c2 == 'B')) {
			isBytes = true
			isRaw = true
			lookahead = 2
		}
	}
	if l.pos+lookahead < l.length {
		quote := l.content.Get(int(l.pos + lookahead))
		if quote == '"' || quote == '\'' {
			l.advance(lookahead)
			return l.consumeStringLiteral(start, quote, isBytes, isRaw), true
		}
	}
	return token{}, false
}

func (l *lexer) consumeNumericLiteral() token {
	start := l.pos
	c := l.content.Get(int(l.pos))
	floatingPoint := false
	if c == '.' {
		floatingPoint = true
		l.advance(1)
		if !l.consumeDigits() {
			return l.setError(start, l.pos, "floating point literal missing digits after decimal separator")
		}
	} else {
		l.advance(1)
		if c == '0' {
			if l.consumeIgnoreCase('x') {
				if !l.consumeHexDigits() {
					return l.setError(start, l.pos, "integral literal missing digits after hexadecimal separator")
				}
				tokType := l.consumeIntegralSuffix()
				return l.makeToken(tokType, start, l.pos)
			}
		}
		_ = l.consumeDigits()
		if l.pos < l.length && l.content.Get(int(l.pos)) == '.' &&
			l.pos+1 < l.length && isDigit(l.content.Get(int(l.pos+1))) {
			floatingPoint = true
			l.advance(1)
			_ = l.consumeDigits()
		}
	}
	if l.consumeIgnoreCase('e') {
		floatingPoint = true
		_ = l.consumeIf(isPlusOrMinus)
		if !l.consumeDigits() {
			return l.setError(start, l.pos, "floating point literal missing digits after exponent separator")
		}
	}
	var tokType tokenKind
	if floatingPoint {
		tokType = tokFloat
	} else {
		tokType = l.consumeIntegralSuffix()
	}
	return l.makeToken(tokType, start, l.pos)
}

func (l *lexer) consumeIdent() token {
	start := l.pos
	for l.pos < l.length {
		c := l.content.Get(int(l.pos))
		if !isIdentTrailing(c) {
			break
		}
		l.advance(1)
	}
	end := l.pos
	word := l.content.Slice(int(start), int(end))
	if kind, ok := keywords[word]; ok {
		return l.makeToken(kind, start, end)
	}
	return l.makeToken(tokIdent, start, end)
}

// Lex scans and returns the next token from the source.
func (l *lexer) Lex() token {
	start := l.pos
	if l.pos >= l.length {
		return l.makeToken(tokEnd, start, start)
	}
	c := l.content.Get(int(l.pos))
	switch c {
	case '\f', '\t', '\r', '\n', ' ':
		l.consumeWhitespace()
		return l.makeToken(tokWhitespace, start, l.pos)
	case '.':
		if l.pos+1 < l.length && isDigit(l.content.Get(int(l.pos+1))) {
			return l.consumeNumericLiteral()
		}
		l.advance(1)
		return l.makeToken(tokDot, start, l.pos)
	case ',':
		l.advance(1)
		return l.makeToken(tokComma, start, l.pos)
	case '!':
		l.advance(1)
		if l.consume('=') {
			return l.makeToken(tokExclamationEqual, start, l.pos)
		}
		return l.makeToken(tokExclamation, start, l.pos)
	case '?':
		l.advance(1)
		return l.makeToken(tokQuestion, start, l.pos)
	case '(':
		l.advance(1)
		return l.makeToken(tokLeftParen, start, l.pos)
	case ')':
		l.advance(1)
		return l.makeToken(tokRightParen, start, l.pos)
	case '{':
		l.advance(1)
		return l.makeToken(tokLeftBrace, start, l.pos)
	case '}':
		l.advance(1)
		return l.makeToken(tokRightBrace, start, l.pos)
	case '[':
		l.advance(1)
		return l.makeToken(tokLeftBracket, start, l.pos)
	case ']':
		l.advance(1)
		return l.makeToken(tokRightBracket, start, l.pos)
	case '=':
		l.advance(1)
		if l.consume('=') {
			return l.makeToken(tokEqualEqual, start, l.pos)
		}
		return l.makeToken(tokEqual, start, l.pos)
	case '<':
		l.advance(1)
		if l.consume('=') {
			return l.makeToken(tokLessEqual, start, l.pos)
		}
		return l.makeToken(tokLess, start, l.pos)
	case '>':
		l.advance(1)
		if l.consume('=') {
			return l.makeToken(tokGreaterEqual, start, l.pos)
		}
		return l.makeToken(tokGreater, start, l.pos)
	case ':':
		l.advance(1)
		return l.makeToken(tokColon, start, l.pos)
	case '%':
		l.advance(1)
		return l.makeToken(tokPercent, start, l.pos)
	case '+':
		l.advance(1)
		return l.makeToken(tokPlus, start, l.pos)
	case '-':
		l.advance(1)
		return l.makeToken(tokMinus, start, l.pos)
	case '*':
		l.advance(1)
		return l.makeToken(tokAsterisk, start, l.pos)
	case '/':
		l.advance(1)
		if l.consume('/') {
			l.consumeLine()
			return l.makeToken(tokComment, start, l.pos)
		}
		return l.makeToken(tokSlash, start, l.pos)
	case '&':
		l.advance(1)
		if l.consume('&') {
			return l.makeToken(tokLogicalAnd, start, l.pos)
		}
		return l.setError(start, l.pos, "unexpected single '&', expected '&&'")
	case '|':
		l.advance(1)
		if l.consume('|') {
			return l.makeToken(tokLogicalOr, start, l.pos)
		}
		return l.setError(start, l.pos, "unexpected single '|', expected '||'")
	case '_':
		return l.consumeIdent()
	case '`':
		return l.consumeQuotedIdent()
	case '\'':
		return l.consumeStringLiteral(start, '\'', false, false)
	case '"':
		return l.consumeStringLiteral(start, '"', false, false)
	case 'r', 'R', 'b', 'B':
		if tok, ok := l.consumePrefixedStringLiteral(); ok {
			return tok
		}
	}
	if isDigit(c) {
		return l.consumeNumericLiteral()
	}
	if isAlpha(c) {
		return l.consumeIdent()
	}
	l.advance(1)
	return l.setError(start, l.pos, "unexpected character")
}

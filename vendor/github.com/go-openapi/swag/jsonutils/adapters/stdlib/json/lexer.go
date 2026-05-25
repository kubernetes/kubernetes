// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package json

import (
	stdjson "encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"

	"github.com/go-openapi/swag/conv"
)

type token struct {
	stdjson.Token
}

func (t token) String() string {
	if t == invalidToken {
		return "invalid token"
	}
	if t == eofToken {
		return "EOF"
	}

	return fmt.Sprintf("%v", t.Token)
}

func (t token) Kind() tokenKind {
	switch t.Token.(type) {
	case nil:
		return tokenNull
	case stdjson.Delim:
		return tokenDelim
	case bool:
		return tokenBool
	case float64:
		return tokenFloat
	case stdjson.Number:
		return tokenNumber
	case string:
		return tokenString
	default:
		return tokenUndef
	}
}

func (t token) Delim() byte {
	r, ok := t.Token.(stdjson.Delim)
	if !ok {
		return 0
	}

	return byte(r)
}

type tokenKind uint8

const (
	tokenUndef tokenKind = iota
	tokenString
	tokenNumber
	tokenFloat
	tokenBool
	tokenNull
	tokenDelim
)

var (
	invalidToken = token{
		Token: stdjson.Token(struct{}{}),
	}

	eofToken = token{
		Token: stdjson.Token(&struct{}{}),
	}

	undefToken = token{
		Token: stdjson.Token(uint8(0)),
	}
)

// jlexer apes easyjson's jlexer, but uses the standard library decoder under the hood.
type jlexer struct {
	buf *bytesReader
	dec *stdjson.Decoder
	err error
	// current token
	next token
	// started bool
}

type bytesReader struct {
	buf    []byte
	offset int
}

func (b *bytesReader) Reset() {
	b.buf = nil
	b.offset = 0
}

func (b *bytesReader) Read(p []byte) (int, error) {
	if b.offset >= len(b.buf) {
		return 0, io.EOF
	}

	n := len(p)
	buf := b.buf[b.offset:]
	m := len(buf)

	if n >= m {
		copy(p, buf)
		b.offset += m

		return m, nil
	}

	copy(p, buf[:n])
	b.offset += n

	return n, nil
}

var _ io.Reader = &bytesReader{}

func newLexer(data []byte) *jlexer {
	l := &jlexer{
		// current: undefToken,
		next: undefToken,
	}
	l.buf = &bytesReader{
		buf: data,
	}
	l.dec = stdjson.NewDecoder(l.buf) // unfortunately, cannot pool this

	return l
}

func (l *jlexer) Reset() {
	l.err = nil
	l.next = undefToken
	// leave l.dec and l.buf alone, since they are replaced at every Borrow
}

func (l *jlexer) Error() error {
	return l.err
}

func (l *jlexer) SetErr(err error) {
	l.err = err
}

func (l *jlexer) Ok() bool {
	return l.err == nil
}

// NextToken consumes a token
func (l *jlexer) NextToken() token {
	if !l.Ok() {
		return invalidToken
	}

	if l.next != undefToken {
		next := l.next
		l.next = undefToken

		return next
	}

	return l.fetchToken()
}

// PeekToken returns the next token without consuming it
func (l *jlexer) PeekToken() token {
	if l.next == undefToken {
		l.next = l.fetchToken()
	}

	return l.next
}

func (l *jlexer) Skip() {
	_ = l.NextToken()
}

func (l *jlexer) IsDelim(c byte) bool {
	if !l.Ok() {
		return false
	}

	next := l.PeekToken()
	if next.Kind() != tokenDelim {
		return false
	}

	if next.Delim() != c {
		return false
	}

	return true
}

func (l *jlexer) IsNull() bool {
	if !l.Ok() {
		return false
	}

	next := l.PeekToken()

	return next.Kind() == tokenNull
}

func (l *jlexer) Delim(c byte) {
	if !l.Ok() {
		return
	}

	tok := l.NextToken()
	if tok.Kind() != tokenDelim {
		l.err = fmt.Errorf("expected a delimiter token but got '%v': %w", tok, ErrStdlib)

		return
	}

	if tok.Delim() != c {
		l.err = fmt.Errorf("expected delimiter '%q' but got '%q': %w", c, tok.Delim(), ErrStdlib)
	}
}

func (l *jlexer) Null() {
	if !l.Ok() {
		return
	}

	tok := l.NextToken()
	if tok.Kind() != tokenNull {
		l.err = fmt.Errorf("expected a null token but got '%v': %w", tok, ErrStdlib)
	}
}

func (l *jlexer) Number() any {
	if !l.Ok() {
		return 0
	}

	tok := l.NextToken()

	switch tok.Kind() { //nolint:exhaustive
	case tokenNumber:
		n := tok.Token.(stdjson.Number).String()
		f, _ := strconv.ParseFloat(n, 64)
		if conv.IsFloat64AJSONInteger(f) {
			return int64(math.Trunc(f))
		}

		return f

	case tokenFloat:
		f := tok.Token.(float64)
		if conv.IsFloat64AJSONInteger(f) {
			return int64(math.Trunc(f))
		}

		return f

	default:
		l.err = fmt.Errorf("expected a number token but got '%v': %w", tok, ErrStdlib)

		return 0
	}
}

func (l *jlexer) Bool() bool {
	if !l.Ok() {
		return false
	}

	tok := l.NextToken()
	if tok.Kind() != tokenBool {
		l.err = fmt.Errorf("expected a bool token but got '%v': %w", tok, ErrStdlib)

		return false
	}

	return tok.Token.(bool)
}

func (l *jlexer) String() string {
	if !l.Ok() {
		return ""
	}

	tok := l.NextToken()
	if tok.Kind() != tokenString {
		l.err = fmt.Errorf("expected a string token but got '%v': %w", tok, ErrStdlib)

		return ""
	}

	return tok.Token.(string)
}

// Commas and colons are elided.
func (l *jlexer) fetchToken() token {
	jtok, err := l.dec.Token()
	if err != nil {
		if errors.Is(err, io.EOF) {
			return eofToken
		}

		l.err = errors.Join(err, ErrStdlib)
		return invalidToken
	}

	return token{Token: jtok}
}

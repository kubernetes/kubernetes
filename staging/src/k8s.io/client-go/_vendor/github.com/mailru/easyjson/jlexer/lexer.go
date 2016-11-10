// Package jlexer contains a JSON lexer implementation.
//
// It is expected that it is mostly used with generated parser code, so the interface is tuned
// for a parser that knows what kind of data is expected.
package jlexer

import (
	"fmt"
	"io"
	"reflect"
	"strconv"
	"unicode/utf8"
	"unsafe"
)

// tokenKind determines type of a token.
type tokenKind byte

const (
	tokenUndef  tokenKind = iota // No token.
	tokenDelim                   // Delimiter: one of '{', '}', '[' or ']'.
	tokenString                  // A string literal, e.g. "abc\u1234"
	tokenNumber                  // Number literal, e.g. 1.5e5
	tokenBool                    // Boolean literal: true or false.
	tokenNull                    // null keyword.
)

// token describes a single token: type, position in the input and value.
type token struct {
	kind tokenKind // Type of a token.

	boolValue  bool   // Value if a boolean literal token.
	byteValue  []byte // Raw value of a token.
	delimValue byte
}

// Lexer is a JSON lexer: it iterates over JSON tokens in a byte slice.
type Lexer struct {
	Data []byte // Input data given to the lexer.

	start int   // Start of the current token.
	pos   int   // Current unscanned position in the input stream.
	token token // Last scanned token, if token.kind != tokenUndef.

	firstElement bool // Whether current element is the first in array or an object.
	wantSep      byte // A comma or a colon character, which need to occur before a token.

	err error // Error encountered during lexing, if any.
}

// fetchToken scans the input for the next token.
func (r *Lexer) fetchToken() {
	r.token.kind = tokenUndef
	r.start = r.pos

	// Check if r.Data has r.pos element
	// If it doesn't, it mean corrupted input data
	if len(r.Data) < r.pos {
		r.errParse("Unexpected end of data")
		return
	}
	// Determine the type of a token by skipping whitespace and reading the
	// first character.
	for _, c := range r.Data[r.pos:] {
		switch c {
		case ':', ',':
			if r.wantSep == c {
				r.pos++
				r.start++
				r.wantSep = 0
			} else {
				r.errSyntax()
			}

		case ' ', '\t', '\r', '\n':
			r.pos++
			r.start++

		case '"':
			if r.wantSep != 0 {
				r.errSyntax()
			}

			r.token.kind = tokenString
			r.fetchString()
			return

		case '{', '[':
			if r.wantSep != 0 {
				r.errSyntax()
			}
			r.firstElement = true
			r.token.kind = tokenDelim
			r.token.delimValue = r.Data[r.pos]
			r.pos++
			return

		case '}', ']':
			if !r.firstElement && (r.wantSep != ',') {
				r.errSyntax()
			}
			r.wantSep = 0
			r.token.kind = tokenDelim
			r.token.delimValue = r.Data[r.pos]
			r.pos++
			return

		case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-':
			if r.wantSep != 0 {
				r.errSyntax()
			}
			r.token.kind = tokenNumber
			r.fetchNumber()
			return

		case 'n':
			if r.wantSep != 0 {
				r.errSyntax()
			}

			r.token.kind = tokenNull
			r.fetchNull()
			return

		case 't':
			if r.wantSep != 0 {
				r.errSyntax()
			}

			r.token.kind = tokenBool
			r.token.boolValue = true
			r.fetchTrue()
			return

		case 'f':
			if r.wantSep != 0 {
				r.errSyntax()
			}

			r.token.kind = tokenBool
			r.token.boolValue = false
			r.fetchFalse()
			return

		default:
			r.errSyntax()
			return
		}
	}
	r.err = io.EOF
	return
}

// isTokenEnd returns true if the char can follow a non-delimiter token
func isTokenEnd(c byte) bool {
	return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '[' || c == ']' || c == '{' || c == '}' || c == ',' || c == ':'
}

// fetchNull fetches and checks remaining bytes of null keyword.
func (r *Lexer) fetchNull() {
	r.pos += 4
	if r.pos > len(r.Data) ||
		r.Data[r.pos-3] != 'u' ||
		r.Data[r.pos-2] != 'l' ||
		r.Data[r.pos-1] != 'l' ||
		(r.pos != len(r.Data) && !isTokenEnd(r.Data[r.pos])) {

		r.pos -= 4
		r.errSyntax()
	}
}

// fetchTrue fetches and checks remaining bytes of true keyword.
func (r *Lexer) fetchTrue() {
	r.pos += 4
	if r.pos > len(r.Data) ||
		r.Data[r.pos-3] != 'r' ||
		r.Data[r.pos-2] != 'u' ||
		r.Data[r.pos-1] != 'e' ||
		(r.pos != len(r.Data) && !isTokenEnd(r.Data[r.pos])) {

		r.pos -= 4
		r.errSyntax()
	}
}

// fetchFalse fetches and checks remaining bytes of false keyword.
func (r *Lexer) fetchFalse() {
	r.pos += 5
	if r.pos > len(r.Data) ||
		r.Data[r.pos-4] != 'a' ||
		r.Data[r.pos-3] != 'l' ||
		r.Data[r.pos-2] != 's' ||
		r.Data[r.pos-1] != 'e' ||
		(r.pos != len(r.Data) && !isTokenEnd(r.Data[r.pos])) {

		r.pos -= 5
		r.errSyntax()
	}
}

// bytesToStr creates a string pointing at the slice to avoid copying.
//
// Warning: the string returned by the function should be used with care, as the whole input data
// chunk may be either blocked from being freed by GC because of a single string or the buffer.Data
// may be garbage-collected even when the string exists.
func bytesToStr(data []byte) string {
	h := (*reflect.SliceHeader)(unsafe.Pointer(&data))
	shdr := reflect.StringHeader{h.Data, h.Len}
	return *(*string)(unsafe.Pointer(&shdr))
}

// fetchNumber scans a number literal token.
func (r *Lexer) fetchNumber() {
	hasE := false
	afterE := false
	hasDot := false

	r.pos++
	for i, c := range r.Data[r.pos:] {
		switch {
		case c >= '0' && c <= '9':
			afterE = false
		case c == '.' && !hasDot:
			hasDot = true
		case (c == 'e' || c == 'E') && !hasE:
			hasE = true
			hasDot = true
			afterE = true
		case (c == '+' || c == '-') && afterE:
			afterE = false
		default:
			r.pos += i
			if !isTokenEnd(c) {
				r.errSyntax()
			} else {
				r.token.byteValue = r.Data[r.start:r.pos]
			}
			return
		}
	}

	r.pos = len(r.Data)
	r.token.byteValue = r.Data[r.start:]
}

// findStringLen tries to scan into the string literal for ending quote char to determine required size.
// The size will be exact if no escapes are present and may be inexact if there are escaped chars.
func findStringLen(data []byte) (hasEscapes bool, length int) {
	delta := 0

	for i := 0; i < len(data); i++ {
		switch data[i] {
		case '\\':
			i++
			delta++
			if i < len(data) && data[i] == 'u' {
				delta++
			}
		case '"':
			return (delta > 0), (i - delta)
		}
	}

	return false, len(data)
}

// processEscape processes a single escape sequence and returns number of bytes processed.
func (r *Lexer) processEscape(data []byte) (int, error) {
	if len(data) < 2 {
		return 0, fmt.Errorf("syntax error at %v", string(data))
	}

	c := data[1]
	switch c {
	case '"', '/', '\\':
		r.token.byteValue = append(r.token.byteValue, c)
		return 2, nil
	case 'b':
		r.token.byteValue = append(r.token.byteValue, '\b')
		return 2, nil
	case 'f':
		r.token.byteValue = append(r.token.byteValue, '\f')
		return 2, nil
	case 'n':
		r.token.byteValue = append(r.token.byteValue, '\n')
		return 2, nil
	case 'r':
		r.token.byteValue = append(r.token.byteValue, '\r')
		return 2, nil
	case 't':
		r.token.byteValue = append(r.token.byteValue, '\t')
		return 2, nil
	case 'u':
	default:
		return 0, fmt.Errorf("syntax error")
	}

	var val rune

	for i := 2; i < len(data) && i < 6; i++ {
		var v byte
		c = data[i]
		switch c {
		case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
			v = c - '0'
		case 'a', 'b', 'c', 'd', 'e', 'f':
			v = c - 'a' + 10
		case 'A', 'B', 'C', 'D', 'E', 'F':
			v = c - 'A' + 10
		default:
			return 0, fmt.Errorf("syntax error")
		}

		val <<= 4
		val |= rune(v)
	}

	l := utf8.RuneLen(val)
	if l == -1 {
		return 0, fmt.Errorf("invalid unicode escape")
	}

	var d [4]byte
	utf8.EncodeRune(d[:], val)
	r.token.byteValue = append(r.token.byteValue, d[:l]...)
	return 6, nil
}

// fetchString scans a string literal token.
func (r *Lexer) fetchString() {
	r.pos++
	data := r.Data[r.pos:]

	hasEscapes, length := findStringLen(data)
	if !hasEscapes {
		r.token.byteValue = data[:length]
		r.pos += length + 1
		return
	}

	r.token.byteValue = make([]byte, 0, length)
	p := 0
	for i := 0; i < len(data); {
		switch data[i] {
		case '"':
			r.pos += i + 1
			r.token.byteValue = append(r.token.byteValue, data[p:i]...)
			i++
			return

		case '\\':
			r.token.byteValue = append(r.token.byteValue, data[p:i]...)
			off, err := r.processEscape(data[i:])
			if err != nil {
				r.errParse(err.Error())
				return
			}
			i += off
			p = i

		default:
			i++
		}
	}
	r.errParse("unterminated string literal")
}

// scanToken scans the next token if no token is currently available in the lexer.
func (r *Lexer) scanToken() {
	if r.token.kind != tokenUndef || r.err != nil {
		return
	}

	r.fetchToken()
}

// consume resets the current token to allow scanning the next one.
func (r *Lexer) consume() {
	r.token.kind = tokenUndef
	r.token.delimValue = 0
}

// Ok returns true if no error (including io.EOF) was encountered during scanning.
func (r *Lexer) Ok() bool {
	return r.err == nil
}

const maxErrorContextLen = 13

func (r *Lexer) errParse(what string) {
	if r.err == nil {
		var str string
		if len(r.Data)-r.pos <= maxErrorContextLen {
			str = string(r.Data)
		} else {
			str = string(r.Data[r.pos:r.pos+maxErrorContextLen-3]) + "..."
		}
		r.err = &LexerError{
			Reason: what,
			Offset: r.pos,
			Data:   str,
		}
	}
}

func (r *Lexer) errSyntax() {
	r.errParse("syntax error")
}

func (r *Lexer) errInvalidToken(expected string) {
	if r.err == nil {
		var str string
		if len(r.token.byteValue) <= maxErrorContextLen {
			str = string(r.token.byteValue)
		} else {
			str = string(r.token.byteValue[:maxErrorContextLen-3]) + "..."
		}
		r.err = &LexerError{
			Reason: fmt.Sprintf("expected %s", expected),
			Offset: r.pos,
			Data:   str,
		}
	}
}

// Delim consumes a token and verifies that it is the given delimiter.
func (r *Lexer) Delim(c byte) {
	if r.token.kind == tokenUndef && r.Ok() {
		r.fetchToken()
	}
	if !r.Ok() || r.token.delimValue != c {
		r.errInvalidToken(string([]byte{c}))
	}
	r.consume()
}

// IsDelim returns true if there was no scanning error and next token is the given delimiter.
func (r *Lexer) IsDelim(c byte) bool {
	if r.token.kind == tokenUndef && r.Ok() {
		r.fetchToken()
	}
	return !r.Ok() || r.token.delimValue == c
}

// Null verifies that the next token is null and consumes it.
func (r *Lexer) Null() {
	if r.token.kind == tokenUndef && r.Ok() {
		r.fetchToken()
	}
	if !r.Ok() || r.token.kind != tokenNull {
		r.errInvalidToken("null")
	}
	r.consume()
}

// IsNull returns true if the next token is a null keyword.
func (r *Lexer) IsNull() bool {
	if r.token.kind == tokenUndef && r.Ok() {
		r.fetchToken()
	}
	return r.Ok() && r.token.kind == tokenNull
}

// Skip skips a single token.
func (r *Lexer) Skip() {
	if r.token.kind == tokenUndef && r.Ok() {
		r.fetchToken()
	}
	r.consume()
}

// SkipRecursive skips next array or object completely, or just skips a single token if not
// an array/object.
//
// Note: no syntax validation is performed on the skipped data.
func (r *Lexer) SkipRecursive() {
	r.scanToken()

	var start, end byte

	if r.token.delimValue == '{' {
		start, end = '{', '}'
	} else if r.token.delimValue == '[' {
		start, end = '[', ']'
	} else {
		r.consume()
		return
	}

	r.consume()

	level := 1
	inQuotes := false
	wasEscape := false

	for i, c := range r.Data[r.pos:] {
		switch {
		case c == start && !inQuotes:
			level++
		case c == end && !inQuotes:
			level--
			if level == 0 {
				r.pos += i + 1
				return
			}
		case c == '\\' && inQuotes:
			wasEscape = true
			continue
		case c == '"' && inQuotes:
			inQuotes = wasEscape
		case c == '"':
			inQuotes = true
		}
		wasEscape = false
	}
	r.pos = len(r.Data)
	r.err = io.EOF
}

// Raw fetches the next item recursively as a data slice
func (r *Lexer) Raw() []byte {
	r.SkipRecursive()
	if !r.Ok() {
		return nil
	}
	return r.Data[r.start:r.pos]
}

// UnsafeString returns the string value if the token is a string literal.
//
// Warning: returned string may point to the input buffer, so the string should not outlive
// the input buffer. Intended pattern of usage is as an argument to a switch statement.
func (r *Lexer) UnsafeString() string {
	if r.token.kind == tokenUndef && r.Ok() {
		r.fetchToken()
	}
	if !r.Ok() || r.token.kind != tokenString {
		r.errInvalidToken("string")
		return ""
	}

	ret := bytesToStr(r.token.byteValue)
	r.consume()
	return ret
}

// String reads a string literal.
func (r *Lexer) String() string {
	if r.token.kind == tokenUndef && r.Ok() {
		r.fetchToken()
	}
	if !r.Ok() || r.token.kind != tokenString {
		r.errInvalidToken("string")
		return ""

	}
	ret := string(r.token.byteValue)
	r.consume()
	return ret
}

// Bool reads a true or false boolean keyword.
func (r *Lexer) Bool() bool {
	if r.token.kind == tokenUndef && r.Ok() {
		r.fetchToken()
	}
	if !r.Ok() || r.token.kind != tokenBool {
		r.errInvalidToken("bool")
		return false

	}
	ret := r.token.boolValue
	r.consume()
	return ret
}

func (r *Lexer) number() string {
	if r.token.kind == tokenUndef && r.Ok() {
		r.fetchToken()
	}
	if !r.Ok() || r.token.kind != tokenNumber {
		r.errInvalidToken("number")
		return ""

	}
	ret := bytesToStr(r.token.byteValue)
	r.consume()
	return ret
}

func (r *Lexer) Uint8() uint8 {
	s := r.number()
	if !r.Ok() {
		return 0
	}

	n, err := strconv.ParseUint(s, 10, 8)
	if err != nil {
		r.err = &LexerError{
			Reason: err.Error(),
		}
	}
	return uint8(n)
}

func (r *Lexer) Uint16() uint16 {
	s := r.number()
	if !r.Ok() {
		return 0
	}

	n, err := strconv.ParseUint(s, 10, 16)
	if err != nil {
		r.err = &LexerError{
			Reason: err.Error(),
		}
	}
	return uint16(n)
}

func (r *Lexer) Uint32() uint32 {
	s := r.number()
	if !r.Ok() {
		return 0
	}

	n, err := strconv.ParseUint(s, 10, 32)
	if err != nil {
		r.err = &LexerError{
			Reason: err.Error(),
		}
	}
	return uint32(n)
}

func (r *Lexer) Uint64() uint64 {
	s := r.number()
	if !r.Ok() {
		return 0
	}

	n, err := strconv.ParseUint(s, 10, 64)
	if err != nil {
		r.err = &LexerError{
			Reason: err.Error(),
		}
	}
	return n
}

func (r *Lexer) Uint() uint {
	return uint(r.Uint64())
}

func (r *Lexer) Int8() int8 {
	s := r.number()
	if !r.Ok() {
		return 0
	}

	n, err := strconv.ParseInt(s, 10, 8)
	if err != nil {
		r.err = &LexerError{
			Reason: err.Error(),
		}
	}
	return int8(n)
}

func (r *Lexer) Int16() int16 {
	s := r.number()
	if !r.Ok() {
		return 0
	}

	n, err := strconv.ParseInt(s, 10, 16)
	if err != nil {
		r.err = &LexerError{
			Reason: err.Error(),
		}
	}
	return int16(n)
}

func (r *Lexer) Int32() int32 {
	s := r.number()
	if !r.Ok() {
		return 0
	}

	n, err := strconv.ParseInt(s, 10, 32)
	if err != nil {
		r.err = &LexerError{
			Reason: err.Error(),
		}
	}
	return int32(n)
}

func (r *Lexer) Int64() int64 {
	s := r.number()
	if !r.Ok() {
		return 0
	}

	n, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		r.err = &LexerError{
			Reason: err.Error(),
		}
	}
	return n
}

func (r *Lexer) Int() int {
	return int(r.Int64())
}

func (r *Lexer) Uint8Str() uint8 {
	s := r.UnsafeString()
	if !r.Ok() {
		return 0
	}

	n, err := strconv.ParseUint(s, 10, 8)
	if err != nil {
		r.err = &LexerError{
			Reason: err.Error(),
		}
	}
	return uint8(n)
}

func (r *Lexer) Uint16Str() uint16 {
	s := r.UnsafeString()
	if !r.Ok() {
		return 0
	}

	n, err := strconv.ParseUint(s, 10, 16)
	if err != nil {
		r.err = &LexerError{
			Reason: err.Error(),
		}
	}
	return uint16(n)
}

func (r *Lexer) Uint32Str() uint32 {
	s := r.UnsafeString()
	if !r.Ok() {
		return 0
	}

	n, err := strconv.ParseUint(s, 10, 32)
	if err != nil {
		r.err = &LexerError{
			Reason: err.Error(),
		}
	}
	return uint32(n)
}

func (r *Lexer) Uint64Str() uint64 {
	s := r.UnsafeString()
	if !r.Ok() {
		return 0
	}

	n, err := strconv.ParseUint(s, 10, 64)
	if err != nil {
		r.err = &LexerError{
			Reason: err.Error(),
		}
	}
	return n
}

func (r *Lexer) UintStr() uint {
	return uint(r.Uint64Str())
}

func (r *Lexer) Int8Str() int8 {
	s := r.UnsafeString()
	if !r.Ok() {
		return 0
	}

	n, err := strconv.ParseInt(s, 10, 8)
	if err != nil {
		r.err = &LexerError{
			Reason: err.Error(),
		}
	}
	return int8(n)
}

func (r *Lexer) Int16Str() int16 {
	s := r.UnsafeString()
	if !r.Ok() {
		return 0
	}

	n, err := strconv.ParseInt(s, 10, 16)
	if err != nil {
		r.err = &LexerError{
			Reason: err.Error(),
		}
	}
	return int16(n)
}

func (r *Lexer) Int32Str() int32 {
	s := r.UnsafeString()
	if !r.Ok() {
		return 0
	}

	n, err := strconv.ParseInt(s, 10, 32)
	if err != nil {
		r.err = &LexerError{
			Reason: err.Error(),
		}
	}
	return int32(n)
}

func (r *Lexer) Int64Str() int64 {
	s := r.UnsafeString()
	if !r.Ok() {
		return 0
	}

	n, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		r.err = &LexerError{
			Reason: err.Error(),
		}
	}
	return n
}

func (r *Lexer) IntStr() int {
	return int(r.Int64Str())
}

func (r *Lexer) Float32() float32 {
	s := r.number()
	if !r.Ok() {
		return 0
	}

	n, err := strconv.ParseFloat(s, 32)
	if err != nil {
		r.err = &LexerError{
			Reason: err.Error(),
		}
	}
	return float32(n)
}

func (r *Lexer) Float64() float64 {
	s := r.number()
	if !r.Ok() {
		return 0
	}

	n, err := strconv.ParseFloat(s, 64)
	if err != nil {
		r.err = &LexerError{
			Reason: err.Error(),
		}
	}
	return n
}

func (r *Lexer) Error() error {
	return r.err
}

func (r *Lexer) AddError(e error) {
	if r.err == nil {
		r.err = e
	}
}

// Interface fetches an interface{} analogous to the 'encoding/json' package.
func (r *Lexer) Interface() interface{} {
	if r.token.kind == tokenUndef && r.Ok() {
		r.fetchToken()
	}

	if !r.Ok() {
		return nil
	}
	switch r.token.kind {
	case tokenString:
		return r.String()
	case tokenNumber:
		return r.Float64()
	case tokenBool:
		return r.Bool()
	case tokenNull:
		r.Null()
		return nil
	}

	if r.token.delimValue == '{' {
		r.consume()

		ret := map[string]interface{}{}
		for !r.IsDelim('}') {
			key := r.String()
			r.WantColon()
			ret[key] = r.Interface()
			r.WantComma()
		}
		r.Delim('}')

		if r.Ok() {
			return ret
		} else {
			return nil
		}
	} else if r.token.delimValue == '[' {
		r.consume()

		var ret []interface{}
		for !r.IsDelim(']') {
			ret = append(ret, r.Interface())
			r.WantComma()
		}
		r.Delim(']')

		if r.Ok() {
			return ret
		} else {
			return nil
		}
	}
	r.errSyntax()
	return nil
}

// WantComma requires a comma to be present before fetching next token.
func (r *Lexer) WantComma() {
	r.wantSep = ','
	r.firstElement = false
}

// WantColon requires a colon to be present before fetching next token.
func (r *Lexer) WantColon() {
	r.wantSep = ':'
	r.firstElement = false
}

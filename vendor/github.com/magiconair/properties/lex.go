// Copyright 2016 Frank Schroeder. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Parts of the lexer are from the template/text/parser package
// For these parts the following applies:
//
// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file of the go 1.2
// distribution.

package properties

import (
	"fmt"
	"strconv"
	"strings"
	"unicode/utf8"
)

// item represents a token or text string returned from the scanner.
type item struct {
	typ itemType // The type of this item.
	pos int      // The starting position, in bytes, of this item in the input string.
	val string   // The value of this item.
}

func (i item) String() string {
	switch {
	case i.typ == itemEOF:
		return "EOF"
	case i.typ == itemError:
		return i.val
	case len(i.val) > 10:
		return fmt.Sprintf("%.10q...", i.val)
	}
	return fmt.Sprintf("%q", i.val)
}

// itemType identifies the type of lex items.
type itemType int

const (
	itemError itemType = iota // error occurred; value is text of error
	itemEOF
	itemKey     // a key
	itemValue   // a value
	itemComment // a comment
)

// defines a constant for EOF
const eof = -1

// permitted whitespace characters space, FF and TAB
const whitespace = " \f\t"

// stateFn represents the state of the scanner as a function that returns the next state.
type stateFn func(*lexer) stateFn

// lexer holds the state of the scanner.
type lexer struct {
	input   string    // the string being scanned
	state   stateFn   // the next lexing function to enter
	pos     int       // current position in the input
	start   int       // start position of this item
	width   int       // width of last rune read from input
	lastPos int       // position of most recent item returned by nextItem
	runes   []rune    // scanned runes for this item
	items   chan item // channel of scanned items
}

// next returns the next rune in the input.
func (l *lexer) next() rune {
	if int(l.pos) >= len(l.input) {
		l.width = 0
		return eof
	}
	r, w := utf8.DecodeRuneInString(l.input[l.pos:])
	l.width = w
	l.pos += l.width
	return r
}

// peek returns but does not consume the next rune in the input.
func (l *lexer) peek() rune {
	r := l.next()
	l.backup()
	return r
}

// backup steps back one rune. Can only be called once per call of next.
func (l *lexer) backup() {
	l.pos -= l.width
}

// emit passes an item back to the client.
func (l *lexer) emit(t itemType) {
	item := item{t, l.start, string(l.runes)}
	l.items <- item
	l.start = l.pos
	l.runes = l.runes[:0]
}

// ignore skips over the pending input before this point.
func (l *lexer) ignore() {
	l.start = l.pos
}

// appends the rune to the current value
func (l *lexer) appendRune(r rune) {
	l.runes = append(l.runes, r)
}

// accept consumes the next rune if it's from the valid set.
func (l *lexer) accept(valid string) bool {
	if strings.IndexRune(valid, l.next()) >= 0 {
		return true
	}
	l.backup()
	return false
}

// acceptRun consumes a run of runes from the valid set.
func (l *lexer) acceptRun(valid string) {
	for strings.IndexRune(valid, l.next()) >= 0 {
	}
	l.backup()
}

// acceptRunUntil consumes a run of runes up to a terminator.
func (l *lexer) acceptRunUntil(term rune) {
	for term != l.next() {
	}
	l.backup()
}

// hasText returns true if the current parsed text is not empty.
func (l *lexer) isNotEmpty() bool {
	return l.pos > l.start
}

// lineNumber reports which line we're on, based on the position of
// the previous item returned by nextItem. Doing it this way
// means we don't have to worry about peek double counting.
func (l *lexer) lineNumber() int {
	return 1 + strings.Count(l.input[:l.lastPos], "\n")
}

// errorf returns an error token and terminates the scan by passing
// back a nil pointer that will be the next state, terminating l.nextItem.
func (l *lexer) errorf(format string, args ...interface{}) stateFn {
	l.items <- item{itemError, l.start, fmt.Sprintf(format, args...)}
	return nil
}

// nextItem returns the next item from the input.
func (l *lexer) nextItem() item {
	item := <-l.items
	l.lastPos = item.pos
	return item
}

// lex creates a new scanner for the input string.
func lex(input string) *lexer {
	l := &lexer{
		input: input,
		items: make(chan item),
		runes: make([]rune, 0, 32),
	}
	go l.run()
	return l
}

// run runs the state machine for the lexer.
func (l *lexer) run() {
	for l.state = lexBeforeKey(l); l.state != nil; {
		l.state = l.state(l)
	}
}

// state functions

// lexBeforeKey scans until a key begins.
func lexBeforeKey(l *lexer) stateFn {
	switch r := l.next(); {
	case isEOF(r):
		l.emit(itemEOF)
		return nil

	case isEOL(r):
		l.ignore()
		return lexBeforeKey

	case isComment(r):
		return lexComment

	case isWhitespace(r):
		l.acceptRun(whitespace)
		l.ignore()
		return lexKey

	default:
		l.backup()
		return lexKey
	}
}

// lexComment scans a comment line. The comment character has already been scanned.
func lexComment(l *lexer) stateFn {
	l.acceptRun(whitespace)
	l.ignore()
	for {
		switch r := l.next(); {
		case isEOF(r):
			l.ignore()
			l.emit(itemEOF)
			return nil
		case isEOL(r):
			l.emit(itemComment)
			return lexBeforeKey
		default:
			l.appendRune(r)
		}
	}
}

// lexKey scans the key up to a delimiter
func lexKey(l *lexer) stateFn {
	var r rune

Loop:
	for {
		switch r = l.next(); {

		case isEscape(r):
			err := l.scanEscapeSequence()
			if err != nil {
				return l.errorf(err.Error())
			}

		case isEndOfKey(r):
			l.backup()
			break Loop

		case isEOF(r):
			break Loop

		default:
			l.appendRune(r)
		}
	}

	if len(l.runes) > 0 {
		l.emit(itemKey)
	}

	if isEOF(r) {
		l.emit(itemEOF)
		return nil
	}

	return lexBeforeValue
}

// lexBeforeValue scans the delimiter between key and value.
// Leading and trailing whitespace is ignored.
// We expect to be just after the key.
func lexBeforeValue(l *lexer) stateFn {
	l.acceptRun(whitespace)
	l.accept(":=")
	l.acceptRun(whitespace)
	l.ignore()
	return lexValue
}

// lexValue scans text until the end of the line. We expect to be just after the delimiter.
func lexValue(l *lexer) stateFn {
	for {
		switch r := l.next(); {
		case isEscape(r):
			r := l.peek()
			if isEOL(r) {
				l.next()
				l.acceptRun(whitespace)
			} else {
				err := l.scanEscapeSequence()
				if err != nil {
					return l.errorf(err.Error())
				}
			}

		case isEOL(r):
			l.emit(itemValue)
			l.ignore()
			return lexBeforeKey

		case isEOF(r):
			l.emit(itemValue)
			l.emit(itemEOF)
			return nil

		default:
			l.appendRune(r)
		}
	}
}

// scanEscapeSequence scans either one of the escaped characters
// or a unicode literal. We expect to be after the escape character.
func (l *lexer) scanEscapeSequence() error {
	switch r := l.next(); {

	case isEscapedCharacter(r):
		l.appendRune(decodeEscapedCharacter(r))
		return nil

	case atUnicodeLiteral(r):
		return l.scanUnicodeLiteral()

	case isEOF(r):
		return fmt.Errorf("premature EOF")

	// silently drop the escape character and append the rune as is
	default:
		l.appendRune(r)
		return nil
	}
}

// scans a unicode literal in the form \uXXXX. We expect to be after the \u.
func (l *lexer) scanUnicodeLiteral() error {
	// scan the digits
	d := make([]rune, 4)
	for i := 0; i < 4; i++ {
		d[i] = l.next()
		if d[i] == eof || !strings.ContainsRune("0123456789abcdefABCDEF", d[i]) {
			return fmt.Errorf("invalid unicode literal")
		}
	}

	// decode the digits into a rune
	r, err := strconv.ParseInt(string(d), 16, 0)
	if err != nil {
		return err
	}

	l.appendRune(rune(r))
	return nil
}

// decodeEscapedCharacter returns the unescaped rune. We expect to be after the escape character.
func decodeEscapedCharacter(r rune) rune {
	switch r {
	case 'f':
		return '\f'
	case 'n':
		return '\n'
	case 'r':
		return '\r'
	case 't':
		return '\t'
	default:
		return r
	}
}

// atUnicodeLiteral reports whether we are at a unicode literal.
// The escape character has already been consumed.
func atUnicodeLiteral(r rune) bool {
	return r == 'u'
}

// isComment reports whether we are at the start of a comment.
func isComment(r rune) bool {
	return r == '#' || r == '!'
}

// isEndOfKey reports whether the rune terminates the current key.
func isEndOfKey(r rune) bool {
	return strings.ContainsRune(" \f\t\r\n:=", r)
}

// isEOF reports whether we are at EOF.
func isEOF(r rune) bool {
	return r == eof
}

// isEOL reports whether we are at a new line character.
func isEOL(r rune) bool {
	return r == '\n' || r == '\r'
}

// isEscape reports whether the rune is the escape character which
// prefixes unicode literals and other escaped characters.
func isEscape(r rune) bool {
	return r == '\\'
}

// isEscapedCharacter reports whether we are at one of the characters that need escaping.
// The escape character has already been consumed.
func isEscapedCharacter(r rune) bool {
	return strings.ContainsRune(" :=fnrt", r)
}

// isWhitespace reports whether the rune is a whitespace character.
func isWhitespace(r rune) bool {
	return strings.ContainsRune(whitespace, r)
}

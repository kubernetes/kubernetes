// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package scanner implements a scanner for gcfg configuration text.
// It takes a []byte as source which can then be tokenized
// through repeated calls to the Scan method.
//
// Note that the API for the scanner package may change to accommodate new
// features or implementation changes in gcfg.
//
package scanner

import (
	"fmt"
	"path/filepath"
	"unicode"
	"unicode/utf8"
)

import (
	"github.com/src-d/gcfg/token"
)

// An ErrorHandler may be provided to Scanner.Init. If a syntax error is
// encountered and a handler was installed, the handler is called with a
// position and an error message. The position points to the beginning of
// the offending token.
//
type ErrorHandler func(pos token.Position, msg string)

// A Scanner holds the scanner's internal state while processing
// a given text.  It can be allocated as part of another data
// structure but must be initialized via Init before use.
//
type Scanner struct {
	// immutable state
	file *token.File  // source file handle
	dir  string       // directory portion of file.Name()
	src  []byte       // source
	err  ErrorHandler // error reporting; or nil
	mode Mode         // scanning mode

	// scanning state
	ch         rune // current character
	offset     int  // character offset
	rdOffset   int  // reading offset (position after current character)
	lineOffset int  // current line offset
	nextVal    bool // next token is expected to be a value

	// public state - ok to modify
	ErrorCount int // number of errors encountered
}

// Read the next Unicode char into s.ch.
// s.ch < 0 means end-of-file.
//
func (s *Scanner) next() {
	if s.rdOffset < len(s.src) {
		s.offset = s.rdOffset
		if s.ch == '\n' {
			s.lineOffset = s.offset
			s.file.AddLine(s.offset)
		}
		r, w := rune(s.src[s.rdOffset]), 1
		switch {
		case r == 0:
			s.error(s.offset, "illegal character NUL")
		case r >= 0x80:
			// not ASCII
			r, w = utf8.DecodeRune(s.src[s.rdOffset:])
			if r == utf8.RuneError && w == 1 {
				s.error(s.offset, "illegal UTF-8 encoding")
			}
		}
		s.rdOffset += w
		s.ch = r
	} else {
		s.offset = len(s.src)
		if s.ch == '\n' {
			s.lineOffset = s.offset
			s.file.AddLine(s.offset)
		}
		s.ch = -1 // eof
	}
}

// A mode value is a set of flags (or 0).
// They control scanner behavior.
//
type Mode uint

const (
	ScanComments Mode = 1 << iota // return comments as COMMENT tokens
)

// Init prepares the scanner s to tokenize the text src by setting the
// scanner at the beginning of src. The scanner uses the file set file
// for position information and it adds line information for each line.
// It is ok to re-use the same file when re-scanning the same file as
// line information which is already present is ignored. Init causes a
// panic if the file size does not match the src size.
//
// Calls to Scan will invoke the error handler err if they encounter a
// syntax error and err is not nil. Also, for each error encountered,
// the Scanner field ErrorCount is incremented by one. The mode parameter
// determines how comments are handled.
//
// Note that Init may call err if there is an error in the first character
// of the file.
//
func (s *Scanner) Init(file *token.File, src []byte, err ErrorHandler, mode Mode) {
	// Explicitly initialize all fields since a scanner may be reused.
	if file.Size() != len(src) {
		panic(fmt.Sprintf("file size (%d) does not match src len (%d)", file.Size(), len(src)))
	}
	s.file = file
	s.dir, _ = filepath.Split(file.Name())
	s.src = src
	s.err = err
	s.mode = mode

	s.ch = ' '
	s.offset = 0
	s.rdOffset = 0
	s.lineOffset = 0
	s.ErrorCount = 0
	s.nextVal = false

	s.next()
}

func (s *Scanner) error(offs int, msg string) {
	if s.err != nil {
		s.err(s.file.Position(s.file.Pos(offs)), msg)
	}
	s.ErrorCount++
}

func (s *Scanner) scanComment() string {
	// initial [;#] already consumed
	offs := s.offset - 1 // position of initial [;#]

	for s.ch != '\n' && s.ch >= 0 {
		s.next()
	}
	return string(s.src[offs:s.offset])
}

func isLetter(ch rune) bool {
	return 'a' <= ch && ch <= 'z' || 'A' <= ch && ch <= 'Z' || ch >= 0x80 && unicode.IsLetter(ch)
}

func isDigit(ch rune) bool {
	return '0' <= ch && ch <= '9' || ch >= 0x80 && unicode.IsDigit(ch)
}

func (s *Scanner) scanIdentifier() string {
	offs := s.offset
	for isLetter(s.ch) || isDigit(s.ch) || s.ch == '-' {
		s.next()
	}
	return string(s.src[offs:s.offset])
}

func (s *Scanner) scanEscape(val bool) {
	offs := s.offset
	ch := s.ch
	s.next() // always make progress
	switch ch {
	case '\\', '"':
		// ok
	case 'n', 't':
		if val {
			break // ok
		}
		fallthrough
	default:
		s.error(offs, "unknown escape sequence")
	}
}

func (s *Scanner) scanString() string {
	// '"' opening already consumed
	offs := s.offset - 1

	for s.ch != '"' {
		ch := s.ch
		s.next()
		if ch == '\n' || ch < 0 {
			s.error(offs, "string not terminated")
			break
		}
		if ch == '\\' {
			s.scanEscape(false)
		}
	}

	s.next()

	return string(s.src[offs:s.offset])
}

func stripCR(b []byte) []byte {
	c := make([]byte, len(b))
	i := 0
	for _, ch := range b {
		if ch != '\r' {
			c[i] = ch
			i++
		}
	}
	return c[:i]
}

func (s *Scanner) scanValString() string {
	offs := s.offset

	hasCR := false
	end := offs
	inQuote := false
loop:
	for inQuote || s.ch >= 0 && s.ch != '\n' && s.ch != ';' && s.ch != '#' {
		ch := s.ch
		s.next()
		switch {
		case inQuote && ch == '\\':
			s.scanEscape(true)
		case !inQuote && ch == '\\':
			if s.ch == '\r' {
				hasCR = true
				s.next()
			}
			if s.ch != '\n' {
				s.error(offs, "unquoted '\\' must be followed by new line")
				break loop
			}
			s.next()
		case ch == '"':
			inQuote = !inQuote
		case ch == '\r':
			hasCR = true
		case ch < 0 || inQuote && ch == '\n':
			s.error(offs, "string not terminated")
			break loop
		}
		if inQuote || !isWhiteSpace(ch) {
			end = s.offset
		}
	}

	lit := s.src[offs:end]
	if hasCR {
		lit = stripCR(lit)
	}

	return string(lit)
}

func isWhiteSpace(ch rune) bool {
	return ch == ' ' || ch == '\t' || ch == '\r'
}

func (s *Scanner) skipWhitespace() {
	for isWhiteSpace(s.ch) {
		s.next()
	}
}

// Scan scans the next token and returns the token position, the token,
// and its literal string if applicable. The source end is indicated by
// token.EOF.
//
// If the returned token is a literal (token.IDENT, token.STRING) or
// token.COMMENT, the literal string has the corresponding value.
//
// If the returned token is token.ILLEGAL, the literal string is the
// offending character.
//
// In all other cases, Scan returns an empty literal string.
//
// For more tolerant parsing, Scan will return a valid token if
// possible even if a syntax error was encountered. Thus, even
// if the resulting token sequence contains no illegal tokens,
// a client may not assume that no error occurred. Instead it
// must check the scanner's ErrorCount or the number of calls
// of the error handler, if there was one installed.
//
// Scan adds line information to the file added to the file
// set with Init. Token positions are relative to that file
// and thus relative to the file set.
//
func (s *Scanner) Scan() (pos token.Pos, tok token.Token, lit string) {
scanAgain:
	s.skipWhitespace()

	// current token start
	pos = s.file.Pos(s.offset)

	// determine token value
	switch ch := s.ch; {
	case s.nextVal:
		lit = s.scanValString()
		tok = token.STRING
		s.nextVal = false
	case isLetter(ch):
		lit = s.scanIdentifier()
		tok = token.IDENT
	default:
		s.next() // always make progress
		switch ch {
		case -1:
			tok = token.EOF
		case '\n':
			tok = token.EOL
		case '"':
			tok = token.STRING
			lit = s.scanString()
		case '[':
			tok = token.LBRACK
		case ']':
			tok = token.RBRACK
		case ';', '#':
			// comment
			lit = s.scanComment()
			if s.mode&ScanComments == 0 {
				// skip comment
				goto scanAgain
			}
			tok = token.COMMENT
		case '=':
			tok = token.ASSIGN
			s.nextVal = true
		default:
			s.error(s.file.Offset(pos), fmt.Sprintf("illegal character %#U", ch))
			tok = token.ILLEGAL
			lit = string(ch)
		}
	}

	return
}

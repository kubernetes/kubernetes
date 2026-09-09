/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package codetags

import (
	"bytes"
	"fmt"
	"strconv"
	"strings"
	"unicode"
)

type scanner struct {
	buf []rune
	pos int
}

func (s *scanner) next() rune {
	if s.pos >= len(s.buf) {
		return EOF
	}
	r := s.buf[s.pos]
	s.pos++
	return r
}

func (s *scanner) peek() rune {
	return s.peekN(0)
}

func (s *scanner) peekN(n int) rune {
	if s.pos+n >= len(s.buf) {
		return EOF
	}
	return s.buf[s.pos+n]
}

func (s *scanner) skipWhitespace() rune {
	for r := s.peek(); unicode.IsSpace(r); r = s.peek() {
		s.next()
	}
	return s.peek()
}

func (s *scanner) remainder() string {
	result := string(s.buf[s.pos:])
	s.pos = len(s.buf)
	return result
}

const (
	EOF = -1
)

func (s *scanner) nextIsTrailingComment() bool {
	i := 0
	for ; unicode.IsSpace(s.peekN(i)); i++ {
	}
	return s.peekN(i) == '#'
}

func (s *scanner) nextNumber() (string, error) {
	const (
		stBegin  = "stBegin"
		stPrefix = "stPrefix"
		stPosNeg = "stPosNeg"
		stNumber = "stNumber"
	)
	var buf bytes.Buffer
	st := stBegin

parseLoop:
	for r := s.peek(); r != EOF; r = s.peek() {
		switch st {
		case stBegin:
			switch {
			case r == '0':
				buf.WriteRune(s.next())
				st = stPrefix
			case r == '+' || r == '-':
				buf.WriteRune(s.next())
				st = stPosNeg
			case unicode.IsDigit(r):
				buf.WriteRune(s.next())
				st = stNumber
			default:
				break parseLoop
			}
		case stPosNeg:
			switch {
			case r == '0':
				buf.WriteRune(s.next())
				st = stPrefix
			case unicode.IsDigit(r):
				buf.WriteRune(s.next())
				st = stNumber
			default:
				break parseLoop
			}
		case stPrefix:
			switch {
			case unicode.IsDigit(r):
				buf.WriteRune(s.next())
				st = stNumber
			case r == 'x' || r == 'o' || r == 'b':
				buf.WriteRune(s.next())
				st = stNumber
			default:
				break parseLoop
			}
		case stNumber:
			const hexits = "abcdefABCDEF"
			switch {
			case unicode.IsDigit(r) || strings.Contains(hexits, string(r)):
				buf.WriteRune(s.next())
			default:
				break parseLoop
			}
		default:
			return "", fmt.Errorf("unexpected internal parser error: unknown state: %s at position %d", st, s.pos)
		}
	}
	numStr := buf.String()
	if _, err := strconv.ParseInt(numStr, 0, 64); err != nil {
		return "", fmt.Errorf("invalid number %q at position %d", numStr, s.pos)
	}
	return numStr, nil
}

func (s *scanner) nextString() (string, error) {
	const (
		stBegin        = "stBegin"
		stQuotedString = "stQuotedString"
		stEscape       = "stEscape"
	)
	var buf bytes.Buffer
	var quote rune
	var incomplete bool
	st := stBegin

parseLoop:
	for r := s.peek(); r != EOF; r = s.peek() {
		switch st {
		case stBegin:
			switch {
			case r == '"' || r == '`':
				incomplete = true
				quote = s.next() // consume quote
				st = stQuotedString
			default:
				return "", fmt.Errorf("expected string at position %d", s.pos)
			}
		case stQuotedString:
			switch {
			case r == '\\':
				s.next() // consume escape
				st = stEscape
			case r == quote:
				incomplete = false
				s.next()
				break parseLoop
			default:
				buf.WriteRune(s.next())
			}
		case stEscape:
			switch {
			case r == quote || r == '\\':
				buf.WriteRune(s.next())
				st = stQuotedString
			default:
				return "", fmt.Errorf("unhandled escaped character %q", r)
			}
		default:
			return "", fmt.Errorf("unexpected internal parser error: unknown state: %s at position %d", st, s.pos)
		}
	}
	if incomplete {
		return "", fmt.Errorf("unterminated string at position %d", s.pos)
	}
	return buf.String(), nil
}

func (s *scanner) nextIdent(isInteriorChar func(r rune) bool) (string, error) {
	const (
		stBegin    = "stBegin"
		stInterior = "stInterior"
	)
	var buf bytes.Buffer
	st := stBegin

parseLoop:
	for r := s.peek(); r != EOF; r = s.peek() {
		switch st {
		case stBegin:
			switch {
			case isIdentBegin(r):
				buf.WriteRune(s.next())
				st = stInterior
			default:
				return "", fmt.Errorf("expected identifier at position %d", s.pos)
			}
		case stInterior:
			switch {
			case isInteriorChar(r):
				buf.WriteRune(s.next())
			default:
				break parseLoop
			}
		default:
			return "", fmt.Errorf("unexpected internal parser error: unknown state: %s at position %d", st, s.pos)
		}
	}
	return buf.String(), nil
}

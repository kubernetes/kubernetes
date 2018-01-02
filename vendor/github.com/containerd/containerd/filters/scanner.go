package filters

import (
	"fmt"
	"unicode"
	"unicode/utf8"
)

const (
	tokenEOF = -(iota + 1)
	tokenQuoted
	tokenValue
	tokenField
	tokenSeparator
	tokenOperator
	tokenIllegal
)

type token rune

func (t token) String() string {
	switch t {
	case tokenEOF:
		return "EOF"
	case tokenQuoted:
		return "Quoted"
	case tokenValue:
		return "Value"
	case tokenField:
		return "Field"
	case tokenSeparator:
		return "Separator"
	case tokenOperator:
		return "Operator"
	case tokenIllegal:
		return "Illegal"
	}

	return string(t)
}

func (t token) GoString() string {
	return "token" + t.String()
}

type scanner struct {
	input string
	pos   int
	ppos  int // bounds the current rune in the string
	value bool
}

func (s *scanner) init(input string) {
	s.input = input
	s.pos = 0
	s.ppos = 0
}

func (s *scanner) next() rune {
	if s.pos >= len(s.input) {
		return tokenEOF
	}
	s.pos = s.ppos

	r, w := utf8.DecodeRuneInString(s.input[s.ppos:])
	s.ppos += w
	if r == utf8.RuneError {
		if w > 0 {
			return tokenIllegal
		}
		return tokenEOF
	}

	if r == 0 {
		return tokenIllegal
	}

	return r
}

func (s *scanner) peek() rune {
	pos := s.pos
	ppos := s.ppos
	ch := s.next()
	s.pos = pos
	s.ppos = ppos
	return ch
}

func (s *scanner) scan() (int, token, string) {
	var (
		ch  = s.next()
		pos = s.pos
	)

chomp:
	switch {
	case ch == tokenEOF:
	case ch == tokenIllegal:
	case isQuoteRune(ch):
		s.scanQuoted(ch)
		return pos, tokenQuoted, s.input[pos:s.ppos]
	case isSeparatorRune(ch):
		return pos, tokenSeparator, s.input[pos:s.ppos]
	case isOperatorRune(ch):
		s.scanOperator()
		s.value = true
		return pos, tokenOperator, s.input[pos:s.ppos]
	case unicode.IsSpace(ch):
		// chomp
		ch = s.next()
		pos = s.pos
		goto chomp
	case s.value:
		s.scanValue()
		s.value = false
		return pos, tokenValue, s.input[pos:s.ppos]
	case isFieldRune(ch):
		s.scanField()
		return pos, tokenField, s.input[pos:s.ppos]
	}

	return s.pos, token(ch), ""
}

func (s *scanner) scanField() {
	for {
		ch := s.peek()
		if !isFieldRune(ch) {
			break
		}
		s.next()
	}
}

func (s *scanner) scanOperator() {
	for {
		ch := s.peek()
		switch ch {
		case '=', '!', '~':
			s.next()
		default:
			return
		}
	}
}

func (s *scanner) scanValue() {
	for {
		ch := s.peek()
		if !isValueRune(ch) {
			break
		}
		s.next()
	}
}

func (s *scanner) scanQuoted(quote rune) {
	ch := s.next() // read character after quote
	for ch != quote {
		if ch == '\n' || ch < 0 {
			s.error("literal not terminated")
			return
		}
		if ch == '\\' {
			ch = s.scanEscape(quote)
		} else {
			ch = s.next()
		}
	}
	return
}

func (s *scanner) scanEscape(quote rune) rune {
	ch := s.next() // read character after '/'
	switch ch {
	case 'a', 'b', 'f', 'n', 'r', 't', 'v', '\\', quote:
		// nothing to do
		ch = s.next()
	case '0', '1', '2', '3', '4', '5', '6', '7':
		ch = s.scanDigits(ch, 8, 3)
	case 'x':
		ch = s.scanDigits(s.next(), 16, 2)
	case 'u':
		ch = s.scanDigits(s.next(), 16, 4)
	case 'U':
		ch = s.scanDigits(s.next(), 16, 8)
	default:
		s.error("illegal char escape")
	}
	return ch
}

func (s *scanner) scanDigits(ch rune, base, n int) rune {
	for n > 0 && digitVal(ch) < base {
		ch = s.next()
		n--
	}
	if n > 0 {
		s.error("illegal char escape")
	}
	return ch
}

func (s *scanner) error(msg string) {
	fmt.Println("error fixme", msg)
}

func digitVal(ch rune) int {
	switch {
	case '0' <= ch && ch <= '9':
		return int(ch - '0')
	case 'a' <= ch && ch <= 'f':
		return int(ch - 'a' + 10)
	case 'A' <= ch && ch <= 'F':
		return int(ch - 'A' + 10)
	}
	return 16 // larger than any legal digit val
}

func isFieldRune(r rune) bool {
	return (r == '_' || isAlphaRune(r) || isDigitRune(r))
}

func isAlphaRune(r rune) bool {
	return r >= 'A' && r <= 'Z' || r >= 'a' && r <= 'z'
}

func isDigitRune(r rune) bool {
	return r >= '0' && r <= '9'
}

func isOperatorRune(r rune) bool {
	switch r {
	case '=', '!', '~':
		return true
	}

	return false
}

func isQuoteRune(r rune) bool {
	switch r {
	case '"': // maybe add single quoting?
		return true
	}

	return false
}

func isSeparatorRune(r rune) bool {
	switch r {
	case ',', '.':
		return true
	}

	return false
}

func isValueRune(r rune) bool {
	return r != ',' && !unicode.IsSpace(r) &&
		(unicode.IsLetter(r) ||
			unicode.IsDigit(r) ||
			unicode.IsNumber(r) ||
			unicode.IsGraphic(r) ||
			unicode.IsPunct(r))
}

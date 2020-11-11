package pattern

import (
	"fmt"
	"go/token"
	"unicode"
	"unicode/utf8"
)

type lexer struct {
	f *token.File

	input string
	start int
	pos   int
	width int
	items chan item
}

type itemType int

const eof = -1

const (
	itemError itemType = iota
	itemLeftParen
	itemRightParen
	itemLeftBracket
	itemRightBracket
	itemTypeName
	itemVariable
	itemAt
	itemColon
	itemBlank
	itemString
	itemEOF
)

func (typ itemType) String() string {
	switch typ {
	case itemError:
		return "ERROR"
	case itemLeftParen:
		return "("
	case itemRightParen:
		return ")"
	case itemLeftBracket:
		return "["
	case itemRightBracket:
		return "]"
	case itemTypeName:
		return "TYPE"
	case itemVariable:
		return "VAR"
	case itemAt:
		return "@"
	case itemColon:
		return ":"
	case itemBlank:
		return "_"
	case itemString:
		return "STRING"
	case itemEOF:
		return "EOF"
	default:
		return fmt.Sprintf("itemType(%d)", typ)
	}
}

type item struct {
	typ itemType
	val string
	pos int
}

type stateFn func(*lexer) stateFn

func (l *lexer) run() {
	for state := lexStart; state != nil; {
		state = state(l)
	}
	close(l.items)
}

func (l *lexer) emitValue(t itemType, value string) {
	l.items <- item{t, value, l.start}
	l.start = l.pos
}

func (l *lexer) emit(t itemType) {
	l.items <- item{t, l.input[l.start:l.pos], l.start}
	l.start = l.pos
}

func lexStart(l *lexer) stateFn {
	switch r := l.next(); {
	case r == eof:
		l.emit(itemEOF)
		return nil
	case unicode.IsSpace(r):
		l.ignore()
	case r == '(':
		l.emit(itemLeftParen)
	case r == ')':
		l.emit(itemRightParen)
	case r == '[':
		l.emit(itemLeftBracket)
	case r == ']':
		l.emit(itemRightBracket)
	case r == '@':
		l.emit(itemAt)
	case r == ':':
		l.emit(itemColon)
	case r == '_':
		l.emit(itemBlank)
	case r == '"':
		l.backup()
		return lexString
	case unicode.IsUpper(r):
		l.backup()
		return lexType
	case unicode.IsLower(r):
		l.backup()
		return lexVariable
	default:
		return l.errorf("unexpected character %c", r)
	}
	return lexStart
}

func (l *lexer) next() (r rune) {
	if l.pos >= len(l.input) {
		l.width = 0
		return eof
	}
	r, l.width = utf8.DecodeRuneInString(l.input[l.pos:])

	if r == '\n' {
		l.f.AddLine(l.pos)
	}

	l.pos += l.width

	return r
}

func (l *lexer) ignore() {
	l.start = l.pos
}

func (l *lexer) backup() {
	l.pos -= l.width
}

func (l *lexer) errorf(format string, args ...interface{}) stateFn {
	// TODO(dh): emit position information in errors
	l.items <- item{
		itemError,
		fmt.Sprintf(format, args...),
		l.start,
	}
	return nil
}

func isAlphaNumeric(r rune) bool {
	return r >= '0' && r <= '9' ||
		r >= 'a' && r <= 'z' ||
		r >= 'A' && r <= 'Z'
}

func lexString(l *lexer) stateFn {
	l.next() // skip quote
	escape := false

	var runes []rune
	for {
		switch r := l.next(); r {
		case eof:
			return l.errorf("unterminated string")
		case '"':
			if !escape {
				l.emitValue(itemString, string(runes))
				return lexStart
			} else {
				runes = append(runes, '"')
				escape = false
			}
		case '\\':
			if escape {
				runes = append(runes, '\\')
				escape = false
			} else {
				escape = true
			}
		default:
			runes = append(runes, r)
		}
	}
}

func lexType(l *lexer) stateFn {
	l.next()
	for {
		if !isAlphaNumeric(l.next()) {
			l.backup()
			l.emit(itemTypeName)
			return lexStart
		}
	}
}

func lexVariable(l *lexer) stateFn {
	l.next()
	for {
		if !isAlphaNumeric(l.next()) {
			l.backup()
			l.emit(itemVariable)
			return lexStart
		}
	}
}

package lexer

import (
	"bytes"
	"fmt"
	"github.com/gobwas/glob/util/runes"
	"unicode/utf8"
)

const (
	char_any           = '*'
	char_comma         = ','
	char_single        = '?'
	char_escape        = '\\'
	char_range_open    = '['
	char_range_close   = ']'
	char_terms_open    = '{'
	char_terms_close   = '}'
	char_range_not     = '!'
	char_range_between = '-'
)

var specials = []byte{
	char_any,
	char_single,
	char_escape,
	char_range_open,
	char_range_close,
	char_terms_open,
	char_terms_close,
}

func Special(c byte) bool {
	return bytes.IndexByte(specials, c) != -1
}

type tokens []Token

func (i *tokens) shift() (ret Token) {
	ret = (*i)[0]
	copy(*i, (*i)[1:])
	*i = (*i)[:len(*i)-1]
	return
}

func (i *tokens) push(v Token) {
	*i = append(*i, v)
}

func (i *tokens) empty() bool {
	return len(*i) == 0
}

var eof rune = 0

type lexer struct {
	data string
	pos  int
	err  error

	tokens     tokens
	termsLevel int

	lastRune     rune
	lastRuneSize int
	hasRune      bool
}

func NewLexer(source string) *lexer {
	l := &lexer{
		data:   source,
		tokens: tokens(make([]Token, 0, 4)),
	}
	return l
}

func (l *lexer) Next() Token {
	if l.err != nil {
		return Token{Error, l.err.Error()}
	}
	if !l.tokens.empty() {
		return l.tokens.shift()
	}

	l.fetchItem()
	return l.Next()
}

func (l *lexer) peek() (r rune, w int) {
	if l.pos == len(l.data) {
		return eof, 0
	}

	r, w = utf8.DecodeRuneInString(l.data[l.pos:])
	if r == utf8.RuneError {
		l.errorf("could not read rune")
		r = eof
		w = 0
	}

	return
}

func (l *lexer) read() rune {
	if l.hasRune {
		l.hasRune = false
		l.seek(l.lastRuneSize)
		return l.lastRune
	}

	r, s := l.peek()
	l.seek(s)

	l.lastRune = r
	l.lastRuneSize = s

	return r
}

func (l *lexer) seek(w int) {
	l.pos += w
}

func (l *lexer) unread() {
	if l.hasRune {
		l.errorf("could not unread rune")
		return
	}
	l.seek(-l.lastRuneSize)
	l.hasRune = true
}

func (l *lexer) errorf(f string, v ...interface{}) {
	l.err = fmt.Errorf(f, v...)
}

func (l *lexer) inTerms() bool {
	return l.termsLevel > 0
}

func (l *lexer) termsEnter() {
	l.termsLevel++
}

func (l *lexer) termsLeave() {
	l.termsLevel--
}

var inTextBreakers = []rune{char_single, char_any, char_range_open, char_terms_open}
var inTermsBreakers = append(inTextBreakers, char_terms_close, char_comma)

func (l *lexer) fetchItem() {
	r := l.read()
	switch {
	case r == eof:
		l.tokens.push(Token{EOF, ""})

	case r == char_terms_open:
		l.termsEnter()
		l.tokens.push(Token{TermsOpen, string(r)})

	case r == char_comma && l.inTerms():
		l.tokens.push(Token{Separator, string(r)})

	case r == char_terms_close && l.inTerms():
		l.tokens.push(Token{TermsClose, string(r)})
		l.termsLeave()

	case r == char_range_open:
		l.tokens.push(Token{RangeOpen, string(r)})
		l.fetchRange()

	case r == char_single:
		l.tokens.push(Token{Single, string(r)})

	case r == char_any:
		if l.read() == char_any {
			l.tokens.push(Token{Super, string(r) + string(r)})
		} else {
			l.unread()
			l.tokens.push(Token{Any, string(r)})
		}

	default:
		l.unread()

		var breakers []rune
		if l.inTerms() {
			breakers = inTermsBreakers
		} else {
			breakers = inTextBreakers
		}
		l.fetchText(breakers)
	}
}

func (l *lexer) fetchRange() {
	var wantHi bool
	var wantClose bool
	var seenNot bool
	for {
		r := l.read()
		if r == eof {
			l.errorf("unexpected end of input")
			return
		}

		if wantClose {
			if r != char_range_close {
				l.errorf("expected close range character")
			} else {
				l.tokens.push(Token{RangeClose, string(r)})
			}
			return
		}

		if wantHi {
			l.tokens.push(Token{RangeHi, string(r)})
			wantClose = true
			continue
		}

		if !seenNot && r == char_range_not {
			l.tokens.push(Token{Not, string(r)})
			seenNot = true
			continue
		}

		if n, w := l.peek(); n == char_range_between {
			l.seek(w)
			l.tokens.push(Token{RangeLo, string(r)})
			l.tokens.push(Token{RangeBetween, string(n)})
			wantHi = true
			continue
		}

		l.unread() // unread first peek and fetch as text
		l.fetchText([]rune{char_range_close})
		wantClose = true
	}
}

func (l *lexer) fetchText(breakers []rune) {
	var data []rune
	var escaped bool

reading:
	for {
		r := l.read()
		if r == eof {
			break
		}

		if !escaped {
			if r == char_escape {
				escaped = true
				continue
			}

			if runes.IndexRune(breakers, r) != -1 {
				l.unread()
				break reading
			}
		}

		escaped = false
		data = append(data, r)
	}

	if len(data) > 0 {
		l.tokens.push(Token{Text, string(data)})
	}
}

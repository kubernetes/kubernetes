// TOML lexer.
//
// Written using the principles developed by Rob Pike in
// http://www.youtube.com/watch?v=HxaD_trXwRE

package toml

import (
	"bytes"
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

var dateRegexp *regexp.Regexp

// Define state functions
type tomlLexStateFn func() tomlLexStateFn

// Define lexer
type tomlLexer struct {
	inputIdx          int
	input             []rune // Textual source
	currentTokenStart int
	currentTokenStop  int
	tokens            []token
	depth             int
	line              int
	col               int
	endbufferLine     int
	endbufferCol      int
}

// Basic read operations on input

func (l *tomlLexer) read() rune {
	r := l.peek()
	if r == '\n' {
		l.endbufferLine++
		l.endbufferCol = 1
	} else {
		l.endbufferCol++
	}
	l.inputIdx++
	return r
}

func (l *tomlLexer) next() rune {
	r := l.read()

	if r != eof {
		l.currentTokenStop++
	}
	return r
}

func (l *tomlLexer) ignore() {
	l.currentTokenStart = l.currentTokenStop
	l.line = l.endbufferLine
	l.col = l.endbufferCol
}

func (l *tomlLexer) skip() {
	l.next()
	l.ignore()
}

func (l *tomlLexer) fastForward(n int) {
	for i := 0; i < n; i++ {
		l.next()
	}
}

func (l *tomlLexer) emitWithValue(t tokenType, value string) {
	l.tokens = append(l.tokens, token{
		Position: Position{l.line, l.col},
		typ:      t,
		val:      value,
	})
	l.ignore()
}

func (l *tomlLexer) emit(t tokenType) {
	l.emitWithValue(t, string(l.input[l.currentTokenStart:l.currentTokenStop]))
}

func (l *tomlLexer) peek() rune {
	if l.inputIdx >= len(l.input) {
		return eof
	}
	return l.input[l.inputIdx]
}

func (l *tomlLexer) peekString(size int) string {
	maxIdx := len(l.input)
	upperIdx := l.inputIdx + size // FIXME: potential overflow
	if upperIdx > maxIdx {
		upperIdx = maxIdx
	}
	return string(l.input[l.inputIdx:upperIdx])
}

func (l *tomlLexer) follow(next string) bool {
	return next == l.peekString(len(next))
}

// Error management

func (l *tomlLexer) errorf(format string, args ...interface{}) tomlLexStateFn {
	l.tokens = append(l.tokens, token{
		Position: Position{l.line, l.col},
		typ:      tokenError,
		val:      fmt.Sprintf(format, args...),
	})
	return nil
}

// State functions

func (l *tomlLexer) lexVoid() tomlLexStateFn {
	for {
		next := l.peek()
		switch next {
		case '[':
			return l.lexTableKey
		case '#':
			return l.lexComment(l.lexVoid)
		case '=':
			return l.lexEqual
		case '\r':
			fallthrough
		case '\n':
			l.skip()
			continue
		}

		if isSpace(next) {
			l.skip()
		}

		if l.depth > 0 {
			return l.lexRvalue
		}

		if isKeyStartChar(next) {
			return l.lexKey
		}

		if next == eof {
			l.next()
			break
		}
	}

	l.emit(tokenEOF)
	return nil
}

func (l *tomlLexer) lexRvalue() tomlLexStateFn {
	for {
		next := l.peek()
		switch next {
		case '.':
			return l.errorf("cannot start float with a dot")
		case '=':
			return l.lexEqual
		case '[':
			l.depth++
			return l.lexLeftBracket
		case ']':
			l.depth--
			return l.lexRightBracket
		case '{':
			return l.lexLeftCurlyBrace
		case '}':
			return l.lexRightCurlyBrace
		case '#':
			return l.lexComment(l.lexRvalue)
		case '"':
			return l.lexString
		case '\'':
			return l.lexLiteralString
		case ',':
			return l.lexComma
		case '\r':
			fallthrough
		case '\n':
			l.skip()
			if l.depth == 0 {
				return l.lexVoid
			}
			return l.lexRvalue
		case '_':
			return l.errorf("cannot start number with underscore")
		}

		if l.follow("true") {
			return l.lexTrue
		}

		if l.follow("false") {
			return l.lexFalse
		}

		if isSpace(next) {
			l.skip()
			continue
		}

		if next == eof {
			l.next()
			break
		}

		possibleDate := l.peekString(35)
		dateMatch := dateRegexp.FindString(possibleDate)
		if dateMatch != "" {
			l.fastForward(len(dateMatch))
			return l.lexDate
		}

		if next == '+' || next == '-' || isDigit(next) {
			return l.lexNumber
		}

		if isAlphanumeric(next) {
			return l.lexKey
		}

		return l.errorf("no value can start with %c", next)
	}

	l.emit(tokenEOF)
	return nil
}

func (l *tomlLexer) lexLeftCurlyBrace() tomlLexStateFn {
	l.next()
	l.emit(tokenLeftCurlyBrace)
	return l.lexRvalue
}

func (l *tomlLexer) lexRightCurlyBrace() tomlLexStateFn {
	l.next()
	l.emit(tokenRightCurlyBrace)
	return l.lexRvalue
}

func (l *tomlLexer) lexDate() tomlLexStateFn {
	l.emit(tokenDate)
	return l.lexRvalue
}

func (l *tomlLexer) lexTrue() tomlLexStateFn {
	l.fastForward(4)
	l.emit(tokenTrue)
	return l.lexRvalue
}

func (l *tomlLexer) lexFalse() tomlLexStateFn {
	l.fastForward(5)
	l.emit(tokenFalse)
	return l.lexRvalue
}

func (l *tomlLexer) lexEqual() tomlLexStateFn {
	l.next()
	l.emit(tokenEqual)
	return l.lexRvalue
}

func (l *tomlLexer) lexComma() tomlLexStateFn {
	l.next()
	l.emit(tokenComma)
	return l.lexRvalue
}

func (l *tomlLexer) lexKey() tomlLexStateFn {
	growingString := ""

	for r := l.peek(); isKeyChar(r) || r == '\n' || r == '\r'; r = l.peek() {
		if r == '"' {
			l.next()
			str, err := l.lexStringAsString(`"`, false, true)
			if err != nil {
				return l.errorf(err.Error())
			}
			growingString += `"` + str + `"`
			l.next()
			continue
		} else if r == '\n' {
			return l.errorf("keys cannot contain new lines")
		} else if isSpace(r) {
			break
		} else if !isValidBareChar(r) {
			return l.errorf("keys cannot contain %c character", r)
		}
		growingString += string(r)
		l.next()
	}
	l.emitWithValue(tokenKey, growingString)
	return l.lexVoid
}

func (l *tomlLexer) lexComment(previousState tomlLexStateFn) tomlLexStateFn {
	return func() tomlLexStateFn {
		for next := l.peek(); next != '\n' && next != eof; next = l.peek() {
			if next == '\r' && l.follow("\r\n") {
				break
			}
			l.next()
		}
		l.ignore()
		return previousState
	}
}

func (l *tomlLexer) lexLeftBracket() tomlLexStateFn {
	l.next()
	l.emit(tokenLeftBracket)
	return l.lexRvalue
}

func (l *tomlLexer) lexLiteralStringAsString(terminator string, discardLeadingNewLine bool) (string, error) {
	growingString := ""

	if discardLeadingNewLine {
		if l.follow("\r\n") {
			l.skip()
			l.skip()
		} else if l.peek() == '\n' {
			l.skip()
		}
	}

	// find end of string
	for {
		if l.follow(terminator) {
			return growingString, nil
		}

		next := l.peek()
		if next == eof {
			break
		}
		growingString += string(l.next())
	}

	return "", errors.New("unclosed string")
}

func (l *tomlLexer) lexLiteralString() tomlLexStateFn {
	l.skip()

	// handle special case for triple-quote
	terminator := "'"
	discardLeadingNewLine := false
	if l.follow("''") {
		l.skip()
		l.skip()
		terminator = "'''"
		discardLeadingNewLine = true
	}

	str, err := l.lexLiteralStringAsString(terminator, discardLeadingNewLine)
	if err != nil {
		return l.errorf(err.Error())
	}

	l.emitWithValue(tokenString, str)
	l.fastForward(len(terminator))
	l.ignore()
	return l.lexRvalue
}

// Lex a string and return the results as a string.
// Terminator is the substring indicating the end of the token.
// The resulting string does not include the terminator.
func (l *tomlLexer) lexStringAsString(terminator string, discardLeadingNewLine, acceptNewLines bool) (string, error) {
	growingString := ""

	if discardLeadingNewLine {
		if l.follow("\r\n") {
			l.skip()
			l.skip()
		} else if l.peek() == '\n' {
			l.skip()
		}
	}

	for {
		if l.follow(terminator) {
			return growingString, nil
		}

		if l.follow("\\") {
			l.next()
			switch l.peek() {
			case '\r':
				fallthrough
			case '\n':
				fallthrough
			case '\t':
				fallthrough
			case ' ':
				// skip all whitespace chars following backslash
				for strings.ContainsRune("\r\n\t ", l.peek()) {
					l.next()
				}
			case '"':
				growingString += "\""
				l.next()
			case 'n':
				growingString += "\n"
				l.next()
			case 'b':
				growingString += "\b"
				l.next()
			case 'f':
				growingString += "\f"
				l.next()
			case '/':
				growingString += "/"
				l.next()
			case 't':
				growingString += "\t"
				l.next()
			case 'r':
				growingString += "\r"
				l.next()
			case '\\':
				growingString += "\\"
				l.next()
			case 'u':
				l.next()
				code := ""
				for i := 0; i < 4; i++ {
					c := l.peek()
					if !isHexDigit(c) {
						return "", errors.New("unfinished unicode escape")
					}
					l.next()
					code = code + string(c)
				}
				intcode, err := strconv.ParseInt(code, 16, 32)
				if err != nil {
					return "", errors.New("invalid unicode escape: \\u" + code)
				}
				growingString += string(rune(intcode))
			case 'U':
				l.next()
				code := ""
				for i := 0; i < 8; i++ {
					c := l.peek()
					if !isHexDigit(c) {
						return "", errors.New("unfinished unicode escape")
					}
					l.next()
					code = code + string(c)
				}
				intcode, err := strconv.ParseInt(code, 16, 64)
				if err != nil {
					return "", errors.New("invalid unicode escape: \\U" + code)
				}
				growingString += string(rune(intcode))
			default:
				return "", errors.New("invalid escape sequence: \\" + string(l.peek()))
			}
		} else {
			r := l.peek()

			if 0x00 <= r && r <= 0x1F && !(acceptNewLines && (r == '\n' || r == '\r')) {
				return "", fmt.Errorf("unescaped control character %U", r)
			}
			l.next()
			growingString += string(r)
		}

		if l.peek() == eof {
			break
		}
	}

	return "", errors.New("unclosed string")
}

func (l *tomlLexer) lexString() tomlLexStateFn {
	l.skip()

	// handle special case for triple-quote
	terminator := `"`
	discardLeadingNewLine := false
	acceptNewLines := false
	if l.follow(`""`) {
		l.skip()
		l.skip()
		terminator = `"""`
		discardLeadingNewLine = true
		acceptNewLines = true
	}

	str, err := l.lexStringAsString(terminator, discardLeadingNewLine, acceptNewLines)

	if err != nil {
		return l.errorf(err.Error())
	}

	l.emitWithValue(tokenString, str)
	l.fastForward(len(terminator))
	l.ignore()
	return l.lexRvalue
}

func (l *tomlLexer) lexTableKey() tomlLexStateFn {
	l.next()

	if l.peek() == '[' {
		// token '[[' signifies an array of tables
		l.next()
		l.emit(tokenDoubleLeftBracket)
		return l.lexInsideTableArrayKey
	}
	// vanilla table key
	l.emit(tokenLeftBracket)
	return l.lexInsideTableKey
}

func (l *tomlLexer) lexInsideTableArrayKey() tomlLexStateFn {
	for r := l.peek(); r != eof; r = l.peek() {
		switch r {
		case ']':
			if l.currentTokenStop > l.currentTokenStart {
				l.emit(tokenKeyGroupArray)
			}
			l.next()
			if l.peek() != ']' {
				break
			}
			l.next()
			l.emit(tokenDoubleRightBracket)
			return l.lexVoid
		case '[':
			return l.errorf("table array key cannot contain ']'")
		default:
			l.next()
		}
	}
	return l.errorf("unclosed table array key")
}

func (l *tomlLexer) lexInsideTableKey() tomlLexStateFn {
	for r := l.peek(); r != eof; r = l.peek() {
		switch r {
		case ']':
			if l.currentTokenStop > l.currentTokenStart {
				l.emit(tokenKeyGroup)
			}
			l.next()
			l.emit(tokenRightBracket)
			return l.lexVoid
		case '[':
			return l.errorf("table key cannot contain ']'")
		default:
			l.next()
		}
	}
	return l.errorf("unclosed table key")
}

func (l *tomlLexer) lexRightBracket() tomlLexStateFn {
	l.next()
	l.emit(tokenRightBracket)
	return l.lexRvalue
}

func (l *tomlLexer) lexNumber() tomlLexStateFn {
	r := l.peek()
	if r == '+' || r == '-' {
		l.next()
	}
	pointSeen := false
	expSeen := false
	digitSeen := false
	for {
		next := l.peek()
		if next == '.' {
			if pointSeen {
				return l.errorf("cannot have two dots in one float")
			}
			l.next()
			if !isDigit(l.peek()) {
				return l.errorf("float cannot end with a dot")
			}
			pointSeen = true
		} else if next == 'e' || next == 'E' {
			expSeen = true
			l.next()
			r := l.peek()
			if r == '+' || r == '-' {
				l.next()
			}
		} else if isDigit(next) {
			digitSeen = true
			l.next()
		} else if next == '_' {
			l.next()
		} else {
			break
		}
		if pointSeen && !digitSeen {
			return l.errorf("cannot start float with a dot")
		}
	}

	if !digitSeen {
		return l.errorf("no digit in that number")
	}
	if pointSeen || expSeen {
		l.emit(tokenFloat)
	} else {
		l.emit(tokenInteger)
	}
	return l.lexRvalue
}

func (l *tomlLexer) run() {
	for state := l.lexVoid; state != nil; {
		state = state()
	}
}

func init() {
	dateRegexp = regexp.MustCompile(`^\d{1,4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,9})?(Z|[+-]\d{2}:\d{2})`)
}

// Entry point
func lexToml(inputBytes []byte) []token {
	runes := bytes.Runes(inputBytes)
	l := &tomlLexer{
		input:         runes,
		tokens:        make([]token, 0, 256),
		line:          1,
		col:           1,
		endbufferLine: 1,
		endbufferCol:  1,
	}
	l.run()
	return l.tokens
}

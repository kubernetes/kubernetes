// TOML lexer.
//
// Written using the principles developped by Rob Pike in
// http://www.youtube.com/watch?v=HxaD_trXwRE

package toml

import (
	"errors"
	"fmt"
	"io"
	"regexp"
	"strconv"
	"strings"

	"github.com/pelletier/go-buffruneio"
)

var dateRegexp *regexp.Regexp

// Define state functions
type tomlLexStateFn func() tomlLexStateFn

// Define lexer
type tomlLexer struct {
	input         *buffruneio.Reader // Textual source
	buffer        []rune             // Runes composing the current token
	tokens        chan token
	depth         int
	line          int
	col           int
	endbufferLine int
	endbufferCol  int
}

// Basic read operations on input

func (l *tomlLexer) read() rune {
	r, err := l.input.ReadRune()
	if err != nil {
		panic(err)
	}
	if r == '\n' {
		l.endbufferLine++
		l.endbufferCol = 1
	} else {
		l.endbufferCol++
	}
	return r
}

func (l *tomlLexer) next() rune {
	r := l.read()

	if r != eof {
		l.buffer = append(l.buffer, r)
	}
	return r
}

func (l *tomlLexer) ignore() {
	l.buffer = make([]rune, 0)
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
	l.tokens <- token{
		Position: Position{l.line, l.col},
		typ:      t,
		val:      value,
	}
	l.ignore()
}

func (l *tomlLexer) emit(t tokenType) {
	l.emitWithValue(t, string(l.buffer))
}

func (l *tomlLexer) peek() rune {
	r, err := l.input.ReadRune()
	if err != nil {
		panic(err)
	}
	l.input.UnreadRune()
	return r
}

func (l *tomlLexer) follow(next string) bool {
	for _, expectedRune := range next {
		r, err := l.input.ReadRune()
		defer l.input.UnreadRune()
		if err != nil {
			panic(err)
		}
		if expectedRune != r {
			return false
		}
	}
	return true
}

// Error management

func (l *tomlLexer) errorf(format string, args ...interface{}) tomlLexStateFn {
	l.tokens <- token{
		Position: Position{l.line, l.col},
		typ:      tokenError,
		val:      fmt.Sprintf(format, args...),
	}
	return nil
}

// State functions

func (l *tomlLexer) lexVoid() tomlLexStateFn {
	for {
		next := l.peek()
		switch next {
		case '[':
			return l.lexKeyGroup
		case '#':
			return l.lexComment
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
			return l.lexComment
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

		possibleDate := string(l.input.Peek(35))
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

func (l *tomlLexer) lexComment() tomlLexStateFn {
	for next := l.peek(); next != '\n' && next != eof; next = l.peek() {
		if next == '\r' && l.follow("\r\n") {
			break
		}
		l.next()
	}
	l.ignore()
	return l.lexVoid
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

func (l *tomlLexer) lexKeyGroup() tomlLexStateFn {
	l.next()

	if l.peek() == '[' {
		// token '[[' signifies an array of anonymous key groups
		l.next()
		l.emit(tokenDoubleLeftBracket)
		return l.lexInsideKeyGroupArray
	}
	// vanilla key group
	l.emit(tokenLeftBracket)
	return l.lexInsideKeyGroup
}

func (l *tomlLexer) lexInsideKeyGroupArray() tomlLexStateFn {
	for r := l.peek(); r != eof; r = l.peek() {
		switch r {
		case ']':
			if len(l.buffer) > 0 {
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
			return l.errorf("group name cannot contain ']'")
		default:
			l.next()
		}
	}
	return l.errorf("unclosed key group array")
}

func (l *tomlLexer) lexInsideKeyGroup() tomlLexStateFn {
	for r := l.peek(); r != eof; r = l.peek() {
		switch r {
		case ']':
			if len(l.buffer) > 0 {
				l.emit(tokenKeyGroup)
			}
			l.next()
			l.emit(tokenRightBracket)
			return l.lexVoid
		case '[':
			return l.errorf("group name cannot contain ']'")
		default:
			l.next()
		}
	}
	return l.errorf("unclosed key group")
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
	close(l.tokens)
}

func init() {
	dateRegexp = regexp.MustCompile(`^\d{1,4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,9})?(Z|[+-]\d{2}:\d{2})`)
}

// Entry point
func lexToml(input io.Reader) chan token {
	bufferedInput := buffruneio.NewReader(input)
	l := &tomlLexer{
		input:         bufferedInput,
		tokens:        make(chan token),
		line:          1,
		col:           1,
		endbufferLine: 1,
		endbufferCol:  1,
	}
	go l.run()
	return l.tokens
}

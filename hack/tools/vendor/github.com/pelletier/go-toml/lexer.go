// TOML lexer.
//
// Written using the principles developed by Rob Pike in
// http://www.youtube.com/watch?v=HxaD_trXwRE

package toml

import (
	"bytes"
	"errors"
	"fmt"
	"strconv"
	"strings"
)

// Define state functions
type tomlLexStateFn func() tomlLexStateFn

// Define lexer
type tomlLexer struct {
	inputIdx          int
	input             []rune // Textual source
	currentTokenStart int
	currentTokenStop  int
	tokens            []token
	brackets          []rune
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
		case '}': // after '{'
			return l.lexRightCurlyBrace
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
			return l.lexLeftBracket
		case ']':
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
			if len(l.brackets) > 0 && l.brackets[len(l.brackets)-1] == '[' {
				return l.lexRvalue
			}
			return l.lexVoid
		}

		if l.follow("true") {
			return l.lexTrue
		}

		if l.follow("false") {
			return l.lexFalse
		}

		if l.follow("inf") {
			return l.lexInf
		}

		if l.follow("nan") {
			return l.lexNan
		}

		if isSpace(next) {
			l.skip()
			continue
		}

		if next == eof {
			l.next()
			break
		}

		if next == '+' || next == '-' {
			return l.lexNumber
		}

		if isDigit(next) {
			return l.lexDateTimeOrNumber
		}

		return l.errorf("no value can start with %c", next)
	}

	l.emit(tokenEOF)
	return nil
}

func (l *tomlLexer) lexDateTimeOrNumber() tomlLexStateFn {
	// Could be either a date/time, or a digit.
	// The options for date/times are:
	//   YYYY-... => date or date-time
	//   HH:... => time
	// Anything else should be a number.

	lookAhead := l.peekString(5)
	if len(lookAhead) < 3 {
		return l.lexNumber()
	}

	for idx, r := range lookAhead {
		if !isDigit(r) {
			if idx == 2 && r == ':' {
				return l.lexDateTimeOrTime()
			}
			if idx == 4 && r == '-' {
				return l.lexDateTimeOrTime()
			}
			return l.lexNumber()
		}
	}
	return l.lexNumber()
}

func (l *tomlLexer) lexLeftCurlyBrace() tomlLexStateFn {
	l.next()
	l.emit(tokenLeftCurlyBrace)
	l.brackets = append(l.brackets, '{')
	return l.lexVoid
}

func (l *tomlLexer) lexRightCurlyBrace() tomlLexStateFn {
	l.next()
	l.emit(tokenRightCurlyBrace)
	if len(l.brackets) == 0 || l.brackets[len(l.brackets)-1] != '{' {
		return l.errorf("cannot have '}' here")
	}
	l.brackets = l.brackets[:len(l.brackets)-1]
	return l.lexRvalue
}

func (l *tomlLexer) lexDateTimeOrTime() tomlLexStateFn {
	// Example matches:
	// 1979-05-27T07:32:00Z
	// 1979-05-27T00:32:00-07:00
	// 1979-05-27T00:32:00.999999-07:00
	// 1979-05-27 07:32:00Z
	// 1979-05-27 00:32:00-07:00
	// 1979-05-27 00:32:00.999999-07:00
	// 1979-05-27T07:32:00
	// 1979-05-27T00:32:00.999999
	// 1979-05-27 07:32:00
	// 1979-05-27 00:32:00.999999
	// 1979-05-27
	// 07:32:00
	// 00:32:00.999999

	// we already know those two are digits
	l.next()
	l.next()

	// Got 2 digits. At that point it could be either a time or a date(-time).

	r := l.next()
	if r == ':' {
		return l.lexTime()
	}

	return l.lexDateTime()
}

func (l *tomlLexer) lexDateTime() tomlLexStateFn {
	// This state accepts an offset date-time, a local date-time, or a local date.
	//
	//   v--- cursor
	// 1979-05-27T07:32:00Z
	// 1979-05-27T00:32:00-07:00
	// 1979-05-27T00:32:00.999999-07:00
	// 1979-05-27 07:32:00Z
	// 1979-05-27 00:32:00-07:00
	// 1979-05-27 00:32:00.999999-07:00
	// 1979-05-27T07:32:00
	// 1979-05-27T00:32:00.999999
	// 1979-05-27 07:32:00
	// 1979-05-27 00:32:00.999999
	// 1979-05-27

	// date

	// already checked by lexRvalue
	l.next() // digit
	l.next() // -

	for i := 0; i < 2; i++ {
		r := l.next()
		if !isDigit(r) {
			return l.errorf("invalid month digit in date: %c", r)
		}
	}

	r := l.next()
	if r != '-' {
		return l.errorf("expected - to separate month of a date, not %c", r)
	}

	for i := 0; i < 2; i++ {
		r := l.next()
		if !isDigit(r) {
			return l.errorf("invalid day digit in date: %c", r)
		}
	}

	l.emit(tokenLocalDate)

	r = l.peek()

	if r == eof {

		return l.lexRvalue
	}

	if r != ' ' && r != 'T' {
		return l.errorf("incorrect date/time separation character: %c", r)
	}

	if r == ' ' {
		lookAhead := l.peekString(3)[1:]
		if len(lookAhead) < 2 {
			return l.lexRvalue
		}
		for _, r := range lookAhead {
			if !isDigit(r) {
				return l.lexRvalue
			}
		}
	}

	l.skip() // skip the T or ' '

	// time

	for i := 0; i < 2; i++ {
		r := l.next()
		if !isDigit(r) {
			return l.errorf("invalid hour digit in time: %c", r)
		}
	}

	r = l.next()
	if r != ':' {
		return l.errorf("time hour/minute separator should be :, not %c", r)
	}

	for i := 0; i < 2; i++ {
		r := l.next()
		if !isDigit(r) {
			return l.errorf("invalid minute digit in time: %c", r)
		}
	}

	r = l.next()
	if r != ':' {
		return l.errorf("time minute/second separator should be :, not %c", r)
	}

	for i := 0; i < 2; i++ {
		r := l.next()
		if !isDigit(r) {
			return l.errorf("invalid second digit in time: %c", r)
		}
	}

	r = l.peek()
	if r == '.' {
		l.next()
		r := l.next()
		if !isDigit(r) {
			return l.errorf("expected at least one digit in time's fraction, not %c", r)
		}

		for {
			r := l.peek()
			if !isDigit(r) {
				break
			}
			l.next()
		}
	}

	l.emit(tokenLocalTime)

	return l.lexTimeOffset

}

func (l *tomlLexer) lexTimeOffset() tomlLexStateFn {
	// potential offset

	// Z
	// -07:00
	// +07:00
	// nothing

	r := l.peek()

	if r == 'Z' {
		l.next()
		l.emit(tokenTimeOffset)
	} else if r == '+' || r == '-' {
		l.next()

		for i := 0; i < 2; i++ {
			r := l.next()
			if !isDigit(r) {
				return l.errorf("invalid hour digit in time offset: %c", r)
			}
		}

		r = l.next()
		if r != ':' {
			return l.errorf("time offset hour/minute separator should be :, not %c", r)
		}

		for i := 0; i < 2; i++ {
			r := l.next()
			if !isDigit(r) {
				return l.errorf("invalid minute digit in time offset: %c", r)
			}
		}

		l.emit(tokenTimeOffset)
	}

	return l.lexRvalue
}

func (l *tomlLexer) lexTime() tomlLexStateFn {
	//   v--- cursor
	// 07:32:00
	// 00:32:00.999999

	for i := 0; i < 2; i++ {
		r := l.next()
		if !isDigit(r) {
			return l.errorf("invalid minute digit in time: %c", r)
		}
	}

	r := l.next()
	if r != ':' {
		return l.errorf("time minute/second separator should be :, not %c", r)
	}

	for i := 0; i < 2; i++ {
		r := l.next()
		if !isDigit(r) {
			return l.errorf("invalid second digit in time: %c", r)
		}
	}

	r = l.peek()
	if r == '.' {
		l.next()
		r := l.next()
		if !isDigit(r) {
			return l.errorf("expected at least one digit in time's fraction, not %c", r)
		}

		for {
			r := l.peek()
			if !isDigit(r) {
				break
			}
			l.next()
		}
	}

	l.emit(tokenLocalTime)
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

func (l *tomlLexer) lexInf() tomlLexStateFn {
	l.fastForward(3)
	l.emit(tokenInf)
	return l.lexRvalue
}

func (l *tomlLexer) lexNan() tomlLexStateFn {
	l.fastForward(3)
	l.emit(tokenNan)
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
	if len(l.brackets) > 0 && l.brackets[len(l.brackets)-1] == '{' {
		return l.lexVoid
	}
	return l.lexRvalue
}

// Parse the key and emits its value without escape sequences.
// bare keys, basic string keys and literal string keys are supported.
func (l *tomlLexer) lexKey() tomlLexStateFn {
	var sb strings.Builder

	for r := l.peek(); isKeyChar(r) || r == '\n' || r == '\r'; r = l.peek() {
		if r == '"' {
			l.next()
			str, err := l.lexStringAsString(`"`, false, true)
			if err != nil {
				return l.errorf(err.Error())
			}
			sb.WriteString("\"")
			sb.WriteString(str)
			sb.WriteString("\"")
			l.next()
			continue
		} else if r == '\'' {
			l.next()
			str, err := l.lexLiteralStringAsString(`'`, false)
			if err != nil {
				return l.errorf(err.Error())
			}
			sb.WriteString("'")
			sb.WriteString(str)
			sb.WriteString("'")
			l.next()
			continue
		} else if r == '\n' {
			return l.errorf("keys cannot contain new lines")
		} else if isSpace(r) {
			var str strings.Builder
			str.WriteString(" ")

			// skip trailing whitespace
			l.next()
			for r = l.peek(); isSpace(r); r = l.peek() {
				str.WriteRune(r)
				l.next()
			}
			// break loop if not a dot
			if r != '.' {
				break
			}
			str.WriteString(".")
			// skip trailing whitespace after dot
			l.next()
			for r = l.peek(); isSpace(r); r = l.peek() {
				str.WriteRune(r)
				l.next()
			}
			sb.WriteString(str.String())
			continue
		} else if r == '.' {
			// skip
		} else if !isValidBareChar(r) {
			return l.errorf("keys cannot contain %c character", r)
		}
		sb.WriteRune(r)
		l.next()
	}
	l.emitWithValue(tokenKey, sb.String())
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
	l.brackets = append(l.brackets, '[')
	return l.lexRvalue
}

func (l *tomlLexer) lexLiteralStringAsString(terminator string, discardLeadingNewLine bool) (string, error) {
	var sb strings.Builder

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
			return sb.String(), nil
		}

		next := l.peek()
		if next == eof {
			break
		}
		sb.WriteRune(l.next())
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
	var sb strings.Builder

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
			return sb.String(), nil
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
				sb.WriteString("\"")
				l.next()
			case 'n':
				sb.WriteString("\n")
				l.next()
			case 'b':
				sb.WriteString("\b")
				l.next()
			case 'f':
				sb.WriteString("\f")
				l.next()
			case '/':
				sb.WriteString("/")
				l.next()
			case 't':
				sb.WriteString("\t")
				l.next()
			case 'r':
				sb.WriteString("\r")
				l.next()
			case '\\':
				sb.WriteString("\\")
				l.next()
			case 'u':
				l.next()
				var code strings.Builder
				for i := 0; i < 4; i++ {
					c := l.peek()
					if !isHexDigit(c) {
						return "", errors.New("unfinished unicode escape")
					}
					l.next()
					code.WriteRune(c)
				}
				intcode, err := strconv.ParseInt(code.String(), 16, 32)
				if err != nil {
					return "", errors.New("invalid unicode escape: \\u" + code.String())
				}
				sb.WriteRune(rune(intcode))
			case 'U':
				l.next()
				var code strings.Builder
				for i := 0; i < 8; i++ {
					c := l.peek()
					if !isHexDigit(c) {
						return "", errors.New("unfinished unicode escape")
					}
					l.next()
					code.WriteRune(c)
				}
				intcode, err := strconv.ParseInt(code.String(), 16, 64)
				if err != nil {
					return "", errors.New("invalid unicode escape: \\U" + code.String())
				}
				sb.WriteRune(rune(intcode))
			default:
				return "", errors.New("invalid escape sequence: \\" + string(l.peek()))
			}
		} else {
			r := l.peek()

			if 0x00 <= r && r <= 0x1F && r != '\t' && !(acceptNewLines && (r == '\n' || r == '\r')) {
				return "", fmt.Errorf("unescaped control character %U", r)
			}
			l.next()
			sb.WriteRune(r)
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

// Parse the key till "]]", but only bare keys are supported
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

// Parse the key till "]" but only bare keys are supported
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
	if len(l.brackets) == 0 || l.brackets[len(l.brackets)-1] != '[' {
		return l.errorf("cannot have ']' here")
	}
	l.brackets = l.brackets[:len(l.brackets)-1]
	return l.lexRvalue
}

type validRuneFn func(r rune) bool

func isValidHexRune(r rune) bool {
	return r >= 'a' && r <= 'f' ||
		r >= 'A' && r <= 'F' ||
		r >= '0' && r <= '9' ||
		r == '_'
}

func isValidOctalRune(r rune) bool {
	return r >= '0' && r <= '7' || r == '_'
}

func isValidBinaryRune(r rune) bool {
	return r == '0' || r == '1' || r == '_'
}

func (l *tomlLexer) lexNumber() tomlLexStateFn {
	r := l.peek()

	if r == '0' {
		follow := l.peekString(2)
		if len(follow) == 2 {
			var isValidRune validRuneFn
			switch follow[1] {
			case 'x':
				isValidRune = isValidHexRune
			case 'o':
				isValidRune = isValidOctalRune
			case 'b':
				isValidRune = isValidBinaryRune
			default:
				if follow[1] >= 'a' && follow[1] <= 'z' || follow[1] >= 'A' && follow[1] <= 'Z' {
					return l.errorf("unknown number base: %s. possible options are x (hex) o (octal) b (binary)", string(follow[1]))
				}
			}

			if isValidRune != nil {
				l.next()
				l.next()
				digitSeen := false
				for {
					next := l.peek()
					if !isValidRune(next) {
						break
					}
					digitSeen = true
					l.next()
				}

				if !digitSeen {
					return l.errorf("number needs at least one digit")
				}

				l.emit(tokenInteger)

				return l.lexRvalue
			}
		}
	}

	if r == '+' || r == '-' {
		l.next()
		if l.follow("inf") {
			return l.lexInf
		}
		if l.follow("nan") {
			return l.lexNan
		}
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

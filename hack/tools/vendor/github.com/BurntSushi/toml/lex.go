package toml

import (
	"fmt"
	"reflect"
	"runtime"
	"strings"
	"unicode"
	"unicode/utf8"
)

type itemType int

const (
	itemError itemType = iota
	itemNIL            // used in the parser to indicate no type
	itemEOF
	itemText
	itemString
	itemRawString
	itemMultilineString
	itemRawMultilineString
	itemBool
	itemInteger
	itemFloat
	itemDatetime
	itemArray // the start of an array
	itemArrayEnd
	itemTableStart
	itemTableEnd
	itemArrayTableStart
	itemArrayTableEnd
	itemKeyStart
	itemKeyEnd
	itemCommentStart
	itemInlineTableStart
	itemInlineTableEnd
)

const eof = 0

type stateFn func(lx *lexer) stateFn

func (p Position) String() string {
	return fmt.Sprintf("at line %d; start %d; length %d", p.Line, p.Start, p.Len)
}

type lexer struct {
	input string
	start int
	pos   int
	line  int
	state stateFn
	items chan item

	// Allow for backing up up to 4 runes. This is necessary because TOML
	// contains 3-rune tokens (""" and ''').
	prevWidths [4]int
	nprev      int  // how many of prevWidths are in use
	atEOF      bool // If we emit an eof, we can still back up, but it is not OK to call next again.

	// A stack of state functions used to maintain context.
	//
	// The idea is to reuse parts of the state machine in various places. For
	// example, values can appear at the top level or within arbitrarily nested
	// arrays. The last state on the stack is used after a value has been lexed.
	// Similarly for comments.
	stack []stateFn
}

type item struct {
	typ itemType
	val string
	err error
	pos Position
}

func (lx *lexer) nextItem() item {
	for {
		select {
		case item := <-lx.items:
			return item
		default:
			lx.state = lx.state(lx)
			//fmt.Printf("     STATE %-24s  current: %-10q	stack: %s\n", lx.state, lx.current(), lx.stack)
		}
	}
}

func lex(input string) *lexer {
	lx := &lexer{
		input: input,
		state: lexTop,
		items: make(chan item, 10),
		stack: make([]stateFn, 0, 10),
		line:  1,
	}
	return lx
}

func (lx *lexer) push(state stateFn) {
	lx.stack = append(lx.stack, state)
}

func (lx *lexer) pop() stateFn {
	if len(lx.stack) == 0 {
		return lx.errorf("BUG in lexer: no states to pop")
	}
	last := lx.stack[len(lx.stack)-1]
	lx.stack = lx.stack[0 : len(lx.stack)-1]
	return last
}

func (lx *lexer) current() string {
	return lx.input[lx.start:lx.pos]
}

func (lx lexer) getPos() Position {
	p := Position{
		Line:  lx.line,
		Start: lx.start,
		Len:   lx.pos - lx.start,
	}
	if p.Len <= 0 {
		p.Len = 1
	}
	return p
}

func (lx *lexer) emit(typ itemType) {
	lx.items <- item{typ: typ, pos: lx.getPos(), val: lx.current()}
	lx.start = lx.pos
}

func (lx *lexer) emitTrim(typ itemType) {
	lx.items <- item{typ: typ, pos: lx.getPos(), val: strings.TrimSpace(lx.current())}
	lx.start = lx.pos
}

func (lx *lexer) next() (r rune) {
	if lx.atEOF {
		panic("BUG in lexer: next called after EOF")
	}
	if lx.pos >= len(lx.input) {
		lx.atEOF = true
		return eof
	}

	if lx.input[lx.pos] == '\n' {
		lx.line++
	}
	lx.prevWidths[3] = lx.prevWidths[2]
	lx.prevWidths[2] = lx.prevWidths[1]
	lx.prevWidths[1] = lx.prevWidths[0]
	if lx.nprev < 4 {
		lx.nprev++
	}

	r, w := utf8.DecodeRuneInString(lx.input[lx.pos:])
	if r == utf8.RuneError {
		lx.error(errLexUTF8{lx.input[lx.pos]})
		return utf8.RuneError
	}

	// Note: don't use peek() here, as this calls next().
	if isControl(r) || (r == '\r' && (len(lx.input)-1 == lx.pos || lx.input[lx.pos+1] != '\n')) {
		lx.errorControlChar(r)
		return utf8.RuneError
	}

	lx.prevWidths[0] = w
	lx.pos += w
	return r
}

// ignore skips over the pending input before this point.
func (lx *lexer) ignore() {
	lx.start = lx.pos
}

// backup steps back one rune. Can be called 4 times between calls to next.
func (lx *lexer) backup() {
	if lx.atEOF {
		lx.atEOF = false
		return
	}
	if lx.nprev < 1 {
		panic("BUG in lexer: backed up too far")
	}
	w := lx.prevWidths[0]
	lx.prevWidths[0] = lx.prevWidths[1]
	lx.prevWidths[1] = lx.prevWidths[2]
	lx.prevWidths[2] = lx.prevWidths[3]
	lx.nprev--

	lx.pos -= w
	if lx.pos < len(lx.input) && lx.input[lx.pos] == '\n' {
		lx.line--
	}
}

// accept consumes the next rune if it's equal to `valid`.
func (lx *lexer) accept(valid rune) bool {
	if lx.next() == valid {
		return true
	}
	lx.backup()
	return false
}

// peek returns but does not consume the next rune in the input.
func (lx *lexer) peek() rune {
	r := lx.next()
	lx.backup()
	return r
}

// skip ignores all input that matches the given predicate.
func (lx *lexer) skip(pred func(rune) bool) {
	for {
		r := lx.next()
		if pred(r) {
			continue
		}
		lx.backup()
		lx.ignore()
		return
	}
}

// error stops all lexing by emitting an error and returning `nil`.
//
// Note that any value that is a character is escaped if it's a special
// character (newlines, tabs, etc.).
func (lx *lexer) error(err error) stateFn {
	if lx.atEOF {
		return lx.errorPrevLine(err)
	}
	lx.items <- item{typ: itemError, pos: lx.getPos(), err: err}
	return nil
}

// errorfPrevline is like error(), but sets the position to the last column of
// the previous line.
//
// This is so that unexpected EOF or NL errors don't show on a new blank line.
func (lx *lexer) errorPrevLine(err error) stateFn {
	pos := lx.getPos()
	pos.Line--
	pos.Len = 1
	pos.Start = lx.pos - 1
	lx.items <- item{typ: itemError, pos: pos, err: err}
	return nil
}

// errorPos is like error(), but allows explicitly setting the position.
func (lx *lexer) errorPos(start, length int, err error) stateFn {
	pos := lx.getPos()
	pos.Start = start
	pos.Len = length
	lx.items <- item{typ: itemError, pos: pos, err: err}
	return nil
}

// errorf is like error, and creates a new error.
func (lx *lexer) errorf(format string, values ...interface{}) stateFn {
	if lx.atEOF {
		pos := lx.getPos()
		pos.Line--
		pos.Len = 1
		pos.Start = lx.pos - 1
		lx.items <- item{typ: itemError, pos: pos, err: fmt.Errorf(format, values...)}
		return nil
	}
	lx.items <- item{typ: itemError, pos: lx.getPos(), err: fmt.Errorf(format, values...)}
	return nil
}

func (lx *lexer) errorControlChar(cc rune) stateFn {
	return lx.errorPos(lx.pos-1, 1, errLexControl{cc})
}

// lexTop consumes elements at the top level of TOML data.
func lexTop(lx *lexer) stateFn {
	r := lx.next()
	if isWhitespace(r) || isNL(r) {
		return lexSkip(lx, lexTop)
	}
	switch r {
	case '#':
		lx.push(lexTop)
		return lexCommentStart
	case '[':
		return lexTableStart
	case eof:
		if lx.pos > lx.start {
			return lx.errorf("unexpected EOF")
		}
		lx.emit(itemEOF)
		return nil
	}

	// At this point, the only valid item can be a key, so we back up
	// and let the key lexer do the rest.
	lx.backup()
	lx.push(lexTopEnd)
	return lexKeyStart
}

// lexTopEnd is entered whenever a top-level item has been consumed. (A value
// or a table.) It must see only whitespace, and will turn back to lexTop
// upon a newline. If it sees EOF, it will quit the lexer successfully.
func lexTopEnd(lx *lexer) stateFn {
	r := lx.next()
	switch {
	case r == '#':
		// a comment will read to a newline for us.
		lx.push(lexTop)
		return lexCommentStart
	case isWhitespace(r):
		return lexTopEnd
	case isNL(r):
		lx.ignore()
		return lexTop
	case r == eof:
		lx.emit(itemEOF)
		return nil
	}
	return lx.errorf(
		"expected a top-level item to end with a newline, comment, or EOF, but got %q instead",
		r)
}

// lexTable lexes the beginning of a table. Namely, it makes sure that
// it starts with a character other than '.' and ']'.
// It assumes that '[' has already been consumed.
// It also handles the case that this is an item in an array of tables.
// e.g., '[[name]]'.
func lexTableStart(lx *lexer) stateFn {
	if lx.peek() == '[' {
		lx.next()
		lx.emit(itemArrayTableStart)
		lx.push(lexArrayTableEnd)
	} else {
		lx.emit(itemTableStart)
		lx.push(lexTableEnd)
	}
	return lexTableNameStart
}

func lexTableEnd(lx *lexer) stateFn {
	lx.emit(itemTableEnd)
	return lexTopEnd
}

func lexArrayTableEnd(lx *lexer) stateFn {
	if r := lx.next(); r != ']' {
		return lx.errorf("expected end of table array name delimiter ']', but got %q instead", r)
	}
	lx.emit(itemArrayTableEnd)
	return lexTopEnd
}

func lexTableNameStart(lx *lexer) stateFn {
	lx.skip(isWhitespace)
	switch r := lx.peek(); {
	case r == ']' || r == eof:
		return lx.errorf("unexpected end of table name (table names cannot be empty)")
	case r == '.':
		return lx.errorf("unexpected table separator (table names cannot be empty)")
	case r == '"' || r == '\'':
		lx.ignore()
		lx.push(lexTableNameEnd)
		return lexQuotedName
	default:
		lx.push(lexTableNameEnd)
		return lexBareName
	}
}

// lexTableNameEnd reads the end of a piece of a table name, optionally
// consuming whitespace.
func lexTableNameEnd(lx *lexer) stateFn {
	lx.skip(isWhitespace)
	switch r := lx.next(); {
	case isWhitespace(r):
		return lexTableNameEnd
	case r == '.':
		lx.ignore()
		return lexTableNameStart
	case r == ']':
		return lx.pop()
	default:
		return lx.errorf("expected '.' or ']' to end table name, but got %q instead", r)
	}
}

// lexBareName lexes one part of a key or table.
//
// It assumes that at least one valid character for the table has already been
// read.
//
// Lexes only one part, e.g. only 'a' inside 'a.b'.
func lexBareName(lx *lexer) stateFn {
	r := lx.next()
	if isBareKeyChar(r) {
		return lexBareName
	}
	lx.backup()
	lx.emit(itemText)
	return lx.pop()
}

// lexBareName lexes one part of a key or table.
//
// It assumes that at least one valid character for the table has already been
// read.
//
// Lexes only one part, e.g. only '"a"' inside '"a".b'.
func lexQuotedName(lx *lexer) stateFn {
	r := lx.next()
	switch {
	case isWhitespace(r):
		return lexSkip(lx, lexValue)
	case r == '"':
		lx.ignore() // ignore the '"'
		return lexString
	case r == '\'':
		lx.ignore() // ignore the "'"
		return lexRawString
	case r == eof:
		return lx.errorf("unexpected EOF; expected value")
	default:
		return lx.errorf("expected value but found %q instead", r)
	}
}

// lexKeyStart consumes all key parts until a '='.
func lexKeyStart(lx *lexer) stateFn {
	lx.skip(isWhitespace)
	switch r := lx.peek(); {
	case r == '=' || r == eof:
		return lx.errorf("unexpected '=': key name appears blank")
	case r == '.':
		return lx.errorf("unexpected '.': keys cannot start with a '.'")
	case r == '"' || r == '\'':
		lx.ignore()
		fallthrough
	default: // Bare key
		lx.emit(itemKeyStart)
		return lexKeyNameStart
	}
}

func lexKeyNameStart(lx *lexer) stateFn {
	lx.skip(isWhitespace)
	switch r := lx.peek(); {
	case r == '=' || r == eof:
		return lx.errorf("unexpected '='")
	case r == '.':
		return lx.errorf("unexpected '.'")
	case r == '"' || r == '\'':
		lx.ignore()
		lx.push(lexKeyEnd)
		return lexQuotedName
	default:
		lx.push(lexKeyEnd)
		return lexBareName
	}
}

// lexKeyEnd consumes the end of a key and trims whitespace (up to the key
// separator).
func lexKeyEnd(lx *lexer) stateFn {
	lx.skip(isWhitespace)
	switch r := lx.next(); {
	case isWhitespace(r):
		return lexSkip(lx, lexKeyEnd)
	case r == eof:
		return lx.errorf("unexpected EOF; expected key separator '='")
	case r == '.':
		lx.ignore()
		return lexKeyNameStart
	case r == '=':
		lx.emit(itemKeyEnd)
		return lexSkip(lx, lexValue)
	default:
		return lx.errorf("expected '.' or '=', but got %q instead", r)
	}
}

// lexValue starts the consumption of a value anywhere a value is expected.
// lexValue will ignore whitespace.
// After a value is lexed, the last state on the next is popped and returned.
func lexValue(lx *lexer) stateFn {
	// We allow whitespace to precede a value, but NOT newlines.
	// In array syntax, the array states are responsible for ignoring newlines.
	r := lx.next()
	switch {
	case isWhitespace(r):
		return lexSkip(lx, lexValue)
	case isDigit(r):
		lx.backup() // avoid an extra state and use the same as above
		return lexNumberOrDateStart
	}
	switch r {
	case '[':
		lx.ignore()
		lx.emit(itemArray)
		return lexArrayValue
	case '{':
		lx.ignore()
		lx.emit(itemInlineTableStart)
		return lexInlineTableValue
	case '"':
		if lx.accept('"') {
			if lx.accept('"') {
				lx.ignore() // Ignore """
				return lexMultilineString
			}
			lx.backup()
		}
		lx.ignore() // ignore the '"'
		return lexString
	case '\'':
		if lx.accept('\'') {
			if lx.accept('\'') {
				lx.ignore() // Ignore """
				return lexMultilineRawString
			}
			lx.backup()
		}
		lx.ignore() // ignore the "'"
		return lexRawString
	case '.': // special error case, be kind to users
		return lx.errorf("floats must start with a digit, not '.'")
	case 'i', 'n':
		if (lx.accept('n') && lx.accept('f')) || (lx.accept('a') && lx.accept('n')) {
			lx.emit(itemFloat)
			return lx.pop()
		}
	case '-', '+':
		return lexDecimalNumberStart
	}
	if unicode.IsLetter(r) {
		// Be permissive here; lexBool will give a nice error if the
		// user wrote something like
		//   x = foo
		// (i.e. not 'true' or 'false' but is something else word-like.)
		lx.backup()
		return lexBool
	}
	if r == eof {
		return lx.errorf("unexpected EOF; expected value")
	}
	return lx.errorf("expected value but found %q instead", r)
}

// lexArrayValue consumes one value in an array. It assumes that '[' or ','
// have already been consumed. All whitespace and newlines are ignored.
func lexArrayValue(lx *lexer) stateFn {
	r := lx.next()
	switch {
	case isWhitespace(r) || isNL(r):
		return lexSkip(lx, lexArrayValue)
	case r == '#':
		lx.push(lexArrayValue)
		return lexCommentStart
	case r == ',':
		return lx.errorf("unexpected comma")
	case r == ']':
		return lexArrayEnd
	}

	lx.backup()
	lx.push(lexArrayValueEnd)
	return lexValue
}

// lexArrayValueEnd consumes everything between the end of an array value and
// the next value (or the end of the array): it ignores whitespace and newlines
// and expects either a ',' or a ']'.
func lexArrayValueEnd(lx *lexer) stateFn {
	switch r := lx.next(); {
	case isWhitespace(r) || isNL(r):
		return lexSkip(lx, lexArrayValueEnd)
	case r == '#':
		lx.push(lexArrayValueEnd)
		return lexCommentStart
	case r == ',':
		lx.ignore()
		return lexArrayValue // move on to the next value
	case r == ']':
		return lexArrayEnd
	default:
		return lx.errorf("expected a comma (',') or array terminator (']'), but got %s", runeOrEOF(r))
	}
}

// lexArrayEnd finishes the lexing of an array.
// It assumes that a ']' has just been consumed.
func lexArrayEnd(lx *lexer) stateFn {
	lx.ignore()
	lx.emit(itemArrayEnd)
	return lx.pop()
}

// lexInlineTableValue consumes one key/value pair in an inline table.
// It assumes that '{' or ',' have already been consumed. Whitespace is ignored.
func lexInlineTableValue(lx *lexer) stateFn {
	r := lx.next()
	switch {
	case isWhitespace(r):
		return lexSkip(lx, lexInlineTableValue)
	case isNL(r):
		return lx.errorPrevLine(errLexInlineTableNL{})
	case r == '#':
		lx.push(lexInlineTableValue)
		return lexCommentStart
	case r == ',':
		return lx.errorf("unexpected comma")
	case r == '}':
		return lexInlineTableEnd
	}
	lx.backup()
	lx.push(lexInlineTableValueEnd)
	return lexKeyStart
}

// lexInlineTableValueEnd consumes everything between the end of an inline table
// key/value pair and the next pair (or the end of the table):
// it ignores whitespace and expects either a ',' or a '}'.
func lexInlineTableValueEnd(lx *lexer) stateFn {
	switch r := lx.next(); {
	case isWhitespace(r):
		return lexSkip(lx, lexInlineTableValueEnd)
	case isNL(r):
		return lx.errorPrevLine(errLexInlineTableNL{})
	case r == '#':
		lx.push(lexInlineTableValueEnd)
		return lexCommentStart
	case r == ',':
		lx.ignore()
		lx.skip(isWhitespace)
		if lx.peek() == '}' {
			return lx.errorf("trailing comma not allowed in inline tables")
		}
		return lexInlineTableValue
	case r == '}':
		return lexInlineTableEnd
	default:
		return lx.errorf("expected a comma or an inline table terminator '}', but got %s instead", runeOrEOF(r))
	}
}

func runeOrEOF(r rune) string {
	if r == eof {
		return "end of file"
	}
	return "'" + string(r) + "'"
}

// lexInlineTableEnd finishes the lexing of an inline table.
// It assumes that a '}' has just been consumed.
func lexInlineTableEnd(lx *lexer) stateFn {
	lx.ignore()
	lx.emit(itemInlineTableEnd)
	return lx.pop()
}

// lexString consumes the inner contents of a string. It assumes that the
// beginning '"' has already been consumed and ignored.
func lexString(lx *lexer) stateFn {
	r := lx.next()
	switch {
	case r == eof:
		return lx.errorf(`unexpected EOF; expected '"'`)
	case isNL(r):
		return lx.errorPrevLine(errLexStringNL{})
	case r == '\\':
		lx.push(lexString)
		return lexStringEscape
	case r == '"':
		lx.backup()
		lx.emit(itemString)
		lx.next()
		lx.ignore()
		return lx.pop()
	}
	return lexString
}

// lexMultilineString consumes the inner contents of a string. It assumes that
// the beginning '"""' has already been consumed and ignored.
func lexMultilineString(lx *lexer) stateFn {
	r := lx.next()
	switch r {
	default:
		return lexMultilineString
	case eof:
		return lx.errorf(`unexpected EOF; expected '"""'`)
	case '\\':
		return lexMultilineStringEscape
	case '"':
		/// Found " → try to read two more "".
		if lx.accept('"') {
			if lx.accept('"') {
				/// Peek ahead: the string can contain " and "", including at the
				/// end: """str"""""
				/// 6 or more at the end, however, is an error.
				if lx.peek() == '"' {
					/// Check if we already lexed 5 's; if so we have 6 now, and
					/// that's just too many man!
					if strings.HasSuffix(lx.current(), `"""""`) {
						return lx.errorf(`unexpected '""""""'`)
					}
					lx.backup()
					lx.backup()
					return lexMultilineString
				}

				lx.backup() /// backup: don't include the """ in the item.
				lx.backup()
				lx.backup()
				lx.emit(itemMultilineString)
				lx.next() /// Read over ''' again and discard it.
				lx.next()
				lx.next()
				lx.ignore()
				return lx.pop()
			}
			lx.backup()
		}
		return lexMultilineString
	}
}

// lexRawString consumes a raw string. Nothing can be escaped in such a string.
// It assumes that the beginning "'" has already been consumed and ignored.
func lexRawString(lx *lexer) stateFn {
	r := lx.next()
	switch {
	default:
		return lexRawString
	case r == eof:
		return lx.errorf(`unexpected EOF; expected "'"`)
	case isNL(r):
		return lx.errorPrevLine(errLexStringNL{})
	case r == '\'':
		lx.backup()
		lx.emit(itemRawString)
		lx.next()
		lx.ignore()
		return lx.pop()
	}
}

// lexMultilineRawString consumes a raw string. Nothing can be escaped in such
// a string. It assumes that the beginning "'''" has already been consumed and
// ignored.
func lexMultilineRawString(lx *lexer) stateFn {
	r := lx.next()
	switch r {
	default:
		return lexMultilineRawString
	case eof:
		return lx.errorf(`unexpected EOF; expected "'''"`)
	case '\'':
		/// Found ' → try to read two more ''.
		if lx.accept('\'') {
			if lx.accept('\'') {
				/// Peek ahead: the string can contain ' and '', including at the
				/// end: '''str'''''
				/// 6 or more at the end, however, is an error.
				if lx.peek() == '\'' {
					/// Check if we already lexed 5 's; if so we have 6 now, and
					/// that's just too many man!
					if strings.HasSuffix(lx.current(), "'''''") {
						return lx.errorf(`unexpected "''''''"`)
					}
					lx.backup()
					lx.backup()
					return lexMultilineRawString
				}

				lx.backup() /// backup: don't include the ''' in the item.
				lx.backup()
				lx.backup()
				lx.emit(itemRawMultilineString)
				lx.next() /// Read over ''' again and discard it.
				lx.next()
				lx.next()
				lx.ignore()
				return lx.pop()
			}
			lx.backup()
		}
		return lexMultilineRawString
	}
}

// lexMultilineStringEscape consumes an escaped character. It assumes that the
// preceding '\\' has already been consumed.
func lexMultilineStringEscape(lx *lexer) stateFn {
	// Handle the special case first:
	if isNL(lx.next()) {
		return lexMultilineString
	}
	lx.backup()
	lx.push(lexMultilineString)
	return lexStringEscape(lx)
}

func lexStringEscape(lx *lexer) stateFn {
	r := lx.next()
	switch r {
	case 'b':
		fallthrough
	case 't':
		fallthrough
	case 'n':
		fallthrough
	case 'f':
		fallthrough
	case 'r':
		fallthrough
	case '"':
		fallthrough
	case ' ', '\t':
		// Inside """ .. """ strings you can use \ to escape newlines, and any
		// amount of whitespace can be between the \ and \n.
		fallthrough
	case '\\':
		return lx.pop()
	case 'u':
		return lexShortUnicodeEscape
	case 'U':
		return lexLongUnicodeEscape
	}
	return lx.error(errLexEscape{r})
}

func lexShortUnicodeEscape(lx *lexer) stateFn {
	var r rune
	for i := 0; i < 4; i++ {
		r = lx.next()
		if !isHexadecimal(r) {
			return lx.errorf(
				`expected four hexadecimal digits after '\u', but got %q instead`,
				lx.current())
		}
	}
	return lx.pop()
}

func lexLongUnicodeEscape(lx *lexer) stateFn {
	var r rune
	for i := 0; i < 8; i++ {
		r = lx.next()
		if !isHexadecimal(r) {
			return lx.errorf(
				`expected eight hexadecimal digits after '\U', but got %q instead`,
				lx.current())
		}
	}
	return lx.pop()
}

// lexNumberOrDateStart processes the first character of a value which begins
// with a digit. It exists to catch values starting with '0', so that
// lexBaseNumberOrDate can differentiate base prefixed integers from other
// types.
func lexNumberOrDateStart(lx *lexer) stateFn {
	r := lx.next()
	switch r {
	case '0':
		return lexBaseNumberOrDate
	}

	if !isDigit(r) {
		// The only way to reach this state is if the value starts
		// with a digit, so specifically treat anything else as an
		// error.
		return lx.errorf("expected a digit but got %q", r)
	}

	return lexNumberOrDate
}

// lexNumberOrDate consumes either an integer, float or datetime.
func lexNumberOrDate(lx *lexer) stateFn {
	r := lx.next()
	if isDigit(r) {
		return lexNumberOrDate
	}
	switch r {
	case '-', ':':
		return lexDatetime
	case '_':
		return lexDecimalNumber
	case '.', 'e', 'E':
		return lexFloat
	}

	lx.backup()
	lx.emit(itemInteger)
	return lx.pop()
}

// lexDatetime consumes a Datetime, to a first approximation.
// The parser validates that it matches one of the accepted formats.
func lexDatetime(lx *lexer) stateFn {
	r := lx.next()
	if isDigit(r) {
		return lexDatetime
	}
	switch r {
	case '-', ':', 'T', 't', ' ', '.', 'Z', 'z', '+':
		return lexDatetime
	}

	lx.backup()
	lx.emitTrim(itemDatetime)
	return lx.pop()
}

// lexHexInteger consumes a hexadecimal integer after seeing the '0x' prefix.
func lexHexInteger(lx *lexer) stateFn {
	r := lx.next()
	if isHexadecimal(r) {
		return lexHexInteger
	}
	switch r {
	case '_':
		return lexHexInteger
	}

	lx.backup()
	lx.emit(itemInteger)
	return lx.pop()
}

// lexOctalInteger consumes an octal integer after seeing the '0o' prefix.
func lexOctalInteger(lx *lexer) stateFn {
	r := lx.next()
	if isOctal(r) {
		return lexOctalInteger
	}
	switch r {
	case '_':
		return lexOctalInteger
	}

	lx.backup()
	lx.emit(itemInteger)
	return lx.pop()
}

// lexBinaryInteger consumes a binary integer after seeing the '0b' prefix.
func lexBinaryInteger(lx *lexer) stateFn {
	r := lx.next()
	if isBinary(r) {
		return lexBinaryInteger
	}
	switch r {
	case '_':
		return lexBinaryInteger
	}

	lx.backup()
	lx.emit(itemInteger)
	return lx.pop()
}

// lexDecimalNumber consumes a decimal float or integer.
func lexDecimalNumber(lx *lexer) stateFn {
	r := lx.next()
	if isDigit(r) {
		return lexDecimalNumber
	}
	switch r {
	case '.', 'e', 'E':
		return lexFloat
	case '_':
		return lexDecimalNumber
	}

	lx.backup()
	lx.emit(itemInteger)
	return lx.pop()
}

// lexDecimalNumber consumes the first digit of a number beginning with a sign.
// It assumes the sign has already been consumed. Values which start with a sign
// are only allowed to be decimal integers or floats.
//
// The special "nan" and "inf" values are also recognized.
func lexDecimalNumberStart(lx *lexer) stateFn {
	r := lx.next()

	// Special error cases to give users better error messages
	switch r {
	case 'i':
		if !lx.accept('n') || !lx.accept('f') {
			return lx.errorf("invalid float: '%s'", lx.current())
		}
		lx.emit(itemFloat)
		return lx.pop()
	case 'n':
		if !lx.accept('a') || !lx.accept('n') {
			return lx.errorf("invalid float: '%s'", lx.current())
		}
		lx.emit(itemFloat)
		return lx.pop()
	case '0':
		p := lx.peek()
		switch p {
		case 'b', 'o', 'x':
			return lx.errorf("cannot use sign with non-decimal numbers: '%s%c'", lx.current(), p)
		}
	case '.':
		return lx.errorf("floats must start with a digit, not '.'")
	}

	if isDigit(r) {
		return lexDecimalNumber
	}

	return lx.errorf("expected a digit but got %q", r)
}

// lexBaseNumberOrDate differentiates between the possible values which
// start with '0'. It assumes that before reaching this state, the initial '0'
// has been consumed.
func lexBaseNumberOrDate(lx *lexer) stateFn {
	r := lx.next()
	// Note: All datetimes start with at least two digits, so we don't
	// handle date characters (':', '-', etc.) here.
	if isDigit(r) {
		return lexNumberOrDate
	}
	switch r {
	case '_':
		// Can only be decimal, because there can't be an underscore
		// between the '0' and the base designator, and dates can't
		// contain underscores.
		return lexDecimalNumber
	case '.', 'e', 'E':
		return lexFloat
	case 'b':
		r = lx.peek()
		if !isBinary(r) {
			lx.errorf("not a binary number: '%s%c'", lx.current(), r)
		}
		return lexBinaryInteger
	case 'o':
		r = lx.peek()
		if !isOctal(r) {
			lx.errorf("not an octal number: '%s%c'", lx.current(), r)
		}
		return lexOctalInteger
	case 'x':
		r = lx.peek()
		if !isHexadecimal(r) {
			lx.errorf("not a hexidecimal number: '%s%c'", lx.current(), r)
		}
		return lexHexInteger
	}

	lx.backup()
	lx.emit(itemInteger)
	return lx.pop()
}

// lexFloat consumes the elements of a float. It allows any sequence of
// float-like characters, so floats emitted by the lexer are only a first
// approximation and must be validated by the parser.
func lexFloat(lx *lexer) stateFn {
	r := lx.next()
	if isDigit(r) {
		return lexFloat
	}
	switch r {
	case '_', '.', '-', '+', 'e', 'E':
		return lexFloat
	}

	lx.backup()
	lx.emit(itemFloat)
	return lx.pop()
}

// lexBool consumes a bool string: 'true' or 'false.
func lexBool(lx *lexer) stateFn {
	var rs []rune
	for {
		r := lx.next()
		if !unicode.IsLetter(r) {
			lx.backup()
			break
		}
		rs = append(rs, r)
	}
	s := string(rs)
	switch s {
	case "true", "false":
		lx.emit(itemBool)
		return lx.pop()
	}
	return lx.errorf("expected value but found %q instead", s)
}

// lexCommentStart begins the lexing of a comment. It will emit
// itemCommentStart and consume no characters, passing control to lexComment.
func lexCommentStart(lx *lexer) stateFn {
	lx.ignore()
	lx.emit(itemCommentStart)
	return lexComment
}

// lexComment lexes an entire comment. It assumes that '#' has been consumed.
// It will consume *up to* the first newline character, and pass control
// back to the last state on the stack.
func lexComment(lx *lexer) stateFn {
	switch r := lx.next(); {
	case isNL(r) || r == eof:
		lx.backup()
		lx.emit(itemText)
		return lx.pop()
	default:
		return lexComment
	}
}

// lexSkip ignores all slurped input and moves on to the next state.
func lexSkip(lx *lexer, nextState stateFn) stateFn {
	lx.ignore()
	return nextState
}

func (s stateFn) String() string {
	name := runtime.FuncForPC(reflect.ValueOf(s).Pointer()).Name()
	if i := strings.LastIndexByte(name, '.'); i > -1 {
		name = name[i+1:]
	}
	if s == nil {
		name = "<nil>"
	}
	return name + "()"
}

func (itype itemType) String() string {
	switch itype {
	case itemError:
		return "Error"
	case itemNIL:
		return "NIL"
	case itemEOF:
		return "EOF"
	case itemText:
		return "Text"
	case itemString, itemRawString, itemMultilineString, itemRawMultilineString:
		return "String"
	case itemBool:
		return "Bool"
	case itemInteger:
		return "Integer"
	case itemFloat:
		return "Float"
	case itemDatetime:
		return "DateTime"
	case itemTableStart:
		return "TableStart"
	case itemTableEnd:
		return "TableEnd"
	case itemKeyStart:
		return "KeyStart"
	case itemKeyEnd:
		return "KeyEnd"
	case itemArray:
		return "Array"
	case itemArrayEnd:
		return "ArrayEnd"
	case itemCommentStart:
		return "CommentStart"
	case itemInlineTableStart:
		return "InlineTableStart"
	case itemInlineTableEnd:
		return "InlineTableEnd"
	}
	panic(fmt.Sprintf("BUG: Unknown type '%d'.", int(itype)))
}

func (item item) String() string {
	return fmt.Sprintf("(%s, %s)", item.typ.String(), item.val)
}

func isWhitespace(r rune) bool { return r == '\t' || r == ' ' }
func isNL(r rune) bool         { return r == '\n' || r == '\r' }
func isControl(r rune) bool { // Control characters except \t, \r, \n
	switch r {
	case '\t', '\r', '\n':
		return false
	default:
		return (r >= 0x00 && r <= 0x1f) || r == 0x7f
	}
}
func isDigit(r rune) bool  { return r >= '0' && r <= '9' }
func isBinary(r rune) bool { return r == '0' || r == '1' }
func isOctal(r rune) bool  { return r >= '0' && r <= '7' }
func isHexadecimal(r rune) bool {
	return (r >= '0' && r <= '9') || (r >= 'a' && r <= 'f') || (r >= 'A' && r <= 'F')
}
func isBareKeyChar(r rune) bool {
	return (r >= 'A' && r <= 'Z') ||
		(r >= 'a' && r <= 'z') ||
		(r >= '0' && r <= '9') ||
		r == '_' || r == '-'
}

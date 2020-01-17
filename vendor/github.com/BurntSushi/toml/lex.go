package toml

import (
	"fmt"
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
	itemCommentStart
	itemInlineTableStart
	itemInlineTableEnd
)

const (
	eof              = 0
	comma            = ','
	tableStart       = '['
	tableEnd         = ']'
	arrayTableStart  = '['
	arrayTableEnd    = ']'
	tableSep         = '.'
	keySep           = '='
	arrayStart       = '['
	arrayEnd         = ']'
	commentStart     = '#'
	stringStart      = '"'
	stringEnd        = '"'
	rawStringStart   = '\''
	rawStringEnd     = '\''
	inlineTableStart = '{'
	inlineTableEnd   = '}'
)

type stateFn func(lx *lexer) stateFn

type lexer struct {
	input string
	start int
	pos   int
	line  int
	state stateFn
	items chan item

	// Allow for backing up up to three runes.
	// This is necessary because TOML contains 3-rune tokens (""" and ''').
	prevWidths [3]int
	nprev      int // how many of prevWidths are in use
	// If we emit an eof, we can still back up, but it is not OK to call
	// next again.
	atEOF bool

	// A stack of state functions used to maintain context.
	// The idea is to reuse parts of the state machine in various places.
	// For example, values can appear at the top level or within arbitrarily
	// nested arrays. The last state on the stack is used after a value has
	// been lexed. Similarly for comments.
	stack []stateFn
}

type item struct {
	typ  itemType
	val  string
	line int
}

func (lx *lexer) nextItem() item {
	for {
		select {
		case item := <-lx.items:
			return item
		default:
			lx.state = lx.state(lx)
		}
	}
}

func lex(input string) *lexer {
	lx := &lexer{
		input: input,
		state: lexTop,
		line:  1,
		items: make(chan item, 10),
		stack: make([]stateFn, 0, 10),
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

func (lx *lexer) emit(typ itemType) {
	lx.items <- item{typ, lx.current(), lx.line}
	lx.start = lx.pos
}

func (lx *lexer) emitTrim(typ itemType) {
	lx.items <- item{typ, strings.TrimSpace(lx.current()), lx.line}
	lx.start = lx.pos
}

func (lx *lexer) next() (r rune) {
	if lx.atEOF {
		panic("next called after EOF")
	}
	if lx.pos >= len(lx.input) {
		lx.atEOF = true
		return eof
	}

	if lx.input[lx.pos] == '\n' {
		lx.line++
	}
	lx.prevWidths[2] = lx.prevWidths[1]
	lx.prevWidths[1] = lx.prevWidths[0]
	if lx.nprev < 3 {
		lx.nprev++
	}
	r, w := utf8.DecodeRuneInString(lx.input[lx.pos:])
	lx.prevWidths[0] = w
	lx.pos += w
	return r
}

// ignore skips over the pending input before this point.
func (lx *lexer) ignore() {
	lx.start = lx.pos
}

// backup steps back one rune. Can be called only twice between calls to next.
func (lx *lexer) backup() {
	if lx.atEOF {
		lx.atEOF = false
		return
	}
	if lx.nprev < 1 {
		panic("backed up too far")
	}
	w := lx.prevWidths[0]
	lx.prevWidths[0] = lx.prevWidths[1]
	lx.prevWidths[1] = lx.prevWidths[2]
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

// errorf stops all lexing by emitting an error and returning `nil`.
// Note that any value that is a character is escaped if it's a special
// character (newlines, tabs, etc.).
func (lx *lexer) errorf(format string, values ...interface{}) stateFn {
	lx.items <- item{
		itemError,
		fmt.Sprintf(format, values...),
		lx.line,
	}
	return nil
}

// lexTop consumes elements at the top level of TOML data.
func lexTop(lx *lexer) stateFn {
	r := lx.next()
	if isWhitespace(r) || isNL(r) {
		return lexSkip(lx, lexTop)
	}
	switch r {
	case commentStart:
		lx.push(lexTop)
		return lexCommentStart
	case tableStart:
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
	case r == commentStart:
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
	return lx.errorf("expected a top-level item to end with a newline, "+
		"comment, or EOF, but got %q instead", r)
}

// lexTable lexes the beginning of a table. Namely, it makes sure that
// it starts with a character other than '.' and ']'.
// It assumes that '[' has already been consumed.
// It also handles the case that this is an item in an array of tables.
// e.g., '[[name]]'.
func lexTableStart(lx *lexer) stateFn {
	if lx.peek() == arrayTableStart {
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
	if r := lx.next(); r != arrayTableEnd {
		return lx.errorf("expected end of table array name delimiter %q, "+
			"but got %q instead", arrayTableEnd, r)
	}
	lx.emit(itemArrayTableEnd)
	return lexTopEnd
}

func lexTableNameStart(lx *lexer) stateFn {
	lx.skip(isWhitespace)
	switch r := lx.peek(); {
	case r == tableEnd || r == eof:
		return lx.errorf("unexpected end of table name " +
			"(table names cannot be empty)")
	case r == tableSep:
		return lx.errorf("unexpected table separator " +
			"(table names cannot be empty)")
	case r == stringStart || r == rawStringStart:
		lx.ignore()
		lx.push(lexTableNameEnd)
		return lexValue // reuse string lexing
	default:
		return lexBareTableName
	}
}

// lexBareTableName lexes the name of a table. It assumes that at least one
// valid character for the table has already been read.
func lexBareTableName(lx *lexer) stateFn {
	r := lx.next()
	if isBareKeyChar(r) {
		return lexBareTableName
	}
	lx.backup()
	lx.emit(itemText)
	return lexTableNameEnd
}

// lexTableNameEnd reads the end of a piece of a table name, optionally
// consuming whitespace.
func lexTableNameEnd(lx *lexer) stateFn {
	lx.skip(isWhitespace)
	switch r := lx.next(); {
	case isWhitespace(r):
		return lexTableNameEnd
	case r == tableSep:
		lx.ignore()
		return lexTableNameStart
	case r == tableEnd:
		return lx.pop()
	default:
		return lx.errorf("expected '.' or ']' to end table name, "+
			"but got %q instead", r)
	}
}

// lexKeyStart consumes a key name up until the first non-whitespace character.
// lexKeyStart will ignore whitespace.
func lexKeyStart(lx *lexer) stateFn {
	r := lx.peek()
	switch {
	case r == keySep:
		return lx.errorf("unexpected key separator %q", keySep)
	case isWhitespace(r) || isNL(r):
		lx.next()
		return lexSkip(lx, lexKeyStart)
	case r == stringStart || r == rawStringStart:
		lx.ignore()
		lx.emit(itemKeyStart)
		lx.push(lexKeyEnd)
		return lexValue // reuse string lexing
	default:
		lx.ignore()
		lx.emit(itemKeyStart)
		return lexBareKey
	}
}

// lexBareKey consumes the text of a bare key. Assumes that the first character
// (which is not whitespace) has not yet been consumed.
func lexBareKey(lx *lexer) stateFn {
	switch r := lx.next(); {
	case isBareKeyChar(r):
		return lexBareKey
	case isWhitespace(r):
		lx.backup()
		lx.emit(itemText)
		return lexKeyEnd
	case r == keySep:
		lx.backup()
		lx.emit(itemText)
		return lexKeyEnd
	default:
		return lx.errorf("bare keys cannot contain %q", r)
	}
}

// lexKeyEnd consumes the end of a key and trims whitespace (up to the key
// separator).
func lexKeyEnd(lx *lexer) stateFn {
	switch r := lx.next(); {
	case r == keySep:
		return lexSkip(lx, lexValue)
	case isWhitespace(r):
		return lexSkip(lx, lexKeyEnd)
	default:
		return lx.errorf("expected key separator %q, but got %q instead",
			keySep, r)
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
	case arrayStart:
		lx.ignore()
		lx.emit(itemArray)
		return lexArrayValue
	case inlineTableStart:
		lx.ignore()
		lx.emit(itemInlineTableStart)
		return lexInlineTableValue
	case stringStart:
		if lx.accept(stringStart) {
			if lx.accept(stringStart) {
				lx.ignore() // Ignore """
				return lexMultilineString
			}
			lx.backup()
		}
		lx.ignore() // ignore the '"'
		return lexString
	case rawStringStart:
		if lx.accept(rawStringStart) {
			if lx.accept(rawStringStart) {
				lx.ignore() // Ignore """
				return lexMultilineRawString
			}
			lx.backup()
		}
		lx.ignore() // ignore the "'"
		return lexRawString
	case '+', '-':
		return lexNumberStart
	case '.': // special error case, be kind to users
		return lx.errorf("floats must start with a digit, not '.'")
	}
	if unicode.IsLetter(r) {
		// Be permissive here; lexBool will give a nice error if the
		// user wrote something like
		//   x = foo
		// (i.e. not 'true' or 'false' but is something else word-like.)
		lx.backup()
		return lexBool
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
	case r == commentStart:
		lx.push(lexArrayValue)
		return lexCommentStart
	case r == comma:
		return lx.errorf("unexpected comma")
	case r == arrayEnd:
		// NOTE(caleb): The spec isn't clear about whether you can have
		// a trailing comma or not, so we'll allow it.
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
	r := lx.next()
	switch {
	case isWhitespace(r) || isNL(r):
		return lexSkip(lx, lexArrayValueEnd)
	case r == commentStart:
		lx.push(lexArrayValueEnd)
		return lexCommentStart
	case r == comma:
		lx.ignore()
		return lexArrayValue // move on to the next value
	case r == arrayEnd:
		return lexArrayEnd
	}
	return lx.errorf(
		"expected a comma or array terminator %q, but got %q instead",
		arrayEnd, r,
	)
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
		return lx.errorf("newlines not allowed within inline tables")
	case r == commentStart:
		lx.push(lexInlineTableValue)
		return lexCommentStart
	case r == comma:
		return lx.errorf("unexpected comma")
	case r == inlineTableEnd:
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
	r := lx.next()
	switch {
	case isWhitespace(r):
		return lexSkip(lx, lexInlineTableValueEnd)
	case isNL(r):
		return lx.errorf("newlines not allowed within inline tables")
	case r == commentStart:
		lx.push(lexInlineTableValueEnd)
		return lexCommentStart
	case r == comma:
		lx.ignore()
		return lexInlineTableValue
	case r == inlineTableEnd:
		return lexInlineTableEnd
	}
	return lx.errorf("expected a comma or an inline table terminator %q, "+
		"but got %q instead", inlineTableEnd, r)
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
		return lx.errorf("unexpected EOF")
	case isNL(r):
		return lx.errorf("strings cannot contain newlines")
	case r == '\\':
		lx.push(lexString)
		return lexStringEscape
	case r == stringEnd:
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
	switch lx.next() {
	case eof:
		return lx.errorf("unexpected EOF")
	case '\\':
		return lexMultilineStringEscape
	case stringEnd:
		if lx.accept(stringEnd) {
			if lx.accept(stringEnd) {
				lx.backup()
				lx.backup()
				lx.backup()
				lx.emit(itemMultilineString)
				lx.next()
				lx.next()
				lx.next()
				lx.ignore()
				return lx.pop()
			}
			lx.backup()
		}
	}
	return lexMultilineString
}

// lexRawString consumes a raw string. Nothing can be escaped in such a string.
// It assumes that the beginning "'" has already been consumed and ignored.
func lexRawString(lx *lexer) stateFn {
	r := lx.next()
	switch {
	case r == eof:
		return lx.errorf("unexpected EOF")
	case isNL(r):
		return lx.errorf("strings cannot contain newlines")
	case r == rawStringEnd:
		lx.backup()
		lx.emit(itemRawString)
		lx.next()
		lx.ignore()
		return lx.pop()
	}
	return lexRawString
}

// lexMultilineRawString consumes a raw string. Nothing can be escaped in such
// a string. It assumes that the beginning "'''" has already been consumed and
// ignored.
func lexMultilineRawString(lx *lexer) stateFn {
	switch lx.next() {
	case eof:
		return lx.errorf("unexpected EOF")
	case rawStringEnd:
		if lx.accept(rawStringEnd) {
			if lx.accept(rawStringEnd) {
				lx.backup()
				lx.backup()
				lx.backup()
				lx.emit(itemRawMultilineString)
				lx.next()
				lx.next()
				lx.next()
				lx.ignore()
				return lx.pop()
			}
			lx.backup()
		}
	}
	return lexMultilineRawString
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
	case '\\':
		return lx.pop()
	case 'u':
		return lexShortUnicodeEscape
	case 'U':
		return lexLongUnicodeEscape
	}
	return lx.errorf("invalid escape character %q; only the following "+
		"escape characters are allowed: "+
		`\b, \t, \n, \f, \r, \", \\, \uXXXX, and \UXXXXXXXX`, r)
}

func lexShortUnicodeEscape(lx *lexer) stateFn {
	var r rune
	for i := 0; i < 4; i++ {
		r = lx.next()
		if !isHexadecimal(r) {
			return lx.errorf(`expected four hexadecimal digits after '\u', `+
				"but got %q instead", lx.current())
		}
	}
	return lx.pop()
}

func lexLongUnicodeEscape(lx *lexer) stateFn {
	var r rune
	for i := 0; i < 8; i++ {
		r = lx.next()
		if !isHexadecimal(r) {
			return lx.errorf(`expected eight hexadecimal digits after '\U', `+
				"but got %q instead", lx.current())
		}
	}
	return lx.pop()
}

// lexNumberOrDateStart consumes either an integer, a float, or datetime.
func lexNumberOrDateStart(lx *lexer) stateFn {
	r := lx.next()
	if isDigit(r) {
		return lexNumberOrDate
	}
	switch r {
	case '_':
		return lexNumber
	case 'e', 'E':
		return lexFloat
	case '.':
		return lx.errorf("floats must start with a digit, not '.'")
	}
	return lx.errorf("expected a digit but got %q", r)
}

// lexNumberOrDate consumes either an integer, float or datetime.
func lexNumberOrDate(lx *lexer) stateFn {
	r := lx.next()
	if isDigit(r) {
		return lexNumberOrDate
	}
	switch r {
	case '-':
		return lexDatetime
	case '_':
		return lexNumber
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
	case '-', 'T', ':', '.', 'Z', '+':
		return lexDatetime
	}

	lx.backup()
	lx.emit(itemDatetime)
	return lx.pop()
}

// lexNumberStart consumes either an integer or a float. It assumes that a sign
// has already been read, but that *no* digits have been consumed.
// lexNumberStart will move to the appropriate integer or float states.
func lexNumberStart(lx *lexer) stateFn {
	// We MUST see a digit. Even floats have to start with a digit.
	r := lx.next()
	if !isDigit(r) {
		if r == '.' {
			return lx.errorf("floats must start with a digit, not '.'")
		}
		return lx.errorf("expected a digit but got %q", r)
	}
	return lexNumber
}

// lexNumber consumes an integer or a float after seeing the first digit.
func lexNumber(lx *lexer) stateFn {
	r := lx.next()
	if isDigit(r) {
		return lexNumber
	}
	switch r {
	case '_':
		return lexNumber
	case '.', 'e', 'E':
		return lexFloat
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
	r := lx.peek()
	if isNL(r) || r == eof {
		lx.emit(itemText)
		return lx.pop()
	}
	lx.next()
	return lexComment
}

// lexSkip ignores all slurped input and moves on to the next state.
func lexSkip(lx *lexer, nextState stateFn) stateFn {
	return func(lx *lexer) stateFn {
		lx.ignore()
		return nextState
	}
}

// isWhitespace returns true if `r` is a whitespace character according
// to the spec.
func isWhitespace(r rune) bool {
	return r == '\t' || r == ' '
}

func isNL(r rune) bool {
	return r == '\n' || r == '\r'
}

func isDigit(r rune) bool {
	return r >= '0' && r <= '9'
}

func isHexadecimal(r rune) bool {
	return (r >= '0' && r <= '9') ||
		(r >= 'a' && r <= 'f') ||
		(r >= 'A' && r <= 'F')
}

func isBareKeyChar(r rune) bool {
	return (r >= 'A' && r <= 'Z') ||
		(r >= 'a' && r <= 'z') ||
		(r >= '0' && r <= '9') ||
		r == '_' ||
		r == '-'
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
	case itemArray:
		return "Array"
	case itemArrayEnd:
		return "ArrayEnd"
	case itemCommentStart:
		return "CommentStart"
	}
	panic(fmt.Sprintf("BUG: Unknown type '%d'.", int(itype)))
}

func (item item) String() string {
	return fmt.Sprintf("(%s, %s)", item.typ.String(), item.val)
}

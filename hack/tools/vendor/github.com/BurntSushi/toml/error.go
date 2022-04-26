package toml

import (
	"fmt"
	"strings"
)

// ParseError is returned when there is an error parsing the TOML syntax.
//
// For example invalid syntax, duplicate keys, etc.
//
// In addition to the error message itself, you can also print detailed location
// information with context by using ErrorWithLocation():
//
//     toml: error: Key 'fruit' was already created and cannot be used as an array.
//
//     At line 4, column 2-7:
//
//           2 | fruit = []
//           3 |
//           4 | [[fruit]] # Not allowed
//                 ^^^^^
//
// Furthermore, the ErrorWithUsage() can be used to print the above with some
// more detailed usage guidance:
//
//    toml: error: newlines not allowed within inline tables
//
//    At line 1, column 18:
//
//          1 | x = [{ key = 42 #
//                               ^
//
//    Error help:
//
//      Inline tables must always be on a single line:
//
//          table = {key = 42, second = 43}
//
//      It is invalid to split them over multiple lines like so:
//
//          # INVALID
//          table = {
//              key    = 42,
//              second = 43
//          }
//
//      Use regular for this:
//
//          [table]
//          key    = 42
//          second = 43
type ParseError struct {
	Message  string   // Short technical message.
	Usage    string   // Longer message with usage guidance; may be blank.
	Position Position // Position of the error
	LastKey  string   // Last parsed key, may be blank.
	Line     int      // Line the error occurred. Deprecated: use Position.

	err   error
	input string
}

// Position of an error.
type Position struct {
	Line  int // Line number, starting at 1.
	Start int // Start of error, as byte offset starting at 0.
	Len   int // Lenght in bytes.
}

func (pe ParseError) Error() string {
	msg := pe.Message
	if msg == "" { // Error from errorf()
		msg = pe.err.Error()
	}

	if pe.LastKey == "" {
		return fmt.Sprintf("toml: line %d: %s", pe.Position.Line, msg)
	}
	return fmt.Sprintf("toml: line %d (last key %q): %s",
		pe.Position.Line, pe.LastKey, msg)
}

// ErrorWithUsage() returns the error with detailed location context.
//
// See the documentation on ParseError.
func (pe ParseError) ErrorWithPosition() string {
	if pe.input == "" { // Should never happen, but just in case.
		return pe.Error()
	}

	var (
		lines = strings.Split(pe.input, "\n")
		col   = pe.column(lines)
		b     = new(strings.Builder)
	)

	msg := pe.Message
	if msg == "" {
		msg = pe.err.Error()
	}

	// TODO: don't show control characters as literals? This may not show up
	// well everywhere.

	if pe.Position.Len == 1 {
		fmt.Fprintf(b, "toml: error: %s\n\nAt line %d, column %d:\n\n",
			msg, pe.Position.Line, col+1)
	} else {
		fmt.Fprintf(b, "toml: error: %s\n\nAt line %d, column %d-%d:\n\n",
			msg, pe.Position.Line, col, col+pe.Position.Len)
	}
	if pe.Position.Line > 2 {
		fmt.Fprintf(b, "% 7d | %s\n", pe.Position.Line-2, lines[pe.Position.Line-3])
	}
	if pe.Position.Line > 1 {
		fmt.Fprintf(b, "% 7d | %s\n", pe.Position.Line-1, lines[pe.Position.Line-2])
	}
	fmt.Fprintf(b, "% 7d | %s\n", pe.Position.Line, lines[pe.Position.Line-1])
	fmt.Fprintf(b, "% 10s%s%s\n", "", strings.Repeat(" ", col), strings.Repeat("^", pe.Position.Len))
	return b.String()
}

// ErrorWithUsage() returns the error with detailed location context and usage
// guidance.
//
// See the documentation on ParseError.
func (pe ParseError) ErrorWithUsage() string {
	m := pe.ErrorWithPosition()
	if u, ok := pe.err.(interface{ Usage() string }); ok && u.Usage() != "" {
		return m + "Error help:\n\n    " +
			strings.ReplaceAll(strings.TrimSpace(u.Usage()), "\n", "\n    ") +
			"\n"
	}
	return m
}

func (pe ParseError) column(lines []string) int {
	var pos, col int
	for i := range lines {
		ll := len(lines[i]) + 1 // +1 for the removed newline
		if pos+ll >= pe.Position.Start {
			col = pe.Position.Start - pos
			if col < 0 { // Should never happen, but just in case.
				col = 0
			}
			break
		}
		pos += ll
	}

	return col
}

type (
	errLexControl       struct{ r rune }
	errLexEscape        struct{ r rune }
	errLexUTF8          struct{ b byte }
	errLexInvalidNum    struct{ v string }
	errLexInvalidDate   struct{ v string }
	errLexInlineTableNL struct{}
	errLexStringNL      struct{}
)

func (e errLexControl) Error() string {
	return fmt.Sprintf("TOML files cannot contain control characters: '0x%02x'", e.r)
}
func (e errLexControl) Usage() string { return "" }

func (e errLexEscape) Error() string        { return fmt.Sprintf(`invalid escape in string '\%c'`, e.r) }
func (e errLexEscape) Usage() string        { return usageEscape }
func (e errLexUTF8) Error() string          { return fmt.Sprintf("invalid UTF-8 byte: 0x%02x", e.b) }
func (e errLexUTF8) Usage() string          { return "" }
func (e errLexInvalidNum) Error() string    { return fmt.Sprintf("invalid number: %q", e.v) }
func (e errLexInvalidNum) Usage() string    { return "" }
func (e errLexInvalidDate) Error() string   { return fmt.Sprintf("invalid date: %q", e.v) }
func (e errLexInvalidDate) Usage() string   { return "" }
func (e errLexInlineTableNL) Error() string { return "newlines not allowed within inline tables" }
func (e errLexInlineTableNL) Usage() string { return usageInlineNewline }
func (e errLexStringNL) Error() string      { return "strings cannot contain newlines" }
func (e errLexStringNL) Usage() string      { return usageStringNewline }

const usageEscape = `
A '\' inside a "-delimited string is interpreted as an escape character.

The following escape sequences are supported:
\b, \t, \n, \f, \r, \", \\, \uXXXX, and \UXXXXXXXX

To prevent a '\' from being recognized as an escape character, use either:

- a ' or '''-delimited string; escape characters aren't processed in them; or
- write two backslashes to get a single backslash: '\\'.

If you're trying to add a Windows path (e.g. "C:\Users\martin") then using '/'
instead of '\' will usually also work: "C:/Users/martin".
`

const usageInlineNewline = `
Inline tables must always be on a single line:

    table = {key = 42, second = 43}

It is invalid to split them over multiple lines like so:

    # INVALID
    table = {
        key    = 42,
        second = 43
    }

Use regular for this:

    [table]
    key    = 42
    second = 43
`

const usageStringNewline = `
Strings must always be on a single line, and cannot span more than one line:

    # INVALID
    string = "Hello,
    world!"

Instead use """ or ''' to split strings over multiple lines:

    string = """Hello,
    world!"""
`

/*
Copyright 2016 Google Inc. All Rights Reserved.

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
// Python quoted strings.

package build

import (
	"bytes"
	"fmt"
	"strconv"
	"strings"
)

// unesc maps single-letter chars following \ to their actual values.
var unesc = [256]byte{
	'a':  '\a',
	'b':  '\b',
	'f':  '\f',
	'n':  '\n',
	'r':  '\r',
	't':  '\t',
	'v':  '\v',
	'\\': '\\',
	'\'': '\'',
	'"':  '"',
}

// esc maps escape-worthy bytes to the char that should follow \.
var esc = [256]byte{
	'\a': 'a',
	'\b': 'b',
	'\f': 'f',
	'\n': 'n',
	'\r': 'r',
	'\t': 't',
	'\v': 'v',
	'\\': '\\',
	'\'': '\'',
	'"':  '"',
}

// notEsc is a list of characters that can follow a \ in a string value
// without having to escape the \. That is, since ( is in this list, we
// quote the Go string "foo\\(bar" as the Python literal "foo\(bar".
// This really does happen in BUILD files, especially in strings
// being used as shell arguments containing regular expressions.
const notEsc = " !#$%&()*+,-./:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ{|}~"

// Unquote unquotes the quoted string, returning the actual
// string value, whether the original was triple-quoted, and
// an error describing invalid input.
func Unquote(quoted string) (s string, triple bool, err error) {
	// Check for raw prefix: means don't interpret the inner \.
	raw := false
	if strings.HasPrefix(quoted, "r") {
		raw = true
		quoted = quoted[1:]
	}

	if len(quoted) < 2 {
		err = fmt.Errorf("string literal too short")
		return
	}

	if quoted[0] != '"' && quoted[0] != '\'' || quoted[0] != quoted[len(quoted)-1] {
		err = fmt.Errorf("string literal has invalid quotes")
	}

	// Check for triple quoted string.
	quote := quoted[0]
	if len(quoted) >= 6 && quoted[1] == quote && quoted[2] == quote && quoted[:3] == quoted[len(quoted)-3:] {
		triple = true
		quoted = quoted[3 : len(quoted)-3]
	} else {
		quoted = quoted[1 : len(quoted)-1]
	}

	// Now quoted is the quoted data, but no quotes.
	// If we're in raw mode or there are no escapes, we're done.
	if raw || !strings.Contains(quoted, `\`) {
		s = quoted
		return
	}

	// Otherwise process quoted string.
	// Each iteration processes one escape sequence along with the
	// plain text leading up to it.
	var buf bytes.Buffer
	for {
		// Remove prefix before escape sequence.
		i := strings.Index(quoted, `\`)
		if i < 0 {
			i = len(quoted)
		}
		buf.WriteString(quoted[:i])
		quoted = quoted[i:]

		if len(quoted) == 0 {
			break
		}

		// Process escape sequence.
		if len(quoted) == 1 {
			err = fmt.Errorf(`truncated escape sequence \`)
			return
		}

		switch quoted[1] {
		default:
			// In Python, if \z (for some byte z) is not a known escape sequence
			// then it appears as literal text in the string.
			buf.WriteString(quoted[:2])
			quoted = quoted[2:]

		case '\n':
			// Ignore the escape and the line break.
			quoted = quoted[2:]

		case 'a', 'b', 'f', 'n', 'r', 't', 'v', '\\', '\'', '"':
			// One-char escape
			buf.WriteByte(unesc[quoted[1]])
			quoted = quoted[2:]

		case '0', '1', '2', '3', '4', '5', '6', '7':
			// Octal escape, up to 3 digits.
			n := int(quoted[1] - '0')
			quoted = quoted[2:]
			for i := 1; i < 3; i++ {
				if len(quoted) == 0 || quoted[0] < '0' || '7' < quoted[0] {
					break
				}
				n = n*8 + int(quoted[0]-'0')
				quoted = quoted[1:]
			}
			if n >= 256 {
				// NOTE: Python silently discards the high bit,
				// so that '\541' == '\141' == 'a'.
				// Let's see if we can avoid doing that in BUILD files.
				err = fmt.Errorf(`invalid escape sequence \%03o`, n)
				return
			}
			buf.WriteByte(byte(n))

		case 'x':
			// Hexadecimal escape, exactly 2 digits.
			if len(quoted) < 4 {
				err = fmt.Errorf(`truncated escape sequence %s`, quoted)
				return
			}
			n, err1 := strconv.ParseInt(quoted[2:4], 16, 0)
			if err1 != nil {
				err = fmt.Errorf(`invalid escape sequence %s`, quoted[:4])
				return
			}
			buf.WriteByte(byte(n))
			quoted = quoted[4:]
		}
	}

	s = buf.String()
	return
}

// indexByte returns the index of the first instance of b in s, or else -1.
func indexByte(s string, b byte) int {
	for i := 0; i < len(s); i++ {
		if s[i] == b {
			return i
		}
	}
	return -1
}

// hex is a list of the hexadecimal digits, for use in quoting.
// We always print lower-case hexadecimal.
const hex = "0123456789abcdef"

// quote returns the quoted form of the string value "x".
// If triple is true, quote uses the triple-quoted form """x""".
func quote(unquoted string, triple bool) string {
	q := `"`
	if triple {
		q = `"""`
	}

	var buf bytes.Buffer
	buf.WriteString(q)

	for i := 0; i < len(unquoted); i++ {
		c := unquoted[i]
		if c == '"' && triple && (i+1 < len(unquoted) && unquoted[i+1] != '"' || i+2 < len(unquoted) && unquoted[i+2] != '"') {
			// Can pass up to two quotes through, because they are followed by a non-quote byte.
			buf.WriteByte(c)
			if i+1 < len(unquoted) && unquoted[i+1] == '"' {
				buf.WriteByte(c)
				i++
			}
			continue
		}
		if triple && c == '\n' {
			// Can allow newline in triple-quoted string.
			buf.WriteByte(c)
			continue
		}
		if c == '\'' {
			// Can allow ' since we always use ".
			buf.WriteByte(c)
			continue
		}
		if c == '\\' {
			if i+1 < len(unquoted) && indexByte(notEsc, unquoted[i+1]) >= 0 {
				// Can pass \ through when followed by a byte that
				// known not to be a valid escape sequence and also
				// that does not trigger an escape sequence of its own.
				// Use this, because various BUILD files do.
				buf.WriteByte('\\')
				buf.WriteByte(unquoted[i+1])
				i++
				continue
			}
		}
		if esc[c] != 0 {
			buf.WriteByte('\\')
			buf.WriteByte(esc[c])
			continue
		}
		if c < 0x20 || c >= 0x80 {
			// BUILD files are supposed to be Latin-1, so escape all control and high bytes.
			// I'd prefer to use \x here, but Blaze does not implement
			// \x in quoted strings (b/7272572).
			buf.WriteByte('\\')
			buf.WriteByte(hex[c>>6]) // actually octal but reusing hex digits 0-7.
			buf.WriteByte(hex[(c>>3)&7])
			buf.WriteByte(hex[c&7])
			/*
				buf.WriteByte('\\')
				buf.WriteByte('x')
				buf.WriteByte(hex[c>>4])
				buf.WriteByte(hex[c&0xF])
			*/
			continue
		}
		buf.WriteByte(c)
		continue
	}

	buf.WriteString(q)
	return buf.String()
}

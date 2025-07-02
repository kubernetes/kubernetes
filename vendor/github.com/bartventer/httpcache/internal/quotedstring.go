// Copyright (c) 2025 Bart Venter <bartventer@proton.me>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package internal

import (
	"errors"
	"fmt"
	"strings"
)

// ParseQuotedString parses an HTTP quoted-string (RFC 9110 §5.6.4).
// It returns the unescaped string, or the original string if parsing fails.
func ParseQuotedString(s string) string {
	result, err := ParseQuotedStringE(s)
	if err != nil {
		return s
	}
	return result
}

var (
	errInvalidQuotedString = errors.New("httpcache: invalid quoted-string")
	errUnfinishedEscape    = errors.New("httpcache: unfinished escape (quoted-pair)")
	errInvalidCharacter    = errors.New("httpcache: invalid character in quoted-string")
)

// ParseQuotedStringE parses an HTTP quoted-string (RFC 9110 §5.6.4).
// It returns the unescaped string, or an error if the input is not valid.
func ParseQuotedStringE(s string) (string, error) {
	if len(s) < 2 || s[0] != '"' || s[len(s)-1] != '"' {
		return "", fmt.Errorf(
			"%w: %q does not start and end with DQUOTE",
			errInvalidQuotedString,
			s,
		)
	}
	in := []byte(s[1 : len(s)-1])
	var b strings.Builder
	for i := 0; i < len(in); {
		c := in[i]
		if c == '\\' {
			i++
			if i >= len(in) {
				return "", errUnfinishedEscape
			}
			// Quoted-pair: any byte allowed after '\'
			b.WriteByte(in[i])
			i++
			continue
		}
		if validQDTextByte(c) {
			b.WriteByte(c)
			i++
			continue
		}
		return "", fmt.Errorf("%w: %q contains invalid character %q at position %d",
			errInvalidCharacter,
			s,
			c,
			i+1, // +1 to account for the leading quote
		)
	}
	return b.String(), nil
}

// validQDTextByte reports whether b is allowed as qdtext, per RFC 9110 §5.6.4.
func validQDTextByte(b byte) bool {
	switch {
	case b == '\t' || b == ' ':
		return true
	case b == 0x21:
		return true
	case 0x23 <= b && b <= 0x5B:
		return true
	case 0x5D <= b && b <= 0x7E:
		return true
	case 0x80 <= b: // obs-text
		return true
	}
	return false
}

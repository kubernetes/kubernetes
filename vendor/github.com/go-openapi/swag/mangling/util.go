// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package mangling

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// Removes leading whitespaces
func trim(str string) string { return strings.TrimSpace(str) }

// upper is strings.ToUpper() combined with trim
func upper(str string) string {
	return strings.ToUpper(trim(str))
}

// lower is strings.ToLower() combined with trim
func lower(str string) string {
	return strings.ToLower(trim(str))
}

// isEqualFoldIgnoreSpace is the same as strings.EqualFold, but
// it ignores leading and trailing blank spaces in the compared
// string.
//
// base is assumed to be composed of upper-cased runes, and be already
// trimmed.
//
// This code is heavily inspired from strings.EqualFold.
func isEqualFoldIgnoreSpace(base []rune, str string) bool {
	var i, baseIndex int
	// equivalent to b := []byte(str), but without data copy
	b := hackStringBytes(str)

	for i < len(b) {
		if c := b[i]; c < utf8.RuneSelf {
			// fast path for ASCII
			if c != ' ' && c != '\t' {
				break
			}
			i++

			continue
		}

		// unicode case
		r, size := utf8.DecodeRune(b[i:])
		if !unicode.IsSpace(r) {
			break
		}
		i += size
	}

	if i >= len(b) {
		return len(base) == 0
	}

	for _, baseRune := range base {
		if i >= len(b) {
			break
		}

		if c := b[i]; c < utf8.RuneSelf {
			// single byte rune case (ASCII)
			if baseRune >= utf8.RuneSelf {
				return false
			}

			baseChar := byte(baseRune)
			if c != baseChar && ((c < 'a') || (c > 'z') || (c-'a'+'A' != baseChar)) {
				return false
			}

			baseIndex++
			i++

			continue
		}

		// unicode case
		r, size := utf8.DecodeRune(b[i:])
		if unicode.ToUpper(r) != baseRune {
			return false
		}
		baseIndex++
		i += size
	}

	if baseIndex != len(base) {
		return false
	}

	// all passed: now we should only have blanks
	for i < len(b) {
		if c := b[i]; c < utf8.RuneSelf {
			// fast path for ASCII
			if c != ' ' && c != '\t' {
				return false
			}
			i++

			continue
		}

		// unicode case
		r, size := utf8.DecodeRune(b[i:])
		if !unicode.IsSpace(r) {
			return false
		}

		i += size
	}

	return true
}

// Copyright 2015 Huan Du. All rights reserved.
// Licensed under the MIT license that can be found in the LICENSE file.

package xstrings

import (
	"unicode"
	"unicode/utf8"
)

// Len returns str's utf8 rune length.
func Len(str string) int {
	return utf8.RuneCountInString(str)
}

// WordCount returns number of words in a string.
//
// Word is defined as a locale dependent string containing alphabetic characters,
// which may also contain but not start with `'` and `-` characters.
func WordCount(str string) int {
	var r rune
	var size, n int

	inWord := false

	for len(str) > 0 {
		r, size = utf8.DecodeRuneInString(str)

		switch {
		case isAlphabet(r):
			if !inWord {
				inWord = true
				n++
			}

		case inWord && (r == '\'' || r == '-'):
			// Still in word.

		default:
			inWord = false
		}

		str = str[size:]
	}

	return n
}

const minCJKCharacter = '\u3400'

// Checks r is a letter but not CJK character.
func isAlphabet(r rune) bool {
	if !unicode.IsLetter(r) {
		return false
	}

	switch {
	// Quick check for non-CJK character.
	case r < minCJKCharacter:
		return true

	// Common CJK characters.
	case r >= '\u4E00' && r <= '\u9FCC':
		return false

	// Rare CJK characters.
	case r >= '\u3400' && r <= '\u4D85':
		return false

	// Rare and historic CJK characters.
	case r >= '\U00020000' && r <= '\U0002B81D':
		return false
	}

	return true
}

// Width returns string width in monotype font.
// Multi-byte characters are usually twice the width of single byte characters.
//
// Algorithm comes from `mb_strwidth` in PHP.
// http://php.net/manual/en/function.mb-strwidth.php
func Width(str string) int {
	var r rune
	var size, n int

	for len(str) > 0 {
		r, size = utf8.DecodeRuneInString(str)
		n += RuneWidth(r)
		str = str[size:]
	}

	return n
}

// RuneWidth returns character width in monotype font.
// Multi-byte characters are usually twice the width of single byte characters.
//
// Algorithm comes from `mb_strwidth` in PHP.
// http://php.net/manual/en/function.mb-strwidth.php
func RuneWidth(r rune) int {
	switch {
	case r == utf8.RuneError || r < '\x20':
		return 0

	case '\x20' <= r && r < '\u2000':
		return 1

	case '\u2000' <= r && r < '\uFF61':
		return 2

	case '\uFF61' <= r && r < '\uFFA0':
		return 1

	case '\uFFA0' <= r:
		return 2
	}

	return 0
}

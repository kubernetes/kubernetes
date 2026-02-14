// Package misspell implements utilities for basic spelling correction.
package misspell

import (
	"unicode/utf8"
)

// AlmostEqual reports whether a and b have Damerau-Levenshtein distance of at
// most 1. That is, it reports whether a can be transformed into b by adding,
// removing or substituting a single rune, or by swapping two adjacent runes.
// Invalid runes are considered equal.
//
// It runs in O(len(a)+len(b)) time.
func AlmostEqual(a, b string) bool {
	for len(a) > 0 && len(b) > 0 {
		ra, tailA := shiftRune(a)
		rb, tailB := shiftRune(b)
		if ra == rb {
			a, b = tailA, tailB
			continue
		}
		// check for addition/deletion/substitution
		if equalValid(a, tailB) || equalValid(tailA, b) || equalValid(tailA, tailB) {
			return true
		}
		if len(tailA) == 0 || len(tailB) == 0 {
			return false
		}
		// check for swap
		a, b = tailA, tailB
		Ra, tailA := shiftRune(tailA)
		Rb, tailB := shiftRune(tailB)
		return ra == Rb && Ra == rb && equalValid(tailA, tailB)
	}
	if len(a) == 0 {
		return len(b) == 0 || singleRune(b)
	}
	return singleRune(a)
}

// singleRune reports whether s consists of a single UTF-8 codepoint.
func singleRune(s string) bool {
	_, n := utf8.DecodeRuneInString(s)
	return n == len(s)
}

// shiftRune splits off the first UTF-8 codepoint from s and returns it and the
// rest of the string. It panics if s is empty.
func shiftRune(s string) (rune, string) {
	if len(s) == 0 {
		panic(s)
	}
	r, n := utf8.DecodeRuneInString(s)
	return r, s[n:]
}

// equalValid reports whether a and b are equal, if invalid code points are considered equal.
func equalValid(a, b string) bool {
	var ra, rb rune
	for len(a) > 0 && len(b) > 0 {
		ra, a = shiftRune(a)
		rb, b = shiftRune(b)
		if ra != rb {
			return false
		}
	}
	return len(a) == 0 && len(b) == 0
}

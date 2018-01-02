/*
Copyright 2013 The Camlistore Authors

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

// Package strutil contains string and byte processing functions.
package strutil // import "go4.org/strutil"

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// Fork of Go's implementation in pkg/strings/strings.go:
// Generic split: splits after each instance of sep,
// including sepSave bytes of sep in the subarrays.
func genSplit(dst []string, s, sep string, sepSave, n int) []string {
	if n == 0 {
		return nil
	}
	if sep == "" {
		panic("sep is empty")
	}
	if n < 0 {
		n = strings.Count(s, sep) + 1
	}
	c := sep[0]
	start := 0
	na := 0
	for i := 0; i+len(sep) <= len(s) && na+1 < n; i++ {
		if s[i] == c && (len(sep) == 1 || s[i:i+len(sep)] == sep) {
			dst = append(dst, s[start:i+sepSave])
			na++
			start = i + len(sep)
			i += len(sep) - 1
		}
	}
	dst = append(dst, s[start:])
	return dst
}

// AppendSplitN is like strings.SplitN but appends to and returns dst.
// Unlike strings.SplitN, an empty separator is not supported.
// The count n determines the number of substrings to return:
//   n > 0: at most n substrings; the last substring will be the unsplit remainder.
//   n == 0: the result is nil (zero substrings)
//   n < 0: all substrings
func AppendSplitN(dst []string, s, sep string, n int) []string {
	return genSplit(dst, s, sep, 0, n)
}

// equalFoldRune compares a and b runes whether they fold equally.
//
// The code comes from strings.EqualFold, but shortened to only one rune.
func equalFoldRune(sr, tr rune) bool {
	if sr == tr {
		return true
	}
	// Make sr < tr to simplify what follows.
	if tr < sr {
		sr, tr = tr, sr
	}
	// Fast check for ASCII.
	if tr < utf8.RuneSelf && 'A' <= sr && sr <= 'Z' {
		// ASCII, and sr is upper case.  tr must be lower case.
		if tr == sr+'a'-'A' {
			return true
		}
		return false
	}

	// General case.  SimpleFold(x) returns the next equivalent rune > x
	// or wraps around to smaller values.
	r := unicode.SimpleFold(sr)
	for r != sr && r < tr {
		r = unicode.SimpleFold(r)
	}
	if r == tr {
		return true
	}
	return false
}

// HasPrefixFold is like strings.HasPrefix but uses Unicode case-folding.
func HasPrefixFold(s, prefix string) bool {
	if prefix == "" {
		return true
	}
	for _, pr := range prefix {
		if s == "" {
			return false
		}
		// step with s, too
		sr, size := utf8.DecodeRuneInString(s)
		if sr == utf8.RuneError {
			return false
		}
		s = s[size:]
		if !equalFoldRune(sr, pr) {
			return false
		}
	}
	return true
}

// HasSuffixFold is like strings.HasPrefix but uses Unicode case-folding.
func HasSuffixFold(s, suffix string) bool {
	if suffix == "" {
		return true
	}
	// count the runes and bytes in s, but only till rune count of suffix
	bo, so := len(s), len(suffix)
	for bo > 0 && so > 0 {
		r, size := utf8.DecodeLastRuneInString(s[:bo])
		if r == utf8.RuneError {
			return false
		}
		bo -= size

		sr, size := utf8.DecodeLastRuneInString(suffix[:so])
		if sr == utf8.RuneError {
			return false
		}
		so -= size

		if !equalFoldRune(r, sr) {
			return false
		}
	}
	return so == 0
}

// ContainsFold is like strings.Contains but uses Unicode case-folding.
func ContainsFold(s, substr string) bool {
	if substr == "" {
		return true
	}
	if s == "" {
		return false
	}
	firstRune := rune(substr[0])
	if firstRune >= utf8.RuneSelf {
		firstRune, _ = utf8.DecodeRuneInString(substr)
	}
	for i, rune := range s {
		if equalFoldRune(rune, firstRune) && HasPrefixFold(s[i:], substr) {
			return true
		}
	}
	return false
}

// IsPlausibleJSON reports whether s likely contains a JSON object, without
// actually parsing it. It's meant to be a light heuristic.
func IsPlausibleJSON(s string) bool {
	return startsWithOpenBrace(s) && endsWithCloseBrace(s)
}

func isASCIIWhite(b byte) bool { return b == ' ' || b == '\n' || b == '\r' || b == '\t' }

func startsWithOpenBrace(s string) bool {
	for len(s) > 0 {
		switch {
		case s[0] == '{':
			return true
		case isASCIIWhite(s[0]):
			s = s[1:]
		default:
			return false
		}
	}
	return false
}

func endsWithCloseBrace(s string) bool {
	for len(s) > 0 {
		last := len(s) - 1
		switch {
		case s[last] == '}':
			return true
		case isASCIIWhite(s[last]):
			s = s[:last]
		default:
			return false
		}
	}
	return false
}

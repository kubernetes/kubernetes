// Copyright 2015 Huan Du. All rights reserved.
// Licensed under the MIT license that can be found in the LICENSE file.

package xstrings

import (
	"bytes"
	"strings"
	"unicode/utf8"
)

// Reverse a utf8 encoded string.
func Reverse(str string) string {
	var size int

	tail := len(str)
	buf := make([]byte, tail)
	s := buf

	for len(str) > 0 {
		_, size = utf8.DecodeRuneInString(str)
		tail -= size
		s = append(s[:tail], []byte(str[:size])...)
		str = str[size:]
	}

	return string(buf)
}

// Slice a string by rune.
//
// Start must satisfy 0 <= start <= rune length.
//
// End can be positive, zero or negative.
// If end >= 0, start and end must satisfy start <= end <= rune length.
// If end < 0, it means slice to the end of string.
//
// Otherwise, Slice will panic as out of range.
func Slice(str string, start, end int) string {
	var size, startPos, endPos int

	origin := str

	if start < 0 || end > len(str) || (end >= 0 && start > end) {
		panic("out of range")
	}

	if end >= 0 {
		end -= start
	}

	for start > 0 && len(str) > 0 {
		_, size = utf8.DecodeRuneInString(str)
		start--
		startPos += size
		str = str[size:]
	}

	if end < 0 {
		return origin[startPos:]
	}

	endPos = startPos

	for end > 0 && len(str) > 0 {
		_, size = utf8.DecodeRuneInString(str)
		end--
		endPos += size
		str = str[size:]
	}

	if len(str) == 0 && (start > 0 || end > 0) {
		panic("out of range")
	}

	return origin[startPos:endPos]
}

// Partition splits a string by sep into three parts.
// The return value is a slice of strings with head, match and tail.
//
// If str contains sep, for example "hello" and "l", Partition returns
//     "he", "l", "lo"
//
// If str doesn't contain sep, for example "hello" and "x", Partition returns
//     "hello", "", ""
func Partition(str, sep string) (head, match, tail string) {
	index := strings.Index(str, sep)

	if index == -1 {
		head = str
		return
	}

	head = str[:index]
	match = str[index : index+len(sep)]
	tail = str[index+len(sep):]
	return
}

// LastPartition splits a string by last instance of sep into three parts.
// The return value is a slice of strings with head, match and tail.
//
// If str contains sep, for example "hello" and "l", LastPartition returns
//     "hel", "l", "o"
//
// If str doesn't contain sep, for example "hello" and "x", LastPartition returns
//     "", "", "hello"
func LastPartition(str, sep string) (head, match, tail string) {
	index := strings.LastIndex(str, sep)

	if index == -1 {
		tail = str
		return
	}

	head = str[:index]
	match = str[index : index+len(sep)]
	tail = str[index+len(sep):]
	return
}

// Insert src into dst at given rune index.
// Index is counted by runes instead of bytes.
//
// If index is out of range of dst, panic with out of range.
func Insert(dst, src string, index int) string {
	return Slice(dst, 0, index) + src + Slice(dst, index, -1)
}

// Scrub scrubs invalid utf8 bytes with repl string.
// Adjacent invalid bytes are replaced only once.
func Scrub(str, repl string) string {
	var buf *bytes.Buffer
	var r rune
	var size, pos int
	var hasError bool

	origin := str

	for len(str) > 0 {
		r, size = utf8.DecodeRuneInString(str)

		if r == utf8.RuneError {
			if !hasError {
				if buf == nil {
					buf = &bytes.Buffer{}
				}

				buf.WriteString(origin[:pos])
				hasError = true
			}
		} else if hasError {
			hasError = false
			buf.WriteString(repl)

			origin = origin[pos:]
			pos = 0
		}

		pos += size
		str = str[size:]
	}

	if buf != nil {
		buf.WriteString(origin)
		return buf.String()
	}

	// No invalid byte.
	return origin
}

// WordSplit splits a string into words. Returns a slice of words.
// If there is no word in a string, return nil.
//
// Word is defined as a locale dependent string containing alphabetic characters,
// which may also contain but not start with `'` and `-` characters.
func WordSplit(str string) []string {
	var word string
	var words []string
	var r rune
	var size, pos int

	inWord := false

	for len(str) > 0 {
		r, size = utf8.DecodeRuneInString(str)

		switch {
		case isAlphabet(r):
			if !inWord {
				inWord = true
				word = str
				pos = 0
			}

		case inWord && (r == '\'' || r == '-'):
			// Still in word.

		default:
			if inWord {
				inWord = false
				words = append(words, word[:pos])
			}
		}

		pos += size
		str = str[size:]
	}

	if inWord {
		words = append(words, word[:pos])
	}

	return words
}

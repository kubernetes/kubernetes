// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package jsonutils

import (
	"bytes"
)

// nullJSON represents a JSON object with null type
var nullJSON = []byte("null")

const comma = byte(',')

var closers map[byte]byte

func init() {
	closers = map[byte]byte{
		'{': '}',
		'[': ']',
	}
}

// ConcatJSON concatenates multiple json objects or arrays efficiently.
//
// Note that [ConcatJSON] performs a very simple (and fast) concatenation
// operation: it does not attempt to merge objects.
func ConcatJSON(blobs ...[]byte) []byte {
	if len(blobs) == 0 {
		return nil
	}

	last := len(blobs) - 1
	for blobs[last] == nil || bytes.Equal(blobs[last], nullJSON) {
		// strips trailing null objects
		last--
		if last < 0 {
			// there was nothing but "null"s or nil...
			return nil
		}
	}
	if last == 0 {
		return blobs[0]
	}

	var opening, closing byte
	var idx, a int
	buf := bytes.NewBuffer(nil)

	for i, b := range blobs[:last+1] {
		if b == nil || bytes.Equal(b, nullJSON) {
			// a null object is in the list: skip it
			continue
		}
		if len(b) > 0 && opening == 0 { // is this an array or an object?
			opening, closing = b[0], closers[b[0]]
		}

		if opening != '{' && opening != '[' {
			continue // don't know how to concatenate non container objects
		}

		const minLengthIfNotEmpty = 3
		if len(b) < minLengthIfNotEmpty { // yep empty but also the last one, so closing this thing
			if i == last && a > 0 {
				_ = buf.WriteByte(closing) // never returns err != nil
			}
			continue
		}

		idx = 0
		if a > 0 { // we need to join with a comma for everything beyond the first non-empty item
			_ = buf.WriteByte(comma) // never returns err != nil
			idx = 1                  // this is not the first or the last so we want to drop the leading bracket
		}

		if i != last { // not the last one, strip brackets
			_, _ = buf.Write(b[idx : len(b)-1]) // never returns err != nil
		} else { // last one, strip only the leading bracket
			_, _ = buf.Write(b[idx:])
		}
		a++
	}

	// somehow it ended up being empty, so provide a default value
	if buf.Len() == 0 && (opening == '{' || opening == '[') {
		_ = buf.WriteByte(opening) // never returns err != nil
		_ = buf.WriteByte(closing)
	}

	return buf.Bytes()
}

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"bytes"
	"errors"
	"io"
	"sort"
	"sync"
	"unicode/utf16"
	"unicode/utf8"
)

// NOTE: RawValue is analogous to v1 json.RawMessage.

// RawValue represents a single raw JSON value, which may be one of the following:
//   - a JSON literal (i.e., null, true, or false)
//   - a JSON string (e.g., "hello, world!")
//   - a JSON number (e.g., 123.456)
//   - an entire JSON object (e.g., {"fizz":"buzz"} )
//   - an entire JSON array (e.g., [1,2,3] )
//
// RawValue can represent entire array or object values, while Token cannot.
// RawValue may contain leading and/or trailing whitespace.
type RawValue []byte

// Clone returns a copy of v.
func (v RawValue) Clone() RawValue {
	if v == nil {
		return nil
	}
	return append(RawValue{}, v...)
}

// String returns the string formatting of v.
func (v RawValue) String() string {
	if v == nil {
		return "null"
	}
	return string(v)
}

// IsValid reports whether the raw JSON value is syntactically valid
// according to RFC 7493.
//
// It verifies whether the input is properly encoded as UTF-8,
// that escape sequences within strings decode to valid Unicode codepoints, and
// that all names in each object are unique.
// It does not verify whether numbers are representable within the limits
// of any common numeric type (e.g., float64, int64, or uint64).
func (v RawValue) IsValid() bool {
	d := getBufferedDecoder(v, DecodeOptions{})
	defer putBufferedDecoder(d)
	_, errVal := d.ReadValue()
	_, errEOF := d.ReadToken()
	return errVal == nil && errEOF == io.EOF
}

// Compact removes all whitespace from the raw JSON value.
//
// It does not reformat JSON strings to use any other representation.
// It is guaranteed to succeed if the input is valid.
// If the value is already compacted, then the buffer is not mutated.
func (v *RawValue) Compact() error {
	return v.reformat(false, false, "", "")
}

// Indent reformats the whitespace in the raw JSON value so that each element
// in a JSON object or array begins on a new, indented line beginning with
// prefix followed by one or more copies of indent according to the nesting.
// The value does not begin with the prefix nor any indention,
// to make it easier to embed inside other formatted JSON data.
//
// It does not reformat JSON strings to use any other representation.
// It is guaranteed to succeed if the input is valid.
// If the value is already indented properly, then the buffer is not mutated.
func (v *RawValue) Indent(prefix, indent string) error {
	return v.reformat(false, true, prefix, indent)
}

// Canonicalize canonicalizes the raw JSON value according to the
// JSON Canonicalization Scheme (JCS) as defined by RFC 8785
// where it produces a stable representation of a JSON value.
//
// The output stability is dependent on the stability of the application data
// (see RFC 8785, Appendix E). It cannot produce stable output from
// fundamentally unstable input. For example, if the JSON value
// contains ephemeral data (e.g., a frequently changing timestamp),
// then the value is still unstable regardless of whether this is called.
//
// Note that JCS treats all JSON numbers as IEEE 754 double precision numbers.
// Any numbers with precision beyond what is representable by that form
// will lose their precision when canonicalized. For example, integer values
// beyond ±2⁵³ will lose their precision. It is recommended that
// int64 and uint64 data types be represented as a JSON string.
//
// It is guaranteed to succeed if the input is valid.
// If the value is already canonicalized, then the buffer is not mutated.
func (v *RawValue) Canonicalize() error {
	return v.reformat(true, false, "", "")
}

// TODO: Instead of implementing the v1 Marshaler/Unmarshaler,
// consider implementing the v2 versions instead.

// MarshalJSON returns v as the JSON encoding of v.
// It returns the stored value as the raw JSON output without any validation.
// If v is nil, then this returns a JSON null.
func (v RawValue) MarshalJSON() ([]byte, error) {
	// NOTE: This matches the behavior of v1 json.RawMessage.MarshalJSON.
	if v == nil {
		return []byte("null"), nil
	}
	return v, nil
}

// UnmarshalJSON sets v as the JSON encoding of b.
// It stores a copy of the provided raw JSON input without any validation.
func (v *RawValue) UnmarshalJSON(b []byte) error {
	// NOTE: This matches the behavior of v1 json.RawMessage.UnmarshalJSON.
	if v == nil {
		return errors.New("json.RawValue: UnmarshalJSON on nil pointer")
	}
	*v = append((*v)[:0], b...)
	return nil
}

// Kind returns the starting token kind.
// For a valid value, this will never include '}' or ']'.
func (v RawValue) Kind() Kind {
	if v := v[consumeWhitespace(v):]; len(v) > 0 {
		return Kind(v[0]).normalize()
	}
	return invalidKind
}

func (v *RawValue) reformat(canonical, multiline bool, prefix, indent string) error {
	var eo EncodeOptions
	if canonical {
		eo.AllowInvalidUTF8 = false    // per RFC 8785, section 3.2.4
		eo.AllowDuplicateNames = false // per RFC 8785, section 3.1
		eo.canonicalizeNumbers = true  // per RFC 8785, section 3.2.2.3
		eo.EscapeRune = nil            // per RFC 8785, section 3.2.2.2
		eo.multiline = false           // per RFC 8785, section 3.2.1
	} else {
		if s := trimLeftSpaceTab(prefix); len(s) > 0 {
			panic("json: invalid character " + quoteRune([]byte(s)) + " in indent prefix")
		}
		if s := trimLeftSpaceTab(indent); len(s) > 0 {
			panic("json: invalid character " + quoteRune([]byte(s)) + " in indent")
		}
		eo.AllowInvalidUTF8 = true
		eo.AllowDuplicateNames = true
		eo.preserveRawStrings = true
		eo.multiline = multiline // in case indent is empty
		eo.IndentPrefix = prefix
		eo.Indent = indent
	}
	eo.omitTopLevelNewline = true

	// Write the entire value to reformat all tokens and whitespace.
	e := getBufferedEncoder(eo)
	defer putBufferedEncoder(e)
	if err := e.WriteValue(*v); err != nil {
		return err
	}

	// For canonical output, we may need to reorder object members.
	if canonical {
		// Obtain a buffered encoder just to use its internal buffer as
		// a scratch buffer in reorderObjects for reordering object members.
		e2 := getBufferedEncoder(EncodeOptions{})
		defer putBufferedEncoder(e2)

		// Disable redundant checks performed earlier during encoding.
		d := getBufferedDecoder(e.buf, DecodeOptions{AllowInvalidUTF8: true, AllowDuplicateNames: true})
		defer putBufferedDecoder(d)
		reorderObjects(d, &e2.buf) // per RFC 8785, section 3.2.3
	}

	// Store the result back into the value if different.
	if !bytes.Equal(*v, e.buf) {
		*v = append((*v)[:0], e.buf...)
	}
	return nil
}

func trimLeftSpaceTab(s string) string {
	for i, r := range s {
		switch r {
		case ' ', '\t':
		default:
			return s[i:]
		}
	}
	return ""
}

type memberName struct {
	// name is the unescaped name.
	name []byte
	// before and after are byte offsets into Decoder.buf that represents
	// the entire name/value pair. It may contain leading commas.
	before, after int64
}

var memberNamePool = sync.Pool{New: func() any { return new(memberNames) }}

func getMemberNames() *memberNames {
	ns := memberNamePool.Get().(*memberNames)
	*ns = (*ns)[:0]
	return ns
}
func putMemberNames(ns *memberNames) {
	if cap(*ns) < 1<<10 {
		for i := range *ns {
			(*ns)[i] = memberName{} // avoid pinning name
		}
		memberNamePool.Put(ns)
	}
}

type memberNames []memberName

func (m *memberNames) Len() int           { return len(*m) }
func (m *memberNames) Less(i, j int) bool { return lessUTF16((*m)[i].name, (*m)[j].name) }
func (m *memberNames) Swap(i, j int)      { (*m)[i], (*m)[j] = (*m)[j], (*m)[i] }

// reorderObjects recursively reorders all object members in place
// according to the ordering specified in RFC 8785, section 3.2.3.
//
// Pre-conditions:
//   - The value is valid (i.e., no decoder errors should ever occur).
//   - The value is compact (i.e., no whitespace is present).
//   - Initial call is provided a Decoder reading from the start of v.
//
// Post-conditions:
//   - Exactly one JSON value is read from the Decoder.
//   - All fully-parsed JSON objects are reordered by directly moving
//     the members in the value buffer.
//
// The runtime is approximately O(n·log(n)) + O(m·log(m)),
// where n is len(v) and m is the total number of object members.
func reorderObjects(d *Decoder, scratch *[]byte) {
	switch tok, _ := d.ReadToken(); tok.Kind() {
	case '{':
		// Iterate and collect the name and offsets for every object member.
		members := getMemberNames()
		defer putMemberNames(members)
		var prevName []byte
		isSorted := true

		beforeBody := d.InputOffset() // offset after '{'
		for d.PeekKind() != '}' {
			beforeName := d.InputOffset()
			var flags valueFlags
			name, _ := d.readValue(&flags)
			name = unescapeStringMayCopy(name, flags.isVerbatim())
			reorderObjects(d, scratch)
			afterValue := d.InputOffset()

			if isSorted && len(*members) > 0 {
				isSorted = lessUTF16(prevName, name)
			}
			*members = append(*members, memberName{name, beforeName, afterValue})
			prevName = name
		}
		afterBody := d.InputOffset() // offset before '}'
		d.ReadToken()

		// Sort the members; return early if it's already sorted.
		if isSorted {
			return
		}
		// TODO(https://go.dev/issue/47619): Use slices.Sort.
		sort.Sort(members)

		// Append the reordered members to a new buffer,
		// then copy the reordered members back over the original members.
		// Avoid swapping in place since each member may be a different size
		// where moving a member over a smaller member may corrupt the data
		// for subsequent members before they have been moved.
		//
		// The following invariant must hold:
		//	sum([m.after-m.before for m in members]) == afterBody-beforeBody
		sorted := (*scratch)[:0]
		for i, member := range *members {
			if d.buf[member.before] == ',' {
				member.before++ // trim leading comma
			}
			sorted = append(sorted, d.buf[member.before:member.after]...)
			if i < len(*members)-1 {
				sorted = append(sorted, ',') // append trailing comma
			}
		}
		if int(afterBody-beforeBody) != len(sorted) {
			panic("BUG: length invariant violated")
		}
		copy(d.buf[beforeBody:afterBody], sorted)

		// Update scratch buffer to the largest amount ever used.
		if len(sorted) > len(*scratch) {
			*scratch = sorted
		}
	case '[':
		for d.PeekKind() != ']' {
			reorderObjects(d, scratch)
		}
		d.ReadToken()
	}
}

// lessUTF16 reports whether x is lexicographically less than y according
// to the UTF-16 codepoints of the UTF-8 encoded input strings.
// This implements the ordering specified in RFC 8785, section 3.2.3.
// The inputs must be valid UTF-8, otherwise this may panic.
func lessUTF16(x, y []byte) bool {
	// NOTE: This is an optimized, allocation-free implementation
	// of lessUTF16Simple in fuzz_test.go. FuzzLessUTF16 verifies that the
	// two implementations agree on the result of comparing any two strings.

	isUTF16Self := func(r rune) bool {
		return ('\u0000' <= r && r <= '\uD7FF') || ('\uE000' <= r && r <= '\uFFFF')
	}

	for {
		if len(x) == 0 || len(y) == 0 {
			return len(x) < len(y)
		}

		// ASCII fast-path.
		if x[0] < utf8.RuneSelf || y[0] < utf8.RuneSelf {
			if x[0] != y[0] {
				return x[0] < y[0]
			}
			x, y = x[1:], y[1:]
			continue
		}

		// Decode next pair of runes as UTF-8.
		rx, nx := utf8.DecodeRune(x)
		ry, ny := utf8.DecodeRune(y)
		switch {

		// Both runes encode as either a single or surrogate pair
		// of UTF-16 codepoints.
		case isUTF16Self(rx) == isUTF16Self(ry):
			if rx != ry {
				return rx < ry
			}

		// The x rune is a single UTF-16 codepoint, while
		// the y rune is a surrogate pair of UTF-16 codepoints.
		case isUTF16Self(rx):
			ry, _ := utf16.EncodeRune(ry)
			if rx != ry {
				return rx < ry
			}
			panic("BUG: invalid UTF-8") // implies rx is an unpaired surrogate half

		// The y rune is a single UTF-16 codepoint, while
		// the x rune is a surrogate pair of UTF-16 codepoints.
		case isUTF16Self(ry):
			rx, _ := utf16.EncodeRune(rx)
			if rx != ry {
				return rx < ry
			}
			panic("BUG: invalid UTF-8") // implies ry is an unpaired surrogate half
		}
		x, y = x[nx:], y[ny:]
	}
}

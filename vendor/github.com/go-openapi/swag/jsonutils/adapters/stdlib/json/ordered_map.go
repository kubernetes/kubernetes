// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package json

import (
	stdjson "encoding/json"
	"fmt"
	"iter"

	"github.com/go-openapi/swag/jsonutils/adapters/ifaces"
)

var _ ifaces.OrderedMap = &MapSlice{}

// MapSlice represents a JSON object, with the order of keys maintained.
type MapSlice []MapItem

func (s MapSlice) OrderedItems() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		for _, item := range s {
			if !yield(item.Key, item.Value) {
				return
			}
		}
	}
}

func (s *MapSlice) SetOrderedItems(items iter.Seq2[string, any]) {
	if items == nil {
		*s = nil

		return
	}

	m := *s
	if len(m) > 0 {
		// update mode
		idx := make(map[string]int, len(m))

		for i, item := range m {
			idx[item.Key] = i
		}

		for k, v := range items {
			idx, ok := idx[k]
			if ok {
				m[idx].Value = v

				continue
			}
			m = append(m, MapItem{Key: k, Value: v})
		}

		*s = m

		return
	}

	for k, v := range items {
		m = append(m, MapItem{Key: k, Value: v})
	}

	*s = m
}

// MarshalJSON renders a [MapSlice] as JSON bytes, preserving the order of keys.
func (s MapSlice) MarshalJSON() ([]byte, error) {
	return s.OrderedMarshalJSON()
}

func (s MapSlice) OrderedMarshalJSON() ([]byte, error) {
	w := poolOfWriters.Borrow()
	defer func() {
		poolOfWriters.Redeem(w)
	}()

	s.marshalObject(w)

	return w.BuildBytes() // this clones data, so it's okay to redeem the writer and its buffer
}

// UnmarshalJSON builds a [MapSlice] from JSON bytes, preserving the order of keys.
//
// Inner objects are unmarshaled as [MapSlice] slices and not map[string]any.
func (s *MapSlice) UnmarshalJSON(data []byte) error {
	return s.OrderedUnmarshalJSON(data)
}

func (s *MapSlice) OrderedUnmarshalJSON(data []byte) error {
	l := poolOfLexers.Borrow(data)
	defer func() {
		poolOfLexers.Redeem(l)
	}()

	s.unmarshalObject(l)

	return l.Error()
}

func (s MapSlice) marshalObject(w *jwriter) {
	if s == nil {
		w.RawString("null")

		return
	}

	w.RawByte('{')

	if len(s) == 0 {
		w.RawByte('}')

		return
	}

	s[0].marshalJSON(w)

	for i := 1; i < len(s); i++ {
		w.RawByte(',')
		s[i].marshalJSON(w)
	}

	w.RawByte('}')
}

func (s *MapSlice) unmarshalObject(in *jlexer) {
	if in.IsNull() {
		in.Skip()

		return
	}

	in.Delim('{') // consume token
	if !in.Ok() {
		return
	}

	result := make(MapSlice, 0)

	for in.Ok() && !in.IsDelim('}') {
		var mi MapItem

		mi.unmarshalKeyValue(in)
		result = append(result, mi)
	}

	in.Delim('}')

	if !in.Ok() {
		return
	}

	*s = result
}

// MapItem represents the value of a key in a JSON object held by [MapSlice].
//
// Notice that [MapItem] should not be marshaled to or unmarshaled from JSON directly,
// use this type as part of a [MapSlice] when dealing with JSON bytes.
type MapItem struct {
	Key   string
	Value any
}

func (s MapItem) marshalJSON(w *jwriter) {
	w.String(s.Key)
	w.RawByte(':')
	w.Raw(stdjson.Marshal(s.Value))
}

func (s *MapItem) unmarshalKeyValue(in *jlexer) {
	key := in.String()         // consume string
	value := s.asInterface(in) // consume any value, including termination tokens '}' or ']'

	if !in.Ok() {
		return
	}

	s.Key = key
	s.Value = value
}

func (s *MapItem) unmarshalArray(in *jlexer) []any {
	if in.IsNull() {
		in.Skip()

		return nil
	}

	in.Delim('[') // consume token
	if !in.Ok() {
		return nil
	}

	ret := make([]any, 0)

	for in.Ok() && !in.IsDelim(']') {
		ret = append(ret, s.asInterface(in))
	}

	in.Delim(']')
	if !in.Ok() {
		return nil
	}

	return ret
}

// asInterface is very much like [jlexer.Lexer.Interface], but unmarshals an object
// into a [MapSlice], not a map[string]any.
//
// We have to force parsing errors somehow, since [jlexer.Lexer] doesn't let us
// set a parsing error directly.
func (s *MapItem) asInterface(in *jlexer) any {
	if !in.Ok() {
		return nil
	}

	tok := in.PeekToken() // look-ahead what the next token looks like
	kind := tok.Kind()

	switch kind {
	case tokenString:
		return in.String() // consume string

	case tokenNumber, tokenFloat:
		return in.Number()

	case tokenBool:
		return in.Bool()

	case tokenNull:
		in.Null()

		return nil

	case tokenDelim:
		switch tok.Delim() {
		case '{': // not consumed yet
			ret := make(MapSlice, 0)
			ret.unmarshalObject(in) // consumes the terminating '}'

			if in.Ok() {
				return ret
			}

			// lexer is in an error state: will exhaust
			return nil

		case '[': // not consumed yet
			return s.unmarshalArray(in) // consumes the terminating ']'
		default:
			in.SetErr(fmt.Errorf("unexpected delimiter: %v: %w", tok, ErrStdlib)) // force error
			return nil
		}

	case tokenUndef:
		fallthrough
	default:
		if in.Ok() {
			in.SetErr(fmt.Errorf("unexpected token: %v: %w", tok, ErrStdlib)) // force error
		}

		return nil
	}
}

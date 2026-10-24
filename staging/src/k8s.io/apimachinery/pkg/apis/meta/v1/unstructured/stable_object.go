/*
Copyright The Kubernetes Authors.

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

package unstructured

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
)

// maxStableDepth mirrors the nesting limit of encoding/json.
const maxStableDepth = 10000

// Field is a single name/value pair of an Object.
type Field struct {
	Name string
	// Value is one of nil, bool, int64, float64, string, []any or Object.
	Value any
}

// Object is a JSON object which preserves the order of its fields. It is the
// order-preserving counterpart of map[string]any.
//
// Duplicate names are tolerated when decoding invalid JSON. They are kept
// as they were received and the last one wins in Get, like in encoding/json.
type Object []Field

// Get returns the value of the named field. ok is false if there is no such
// field. A field that is explicitly null returns (nil, true).
//
// The lookup starts at the end so that the last of several duplicates wins.
func (o Object) Get(name string) (value any, ok bool) {
	for i := len(o) - 1; i >= 0; i-- {
		if o[i].Name == name {
			return o[i].Value, true
		}
	}
	return nil, false
}

// Set overwrites the value of the named field, or appends a new field at the end.
func (o *Object) Set(name string, value any) {
	for i := len(*o) - 1; i >= 0; i-- {
		if (*o)[i].Name == name {
			(*o)[i].Value = value
			return
		}
	}
	*o = append(*o, Field{Name: name, Value: value})
}

// Delete removes all fields with the given name.
func (o *Object) Delete(name string) {
	out := (*o)[:0]
	for _, f := range *o {
		if f.Name != name {
			out = append(out, f)
		}
	}
	// Don't keep references to removed values alive.
	clear((*o)[len(out):])
	*o = out
}

// MarshalJSON emits the fields in order.
func (o Object) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	buf.WriteByte('{')
	for i, f := range o {
		if i > 0 {
			buf.WriteByte(',')
		}
		name, err := json.Marshal(f.Name)
		if err != nil {
			return nil, err
		}
		buf.Write(name)
		buf.WriteByte(':')
		value, err := json.Marshal(f.Value)
		if err != nil {
			return nil, fmt.Errorf("field %q: %w", f.Name, err)
		}
		buf.Write(value)
	}
	buf.WriteByte('}')
	return buf.Bytes(), nil
}

// UnmarshalJSON decodes a JSON object and keeps the order of its fields.
// Numbers are decoded like Unstructured does: int64 if possible, else float64.
func (o *Object) UnmarshalJSON(data []byte) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	value, err := decodeStableValue(dec, 0)
	if err != nil {
		return err
	}
	if _, err := dec.Token(); err != io.EOF {
		return errors.New("unexpected data after top-level JSON value")
	}
	switch v := value.(type) {
	case Object:
		*o = v
	case nil:
		*o = nil
	default:
		return fmt.Errorf("cannot decode JSON %T into an object", value)
	}
	return nil
}

func decodeStableValue(dec *json.Decoder, depth int) (any, error) {
	if depth > maxStableDepth {
		return nil, errors.New("exceeded maximum JSON nesting depth")
	}
	tok, err := dec.Token()
	if err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return nil, err
	}
	switch t := tok.(type) {
	case json.Delim:
		switch t {
		case '{':
			obj := Object{}
			for dec.More() {
				keyTok, err := dec.Token()
				if err != nil {
					return nil, err
				}
				key, ok := keyTok.(string)
				if !ok {
					return nil, fmt.Errorf("unexpected object key %v", keyTok)
				}
				value, err := decodeStableValue(dec, depth+1)
				if err != nil {
					return nil, err
				}
				obj = append(obj, Field{Name: key, Value: value})
			}
			if _, err := dec.Token(); err != nil { // closing '}'
				return nil, err
			}
			return obj, nil
		case '[':
			list := []any{}
			for dec.More() {
				value, err := decodeStableValue(dec, depth+1)
				if err != nil {
					return nil, err
				}
				list = append(list, value)
			}
			if _, err := dec.Token(); err != nil { // closing ']'
				return nil, err
			}
			return list, nil
		}
		return nil, fmt.Errorf("unexpected delimiter %q", rune(t))
	case json.Number:
		if i, err := t.Int64(); err == nil {
			return i, nil
		}
		f, err := t.Float64()
		if err != nil {
			return nil, err
		}
		// Same as the regular decoder: integral floats which fit become int64.
		if f == math.Trunc(f) && f >= -(1<<63) && f < (1<<63) {
			return int64(f), nil
		}
		return f, nil
	default:
		// string, bool, nil
		return t, nil
	}
}

// deepCopyStableValue copies values as they occur in an Object.
func deepCopyStableValue(v any) any {
	switch t := v.(type) {
	case Object:
		return t.DeepCopy()
	case []any:
		if t == nil {
			return t
		}
		out := make([]any, len(t))
		for i := range t {
			out[i] = deepCopyStableValue(t[i])
		}
		return out
	default:
		return v
	}
}

// DeepCopy returns a deep copy of the object.
func (o Object) DeepCopy() Object {
	if o == nil {
		return nil
	}
	out := make(Object, len(o))
	for i, f := range o {
		out[i] = Field{Name: f.Name, Value: deepCopyStableValue(f.Value)}
	}
	return out
}

// ToMap converts the object into the map[string]any form used by Unstructured.
// The order of fields is lost.
func (o Object) ToMap() map[string]any {
	out := make(map[string]any, len(o))
	for _, f := range o {
		out[f.Name] = stableToPlain(f.Value)
	}
	return out
}

func stableToPlain(v any) any {
	switch t := v.(type) {
	case Object:
		return t.ToMap()
	case []any:
		if t == nil {
			return t
		}
		out := make([]any, len(t))
		for i := range t {
			out[i] = stableToPlain(t[i])
		}
		return out
	default:
		return v
	}
}

// ObjectFromMap converts the map[string]any form used by Unstructured into an
// Object. Fields are sorted by name because maps have no order.
func ObjectFromMap(m map[string]any) Object {
	if m == nil {
		return nil
	}
	names := make([]string, 0, len(m))
	for name := range m {
		names = append(names, name)
	}
	sort.Strings(names)
	out := make(Object, 0, len(m))
	for _, name := range names {
		out = append(out, Field{Name: name, Value: stableFromPlain(m[name])})
	}
	return out
}

func stableFromPlain(v any) any {
	switch t := v.(type) {
	case map[string]any:
		return ObjectFromMap(t)
	case []any:
		if t == nil {
			return t
		}
		out := make([]any, len(t))
		for i := range t {
			out[i] = stableFromPlain(t[i])
		}
		return out
	default:
		return v
	}
}

/*
Copyright 2019 The Kubernetes Authors.

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

package value

import (
	"encoding/json/jsontext"
	"encoding/json/v2"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"

	internaljson "sigs.k8s.io/structured-merge-diff/v7/internal/json"
)

// Field is an individual key-value pair.
type Field struct {
	Name  string
	Value Value
}

func ValueMarshalJSONTo(enc *jsontext.Encoder, v Value) error {
	switch {
	case v.IsNull():
		return enc.WriteToken(jsontext.Null)
	case v.IsFloat():
		f := v.AsFloat()
		if math.IsInf(f, 0) || math.IsNaN(f) {
			return fmt.Errorf("unsupported value: %v", f)
		}
		return enc.WriteToken(jsontext.Float(f))
	case v.IsInt():
		return enc.WriteToken(jsontext.Int(v.AsInt()))
	case v.IsString():
		return enc.WriteToken(jsontext.String(v.AsString()))
	case v.IsBool():
		return enc.WriteToken(jsontext.Bool(v.AsBool()))
	case v.IsList():
		if err := enc.WriteToken(jsontext.BeginArray); err != nil {
			return err
		}
		list := v.AsList()
		for i := 0; i < list.Length(); i++ {
			if err := ValueMarshalJSONTo(enc, list.At(i)); err != nil {
				return err
			}
		}
		return enc.WriteToken(jsontext.EndArray)
	case v.IsMap():
		// use the json marshaller to make sure the key ordering is deterministic
		return json.MarshalEncode(enc, v.Unstructured(), json.Deterministic(true), jsontext.AllowInvalidUTF8(true))
	default:
		return fmt.Errorf("cannot marshal unknown value type to json")
	}
}

// FieldList is a list of key-value pairs. Each field is expected to
// have a different name.
type FieldList []Field

var _ json.MarshalerTo = (*FieldList)(nil)
var _ json.UnmarshalerFrom = (*FieldList)(nil)

func (fl *FieldList) MarshalJSONTo(enc *jsontext.Encoder) error {
	if err := enc.WriteToken(jsontext.BeginObject); err != nil {
		return err
	}
	for _, f := range *fl {
		if err := enc.WriteToken(jsontext.String(f.Name)); err != nil {
			return err
		}
		if err := ValueMarshalJSONTo(enc, f.Value); err != nil {
			return err
		}
	}
	if err := enc.WriteToken(jsontext.EndObject); err != nil {
		return err
	}

	return nil
}

// FieldListFromJSON is a helper function for reading a JSON document.
func (fl *FieldList) UnmarshalJSONFrom(parser *jsontext.Decoder) error {
	objStart, err := parser.ReadToken()
	if err != nil {
		return fmt.Errorf("parsing JSON: %v", err)
	}
	switch objStart.Kind() {
	case jsontext.BeginObject.Kind():
		// Continue below.
	case jsontext.Null.Kind():
		// A null is equivalent to an empty object.
		*fl = nil
		return nil
	default:
		return fmt.Errorf("expected object")
	}

	var fields FieldList
	for {
		rawKey, err := parser.ReadToken()
		if err == io.EOF {
			return fmt.Errorf("unexpected EOF")
		} else if err != nil {
			return fmt.Errorf("parsing JSON: %v", err)
		}

		if rawKey.Kind() == jsontext.EndObject.Kind() {
			break
		}

		k := rawKey.String()

		v, err := internaljson.ReadValueToAnyMergingDuplicates(parser)
		if err == io.EOF {
			return fmt.Errorf("unexpected EOF")
		} else if err != nil {
			return fmt.Errorf("parsing JSON: %v", err)
		}

		fields = append(fields, Field{Name: k, Value: NewValueInterface(v)})
	}

	fields.Sort()
	*fl = fields

	return nil
}

// Copy returns a copy of the FieldList.
// Values are not copied.
func (f FieldList) Copy() FieldList {
	c := make(FieldList, len(f))
	copy(c, f)
	return c
}

// Sort sorts the field list by Name.
func (f FieldList) Sort() {
	if len(f) < 2 {
		return
	}
	if len(f) == 2 {
		if f[1].Name < f[0].Name {
			f[0], f[1] = f[1], f[0]
		}
		return
	}
	sort.SliceStable(f, func(i, j int) bool {
		return f[i].Name < f[j].Name
	})
}

// Less compares two lists lexically.
func (f FieldList) Less(rhs FieldList) bool {
	return f.Compare(rhs) == -1
}

// Compare compares two lists lexically. The result will be 0 if f==rhs, -1
// if f < rhs, and +1 if f > rhs.
func (f FieldList) Compare(rhs FieldList) int {
	i := 0
	for {
		if i >= len(f) && i >= len(rhs) {
			// Maps are the same length and all items are equal.
			return 0
		}
		if i >= len(f) {
			// F is shorter.
			return -1
		}
		if i >= len(rhs) {
			// RHS is shorter.
			return 1
		}
		if c := strings.Compare(f[i].Name, rhs[i].Name); c != 0 {
			return c
		}
		if c := Compare(f[i].Value, rhs[i].Value); c != 0 {
			return c
		}
		// The items are equal; continue.
		i++
	}
}

// Equals returns true if the two fieldslist are equals, false otherwise.
func (f FieldList) Equals(rhs FieldList) bool {
	if len(f) != len(rhs) {
		return false
	}
	for i := range f {
		if f[i].Name != rhs[i].Name {
			return false
		}
		if !Equals(f[i].Value, rhs[i].Value) {
			return false
		}
	}
	return true
}

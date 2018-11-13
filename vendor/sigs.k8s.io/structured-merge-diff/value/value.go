/*
Copyright 2018 The Kubernetes Authors.

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
	"fmt"
	"strings"
)

// A Value is an object; it corresponds to an 'atom' in the schema.
type Value struct {
	// Exactly one of the below must be set.
	FloatValue   *Float
	IntValue     *Int
	StringValue  *String
	BooleanValue *Boolean
	ListValue    *List
	MapValue     *Map
	Null         bool // represents an explicit `"foo" = null`
}

type Int int64
type Float float64
type String string
type Boolean bool

// Field is an individual key-value pair.
type Field struct {
	Name  string
	Value Value
}

// List is a list of items.
type List struct {
	Items []Value
}

// Map is a map of key-value pairs. It represents both structs and maps. We use
// a list and a go-language map to preserve order.
//
// Set and Get helpers are provided.
type Map struct {
	Items []Field

	// may be nil; lazily constructed.
	// TODO: Direct modifications to Items above will cause serious problems.
	index map[string]*Field
}

// Get returns the (Field, true) or (nil, false) if it is not present
func (m *Map) Get(key string) (*Field, bool) {
	if m.index == nil {
		m.index = map[string]*Field{}
		for i := range m.Items {
			f := &m.Items[i]
			m.index[f.Name] = f
		}
	}
	f, ok := m.index[key]
	return f, ok
}

// Set inserts or updates the given item.
func (m *Map) Set(key string, value Value) {
	if f, ok := m.Get(key); ok {
		f.Value = value
		return
	}
	m.Items = append(m.Items, Field{Name: key, Value: value})
	m.index = nil // Since the append might have reallocated
}

// StringValue returns s as a scalar string Value.
func StringValue(s string) Value {
	s2 := String(s)
	return Value{StringValue: &s2}
}

// IntValue returns i as a scalar numeric (integer) Value.
func IntValue(i int) Value {
	i2 := Int(i)
	return Value{IntValue: &i2}
}

// FloatValue returns f as a scalar numeric (float) Value.
func FloatValue(f float64) Value {
	f2 := Float(f)
	return Value{FloatValue: &f2}
}

// BooleanValue returns b as a scalar boolean Value.
func BooleanValue(b bool) Value {
	b2 := Boolean(b)
	return Value{BooleanValue: &b2}
}

// String returns a human-readable representation of the value.
func (v Value) String() string {
	switch {
	case v.FloatValue != nil:
		return fmt.Sprintf("%v", *v.FloatValue)
	case v.IntValue != nil:
		return fmt.Sprintf("%v", *v.IntValue)
	case v.StringValue != nil:
		return fmt.Sprintf("%q", *v.StringValue)
	case v.BooleanValue != nil:
		return fmt.Sprintf("%v", *v.BooleanValue)
	case v.ListValue != nil:
		strs := []string{}
		for _, item := range v.ListValue.Items {
			strs = append(strs, item.String())
		}
		return "[" + strings.Join(strs, ",") + "]"
	case v.MapValue != nil:
		strs := []string{}
		for _, i := range v.MapValue.Items {
			strs = append(strs, fmt.Sprintf("%v=%v", i.Name, i.Value))
		}
		return "{" + strings.Join(strs, ";") + "}"
	default:
		fallthrough
	case v.Null == true:
		return "null"
	}
}

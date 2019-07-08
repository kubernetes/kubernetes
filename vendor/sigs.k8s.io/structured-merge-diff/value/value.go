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
	"sort"
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

// Compare returns an integer comparing two Fields. The result will be 0
// if f==other, -1 if f < other, and +1 if f > other.
func (f Field) Compare(other Field) int {
	if comp := strings.Compare(f.Name, other.Name); comp != 0 {
		return comp
	}
	return f.Value.Compare(other.Value)
}

type byField []Field

func (f byField) Len() int           { return len(f) }
func (f byField) Swap(i, j int)      { f[i], f[j] = f[j], f[i] }
func (f byField) Less(i, j int) bool { return f[i].Compare(f[j]) < 0 }

type Fields []Field

// This is doing a shallow copy, and sorting.
func (f Fields) SortCopy() Fields {
	fields := []Field{}
	for _, field := range f {
		fields = append(fields, field)
	}
	sort.Sort(byField(fields))
	return fields
}

// Compare returns an integer comparing two list of Fields. The result
// will be 0 if f==other, -1 if f < other, and +1 if f > other.
func (f Fields) Compare(other Fields) int {
	onefields := f.SortCopy()
	otherfields := other.SortCopy()

	for i, field := range onefields {
		if i == len(otherfields) {
			return 1
		}
		if comp := field.Compare(otherfields[i]); comp != 0 {
			return comp
		}
	}
	if len(onefields) == len(otherfields) {
		return 0
	}
	return -1
}

// List is a list of items.
type List struct {
	Items []Value
}

func (l List) SortedItems() []Value {
	copy := []Value{}
	for _, item := range l.Items {
		copy = append(copy, item)
	}
	sort.Slice(copy, func(i, j int) bool {
		return copy[i].Compare(copy[j]) < 0
	})
	return copy
}

// Compare returns an integer comparing two Lists. The result will be 0
// if l==other, -1 if l < other, and +1 if l > other.
func (l List) Compare(other List) int {
	for i, value := range l.Items {
		if i == len(other.Items) {
			return 1
		}
		if comp := value.Compare(other.Items[i]); comp != 0 {
			return comp
		}
	}
	if len(l.Items) == len(other.Items) {
		return 0
	}
	return -1
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

// Compare returns an integer comparing two Maps. The result will be 0
// if m==other, -1 if m < other, and +1 if m > other.
func (m Map) Compare(other Map) int {
	return Fields(m.Items).Compare(Fields(other.Items))
}

func (m *Map) SortedItems() []Field {
	return Fields(m.Items).SortCopy()
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

// Delete removes the key from the set.
func (m *Map) Delete(key string) {
	items := []Field{}
	for i := range m.Items {
		if m.Items[i].Name != key {
			items = append(items, m.Items[i])
		}
	}
	m.Items = items
	m.index = nil // Since the list has changed
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

// Compare returns an integer comparing two Values. The result will be 0
// if v==other, -1 if v < other, and +1 if v > other.
func (v Value) Compare(other Value) int {
	if v.FloatValue != nil {
		if other.FloatValue == nil {
			return -1
		}
		if *v.FloatValue == *other.FloatValue {
			return 0
		} else if *v.FloatValue < *other.FloatValue {
			return 1
		}
		return 1
	}
	if other.FloatValue != nil {
		return 1
	}

	if v.IntValue != nil {
		if other.IntValue == nil {
			return -1
		}
		if *v.IntValue == *other.IntValue {
			return 0
		} else if *v.IntValue < *other.IntValue {
			return -1
		}
		return 1
	}
	if other.IntValue != nil {
		return 1
	}

	if v.StringValue != nil {
		if other.StringValue == nil {
			return -1
		}
		return strings.Compare(string(*v.StringValue), string(*other.StringValue))
	}
	if other.StringValue != nil {
		return 1
	}

	if v.BooleanValue != nil {
		if other.BooleanValue == nil {
			return -1
		}
		if *v.BooleanValue == *other.BooleanValue {
			return 0
		} else if !*v.BooleanValue {
			return -1
		}
		return 1
	}
	if other.BooleanValue != nil {
		return 1
	}

	if v.ListValue != nil {
		if other.ListValue == nil {
			return -1
		}
		return v.ListValue.Compare(*other.ListValue)
	}
	if other.ListValue != nil {
		return 1
	}

	if v.MapValue != nil {
		if other.MapValue == nil {
			return -1
		}
		return v.MapValue.Compare(*other.MapValue)
	}
	if other.MapValue != nil {
		return 1
	}

	// At this point they should both be null. But there is nothing
	// we can do if they are not.
	if v.Null && other.Null {
		return 0
	}

	panic("shouldn't reach")
}

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

// Equals returns true iff the two values are equal.
func (v Value) Equals(rhs Value) bool {
	return !v.Less(rhs) && !rhs.Less(v)
}

// Less provides a total ordering for Value (so that they can be sorted, even
// if they are of different types).
func (v Value) Less(rhs Value) bool {
	if v.FloatValue != nil {
		if rhs.FloatValue == nil {
			// Extra: compare floats and ints numerically.
			if rhs.IntValue != nil {
				return float64(*v.FloatValue) < float64(*rhs.IntValue)
			}
			return true
		}
		return *v.FloatValue < *rhs.FloatValue
	} else if rhs.FloatValue != nil {
		// Extra: compare floats and ints numerically.
		if v.IntValue != nil {
			return float64(*v.IntValue) < float64(*rhs.FloatValue)
		}
		return false
	}

	if v.IntValue != nil {
		if rhs.IntValue == nil {
			return true
		}
		return *v.IntValue < *rhs.IntValue
	} else if rhs.IntValue != nil {
		return false
	}

	if v.StringValue != nil {
		if rhs.StringValue == nil {
			return true
		}
		return *v.StringValue < *rhs.StringValue
	} else if rhs.StringValue != nil {
		return false
	}

	if v.BooleanValue != nil {
		if rhs.BooleanValue == nil {
			return true
		}
		if *v.BooleanValue == *rhs.BooleanValue {
			return false
		}
		return *v.BooleanValue == false
	} else if rhs.BooleanValue != nil {
		return false
	}

	if v.ListValue != nil {
		if rhs.ListValue == nil {
			return true
		}
		return v.ListValue.Less(rhs.ListValue)
	} else if rhs.ListValue != nil {
		return false
	}
	if v.MapValue != nil {
		if rhs.MapValue == nil {
			return true
		}
		return v.MapValue.Less(rhs.MapValue)
	} else if rhs.MapValue != nil {
		return false
	}
	if v.Null {
		if !rhs.Null {
			return true
		}
		return false
	} else if rhs.Null {
		return false
	}

	// Invalid Value-- nothing is set.
	return false
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

// Less compares two lists lexically.
func (l *List) Less(rhs *List) bool {
	i := 0
	for {
		if i >= len(l.Items) && i >= len(rhs.Items) {
			// Lists are the same length and all items are equal.
			return false
		}
		if i >= len(l.Items) {
			// LHS is shorter.
			return true
		}
		if i >= len(rhs.Items) {
			// RHS is shorter.
			return false
		}
		if l.Items[i].Less(rhs.Items[i]) {
			// LHS is less; return
			return true
		}
		if rhs.Items[i].Less(l.Items[i]) {
			// RHS is less; return
			return false
		}
		// The items are equal; continue.
		i++
	}
}

// Map is a map of key-value pairs. It represents both structs and maps. We use
// a list and a go-language map to preserve order.
//
// Set and Get helpers are provided.
type Map struct {
	Items []Field

	// may be nil; lazily constructed.
	// TODO: Direct modifications to Items above will cause serious problems.
	index map[string]int
	// may be empty; lazily constructed.
	// TODO: Direct modifications to Items above will cause serious problems.
	order []int
}

func (m *Map) computeOrder() []int {
	if len(m.order) != len(m.Items) {
		m.order = make([]int, len(m.Items))
		for i := range m.order {
			m.order[i] = i
		}
		sort.SliceStable(m.order, func(i, j int) bool {
			return m.Items[m.order[i]].Name < m.Items[m.order[j]].Name
		})
	}
	return m.order
}

// Less compares two maps lexically.
func (m *Map) Less(rhs *Map) bool {
	var noAllocL, noAllocR [2]int
	var morder, rorder []int

	// For very short maps (<2 elements) this permits us to avoid
	// allocating the order array. We could make this accomodate larger
	// maps, but 2 items should be enough to cover most path element
	// comparisons, and at some point there will be diminishing returns.
	// This has a large effect on the path element deserialization test,
	// because everything is sorted / compared, but only once.
	switch len(m.Items) {
	case 0:
		morder = noAllocL[0:0]
	case 1:
		morder = noAllocL[0:1]
	case 2:
		morder = noAllocL[0:2]
		if m.Items[0].Name > m.Items[1].Name {
			morder[0] = 1
		} else {
			morder[1] = 1
		}
	default:
		morder = m.computeOrder()
	}

	switch len(rhs.Items) {
	case 0:
		rorder = noAllocR[0:0]
	case 1:
		rorder = noAllocR[0:1]
	case 2:
		rorder = noAllocR[0:2]
		if rhs.Items[0].Name > rhs.Items[1].Name {
			rorder[0] = 1
		} else {
			rorder[1] = 1
		}
	default:
		rorder = rhs.computeOrder()
	}

	i := 0
	for {
		if i >= len(morder) && i >= len(rorder) {
			// Maps are the same length and all items are equal.
			return false
		}
		if i >= len(morder) {
			// LHS is shorter.
			return true
		}
		if i >= len(rorder) {
			// RHS is shorter.
			return false
		}
		fa, fb := &m.Items[morder[i]], &rhs.Items[rorder[i]]
		if fa.Name != fb.Name {
			// the map having the field name that sorts lexically less is "less"
			return fa.Name < fb.Name
		}
		if fa.Value.Less(fb.Value) {
			// LHS is less; return
			return true
		}
		if fb.Value.Less(fa.Value) {
			// RHS is less; return
			return false
		}
		// The items are equal; continue.
		i++
	}
}

// Get returns the (Field, true) or (nil, false) if it is not present
func (m *Map) Get(key string) (*Field, bool) {
	if m.index == nil {
		m.index = map[string]int{}
		for i := range m.Items {
			m.index[m.Items[i].Name] = i
		}
	}
	f, ok := m.index[key]
	if !ok {
		return nil, false
	}
	return &m.Items[f], true
}

// Set inserts or updates the given item.
func (m *Map) Set(key string, value Value) {
	if f, ok := m.Get(key); ok {
		f.Value = value
		return
	}
	m.Items = append(m.Items, Field{Name: key, Value: value})
	i := len(m.Items) - 1
	m.index[key] = i
	m.order = nil
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
	m.order = nil
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

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
	if v.FloatValue != nil || rhs.FloatValue != nil {
		var lf float64
		if v.FloatValue != nil {
			lf = float64(*v.FloatValue)
		} else if v.IntValue != nil {
			lf = float64(*v.IntValue)
		} else {
			return false
		}
		var rf float64
		if rhs.FloatValue != nil {
			rf = float64(*rhs.FloatValue)
		} else if rhs.IntValue != nil {
			rf = float64(*rhs.IntValue)
		} else {
			return false
		}
		return lf == rf
	}
	if v.IntValue != nil {
		if rhs.IntValue != nil {
			return *v.IntValue == *rhs.IntValue
		}
		return false
	}
	if v.StringValue != nil {
		if rhs.StringValue != nil {
			return *v.StringValue == *rhs.StringValue
		}
		return false
	}
	if v.BooleanValue != nil {
		if rhs.BooleanValue != nil {
			return *v.BooleanValue == *rhs.BooleanValue
		}
		return false
	}
	if v.ListValue != nil {
		if rhs.ListValue != nil {
			return v.ListValue.Equals(rhs.ListValue)
		}
		return false
	}
	if v.MapValue != nil {
		if rhs.MapValue != nil {
			return v.MapValue.Equals(rhs.MapValue)
		}
		return false
	}
	if v.Null {
		if rhs.Null {
			return true
		}
		return false
	}
	// No field is set, on either objects.
	return true
}

// Less provides a total ordering for Value (so that they can be sorted, even
// if they are of different types).
func (v Value) Less(rhs Value) bool {
	return v.Compare(rhs) == -1
}

// Compare provides a total ordering for Value (so that they can be
// sorted, even if they are of different types). The result will be 0 if
// v==rhs, -1 if v < rhs, and +1 if v > rhs.
func (v Value) Compare(rhs Value) int {
	if v.FloatValue != nil {
		if rhs.FloatValue == nil {
			// Extra: compare floats and ints numerically.
			if rhs.IntValue != nil {
				return v.FloatValue.Compare(Float(*rhs.IntValue))
			}
			return -1
		}
		return v.FloatValue.Compare(*rhs.FloatValue)
	} else if rhs.FloatValue != nil {
		// Extra: compare floats and ints numerically.
		if v.IntValue != nil {
			return Float(*v.IntValue).Compare(*rhs.FloatValue)
		}
		return 1
	}

	if v.IntValue != nil {
		if rhs.IntValue == nil {
			return -1
		}
		return v.IntValue.Compare(*rhs.IntValue)
	} else if rhs.IntValue != nil {
		return 1
	}

	if v.StringValue != nil {
		if rhs.StringValue == nil {
			return -1
		}
		return strings.Compare(string(*v.StringValue), string(*rhs.StringValue))
	} else if rhs.StringValue != nil {
		return 1
	}

	if v.BooleanValue != nil {
		if rhs.BooleanValue == nil {
			return -1
		}
		return v.BooleanValue.Compare(*rhs.BooleanValue)
	} else if rhs.BooleanValue != nil {
		return 1
	}

	if v.ListValue != nil {
		if rhs.ListValue == nil {
			return -1
		}
		return v.ListValue.Compare(rhs.ListValue)
	} else if rhs.ListValue != nil {
		return 1
	}
	if v.MapValue != nil {
		if rhs.MapValue == nil {
			return -1
		}
		return v.MapValue.Compare(rhs.MapValue)
	} else if rhs.MapValue != nil {
		return 1
	}
	if v.Null {
		if !rhs.Null {
			return -1
		}
		return 0
	} else if rhs.Null {
		return 1
	}

	// Invalid Value-- nothing is set.
	return 0
}

type Int int64
type Float float64
type String string
type Boolean bool

// Compare compares integers. The result will be 0 if i==rhs, -1 if i <
// rhs, and +1 if i > rhs.
func (i Int) Compare(rhs Int) int {
	if i > rhs {
		return 1
	} else if i < rhs {
		return -1
	}
	return 0
}

// Compare compares floats. The result will be 0 if f==rhs, -1 if f <
// rhs, and +1 if f > rhs.
func (f Float) Compare(rhs Float) int {
	if f > rhs {
		return 1
	} else if f < rhs {
		return -1
	}
	return 0
}

// Compare compares booleans. The result will be 0 if b==rhs, -1 if b <
// rhs, and +1 if b > rhs.
func (b Boolean) Compare(rhs Boolean) int {
	if b == rhs {
		return 0
	} else if b == false {
		return -1
	}
	return 1
}

// Field is an individual key-value pair.
type Field struct {
	Name  string
	Value Value
}

// FieldList is a list of key-value pairs. Each field is expected to
// have a different name.
type FieldList []Field

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

// Less compares two lists lexically. The result will be 0 if f==rhs, -1
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
		if c := f[i].Value.Compare(rhs[i].Value); c != 0 {
			return c
		}
		// The items are equal; continue.
		i++
	}
}

// List is a list of items.
type List struct {
	Items []Value
}

// Equals compares two lists lexically.
func (l *List) Equals(rhs *List) bool {
	if len(l.Items) != len(rhs.Items) {
		return false
	}

	for i, lv := range l.Items {
		if !lv.Equals(rhs.Items[i]) {
			return false
		}
	}
	return true
}

// Less compares two lists lexically.
func (l *List) Less(rhs *List) bool {
	return l.Compare(rhs) == -1
}

// Compare compares two lists lexically. The result will be 0 if l==rhs, -1
// if l < rhs, and +1 if l > rhs.
func (l *List) Compare(rhs *List) int {
	i := 0
	for {
		if i >= len(l.Items) && i >= len(rhs.Items) {
			// Lists are the same length and all items are equal.
			return 0
		}
		if i >= len(l.Items) {
			// LHS is shorter.
			return -1
		}
		if i >= len(rhs.Items) {
			// RHS is shorter.
			return 1
		}
		if c := l.Items[i].Compare(rhs.Items[i]); c != 0 {
			return c
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

// Equals compares two maps lexically.
func (m *Map) Equals(rhs *Map) bool {
	if len(m.Items) != len(rhs.Items) {
		return false
	}
	for _, lfield := range m.Items {
		rfield, ok := rhs.Get(lfield.Name)
		if !ok {
			return false
		}
		if !lfield.Value.Equals(rfield.Value) {
			return false
		}
	}
	return true
}

// Less compares two maps lexically.
func (m *Map) Less(rhs *Map) bool {
	return m.Compare(rhs) == -1
}

// Compare compares two maps lexically.
func (m *Map) Compare(rhs *Map) int {
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
			return 0
		}
		if i >= len(morder) {
			// LHS is shorter.
			return -1
		}
		if i >= len(rorder) {
			// RHS is shorter.
			return 1
		}
		fa, fb := &m.Items[morder[i]], &rhs.Items[rorder[i]]
		if c := strings.Compare(fa.Name, fb.Name); c != 0 {
			return c
		}
		if c := fa.Value.Compare(fb.Value); c != 0 {
			return c
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

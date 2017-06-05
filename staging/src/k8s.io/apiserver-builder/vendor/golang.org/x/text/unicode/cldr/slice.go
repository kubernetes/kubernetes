// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cldr

import (
	"fmt"
	"reflect"
	"sort"
)

// Slice provides utilities for modifying slices of elements.
// It can be wrapped around any slice of which the element type implements
// interface Elem.
type Slice struct {
	ptr reflect.Value
	typ reflect.Type
}

// Value returns the reflect.Value of the underlying slice.
func (s *Slice) Value() reflect.Value {
	return s.ptr.Elem()
}

// MakeSlice wraps a pointer to a slice of Elems.
// It replaces the array pointed to by the slice so that subsequent modifications
// do not alter the data in a CLDR type.
// It panics if an incorrect type is passed.
func MakeSlice(slicePtr interface{}) Slice {
	ptr := reflect.ValueOf(slicePtr)
	if ptr.Kind() != reflect.Ptr {
		panic(fmt.Sprintf("MakeSlice: argument must be pointer to slice, found %v", ptr.Type()))
	}
	sl := ptr.Elem()
	if sl.Kind() != reflect.Slice {
		panic(fmt.Sprintf("MakeSlice: argument must point to a slice, found %v", sl.Type()))
	}
	intf := reflect.TypeOf((*Elem)(nil)).Elem()
	if !sl.Type().Elem().Implements(intf) {
		panic(fmt.Sprintf("MakeSlice: element type of slice (%v) does not implement Elem", sl.Type().Elem()))
	}
	nsl := reflect.MakeSlice(sl.Type(), sl.Len(), sl.Len())
	reflect.Copy(nsl, sl)
	sl.Set(nsl)
	return Slice{
		ptr: ptr,
		typ: sl.Type().Elem().Elem(),
	}
}

func (s Slice) indexForAttr(a string) []int {
	for i := iter(reflect.Zero(s.typ)); !i.done(); i.next() {
		if n, _ := xmlName(i.field()); n == a {
			return i.index
		}
	}
	panic(fmt.Sprintf("MakeSlice: no attribute %q for type %v", a, s.typ))
}

// Filter filters s to only include elements for which fn returns true.
func (s Slice) Filter(fn func(e Elem) bool) {
	k := 0
	sl := s.Value()
	for i := 0; i < sl.Len(); i++ {
		vi := sl.Index(i)
		if fn(vi.Interface().(Elem)) {
			sl.Index(k).Set(vi)
			k++
		}
	}
	sl.Set(sl.Slice(0, k))
}

// Group finds elements in s for which fn returns the same value and groups
// them in a new Slice.
func (s Slice) Group(fn func(e Elem) string) []Slice {
	m := make(map[string][]reflect.Value)
	sl := s.Value()
	for i := 0; i < sl.Len(); i++ {
		vi := sl.Index(i)
		key := fn(vi.Interface().(Elem))
		m[key] = append(m[key], vi)
	}
	keys := []string{}
	for k, _ := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	res := []Slice{}
	for _, k := range keys {
		nsl := reflect.New(sl.Type())
		nsl.Elem().Set(reflect.Append(nsl.Elem(), m[k]...))
		res = append(res, MakeSlice(nsl.Interface()))
	}
	return res
}

// SelectAnyOf filters s to contain only elements for which attr matches
// any of the values.
func (s Slice) SelectAnyOf(attr string, values ...string) {
	index := s.indexForAttr(attr)
	s.Filter(func(e Elem) bool {
		vf := reflect.ValueOf(e).Elem().FieldByIndex(index)
		return in(values, vf.String())
	})
}

// SelectOnePerGroup filters s to include at most one element e per group of
// elements matching Key(attr), where e has an attribute a that matches any
// the values in v.
// If more than one element in a group matches a value in v preference
// is given to the element that matches the first value in v.
func (s Slice) SelectOnePerGroup(a string, v []string) {
	index := s.indexForAttr(a)
	grouped := s.Group(func(e Elem) string { return Key(e, a) })
	sl := s.Value()
	sl.Set(sl.Slice(0, 0))
	for _, g := range grouped {
		e := reflect.Value{}
		found := len(v)
		gsl := g.Value()
		for i := 0; i < gsl.Len(); i++ {
			vi := gsl.Index(i).Elem().FieldByIndex(index)
			j := 0
			for ; j < len(v) && v[j] != vi.String(); j++ {
			}
			if j < found {
				found = j
				e = gsl.Index(i)
			}
		}
		if found < len(v) {
			sl.Set(reflect.Append(sl, e))
		}
	}
}

// SelectDraft drops all elements from the list with a draft level smaller than d
// and selects the highest draft level of the remaining.
// This method assumes that the input CLDR is canonicalized.
func (s Slice) SelectDraft(d Draft) {
	s.SelectOnePerGroup("draft", drafts[len(drafts)-2-int(d):])
}

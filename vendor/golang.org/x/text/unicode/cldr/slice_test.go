// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cldr

import (
	"reflect"
	"testing"
)

type testSlice []*Common

func mkElem(alt, typ, ref string) *Common {
	return &Common{
		Type:      typ,
		Reference: ref,
		Alt:       alt,
	}
}

var (
	testSlice1 = testSlice{
		mkElem("1", "a", "i.a"),
		mkElem("1", "b", "i.b"),
		mkElem("1", "c", "i.c"),
		mkElem("2", "b", "ii"),
		mkElem("3", "c", "iii"),
		mkElem("4", "a", "iv.a"),
		mkElem("4", "d", "iv.d"),
	}
	testSliceE = testSlice{}
)

func panics(f func()) (panics bool) {
	defer func() {
		if err := recover(); err != nil {
			panics = true
		}
	}()
	f()
	return panics
}

func TestMakeSlice(t *testing.T) {
	foo := 1
	bar := []int{}
	tests := []struct {
		i      interface{}
		panics bool
		err    string
	}{
		{&foo, true, "should panic when passed a pointer to the wrong type"},
		{&bar, true, "should panic when slice element of the wrong type"},
		{testSlice1, true, "should panic when passed a slice"},
		{&testSlice1, false, "should not panic"},
	}
	for i, tt := range tests {
		if panics(func() { MakeSlice(tt.i) }) != tt.panics {
			t.Errorf("%d: %s", i, tt.err)
		}
	}
}

var anyOfTests = []struct {
	sl     testSlice
	values []string
	n      int
}{
	{testSliceE, []string{}, 0},
	{testSliceE, []string{"1", "2", "3"}, 0},
	{testSlice1, []string{}, 0},
	{testSlice1, []string{"1"}, 3},
	{testSlice1, []string{"2"}, 1},
	{testSlice1, []string{"5"}, 0},
	{testSlice1, []string{"1", "2", "3"}, 5},
}

func TestSelectAnyOf(t *testing.T) {
	for i, tt := range anyOfTests {
		sl := tt.sl
		s := MakeSlice(&sl)
		s.SelectAnyOf("alt", tt.values...)
		if len(sl) != tt.n {
			t.Errorf("%d: found len == %d; want %d", i, len(sl), tt.n)
		}
	}
	sl := testSlice1
	s := MakeSlice(&sl)
	if !panics(func() { s.SelectAnyOf("foo") }) {
		t.Errorf("should panic on non-existing attribute")
	}
}

func TestFilter(t *testing.T) {
	for i, tt := range anyOfTests {
		sl := tt.sl
		s := MakeSlice(&sl)
		s.Filter(func(e Elem) bool {
			v, _ := findField(reflect.ValueOf(e), "alt")
			return in(tt.values, v.String())
		})
		if len(sl) != tt.n {
			t.Errorf("%d: found len == %d; want %d", i, len(sl), tt.n)
		}
	}
}

func TestGroup(t *testing.T) {
	f := func(excl ...string) func(Elem) string {
		return func(e Elem) string {
			return Key(e, excl...)
		}
	}
	tests := []struct {
		sl   testSlice
		f    func(Elem) string
		lens []int
	}{
		{testSliceE, f(), []int{}},
		{testSlice1, f(), []int{1, 1, 1, 1, 1, 1, 1}},
		{testSlice1, f("type"), []int{3, 1, 1, 2}},
		{testSlice1, f("alt"), []int{2, 2, 2, 1}},
		{testSlice1, f("alt", "type"), []int{7}},
		{testSlice1, f("alt", "type"), []int{7}},
	}
	for i, tt := range tests {
		sl := tt.sl
		s := MakeSlice(&sl)
		g := s.Group(tt.f)
		if len(tt.lens) != len(g) {
			t.Errorf("%d: found %d; want %d", i, len(g), len(tt.lens))
			continue
		}
		for j, v := range tt.lens {
			if n := g[j].Value().Len(); n != v {
				t.Errorf("%d: found %d for length of group %d; want %d", i, n, j, v)
			}
		}
	}
}

func TestSelectOnePerGroup(t *testing.T) {
	tests := []struct {
		sl     testSlice
		attr   string
		values []string
		refs   []string
	}{
		{testSliceE, "alt", []string{"1"}, []string{}},
		{testSliceE, "type", []string{"a"}, []string{}},
		{testSlice1, "alt", []string{"2", "3", "1"}, []string{"i.a", "ii", "iii"}},
		{testSlice1, "alt", []string{"1", "4"}, []string{"i.a", "i.b", "i.c", "iv.d"}},
		{testSlice1, "type", []string{"c", "d"}, []string{"i.c", "iii", "iv.d"}},
	}
	for i, tt := range tests {
		sl := tt.sl
		s := MakeSlice(&sl)
		s.SelectOnePerGroup(tt.attr, tt.values)
		if len(sl) != len(tt.refs) {
			t.Errorf("%d: found result length %d; want %d", i, len(sl), len(tt.refs))
			continue
		}
		for j, e := range sl {
			if tt.refs[j] != e.Reference {
				t.Errorf("%d:%d found %s; want %s", i, j, e.Reference, tt.refs[i])
			}
		}
	}
	sl := testSlice1
	s := MakeSlice(&sl)
	if !panics(func() { s.SelectOnePerGroup("foo", nil) }) {
		t.Errorf("should panic on non-existing attribute")
	}
}

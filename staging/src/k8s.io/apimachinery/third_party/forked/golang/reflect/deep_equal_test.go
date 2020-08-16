// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflect

import (
	"testing"
)

func TestEqualities(t *testing.T) {
	e := Equalities{}
	type Bar struct {
		X int
	}
	type Baz struct {
		Y Bar
	}
	err := e.AddFuncs(
		func(a, b int) bool {
			return a+1 == b
		},
		func(a, b Bar) bool {
			return a.X*10 == b.X
		},
	)
	if err != nil {
		t.Fatalf("Unexpected: %v", err)
	}

	type Foo struct {
		X int
	}

	table := []struct {
		a, b  interface{}
		equal bool
	}{
		{1, 2, true},
		{2, 1, false},
		{"foo", "fo", false},
		{"foo", "foo", true},
		{"foo", "foobar", false},
		{Foo{1}, Foo{2}, true},
		{Foo{2}, Foo{1}, false},
		{Bar{1}, Bar{10}, true},
		{&Bar{1}, &Bar{10}, true},
		{Baz{Bar{1}}, Baz{Bar{10}}, true},
		{[...]string{}, [...]string{"1", "2", "3"}, false},
		{[...]string{"1"}, [...]string{"1", "2", "3"}, false},
		{[...]string{"1", "2", "3"}, [...]string{}, false},
		{[...]string{"1", "2", "3"}, [...]string{"1", "2", "3"}, true},
		{map[string]int{"foo": 1}, map[string]int{}, false},
		{map[string]int{"foo": 1}, map[string]int{"foo": 2}, true},
		{map[string]int{"foo": 2}, map[string]int{"foo": 1}, false},
		{map[string]int{"foo": 1}, map[string]int{"foo": 2, "bar": 6}, false},
		{map[string]int{"foo": 1, "bar": 6}, map[string]int{"foo": 2}, false},
		{map[string]int{}, map[string]int(nil), true},
		{[]string(nil), []string(nil), true},
		{[]string{}, []string(nil), true},
		{[]string(nil), []string{}, true},
		{[]string{"1"}, []string(nil), false},
		{[]string{}, []string{"1", "2", "3"}, false},
		{[]string{"1"}, []string{"1", "2", "3"}, false},
		{[]string{"1", "2", "3"}, []string{}, false},
	}

	for _, item := range table {
		if e, a := item.equal, e.DeepEqual(item.a, item.b); e != a {
			t.Errorf("Expected (%+v == %+v) == %v, but got %v", item.a, item.b, e, a)
		}
	}
}

func TestDerivatives(t *testing.T) {
	e := Equalities{}
	type Bar struct {
		X int
	}
	type Baz struct {
		Y Bar
	}
	err := e.AddFuncs(
		func(a, b int) bool {
			return a+1 == b
		},
		func(a, b Bar) bool {
			return a.X*10 == b.X
		},
	)
	if err != nil {
		t.Fatalf("Unexpected: %v", err)
	}

	type Foo struct {
		X int
	}

	table := []struct {
		a, b  interface{}
		equal bool
	}{
		{1, 2, true},
		{2, 1, false},
		{"foo", "fo", false},
		{"foo", "foo", true},
		{"foo", "foobar", false},
		{Foo{1}, Foo{2}, true},
		{Foo{2}, Foo{1}, false},
		{Bar{1}, Bar{10}, true},
		{&Bar{1}, &Bar{10}, true},
		{Baz{Bar{1}}, Baz{Bar{10}}, true},
		{[...]string{}, [...]string{"1", "2", "3"}, false},
		{[...]string{"1"}, [...]string{"1", "2", "3"}, false},
		{[...]string{"1", "2", "3"}, [...]string{}, false},
		{[...]string{"1", "2", "3"}, [...]string{"1", "2", "3"}, true},
		{map[string]int{"foo": 1}, map[string]int{}, false},
		{map[string]int{"foo": 1}, map[string]int{"foo": 2}, true},
		{map[string]int{"foo": 2}, map[string]int{"foo": 1}, false},
		{map[string]int{"foo": 1}, map[string]int{"foo": 2, "bar": 6}, true},
		{map[string]int{"foo": 1, "bar": 6}, map[string]int{"foo": 2}, false},
		{map[string]int{}, map[string]int(nil), true},
		{[]string(nil), []string(nil), true},
		{[]string{}, []string(nil), true},
		{[]string(nil), []string{}, true},
		{[]string{"1"}, []string(nil), false},
		{[]string{}, []string{"1", "2", "3"}, true},
		{[]string{"1"}, []string{"1", "2", "3"}, true},
		{[]string{"1", "2", "3"}, []string{}, false},
	}

	for _, item := range table {
		if e, a := item.equal, e.DeepDerivative(item.a, item.b); e != a {
			t.Errorf("Expected (%+v ~ %+v) == %v, but got %v", item.a, item.b, e, a)
		}
	}
}

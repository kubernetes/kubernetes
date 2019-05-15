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

package conversion

import "testing"

func TestDerivates(t *testing.T) {
	type Bar struct {
		X int
	}
	type Baz struct {
		Y Bar
	}
	type Foo struct {
		X int
	}

	e := EqualitiesOrDie(
		func(a, b int) bool {
			return a+1 == b || b+1 == a
		},
		func(a, b Bar) bool {
			return a.X*10 == b.X || b.X*10 == a.X
		},
	)

	table := []struct {
		a, b  interface{}
		equal bool
	}{
		{1, 2, true},
		{1, 3, false},
		{"foo", "fo", false},
		{"foo", "foo", true},
		{"foo", "foobar", false},
		{Foo{1}, Foo{2}, true},
		{Foo{1}, Foo{3}, false},
		{Bar{1}, Bar{10}, true},
		{&Bar{1}, &Bar{10}, true},
		{Baz{Bar{1}}, Baz{Bar{10}}, true},
		{[...]string{}, [...]string{"1", "2", "3"}, false},
		{[...]string{"1"}, [...]string{"1", "2", "3"}, false},
		{[...]string{"1", "2", "3"}, [...]string{}, false},
		{[...]string{"1", "2", "3"}, [...]string{"1", "2", "3"}, true},
		{map[string]int{"foo": 1}, map[string]int{}, false},
		{map[string]int{"foo": 1}, map[string]int{"foo": 2}, true},
		{map[string]int{"foo": 1}, map[string]int{"foo": 3}, false},
		{map[string]int{"foo": 1}, map[string]int{"foo": 2, "bar": 6}, true},
		{map[string]int{"foo": 1, "bar": 6}, map[string]int{"foo": 2}, false},
		{map[string]int{}, map[string]int(nil), true},
		{[]string(nil), []string(nil), true},
		{[]string{}, []string(nil), true},
		{[]string(nil), []string{}, true},
		{[]string{"1"}, []string(nil), false},
		{[]string{}, []string{"1", "2", "3"}, true},
		{[]string{"1"}, []string{"1", "2", "3"}, false},
		{[]string{"1", "2", "3"}, []string{}, false},
	}

	for _, item := range table {
		t.Run("", func(t *testing.T) {
			if e, a := item.equal, e.DeepDerivative(item.a, item.b); e != a {
				t.Errorf("Expected (%+v ~ %+v) == %v, but got %v", item.a, item.b, e, a)
			}
		})
	}
}

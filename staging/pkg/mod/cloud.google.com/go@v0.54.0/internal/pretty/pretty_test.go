// Copyright 2016 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package pretty

import (
	"fmt"
	"strings"
	"testing"
)

type S struct {
	X int
	Y bool
	z *string
}

func TestSprint(t *testing.T) {
	Indent = "~"
	i := 17

	for _, test := range []struct {
		value interface{}
		want  string
	}{
		// primitives and pointer
		{nil, "nil"},
		{3, "3"},
		{9.8, "9.8"},
		{true, "true"},
		{"foo", `"foo"`},
		{&i, "&17"},
		// array and slice
		{[3]int{1, 2, 3}, "[3]int{\n~1,\n~2,\n~3,\n}"},
		{[]int{1, 2, 3}, "[]int{\n~1,\n~2,\n~3,\n}"},
		{[]int{}, "[]int{}"},
		{[]string{"foo"}, "[]string{\n~\"foo\",\n}"},
		// map
		{map[int]bool{}, "map[int]bool{}"},
		{map[int]bool{1: true, 2: false, 3: true},
			"map[int]bool{\n~1: true,\n~3: true,\n}"},
		// struct
		{S{}, "pretty.S{\n}"},
		{S{3, true, ptr("foo")},
			"pretty.S{\n~X: 3,\n~Y: true,\n~z: &\"foo\",\n}"},
		// interface
		{[]interface{}{&i}, "[]interface {}{\n~&17,\n}"},
		// nesting
		{[]S{{1, false, ptr("a")}, {2, true, ptr("b")}},
			`[]pretty.S{
~pretty.S{
~~X: 1,
~~z: &"a",
~},
~pretty.S{
~~X: 2,
~~Y: true,
~~z: &"b",
~},
}`},
	} {
		got := fmt.Sprintf("%v", Value(test.value))
		if got != test.want {
			t.Errorf("%v: got:\n%q\nwant:\n%q", test.value, got, test.want)
		}
	}
}

func TestWithDefaults(t *testing.T) {
	Indent = "~"
	for _, test := range []struct {
		value interface{}
		want  string
	}{
		{map[int]bool{1: true, 2: false, 3: true},
			"map[int]bool{\n~1: true,\n~2: false,\n~3: true,\n}"},
		{S{}, "pretty.S{\n~X: 0,\n~Y: false,\n~z: nil,\n}"},
	} {
		got := fmt.Sprintf("%+v", Value(test.value))
		if got != test.want {
			t.Errorf("%v: got:\n%q\nwant:\n%q", test.value, got, test.want)
		}
	}
}

func TestBadVerb(t *testing.T) {
	got := fmt.Sprintf("%d", Value(8))
	want := "%!d("
	if !strings.HasPrefix(got, want) {
		t.Errorf("got %q, want prefix %q", got, want)
	}
}

func ptr(s string) *string { return &s }

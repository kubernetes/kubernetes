/*
Copyright 2016 The Kubernetes Authors.

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

package diff

import (
	"testing"
)

func TestObjectReflectDiff(t *testing.T) {
	type struct1 struct{ A []int }

	testCases := map[string]struct {
		a, b interface{}
		out  string
	}{
		"map": {
			a: map[string]int{},
			b: map[string]int{},
		},
		"detect nil map": {
			a: map[string]int(nil),
			b: map[string]int{},
			out: `
object:
  a: map[string]int(nil)
  b: map[string]int{}`,
		},
		"detect map changes": {
			a: map[string]int{"test": 1, "other": 2},
			b: map[string]int{"test": 2, "third": 3},
			out: `
object[other]:
  a: 2
  b: <nil>
object[test]:
  a: 1
  b: 2
object[third]:
  a: <nil>
  b: 3`,
		},
		"nil slice":   {a: struct1{A: nil}, b: struct1{A: nil}},
		"empty slice": {a: struct1{A: []int{}}, b: struct1{A: []int{}}},
		"detect slice changes 1": {a: struct1{A: []int{1}}, b: struct1{A: []int{2}}, out: `
object.A[0]:
  a: 1
  b: 2`,
		},
		"detect slice changes 2": {a: struct1{A: []int{}}, b: struct1{A: []int{2}}, out: `
object.A[0]:
  a: <nil>
  b: 2`,
		},
		"detect slice changes 3": {a: struct1{A: []int{1}}, b: struct1{A: []int{}}, out: `
object.A[0]:
  a: 1
  b: <nil>`,
		},
		"detect nil vs empty slices": {a: struct1{A: nil}, b: struct1{A: []int{}}, out: `
object.A:
  a: []int(nil)
  b: []int{}`,
		},
		"display type differences": {a: []interface{}{int64(1)}, b: []interface{}{uint64(1)}, out: `
object[0]:
  a: 1 (int64)
  b: 0x1 (uint64)`,
		},
	}
	for name, test := range testCases {
		expect := test.out
		if len(expect) == 0 {
			expect = "<no diffs>"
		}
		if actual := ObjectReflectDiff(test.a, test.b); actual != expect {
			t.Errorf("%s: unexpected output: %s", name, actual)
		}
	}
}

func TestStringDiff(t *testing.T) {
	diff := StringDiff("aaabb", "aaacc")
	expect := "aaa\n\nA: bb\n\nB: cc\n\n"
	if diff != expect {
		t.Errorf("diff returned %v", diff)
	}
}

func TestLimit(t *testing.T) {
	testcases := []struct {
		a       interface{}
		b       interface{}
		expectA string
		expectB string
	}{
		{
			a:       `short a`,
			b:       `short b`,
			expectA: `"short a"`,
			expectB: `"short b"`,
		},
		{
			a:       `short a`,
			b:       `long b needs truncating`,
			expectA: `"short a"`,
			expectB: `"long b ne...`,
		},
		{
			a:       `long a needs truncating`,
			b:       `long b needs truncating`,
			expectA: `...g a needs ...`,
			expectB: `...g b needs ...`,
		},
		{
			a:       `long common prefix with different stuff at the end of a`,
			b:       `long common prefix with different stuff at the end of b`,
			expectA: `...end of a"`,
			expectB: `...end of b"`,
		},
		{
			a:       `long common prefix with different stuff at the end of a`,
			b:       `long common prefix with different stuff at the end of b which continues`,
			expectA: `...of a"`,
			expectB: `...of b which...`,
		},
	}

	for _, tc := range testcases {
		a, b := limit(tc.a, tc.b, 10)
		if a != tc.expectA || b != tc.expectB {
			t.Errorf("limit(%q, %q)\n\texpected: %s, %s\n\tgot:      %s, %s", tc.a, tc.b, tc.expectA, tc.expectB, a, b)
		}
	}
}

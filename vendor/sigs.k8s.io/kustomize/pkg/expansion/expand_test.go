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

package expansion_test

import (
	"fmt"
	"testing"

	. "sigs.k8s.io/kustomize/pkg/expansion"
)

type expected struct {
	count  int
	edited string
}

func TestMapReference(t *testing.T) {
	type env struct {
		Name  string
		Value interface{}
	}
	envs := []env{
		{
			Name:  "FOO",
			Value: "bar",
		},
		{
			Name:  "ZOO",
			Value: "$(FOO)-1",
		},
		{
			Name:  "BLU",
			Value: "$(ZOO)-2",
		},
		{
			Name:  "INT",
			Value: 2,
		},
		{
			Name:  "ZINT",
			Value: "$(INT)",
		},
		{
			Name:  "BOOL",
			Value: true,
		},
		{
			Name:  "ZBOOL",
			Value: "$(BOOL)",
		},
	}

	declaredEnv := map[string]interface{}{
		"FOO":   "bar",
		"ZOO":   "$(FOO)-1",
		"BLU":   "$(ZOO)-2",
		"INT":   "2",
		"ZINT":  "$(INT)",
		"BOOL":  "true",
		"ZBOOL": "$(BOOL)",
	}

	counts := make(map[string]int)
	mapping := MappingFuncFor(counts, declaredEnv)

	for _, env := range envs {
		declaredEnv[env.Name] = Expand(fmt.Sprintf("%v", env.Value), mapping)
	}

	expectedEnv := map[string]expected{
		"FOO":   {count: 1, edited: "bar"},
		"ZOO":   {count: 1, edited: "bar-1"},
		"BLU":   {count: 0, edited: "bar-1-2"},
		"INT":   {count: 1, edited: "2"},
		"ZINT":  {count: 0, edited: "2"},
		"BOOL":  {count: 1, edited: "true"},
		"ZBOOL": {count: 0, edited: "true"},
	}

	for k, v := range expectedEnv {
		if e, a := v, declaredEnv[k]; e.edited != a || e.count != counts[k] {
			t.Errorf("Expected %v count=%d, got %v count=%d",
				e.edited, e.count, a, counts[k])
		} else {
			delete(declaredEnv, k)
		}
	}

	if len(declaredEnv) != 0 {
		t.Errorf("Unexpected keys in declared env: %v", declaredEnv)
	}
}

func TestMapping(t *testing.T) {
	context := map[string]interface{}{
		"VAR_A":     "A",
		"VAR_B":     "B",
		"VAR_C":     "C",
		"VAR_REF":   "$(VAR_A)",
		"VAR_EMPTY": "",
	}
	doExpansionTest(t, context)
}

func TestMappingDual(t *testing.T) {
	context := map[string]interface{}{
		"VAR_A":     "A",
		"VAR_EMPTY": "",
	}
	context2 := map[string]interface{}{
		"VAR_B":   "B",
		"VAR_C":   "C",
		"VAR_REF": "$(VAR_A)",
	}

	doExpansionTest(t, context, context2)
}

func doExpansionTest(t *testing.T, context ...map[string]interface{}) {
	cases := []struct {
		name     string
		input    string
		expected string
		counts   map[string]int
	}{
		{
			name:     "whole string",
			input:    "$(VAR_A)",
			expected: "A",
			counts:   map[string]int{"VAR_A": 1},
		},
		{
			name:     "repeat",
			input:    "$(VAR_A)-$(VAR_A)",
			expected: "A-A",
			counts:   map[string]int{"VAR_A": 2},
		},
		{
			name:     "multiple repeats",
			input:    "$(VAR_A)-$(VAR_B)-$(VAR_B)-$(VAR_B)-$(VAR_A)",
			expected: "A-B-B-B-A",
			counts:   map[string]int{"VAR_A": 2, "VAR_B": 3},
		},
		{
			name:     "beginning",
			input:    "$(VAR_A)-1",
			expected: "A-1",
			counts:   map[string]int{"VAR_A": 1},
		},
		{
			name:     "middle",
			input:    "___$(VAR_B)___",
			expected: "___B___",
			counts:   map[string]int{"VAR_B": 1},
		},
		{
			name:     "end",
			input:    "___$(VAR_C)",
			expected: "___C",
			counts:   map[string]int{"VAR_C": 1},
		},
		{
			name:     "compound",
			input:    "$(VAR_A)_$(VAR_B)_$(VAR_C)",
			expected: "A_B_C",
			counts:   map[string]int{"VAR_A": 1, "VAR_B": 1, "VAR_C": 1},
		},
		{
			name:     "escape & expand",
			input:    "$$(VAR_B)_$(VAR_A)",
			expected: "$(VAR_B)_A",
			counts:   map[string]int{"VAR_A": 1},
		},
		{
			name:     "compound escape",
			input:    "$$(VAR_A)_$$(VAR_B)",
			expected: "$(VAR_A)_$(VAR_B)",
		},
		{
			name:     "mixed in escapes",
			input:    "f000-$$VAR_A",
			expected: "f000-$VAR_A",
		},
		{
			name:     "backslash escape ignored",
			input:    "foo\\$(VAR_C)bar",
			expected: "foo\\Cbar",
			counts:   map[string]int{"VAR_C": 1},
		},
		{
			name:     "backslash escape ignored",
			input:    "foo\\\\$(VAR_C)bar",
			expected: "foo\\\\Cbar",
			counts:   map[string]int{"VAR_C": 1},
		},
		{
			name:     "lots of backslashes",
			input:    "foo\\\\\\\\$(VAR_A)bar",
			expected: "foo\\\\\\\\Abar",
			counts:   map[string]int{"VAR_A": 1},
		},
		{
			name:     "nested var references",
			input:    "$(VAR_A$(VAR_B))",
			expected: "$(VAR_A$(VAR_B))",
		},
		{
			name:     "nested var references second type",
			input:    "$(VAR_A$(VAR_B)",
			expected: "$(VAR_A$(VAR_B)",
		},
		{
			name:     "value is a reference",
			input:    "$(VAR_REF)",
			expected: "$(VAR_A)",
			counts:   map[string]int{"VAR_REF": 1},
		},
		{
			name:     "value is a reference x 2",
			input:    "%%$(VAR_REF)--$(VAR_REF)%%",
			expected: "%%$(VAR_A)--$(VAR_A)%%",
			counts:   map[string]int{"VAR_REF": 2},
		},
		{
			name:     "empty var",
			input:    "foo$(VAR_EMPTY)bar",
			expected: "foobar",
			counts:   map[string]int{"VAR_EMPTY": 1},
		},
		{
			name:     "unterminated expression",
			input:    "foo$(VAR_Awhoops!",
			expected: "foo$(VAR_Awhoops!",
		},
		{
			name:     "expression without operator",
			input:    "f00__(VAR_A)__",
			expected: "f00__(VAR_A)__",
		},
		{
			name:     "shell special vars pass through",
			input:    "$?_boo_$!",
			expected: "$?_boo_$!",
		},
		{
			name:     "bare operators are ignored",
			input:    "$VAR_A",
			expected: "$VAR_A",
		},
		{
			name:     "undefined vars are passed through",
			input:    "$(VAR_DNE)",
			expected: "$(VAR_DNE)",
		},
		{
			name:     "multiple (even) operators, var undefined",
			input:    "$$$$$$(BIG_MONEY)",
			expected: "$$$(BIG_MONEY)",
		},
		{
			name:     "multiple (even) operators, var defined",
			input:    "$$$$$$(VAR_A)",
			expected: "$$$(VAR_A)",
		},
		{
			name:     "multiple (odd) operators, var undefined",
			input:    "$$$$$$$(GOOD_ODDS)",
			expected: "$$$$(GOOD_ODDS)",
		},
		{
			name:     "multiple (odd) operators, var defined",
			input:    "$$$$$$$(VAR_A)",
			expected: "$$$A",
			counts:   map[string]int{"VAR_A": 1},
		},
		{
			name:     "missing open expression",
			input:    "$VAR_A)",
			expected: "$VAR_A)",
		},
		{
			name:     "shell syntax ignored",
			input:    "${VAR_A}",
			expected: "${VAR_A}",
		},
		{
			name:     "trailing incomplete expression not consumed",
			input:    "$(VAR_B)_______$(A",
			expected: "B_______$(A",
			counts:   map[string]int{"VAR_B": 1},
		},
		{
			name:     "trailing incomplete expression, no content, is not consumed",
			input:    "$(VAR_C)_______$(",
			expected: "C_______$(",
			counts:   map[string]int{"VAR_C": 1},
		},
		{
			name:     "operator at end of input string is preserved",
			input:    "$(VAR_A)foobarzab$",
			expected: "Afoobarzab$",
			counts:   map[string]int{"VAR_A": 1},
		},
		{
			name:     "shell escaped incomplete expr",
			input:    "foo-\\$(VAR_A",
			expected: "foo-\\$(VAR_A",
		},
		{
			name:     "lots of $( in middle",
			input:    "--$($($($($--",
			expected: "--$($($($($--",
		},
		{
			name:     "lots of $( in beginning",
			input:    "$($($($($--foo$(",
			expected: "$($($($($--foo$(",
		},
		{
			name:     "lots of $( at end",
			input:    "foo0--$($($($(",
			expected: "foo0--$($($($(",
		},
		{
			name:     "escaped operators in variable names are not escaped",
			input:    "$(foo$$var)",
			expected: "$(foo$$var)",
		},
		{
			name:     "newline not expanded",
			input:    "\n",
			expected: "\n",
		},
	}

	for _, tc := range cases {
		counts := make(map[string]int)
		mapping := MappingFuncFor(counts, context...)
		expanded := Expand(fmt.Sprintf("%v", tc.input), mapping)
		if e, a := tc.expected, expanded; e != a {
			t.Errorf("%v: expected %q, got %q", tc.name, e, a)
		}
		if len(counts) != len(tc.counts) {
			t.Errorf("%v: len(counts)=%d != len(tc.counts)=%d",
				tc.name, len(counts), len(tc.counts))
		}
		if len(tc.counts) > 0 {
			for k, expectedCount := range tc.counts {
				c, ok := counts[k]
				if ok {
					if c != expectedCount {
						t.Errorf(
							"%v: k=%s, expected count %d, got %d",
							tc.name, k, expectedCount, c)
					}
				} else {
					t.Errorf(
						"%v: k=%s, expected count %d, got zero",
						tc.name, k, expectedCount)
				}
			}
		}
	}
}

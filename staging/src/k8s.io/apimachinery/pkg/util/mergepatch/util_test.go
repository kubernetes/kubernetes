/*
Copyright 2017 The Kubernetes Authors.

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

package mergepatch

import (
	"fmt"
	"testing"
)

func TestHasConflicts(t *testing.T) {
	testCases := []struct {
		A   interface{}
		B   interface{}
		Ret bool
	}{
		{A: "hello", B: "hello", Ret: false},
		{A: "hello", B: "hell", Ret: true},
		{A: "hello", B: nil, Ret: true},
		{A: "hello", B: int64(1), Ret: true},
		{A: "hello", B: float64(1.0), Ret: true},
		{A: "hello", B: false, Ret: true},
		{A: int64(1), B: int64(1), Ret: false},
		{A: nil, B: nil, Ret: false},
		{A: false, B: false, Ret: false},
		{A: float64(3), B: float64(3), Ret: false},

		{A: "hello", B: []interface{}{}, Ret: true},
		{A: []interface{}{int64(1)}, B: []interface{}{}, Ret: true},
		{A: []interface{}{}, B: []interface{}{}, Ret: false},
		{A: []interface{}{int64(1)}, B: []interface{}{int64(1)}, Ret: false},
		{A: map[string]interface{}{}, B: []interface{}{int64(1)}, Ret: true},

		{A: map[string]interface{}{}, B: map[string]interface{}{"a": int64(1)}, Ret: false},
		{A: map[string]interface{}{"a": int64(1)}, B: map[string]interface{}{"a": int64(1)}, Ret: false},
		{A: map[string]interface{}{"a": int64(1)}, B: map[string]interface{}{"a": int64(2)}, Ret: true},
		{A: map[string]interface{}{"a": int64(1)}, B: map[string]interface{}{"b": int64(2)}, Ret: false},

		{
			A:   map[string]interface{}{"a": []interface{}{int64(1)}},
			B:   map[string]interface{}{"a": []interface{}{int64(1)}},
			Ret: false,
		},
		{
			A:   map[string]interface{}{"a": []interface{}{int64(1)}},
			B:   map[string]interface{}{"a": []interface{}{}},
			Ret: true,
		},
		{
			A:   map[string]interface{}{"a": []interface{}{int64(1)}},
			B:   map[string]interface{}{"a": int64(1)},
			Ret: true,
		},

		// Maps and lists with multiple entries.
		{
			A:   map[string]interface{}{"a": int64(1), "b": int64(2)},
			B:   map[string]interface{}{"a": int64(1), "b": int64(0)},
			Ret: true,
		},
		{
			A:   map[string]interface{}{"a": int64(1), "b": int64(2)},
			B:   map[string]interface{}{"a": int64(1), "b": int64(2)},
			Ret: false,
		},
		{
			A:   map[string]interface{}{"a": int64(1), "b": int64(2)},
			B:   map[string]interface{}{"a": int64(1), "b": int64(0), "c": int64(3)},
			Ret: true,
		},
		{
			A:   map[string]interface{}{"a": int64(1), "b": int64(2)},
			B:   map[string]interface{}{"a": int64(1), "b": int64(2), "c": int64(3)},
			Ret: false,
		},
		{
			A:   map[string]interface{}{"a": []interface{}{int64(1), int64(2)}},
			B:   map[string]interface{}{"a": []interface{}{int64(1), int64(0)}},
			Ret: true,
		},
		{
			A:   map[string]interface{}{"a": []interface{}{int64(1), int64(2)}},
			B:   map[string]interface{}{"a": []interface{}{int64(1), int64(2)}},
			Ret: false,
		},

		// Numeric types are not interchangeable.
		// Callers are expected to ensure numeric types are consistent in 'left' and 'right'.
		{A: int64(0), B: float64(0), Ret: true},
		// Other types are not interchangeable.
		{A: int64(0), B: "0", Ret: true},
		{A: int64(0), B: nil, Ret: true},
		{A: int64(0), B: false, Ret: true},
		{A: "true", B: true, Ret: true},
		{A: "null", B: nil, Ret: true},
	}

	for _, testCase := range testCases {
		testStr := fmt.Sprintf("A = %#v, B = %#v", testCase.A, testCase.B)
		// Run each test case multiple times if it passes because HasConflicts()
		// uses map iteration, which returns keys in nondeterministic order.
		for try := 0; try < 10; try++ {
			out, err := HasConflicts(testCase.A, testCase.B)
			if err != nil {
				t.Errorf("%v: unexpected error: %v", testStr, err)
				break
			}
			if out != testCase.Ret {
				t.Errorf("%v: expected %t got %t", testStr, testCase.Ret, out)
				break
			}
			out, err = HasConflicts(testCase.B, testCase.A)
			if err != nil {
				t.Errorf("%v: unexpected error: %v", testStr, err)
				break
			}
			if out != testCase.Ret {
				t.Errorf("%v: expected reversed %t got %t", testStr, testCase.Ret, out)
				break
			}
		}
	}
}

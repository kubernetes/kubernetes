/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package jsonmerge

import (
	"testing"
)

func TestHasConflicts(t *testing.T) {
	testCases := []struct {
		A   interface{}
		B   interface{}
		Ret bool
	}{
		{A: "hello", B: "hello", Ret: false}, // 0
		{A: "hello", B: "hell", Ret: true},
		{A: "hello", B: nil, Ret: true},
		{A: "hello", B: 1, Ret: true},
		{A: "hello", B: float64(1.0), Ret: true},
		{A: "hello", B: false, Ret: true},

		{A: "hello", B: []interface{}{}, Ret: true}, // 6
		{A: []interface{}{1}, B: []interface{}{}, Ret: true},
		{A: []interface{}{}, B: []interface{}{}, Ret: false},
		{A: []interface{}{1}, B: []interface{}{1}, Ret: false},
		{A: map[string]interface{}{}, B: []interface{}{1}, Ret: true},

		{A: map[string]interface{}{}, B: map[string]interface{}{"a": 1}, Ret: false}, // 11
		{A: map[string]interface{}{"a": 1}, B: map[string]interface{}{"a": 1}, Ret: false},
		{A: map[string]interface{}{"a": 1}, B: map[string]interface{}{"a": 2}, Ret: true},
		{A: map[string]interface{}{"a": 1}, B: map[string]interface{}{"b": 2}, Ret: false},

		{ // 15
			A:   map[string]interface{}{"a": []interface{}{1}},
			B:   map[string]interface{}{"a": []interface{}{1}},
			Ret: false,
		},
		{
			A:   map[string]interface{}{"a": []interface{}{1}},
			B:   map[string]interface{}{"a": []interface{}{}},
			Ret: true,
		},
		{
			A:   map[string]interface{}{"a": []interface{}{1}},
			B:   map[string]interface{}{"a": 1},
			Ret: true,
		},
	}

	for i, testCase := range testCases {
		out := hasConflicts(testCase.A, testCase.B)
		if out != testCase.Ret {
			t.Errorf("%d: expected %t got %t", i, testCase.Ret, out)
			continue
		}
		out = hasConflicts(testCase.B, testCase.A)
		if out != testCase.Ret {
			t.Errorf("%d: expected reversed %t got %t", i, testCase.Ret, out)
		}
	}
}

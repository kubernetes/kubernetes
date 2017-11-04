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

package pointer

import (
	"testing"
)

func TestAllPtrFieldsNil(t *testing.T) {
	testCases := []struct {
		obj      interface{}
		expected bool
	}{
		{struct{}{}, true},
		{struct{ Foo int }{12345}, true},
		{&struct{ Foo int }{12345}, true},
		{struct{ Foo *int }{nil}, true},
		{&struct{ Foo *int }{nil}, true},
		{struct {
			Foo int
			Bar *int
		}{12345, nil}, true},
		{&struct {
			Foo int
			Bar *int
		}{12345, nil}, true},
		{struct {
			Foo *int
			Bar *int
		}{nil, nil}, true},
		{&struct {
			Foo *int
			Bar *int
		}{nil, nil}, true},
		{struct{ Foo *int }{new(int)}, false},
		{&struct{ Foo *int }{new(int)}, false},
		{struct {
			Foo *int
			Bar *int
		}{nil, new(int)}, false},
		{&struct {
			Foo *int
			Bar *int
		}{nil, new(int)}, false},
		{(*struct{})(nil), true},
	}
	for i, tc := range testCases {
		if AllPtrFieldsNil(tc.obj) != tc.expected {
			t.Errorf("case[%d]: expected %t, got %t", i, tc.expected, !tc.expected)
		}
	}
}

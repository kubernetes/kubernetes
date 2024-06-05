/*
Copyright 2015 The Kubernetes Authors.

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

package slice

import (
	"reflect"
	"slices"
	"testing"
)

type testType[T comparable] struct {
	typeName       string
	baseSlice      []T
	notPresentItem T
	presentItem    T
}

type testCase[T comparable] struct {
	testName string
	input    []T
	remove   T
	want     []T
}

func TestRemove(t *testing.T) {
	t.Run("Strings", func(t *testing.T) {
		SuiteRemoveT(t, testType[string]{
			typeName:       "string",
			baseSlice:      []string{"a", "b", "c", "d"},
			notPresentItem: "notPresent",
			presentItem:    "b",
		})
	})

	t.Run("Ints", func(t *testing.T) {
		SuiteRemoveT(t, testType[int]{
			typeName:       "int",
			baseSlice:      []int{1, 2, 3, 4},
			notPresentItem: 5,
			presentItem:    2,
		})
	})

	t.Run("Floats", func(t *testing.T) {
		SuiteRemoveT(t, testType[float64]{
			typeName:       "float64",
			baseSlice:      []float64{1.1, 2.2, 3.3, 4.4},
			notPresentItem: 5.5,
			presentItem:    2.2,
		})
	})

	t.Run("Structs", func(t *testing.T) {
		type testStruct struct {
			A int
			B string
		}

		SuiteRemoveT(t, testType[testStruct]{
			typeName:       "testStruct",
			baseSlice:      []testStruct{{1, "a"}, {2, "b"}, {3, "c"}, {4, "d"}},
			notPresentItem: testStruct{5, "e"},
			presentItem:    testStruct{2, "b"},
		})
	})
}

func SuiteRemoveT[T comparable](t *testing.T, toTest testType[T]) {
	tests := []testCase[T]{
		{
			testName: "Nil input slice with item that is known to not be present",
			input:    nil,
			remove:   toTest.notPresentItem,
			want:     nil,
		},
		{
			testName: "Nil input slice with item that is known to be present",
			input:    nil,
			remove:   toTest.presentItem,
			want:     nil,
		},
		{
			testName: "Nothing changed when using an item that is known to be not present",
			input:    slices.Clone(toTest.baseSlice),
			remove:   toTest.notPresentItem,
			want:     slices.Clone(toTest.baseSlice),
		},
		{
			testName: "Removing only the matching items",
			input:    slices.Clone(toTest.baseSlice),
			remove:   toTest.presentItem,
			want: slices.DeleteFunc(slices.Clone(toTest.baseSlice), func(item T) bool {
				return item == toTest.presentItem
			}),
		},
		{
			testName: "All items remove",
			input:    []T{toTest.presentItem, toTest.presentItem, toTest.presentItem},
			remove:   toTest.presentItem,
			want:     nil,
		},
	}

	for _, tt := range tests {
		if got := Remove[[]T, T](tt.input, tt.remove); !reflect.DeepEqual(got, tt.want) {
			t.Errorf("%v: Remove[%s](%v, %v) = %v WANT %v", tt.testName, toTest.typeName, tt.input, tt.remove, got, tt.want)
		}
	}
}

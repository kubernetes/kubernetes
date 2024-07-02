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

package finalizers

import (
	"reflect"
	"testing"
)

func TestRemove(t *testing.T) {
	tests := []struct {
		testName string
		input    []string
		remove   string
		want     []string
	}{
		{
			testName: "Nil input slice",
			input:    nil,
			remove:   "",
			want:     nil,
		},
		{
			testName: "Slice doesn't contain the string",
			input:    []string{"a", "ab", "cdef"},
			remove:   "NotPresentInSlice",
			want:     []string{"a", "ab", "cdef"},
		},
		{
			testName: "All strings removed, result is nil",
			input:    []string{"a"},
			remove:   "a",
			want:     nil,
		},
		{
			testName: "No modifier func, one string removed",
			input:    []string{"a", "ab", "cdef"},
			remove:   "ab",
			want:     []string{"a", "cdef"},
		},
		{
			testName: "No modifier func, all(three) strings removed",
			input:    []string{"ab", "a", "ab", "cdef", "ab"},
			remove:   "ab",
			want:     []string{"a", "cdef"},
		},
		{
			testName: "Removed both the string and the modifier func result",
			input:    []string{"a", "cd", "ab", "ee"},
			remove:   "ee",
			want:     []string{"a", "cd", "ab"},
		},
	}
	for _, tt := range tests {
		if got := Remove(tt.input, tt.remove); !reflect.DeepEqual(got, tt.want) {
			t.Errorf("%v: Remove(%v, %q) = %v WANT %v", tt.testName, tt.input, tt.remove, got, tt.want)
		}
	}
}

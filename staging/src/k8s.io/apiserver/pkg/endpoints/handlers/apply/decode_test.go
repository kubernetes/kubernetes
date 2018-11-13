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

package apply

import (
	"testing"
)

// TestValidateOneOf tests that validateOneOf returns true if <= 1
// non-nil fields are passed to it are and false otherwise
func TestValidateOneOf(t *testing.T) {
	var nilValue *string
	nonNilValue := strPtr("test")

	tests := []struct {
		fields   []interface{}
		expected bool
	}{
		{
			fields:   []interface{}{},
			expected: true,
		}, {
			fields:   []interface{}{nilValue},
			expected: true,
		}, {
			fields:   []interface{}{nonNilValue},
			expected: true,
		}, {
			fields:   []interface{}{nilValue, nonNilValue, nilValue},
			expected: true,
		}, {
			fields:   []interface{}{nonNilValue, nonNilValue},
			expected: false,
		}, {
			fields:   []interface{}{nilValue, nonNilValue, nonNilValue, nilValue},
			expected: false,
		},
	}

	for i, tc := range tests {
		actual := validateOneOf(tc.fields...)
		if actual != tc.expected {
			t.Errorf("[%v]expected validateOneOf(%v) to return:\n\t%+v\nbut got:\n\t%+v", i, tc.fields, tc.expected, actual)
		}
	}
}

func strPtr(s string) *string {
	return &s
}

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

package util

import (
	"reflect"
	"testing"
)

func TestMapValueFlag(t *testing.T) {
	for i, tc := range []struct {
		description string
		passed      []string
		expected    map[string]string
		expectedErr bool
	}{
		{
			description: "wont allow duplicates",
			passed:      []string{"key1=val1", "key1=val1"},
			expected:    map[string]string{"key1": "val1"},
		},
		{
			description: "set multiple key values pairs",
			passed:      []string{"key1=val1", "key2=val2"},
			expected:    map[string]string{"key1": "val1", "key2": "val2"},
		},
		{
			description: "trim empty spaces",
			passed:      []string{" key1 = val1 "},
			expected:    map[string]string{"key1": "val1"},
		},
		{
			description: "error on absense of separator (=)",
			passed:      []string{"key1"},
			expectedErr: true,
		},
		{
			description: "error on multiple separators (=)",
			passed:      []string{"key1=val1,key2=val2"},
			expectedErr: true,
		},
		{
			description: "leave empty if nothing is passed",
			expected:    map[string]string{},
		},
	} {
		t.Run(string(i), func(t *testing.T) {
			t.Log("Running test ", tc.description)
			mv := &MapValue{}
			for _, i := range tc.passed {
				if err := mv.Set(i); err == nil && tc.expectedErr {
					t.Errorf("Unexpected success for value %s", i)
				} else if err != nil && !tc.expectedErr {
					t.Errorf("Unexpected error for value %s: %v", i, err)
				}
			}
			if !tc.expectedErr && !reflect.DeepEqual(map[string]string(*mv), tc.expected) {
				t.Errorf("%+v != %+v", *mv, tc.expected)
			}
		})
	}
}

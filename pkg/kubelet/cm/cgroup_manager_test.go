//go:build linux
// +build linux

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

package cm

import (
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
)

func Test(t *testing.T) {
	tests := []struct {
		input    map[string]string
		expected *map[v1.ResourceName]int64
	}{
		{
			input:    map[string]string{"memory": ""},
			expected: nil,
		},
		{
			input:    map[string]string{"memory": "a"},
			expected: nil,
		},
		{
			input:    map[string]string{"memory": "a%"},
			expected: nil,
		},
		{
			input:    map[string]string{"memory": "200%"},
			expected: nil,
		},
		{
			input: map[string]string{"memory": "0%"},
			expected: &map[v1.ResourceName]int64{
				v1.ResourceMemory: 0,
			},
		},
		{
			input: map[string]string{"memory": "100%"},
			expected: &map[v1.ResourceName]int64{
				v1.ResourceMemory: 100,
			},
		},
		{
			// need to change this when CPU is added as a supported resource
			input:    map[string]string{"memory": "100%", "cpu": "50%"},
			expected: nil,
		},
	}
	for _, test := range tests {
		actual, err := ParseQOSReserved(test.input)
		if actual != nil && test.expected == nil {
			t.Errorf("Unexpected success, input: %v, expected: %v, actual: %v, err: %v", test.input, test.expected, actual, err)
		}
		if actual == nil && test.expected != nil {
			t.Errorf("Unexpected failure, input: %v, expected: %v, actual: %v, err: %v", test.input, test.expected, actual, err)
		}
		if (actual == nil && test.expected == nil) || reflect.DeepEqual(*actual, *test.expected) {
			continue
		}
		t.Errorf("Unexpected result, input: %v, expected: %v, actual: %v, err: %v", test.input, test.expected, actual, err)
	}
}

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

	"k8s.io/kubernetes/pkg/api"
)

func Test(t *testing.T) {
	testCases := []struct {
		input    string
		expected []QOSReserveLimit
	}{
		{
			input:    "",
			expected: nil,
		},
		{
			input:    "a",
			expected: nil,
		},
		{
			input:    "memory",
			expected: nil,
		},
		{
			input:    "memory=a",
			expected: nil,
		},
		{
			input:    "memory=a%",
			expected: nil,
		},
		{
			input:    "memory=0x1%",
			expected: nil,
		},
		{
			input:    "memory=-1%",
			expected: nil,
		},
		{
			input:    "memory=101%",
			expected: nil,
		},
		{
			input: "memory=100%",
			expected: []QOSReserveLimit{
				{
					resource:       string(api.ResourceMemory),
					reservePercent: 100,
				},
			},
		},
		{
			input: "memory=0%",
			expected: []QOSReserveLimit{
				{
					resource:       string(api.ResourceMemory),
					reservePercent: 0,
				},
			},
		},
		{
			input:    "memory=100%,cpu",
			expected: nil,
		},
		{
			input:    "memory=100%,memory=50%",
			expected: nil,
		},
	}
	for _, testCase := range testCases {
		if actual, err := ParseQOSReserveLimits(testCase.input); !reflect.DeepEqual(actual, testCase.expected) {
			t.Errorf("Unexpected result, input: %v, expected: %v, actual: %v, err: %v", testCase.input, testCase.expected, actual, err)
		}
	}
}

/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package quota

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
)

func TestEquals(t *testing.T) {
	testCases := map[string]struct {
		a        resource.List
		b        resource.List
		expected bool
	}{
		"isEqual": {
			a:        resource.List{},
			b:        resource.List{},
			expected: true,
		},
		"isEqualWithKeys": {
			a: resource.List{
				api.ResourceCPU:    resource.MustParse("100m"),
				api.ResourceMemory: resource.MustParse("1Gi"),
			},
			b: resource.List{
				api.ResourceCPU:    resource.MustParse("100m"),
				api.ResourceMemory: resource.MustParse("1Gi"),
			},
			expected: true,
		},
		"isNotEqualSameKeys": {
			a: resource.List{
				api.ResourceCPU:    resource.MustParse("200m"),
				api.ResourceMemory: resource.MustParse("1Gi"),
			},
			b: resource.List{
				api.ResourceCPU:    resource.MustParse("100m"),
				api.ResourceMemory: resource.MustParse("1Gi"),
			},
			expected: false,
		},
		"isNotEqualDiffKeys": {
			a: resource.List{
				api.ResourceCPU:    resource.MustParse("100m"),
				api.ResourceMemory: resource.MustParse("1Gi"),
			},
			b: resource.List{
				api.ResourceCPU:    resource.MustParse("100m"),
				api.ResourceMemory: resource.MustParse("1Gi"),
				api.ResourcePods:   resource.MustParse("1"),
			},
			expected: false,
		},
	}
	for testName, testCase := range testCases {
		if result := Equals(testCase.a, testCase.b); result != testCase.expected {
			t.Errorf("%s expected: %v, actual: %v, a=%v, b=%v", testName, testCase.expected, result, testCase.a, testCase.b)
		}
	}
}

func TestAdd(t *testing.T) {
	testCases := map[string]struct {
		a        resource.List
		b        resource.List
		expected resource.List
	}{
		"noKeys": {
			a:        resource.List{},
			b:        resource.List{},
			expected: resource.List{},
		},
		"toEmpty": {
			a:        resource.List{api.ResourceCPU: resource.MustParse("100m")},
			b:        resource.List{},
			expected: resource.List{api.ResourceCPU: resource.MustParse("100m")},
		},
		"matching": {
			a:        resource.List{api.ResourceCPU: resource.MustParse("100m")},
			b:        resource.List{api.ResourceCPU: resource.MustParse("100m")},
			expected: resource.List{api.ResourceCPU: resource.MustParse("200m")},
		},
	}
	for testName, testCase := range testCases {
		sum := Add(testCase.a, testCase.b)
		if result := Equals(testCase.expected, sum); !result {
			t.Errorf("%s expected: %v, actual: %v", testName, testCase.expected, sum)
		}
	}
}

func TestSubtract(t *testing.T) {
	testCases := map[string]struct {
		a        resource.List
		b        resource.List
		expected resource.List
	}{
		"noKeys": {
			a:        resource.List{},
			b:        resource.List{},
			expected: resource.List{},
		},
		"value-empty": {
			a:        resource.List{api.ResourceCPU: resource.MustParse("100m")},
			b:        resource.List{},
			expected: resource.List{api.ResourceCPU: resource.MustParse("100m")},
		},
		"empty-value": {
			a:        resource.List{},
			b:        resource.List{api.ResourceCPU: resource.MustParse("100m")},
			expected: resource.List{api.ResourceCPU: resource.MustParse("-100m")},
		},
		"value-value": {
			a:        resource.List{api.ResourceCPU: resource.MustParse("200m")},
			b:        resource.List{api.ResourceCPU: resource.MustParse("100m")},
			expected: resource.List{api.ResourceCPU: resource.MustParse("100m")},
		},
	}
	for testName, testCase := range testCases {
		sub := Subtract(testCase.a, testCase.b)
		if result := Equals(testCase.expected, sub); !result {
			t.Errorf("%s expected: %v, actual: %v", testName, testCase.expected, sub)
		}
	}
}

func TestResourceNames(t *testing.T) {
	testCases := map[string]struct {
		a        resource.List
		expected []resource.Name
	}{
		"empty": {
			a:        resource.List{},
			expected: []resource.Name{},
		},
		"values": {
			a: resource.List{
				api.ResourceCPU:    resource.MustParse("100m"),
				api.ResourceMemory: resource.MustParse("1Gi"),
			},
			expected: []resource.Name{api.ResourceMemory, api.ResourceCPU},
		},
	}
	for testName, testCase := range testCases {
		actualSet := ToSet(ResourceNames(testCase.a))
		expectedSet := ToSet(testCase.expected)
		if !actualSet.Equal(expectedSet) {
			t.Errorf("%s expected: %v, actual: %v", testName, expectedSet, actualSet)
		}
	}
}

func TestContains(t *testing.T) {
	testCases := map[string]struct {
		a        []resource.Name
		b        resource.Name
		expected bool
	}{
		"does-not-contain": {
			a:        []resource.Name{api.ResourceMemory},
			b:        api.ResourceCPU,
			expected: false,
		},
		"does-contain": {
			a:        []resource.Name{api.ResourceMemory, api.ResourceCPU},
			b:        api.ResourceCPU,
			expected: true,
		},
	}
	for testName, testCase := range testCases {
		if actual := Contains(testCase.a, testCase.b); actual != testCase.expected {
			t.Errorf("%s expected: %v, actual: %v", testName, testCase.expected, actual)
		}
	}
}

func TestIsZero(t *testing.T) {
	testCases := map[string]struct {
		a        resource.List
		expected bool
	}{
		"empty": {
			a:        resource.List{},
			expected: true,
		},
		"zero": {
			a: resource.List{
				api.ResourceCPU:    resource.MustParse("0"),
				api.ResourceMemory: resource.MustParse("0"),
			},
			expected: true,
		},
		"non-zero": {
			a: resource.List{
				api.ResourceCPU:    resource.MustParse("200m"),
				api.ResourceMemory: resource.MustParse("1Gi"),
			},
			expected: false,
		},
	}
	for testName, testCase := range testCases {
		if result := IsZero(testCase.a); result != testCase.expected {
			t.Errorf("%s expected: %v, actual: %v", testName, testCase.expected, result)
		}
	}
}

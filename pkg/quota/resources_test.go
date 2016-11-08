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

package quota

import (
	"testing"

	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/v1"
)

func TestEquals(t *testing.T) {
	testCases := map[string]struct {
		a        v1.ResourceList
		b        v1.ResourceList
		expected bool
	}{
		"isEqual": {
			a:        v1.ResourceList{},
			b:        v1.ResourceList{},
			expected: true,
		},
		"isEqualWithKeys": {
			a: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("100m"),
				v1.ResourceMemory: resource.MustParse("1Gi"),
			},
			b: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("100m"),
				v1.ResourceMemory: resource.MustParse("1Gi"),
			},
			expected: true,
		},
		"isNotEqualSameKeys": {
			a: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("200m"),
				v1.ResourceMemory: resource.MustParse("1Gi"),
			},
			b: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("100m"),
				v1.ResourceMemory: resource.MustParse("1Gi"),
			},
			expected: false,
		},
		"isNotEqualDiffKeys": {
			a: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("100m"),
				v1.ResourceMemory: resource.MustParse("1Gi"),
			},
			b: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("100m"),
				v1.ResourceMemory: resource.MustParse("1Gi"),
				v1.ResourcePods:   resource.MustParse("1"),
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

func TestMax(t *testing.T) {
	testCases := map[string]struct {
		a        v1.ResourceList
		b        v1.ResourceList
		expected v1.ResourceList
	}{
		"noKeys": {
			a:        v1.ResourceList{},
			b:        v1.ResourceList{},
			expected: v1.ResourceList{},
		},
		"toEmpty": {
			a:        v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
			b:        v1.ResourceList{},
			expected: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
		},
		"matching": {
			a:        v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
			b:        v1.ResourceList{v1.ResourceCPU: resource.MustParse("150m")},
			expected: v1.ResourceList{v1.ResourceCPU: resource.MustParse("150m")},
		},
		"matching(reverse)": {
			a:        v1.ResourceList{v1.ResourceCPU: resource.MustParse("150m")},
			b:        v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
			expected: v1.ResourceList{v1.ResourceCPU: resource.MustParse("150m")},
		},
	}
	for testName, testCase := range testCases {
		sum := Max(testCase.a, testCase.b)
		if result := Equals(testCase.expected, sum); !result {
			t.Errorf("%s expected: %v, actual: %v", testName, testCase.expected, sum)
		}
	}
}

func TestAdd(t *testing.T) {
	testCases := map[string]struct {
		a        v1.ResourceList
		b        v1.ResourceList
		expected v1.ResourceList
	}{
		"noKeys": {
			a:        v1.ResourceList{},
			b:        v1.ResourceList{},
			expected: v1.ResourceList{},
		},
		"toEmpty": {
			a:        v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
			b:        v1.ResourceList{},
			expected: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
		},
		"matching": {
			a:        v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
			b:        v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
			expected: v1.ResourceList{v1.ResourceCPU: resource.MustParse("200m")},
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
		a        v1.ResourceList
		b        v1.ResourceList
		expected v1.ResourceList
	}{
		"noKeys": {
			a:        v1.ResourceList{},
			b:        v1.ResourceList{},
			expected: v1.ResourceList{},
		},
		"value-empty": {
			a:        v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
			b:        v1.ResourceList{},
			expected: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
		},
		"empty-value": {
			a:        v1.ResourceList{},
			b:        v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
			expected: v1.ResourceList{v1.ResourceCPU: resource.MustParse("-100m")},
		},
		"value-value": {
			a:        v1.ResourceList{v1.ResourceCPU: resource.MustParse("200m")},
			b:        v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
			expected: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
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
		a        v1.ResourceList
		expected []v1.ResourceName
	}{
		"empty": {
			a:        v1.ResourceList{},
			expected: []v1.ResourceName{},
		},
		"values": {
			a: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("100m"),
				v1.ResourceMemory: resource.MustParse("1Gi"),
			},
			expected: []v1.ResourceName{v1.ResourceMemory, v1.ResourceCPU},
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
		a        []v1.ResourceName
		b        v1.ResourceName
		expected bool
	}{
		"does-not-contain": {
			a:        []v1.ResourceName{v1.ResourceMemory},
			b:        v1.ResourceCPU,
			expected: false,
		},
		"does-contain": {
			a:        []v1.ResourceName{v1.ResourceMemory, v1.ResourceCPU},
			b:        v1.ResourceCPU,
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
		a        v1.ResourceList
		expected bool
	}{
		"empty": {
			a:        v1.ResourceList{},
			expected: true,
		},
		"zero": {
			a: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("0"),
				v1.ResourceMemory: resource.MustParse("0"),
			},
			expected: true,
		},
		"non-zero": {
			a: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("200m"),
				v1.ResourceMemory: resource.MustParse("1Gi"),
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

func TestIsNegative(t *testing.T) {
	testCases := map[string]struct {
		a        v1.ResourceList
		expected []v1.ResourceName
	}{
		"empty": {
			a:        v1.ResourceList{},
			expected: []v1.ResourceName{},
		},
		"some-negative": {
			a: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("-10"),
				v1.ResourceMemory: resource.MustParse("0"),
			},
			expected: []v1.ResourceName{v1.ResourceCPU},
		},
		"all-negative": {
			a: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("-200m"),
				v1.ResourceMemory: resource.MustParse("-1Gi"),
			},
			expected: []v1.ResourceName{v1.ResourceCPU, v1.ResourceMemory},
		},
	}
	for testName, testCase := range testCases {
		actual := IsNegative(testCase.a)
		actualSet := ToSet(actual)
		expectedSet := ToSet(testCase.expected)
		if !actualSet.Equal(expectedSet) {
			t.Errorf("%s expected: %v, actual: %v", testName, expectedSet, actualSet)
		}
	}
}

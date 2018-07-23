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

	"k8s.io/apimachinery/pkg/api/resource"
	api "k8s.io/kubernetes/pkg/apis/core"
	"reflect"
)

func TestEquals(t *testing.T) {
	testCases := map[string]struct {
		a        api.ResourceList
		b        api.ResourceList
		expected bool
	}{
		"isEqual": {
			a:        api.ResourceList{},
			b:        api.ResourceList{},
			expected: true,
		},
		"isEqualWithKeys": {
			a: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("100m"),
				api.ResourceMemory: resource.MustParse("1Gi"),
			},
			b: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("100m"),
				api.ResourceMemory: resource.MustParse("1Gi"),
			},
			expected: true,
		},
		"isNotEqualSameKeys": {
			a: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("200m"),
				api.ResourceMemory: resource.MustParse("1Gi"),
			},
			b: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("100m"),
				api.ResourceMemory: resource.MustParse("1Gi"),
			},
			expected: false,
		},
		"isNotEqualDiffKeys": {
			a: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("100m"),
				api.ResourceMemory: resource.MustParse("1Gi"),
			},
			b: api.ResourceList{
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

func TestLessThanOrEqual(t *testing.T) {
	testCases := map[string]struct {
		a            api.ResourceList
		b            api.ResourceList
		expectedBool bool
		expectedRes  []api.ResourceName
	}{
		"isEqualWithEmpty": {
			a:            api.ResourceList{},
			b:            api.ResourceList{},
			expectedBool: true,
			expectedRes:  []api.ResourceName{},
		},
		"isEqualWithKeys": {
			a: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("100m"),
				api.ResourceMemory: resource.MustParse("1Gi"),
			},
			b: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("100m"),
				api.ResourceMemory: resource.MustParse("1Gi"),
			},
			expectedBool: true,
			expectedRes:  []api.ResourceName{},
		},
		"isLess": {
			a: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("100m"),
				api.ResourceMemory: resource.MustParse("1Gi"),
			},
			b: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("200m"),
				api.ResourceMemory: resource.MustParse("1Gi"),
			},
			expectedBool: true,
			expectedRes:  []api.ResourceName{},
		},
		"isExceeded": {
			a: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("200m"),
				api.ResourceMemory: resource.MustParse("2Gi"),
			},
			b: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("100m"),
				api.ResourceMemory: resource.MustParse("1Gi"),
			},
			expectedBool: false,
			expectedRes:  []api.ResourceName{api.ResourceCPU, api.ResourceMemory},
		},
		"isExceededWithMoreKeys": {
			a: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("200m"),
				api.ResourceMemory: resource.MustParse("1Gi"),
			},
			b: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("100m"),
				api.ResourceMemory: resource.MustParse("1Gi"),
				api.ResourcePods:   resource.MustParse("1"),
			},
			expectedBool: false,
			expectedRes:  []api.ResourceName{api.ResourceCPU},
		},
	}
	for testName, testCase := range testCases {
		resultBool, resultRes := LessThanOrEqual(testCase.a, testCase.b)
		if resultBool != testCase.expectedBool {
			t.Errorf("%s expected: %v, actual: %v, a=%v, b=%v", testName, testCase.expectedBool, resultBool, testCase.a, testCase.b)
		}
		if !reflect.DeepEqual(resultRes, testCase.expectedRes) {
			t.Errorf("%s expected: %v, actual: %v, a=%v, b=%v", testName, testCase.expectedRes, resultRes, testCase.a, testCase.b)
		}
	}
}

func TestMax(t *testing.T) {
	testCases := map[string]struct {
		a        api.ResourceList
		b        api.ResourceList
		expected api.ResourceList
	}{
		"noKeys": {
			a:        api.ResourceList{},
			b:        api.ResourceList{},
			expected: api.ResourceList{},
		},
		"toEmpty": {
			a:        api.ResourceList{api.ResourceCPU: resource.MustParse("100m")},
			b:        api.ResourceList{},
			expected: api.ResourceList{api.ResourceCPU: resource.MustParse("100m")},
		},
		"matching": {
			a:        api.ResourceList{api.ResourceCPU: resource.MustParse("100m")},
			b:        api.ResourceList{api.ResourceCPU: resource.MustParse("150m")},
			expected: api.ResourceList{api.ResourceCPU: resource.MustParse("150m")},
		},
		"matching(reverse)": {
			a:        api.ResourceList{api.ResourceCPU: resource.MustParse("150m")},
			b:        api.ResourceList{api.ResourceCPU: resource.MustParse("100m")},
			expected: api.ResourceList{api.ResourceCPU: resource.MustParse("150m")},
		},
		"matching-equal": {
			a:        api.ResourceList{api.ResourceCPU: resource.MustParse("100m")},
			b:        api.ResourceList{api.ResourceCPU: resource.MustParse("100m")},
			expected: api.ResourceList{api.ResourceCPU: resource.MustParse("100m")},
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
		a        api.ResourceList
		b        api.ResourceList
		expected api.ResourceList
	}{
		"noKeys": {
			a:        api.ResourceList{},
			b:        api.ResourceList{},
			expected: api.ResourceList{},
		},
		"toEmpty": {
			a:        api.ResourceList{api.ResourceCPU: resource.MustParse("100m")},
			b:        api.ResourceList{},
			expected: api.ResourceList{api.ResourceCPU: resource.MustParse("100m")},
		},
		"matching": {
			a:        api.ResourceList{api.ResourceCPU: resource.MustParse("100m")},
			b:        api.ResourceList{api.ResourceCPU: resource.MustParse("100m")},
			expected: api.ResourceList{api.ResourceCPU: resource.MustParse("200m")},
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
		a        api.ResourceList
		b        api.ResourceList
		expected api.ResourceList
	}{
		"noKeys": {
			a:        api.ResourceList{},
			b:        api.ResourceList{},
			expected: api.ResourceList{},
		},
		"value-empty": {
			a:        api.ResourceList{api.ResourceCPU: resource.MustParse("100m")},
			b:        api.ResourceList{},
			expected: api.ResourceList{api.ResourceCPU: resource.MustParse("100m")},
		},
		"empty-value": {
			a:        api.ResourceList{},
			b:        api.ResourceList{api.ResourceCPU: resource.MustParse("100m")},
			expected: api.ResourceList{api.ResourceCPU: resource.MustParse("-100m")},
		},
		"value-value": {
			a:        api.ResourceList{api.ResourceCPU: resource.MustParse("200m")},
			b:        api.ResourceList{api.ResourceCPU: resource.MustParse("100m")},
			expected: api.ResourceList{api.ResourceCPU: resource.MustParse("100m")},
		},
	}
	for testName, testCase := range testCases {
		sub := Subtract(testCase.a, testCase.b)
		if result := Equals(testCase.expected, sub); !result {
			t.Errorf("%s expected: %v, actual: %v", testName, testCase.expected, sub)
		}
	}
}

func TestSubtractWithNonNegativeResult(t *testing.T) {
	zero := resource.MustParse("0")

	testCases := map[string]struct {
		a        api.ResourceList
		b        api.ResourceList
		expected api.ResourceList
	}{
		"noKeys": {
			a:        api.ResourceList{},
			b:        api.ResourceList{},
			expected: api.ResourceList{},
		},
		"value-value-positive": {
			a:        api.ResourceList{api.ResourceCPU: resource.MustParse("200m")},
			b:        api.ResourceList{api.ResourceCPU: resource.MustParse("100m")},
			expected: api.ResourceList{api.ResourceCPU: resource.MustParse("100m")},
		},
		"value-value-negative": {
			a:        api.ResourceList{api.ResourceCPU: resource.MustParse("100m")},
			b:        api.ResourceList{api.ResourceCPU: resource.MustParse("200m")},
			expected: api.ResourceList{api.ResourceCPU: zero},
		},
		"value-empty": {
			a:        api.ResourceList{api.ResourceCPU: resource.MustParse("100m")},
			b:        api.ResourceList{},
			expected: api.ResourceList{api.ResourceCPU: resource.MustParse("100m")},
		},
		"empty-value": {
			a:        api.ResourceList{},
			b:        api.ResourceList{api.ResourceCPU: resource.MustParse("100m")},
			expected: api.ResourceList{api.ResourceCPU: zero},
		},
		"partial-zero": {
			a:        api.ResourceList{api.ResourceCPU: resource.MustParse("200m")},
			b:        api.ResourceList{api.ResourceCPU: resource.MustParse("100m"), api.ResourceMemory: resource.MustParse("1Gi")},
			expected: api.ResourceList{api.ResourceCPU: resource.MustParse("100m"), api.ResourceMemory: zero},
		},
	}
	for testName, testCase := range testCases {
		sub := SubtractWithNonNegativeResult(testCase.a, testCase.b)
		if result := Equals(testCase.expected, sub); !result {
			t.Errorf("%s expected: %v, actual: %v", testName, testCase.expected, sub)
		}
	}
}

func TestMask(t *testing.T) {
	testCases := map[string]struct {
		inputResourceList api.ResourceList
		inputResourceName []api.ResourceName
		expected          api.ResourceList
	}{
		"empty-empty": {
			inputResourceList: api.ResourceList{},
			inputResourceName: []api.ResourceName{},
			expected:          api.ResourceList{},
		},
		"value-empty": {
			inputResourceList: api.ResourceList{api.ResourceCPU: resource.MustParse("200m")},
			inputResourceName: []api.ResourceName{},
			expected:          api.ResourceList{},
		},
		"empty-value": {
			inputResourceList: api.ResourceList{},
			inputResourceName: []api.ResourceName{api.ResourceCPU},
			expected:          api.ResourceList{},
		},
		"value-value": {
			inputResourceList: api.ResourceList{api.ResourceCPU: resource.MustParse("200m"), api.ResourceMemory: resource.MustParse("1Gi")},
			inputResourceName: []api.ResourceName{api.ResourceCPU},
			expected:          api.ResourceList{api.ResourceCPU: resource.MustParse("200m")},
		},
	}

	for testName, testCase := range testCases {
		sub := Mask(testCase.inputResourceList, testCase.inputResourceName)
		if result := Equals(testCase.expected, sub); !result {
			t.Errorf("%s expected: %v, actual: %v", testName, testCase.expected, sub)
		}
	}
}

func TestResourceNames(t *testing.T) {
	testCases := map[string]struct {
		a        api.ResourceList
		expected []api.ResourceName
	}{
		"empty": {
			a:        api.ResourceList{},
			expected: []api.ResourceName{},
		},
		"values": {
			a: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("100m"),
				api.ResourceMemory: resource.MustParse("1Gi"),
			},
			expected: []api.ResourceName{api.ResourceMemory, api.ResourceCPU},
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
		a        []api.ResourceName
		b        api.ResourceName
		expected bool
	}{
		"does-not-contain": {
			a:        []api.ResourceName{api.ResourceMemory},
			b:        api.ResourceCPU,
			expected: false,
		},
		"does-contain": {
			a:        []api.ResourceName{api.ResourceMemory, api.ResourceCPU},
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

func TestContainsPrefix(t *testing.T) {
	testCases := map[string]struct {
		a        []string
		b        api.ResourceName
		expected bool
	}{
		"does-not-contain": {
			a:        []string{api.ResourceHugePagesPrefix},
			b:        api.ResourceCPU,
			expected: false,
		},
		"does-contain": {
			a:        []string{api.ResourceHugePagesPrefix},
			b:        api.ResourceName(api.ResourceHugePagesPrefix + "2Mi"),
			expected: true,
		},
	}
	for testName, testCase := range testCases {
		if actual := ContainsPrefix(testCase.a, testCase.b); actual != testCase.expected {
			t.Errorf("%s expected: %v, actual: %v", testName, testCase.expected, actual)
		}
	}
}

func TestIntersection(t *testing.T) {
	testCases := map[string]struct {
		a        []api.ResourceName
		b        []api.ResourceName
		expected []api.ResourceName
	}{
		"empty-empty": {
			a:        []api.ResourceName{},
			b:        []api.ResourceName{},
			expected: []api.ResourceName{},
		},
		"empty-value": {
			a:        []api.ResourceName{},
			b:        []api.ResourceName{api.ResourceMemory},
			expected: []api.ResourceName{},
		},
		"value-value-no-intersection": {
			a:        []api.ResourceName{api.ResourceCPU},
			b:        []api.ResourceName{api.ResourceMemory},
			expected: []api.ResourceName{},
		},
		"value-value-with-intersection": {
			a:        []api.ResourceName{api.ResourceCPU},
			b:        []api.ResourceName{api.ResourceCPU, api.ResourceMemory},
			expected: []api.ResourceName{api.ResourceCPU},
		},
	}

	for testName, testCase := range testCases {
		if actual := Intersection(testCase.a, testCase.b); !reflect.DeepEqual(actual, testCase.expected) {
			t.Errorf("%s expected: %v, actual: %v", testName, testCase.expected, actual)
		}
	}
}

func TestIsZero(t *testing.T) {
	testCases := map[string]struct {
		a        api.ResourceList
		expected bool
	}{
		"empty": {
			a:        api.ResourceList{},
			expected: true,
		},
		"zero": {
			a: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("0"),
				api.ResourceMemory: resource.MustParse("0"),
			},
			expected: true,
		},
		"non-zero": {
			a: api.ResourceList{
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

func TestIsNegative(t *testing.T) {
	testCases := map[string]struct {
		a        api.ResourceList
		expected []api.ResourceName
	}{
		"empty": {
			a:        api.ResourceList{},
			expected: []api.ResourceName{},
		},
		"some-negative": {
			a: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("-10"),
				api.ResourceMemory: resource.MustParse("0"),
			},
			expected: []api.ResourceName{api.ResourceCPU},
		},
		"all-negative": {
			a: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("-200m"),
				api.ResourceMemory: resource.MustParse("-1Gi"),
			},
			expected: []api.ResourceName{api.ResourceCPU, api.ResourceMemory},
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

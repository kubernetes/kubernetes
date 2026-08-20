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

package helper

import (
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/kubernetes/pkg/apis/core"
)

func TestSemantic(t *testing.T) {
	table := []struct {
		a, b        interface{}
		shouldEqual bool
	}{
		{resource.MustParse("0"), resource.Quantity{}, true},
		{resource.Quantity{}, resource.MustParse("0"), true},
		{resource.Quantity{}, resource.MustParse("1m"), false},
		{
			resource.NewQuantity(5, resource.BinarySI),
			resource.NewQuantity(5, resource.DecimalSI),
			true,
		},
		{resource.MustParse("2m"), resource.MustParse("1m"), false},
	}

	for index, item := range table {
		if e, a := item.shouldEqual, Semantic.DeepEqual(item.a, item.b); e != a {
			t.Errorf("case[%d], expected %v, got %v.", index, e, a)
		}
	}
}

func TestIsStandardResource(t *testing.T) {
	testCases := []struct {
		input  string
		output bool
	}{
		{"cpu", true},
		{"memory", true},
		{"disk", false},
		{"blah", false},
		{"x.y.z", false},
		{"hugepages-2Mi", true},
		{"requests.hugepages-2Mi", true},
	}
	for i, tc := range testCases {
		if IsStandardResourceName(core.ResourceName(tc.input)) != tc.output {
			t.Errorf("case[%d], input: %s, expected: %t, got: %t", i, tc.input, tc.output, !tc.output)
		}
	}
}

func TestIsStandardContainerResource(t *testing.T) {
	testCases := []struct {
		input  string
		output bool
	}{
		{"cpu", true},
		{"memory", true},
		{"disk", false},
		{"hugepages-2Mi", true},
	}
	for i, tc := range testCases {
		if IsStandardContainerResourceName(core.ResourceName(tc.input)) != tc.output {
			t.Errorf("case[%d], input: %s, expected: %t, got: %t", i, tc.input, tc.output, !tc.output)
		}
	}
}

func TestIsNodeAllocatableResource(t *testing.T) {
	testCases := []struct {
		input  string
		output bool
	}{
		{"cpu", true},
		{"memory", true},
		{"ephemeral-storage", false},
		{"hugepages-2Mi", true},
		{"disk", false},
		{"blah", false},
	}
	for i, tc := range testCases {
		if IsNodeAllocatableResourceName(core.ResourceName(tc.input)) != tc.output {
			t.Errorf("case[%d], input: %s, expected: %t, got: %t", i, tc.input, tc.output, !tc.output)
		}
	}
}

func TestGetAccessModesFromString(t *testing.T) {
	modes := GetAccessModesFromString("ROX")
	if !ContainsAccessMode(modes, core.ReadOnlyMany) {
		t.Errorf("Expected mode %s, but got %+v", core.ReadOnlyMany, modes)
	}

	modes = GetAccessModesFromString("ROX,RWX")
	if !ContainsAccessMode(modes, core.ReadOnlyMany) {
		t.Errorf("Expected mode %s, but got %+v", core.ReadOnlyMany, modes)
	}
	if !ContainsAccessMode(modes, core.ReadWriteMany) {
		t.Errorf("Expected mode %s, but got %+v", core.ReadWriteMany, modes)
	}

	modes = GetAccessModesFromString("RWO,ROX,RWX")
	if !ContainsAccessMode(modes, core.ReadWriteOnce) {
		t.Errorf("Expected mode %s, but got %+v", core.ReadWriteOnce, modes)
	}
	if !ContainsAccessMode(modes, core.ReadOnlyMany) {
		t.Errorf("Expected mode %s, but got %+v", core.ReadOnlyMany, modes)
	}
	if !ContainsAccessMode(modes, core.ReadWriteMany) {
		t.Errorf("Expected mode %s, but got %+v", core.ReadWriteMany, modes)
	}

	modes = GetAccessModesFromString("RWO,ROX,RWX,RWOP")
	if !ContainsAccessMode(modes, core.ReadWriteOnce) {
		t.Errorf("Expected mode %s, but got %+v", core.ReadWriteOnce, modes)
	}
	if !ContainsAccessMode(modes, core.ReadOnlyMany) {
		t.Errorf("Expected mode %s, but got %+v", core.ReadOnlyMany, modes)
	}
	if !ContainsAccessMode(modes, core.ReadWriteMany) {
		t.Errorf("Expected mode %s, but got %+v", core.ReadWriteMany, modes)
	}
	if !ContainsAccessMode(modes, core.ReadWriteOncePod) {
		t.Errorf("Expected mode %s, but got %+v", core.ReadWriteOncePod, modes)
	}
}

func TestRemoveDuplicateAccessModes(t *testing.T) {
	modes := []core.PersistentVolumeAccessMode{
		core.ReadWriteOnce, core.ReadOnlyMany, core.ReadOnlyMany, core.ReadOnlyMany,
	}
	modes = removeDuplicateAccessModes(modes)
	if len(modes) != 2 {
		t.Errorf("Expected 2 distinct modes in set but found %v", len(modes))
	}
}

func TestIsHugePageResourceName(t *testing.T) {
	testCases := []struct {
		name   core.ResourceName
		result bool
	}{
		{
			name:   core.ResourceName("hugepages-2Mi"),
			result: true,
		},
		{
			name:   core.ResourceName("hugepages-1Gi"),
			result: true,
		},
		{
			name:   core.ResourceName("cpu"),
			result: false,
		},
		{
			name:   core.ResourceName("memory"),
			result: false,
		},
	}
	for _, testCase := range testCases {
		if testCase.result != IsHugePageResourceName(testCase.name) {
			t.Errorf("resource: %v expected result: %v", testCase.name, testCase.result)
		}
	}
}

func TestIsHugePageResourceValueDivisible(t *testing.T) {
	testCases := []struct {
		desc     string
		name     core.ResourceName
		quantity resource.Quantity
		result   bool
	}{
		{"multiple of the page size", "hugepages-2Mi", resource.MustParse("4Mi"), true},
		{"not a multiple of the page size", "hugepages-2Mi", resource.MustParse("5Mi"), false},
		{"multiple of a 1Gi page", "hugepages-1Gi", resource.MustParse("2Gi"), true},
		{"fractional multiple of 1Gi", "hugepages-1Gi", resource.MustParse("2.1Gi"), false},
		{"fractional multiple of 1Mi", "hugepages-1Mi", resource.MustParse("2.1Mi"), false},
		{"multiple of a 64Ki page", "hugepages-64Ki", resource.MustParse("128Ki"), true},
		{"empty page size", "hugepages-", resource.MustParse("128Ki"), false},
		{"not a hugepage resource name", "hugepages", resource.MustParse("128Ki"), false},
	}
	for _, testCase := range testCases {
		if testCase.result != IsHugePageResourceValueDivisible(testCase.name, testCase.quantity) {
			t.Errorf("%s: resource: %v quantity: %v expected result: %v", testCase.desc, testCase.name, testCase.quantity.String(), testCase.result)
		}
	}
}

func TestIsHugePageResourceValueDivisibleOverflow(t *testing.T) {
	// These pin values by magnitude: the same number must get the same answer
	// however it is written, and a value past int64 must not read as divisible
	// just because Quantity.Value() wrapped it or collapsed it to zero.
	testCases := []struct {
		desc     string
		name     core.ResourceName
		quantity resource.Quantity
		result   bool
	}{
		{"multiple of a non power of two page size, past 2^63", "hugepages-3Mi", resource.MustParse("9437184000000000000"), true},
		{"same number written with an exponent", "hugepages-3Mi", resource.MustParse("9437184e12"), true},
		{"same number written with a decimal point", "hugepages-3Mi", resource.MustParse("9.437184e18"), true},
		{"one byte past a multiple, past 2^63", "hugepages-3Mi", resource.MustParse("9437184000000000001"), false},
		{"multiple of a power of two page size, past 2^64", "hugepages-2Mi", resource.MustParse("18446744073709551616"), true},
		{"one byte past 2^64", "hugepages-2Mi", resource.MustParse("18446744073709551617"), false},
		{"not a multiple, but Value() collapses it to 0", "hugepages-3Mi", resource.MustParse("1e30"), false},
		{"not a multiple, smallest value where Value() collapses", "hugepages-3Mi", resource.MustParse("1e19"), false},
		{"beyond any byte count a node could serve", "hugepages-2Mi", resource.MustParse("1e100000000"), false},
	}
	for _, testCase := range testCases {
		if testCase.result != IsHugePageResourceValueDivisible(testCase.name, testCase.quantity) {
			t.Errorf("%s: resource: %v quantity: %v expected result: %v", testCase.desc, testCase.name, testCase.quantity.String(), testCase.result)
		}
	}
}

func TestIsHugePageResourceValueDivisiblePageSizeOverflow(t *testing.T) {
	// A page size comes out of the resource name, so it is caller supplied.
	// pageSize.Value() collapsed to 0 for these and the check divided by zero,
	// and the MilliValue guard disagreed with itself on the two spellings.
	testCases := []struct {
		desc string
		name core.ResourceName
	}{
		{"page size collapses Value() to 0", "hugepages-1e19"},
		{"same page size written as a literal", "hugepages-10000000000000000000"},
		{"page size well past int64", "hugepages-1e30"},
		{"page size at 2^64", "hugepages-18446744073709551616"},
	}
	for _, testCase := range testCases {
		if IsHugePageResourceValueDivisible(testCase.name, resource.MustParse("1")) {
			t.Errorf("%s: resource: %v expected result: false", testCase.desc, testCase.name)
		}
	}
}

func TestIsHugePageResourceValueDivisibleFractional(t *testing.T) {
	// Value() rounds away from zero, so a request half a byte short of a whole
	// page used to read back as a whole page. Trailing zeros are not precision
	// and must still count as whole bytes.
	testCases := []struct {
		desc     string
		name     core.ResourceName
		quantity resource.Quantity
		result   bool
	}{
		{"2Mi written in milli", "hugepages-2Mi", resource.MustParse("2097152000m"), true},
		{"2Mi written scaled", "hugepages-2Mi", resource.MustParse("20971520e-1"), true},
		{"half a byte short of 2Mi", "hugepages-2Mi", resource.MustParse("2097151500m"), false},
		{"half a byte short of 1Ki", "hugepages-1Ki", resource.MustParse("1023500m"), false},
		{"negative, half a byte short of 2Mi", "hugepages-2Mi", resource.MustParse("-2097151500m"), false},
	}
	for _, testCase := range testCases {
		if testCase.result != IsHugePageResourceValueDivisible(testCase.name, testCase.quantity) {
			t.Errorf("%s: resource: %v quantity: %v expected result: %v", testCase.desc, testCase.name, testCase.quantity.String(), testCase.result)
		}
	}
}

func TestHugePageResourceName(t *testing.T) {
	testCases := []struct {
		pageSize resource.Quantity
		name     core.ResourceName
	}{
		{
			pageSize: resource.MustParse("2Mi"),
			name:     core.ResourceName("hugepages-2Mi"),
		},
		{
			pageSize: resource.MustParse("1Gi"),
			name:     core.ResourceName("hugepages-1Gi"),
		},
		{
			// verify we do not regress our canonical representation
			pageSize: *resource.NewQuantity(int64(2097152), resource.BinarySI),
			name:     core.ResourceName("hugepages-2Mi"),
		},
	}
	for _, testCase := range testCases {
		if result := HugePageResourceName(testCase.pageSize); result != testCase.name {
			t.Errorf("pageSize: %v, expected: %v, but got: %v", testCase.pageSize.String(), testCase.name, result.String())
		}
	}
}

func TestHugePageSizeFromResourceName(t *testing.T) {
	testCases := []struct {
		name      core.ResourceName
		expectErr bool
		pageSize  resource.Quantity
	}{
		{
			name:      core.ResourceName("hugepages-2Mi"),
			pageSize:  resource.MustParse("2Mi"),
			expectErr: false,
		},
		{
			name:      core.ResourceName("hugepages-1Gi"),
			pageSize:  resource.MustParse("1Gi"),
			expectErr: false,
		},
		{
			name:      core.ResourceName("hugepages-bad"),
			expectErr: true,
		},
	}
	for _, testCase := range testCases {
		value, err := HugePageSizeFromResourceName(testCase.name)
		if testCase.expectErr && err == nil {
			t.Errorf("Expected an error for %v", testCase.name)
		} else if !testCase.expectErr && err != nil {
			t.Errorf("Unexpected error for %v, got %v", testCase.name, err)
		} else if testCase.pageSize.Value() != value.Value() {
			t.Errorf("Unexpected pageSize for resource %v got %v", testCase.name, value.String())
		}
	}
}

func TestIsOvercommitAllowed(t *testing.T) {
	testCases := []struct {
		name    core.ResourceName
		allowed bool
	}{
		{
			name:    core.ResourceCPU,
			allowed: true,
		},
		{
			name:    core.ResourceMemory,
			allowed: true,
		},
		{
			name:    HugePageResourceName(resource.MustParse("2Mi")),
			allowed: false,
		},
	}
	for _, testCase := range testCases {
		if testCase.allowed != IsOvercommitAllowed(testCase.name) {
			t.Errorf("Unexpected result for %v", testCase.name)
		}
	}
}

func TestIsServiceIPSet(t *testing.T) {
	testCases := []struct {
		input  core.ServiceSpec
		output bool
		name   string
	}{
		{
			name: "nil cluster ip",
			input: core.ServiceSpec{
				ClusterIPs: nil,
			},

			output: false,
		},
		{
			name: "headless service",
			input: core.ServiceSpec{
				ClusterIP:  "None",
				ClusterIPs: []string{"None"},
			},
			output: false,
		},
		// true cases
		{
			name: "one ipv4",
			input: core.ServiceSpec{
				ClusterIP:  "1.2.3.4",
				ClusterIPs: []string{"1.2.3.4"},
			},
			output: true,
		},
		{
			name: "one ipv6",
			input: core.ServiceSpec{
				ClusterIP:  "2001::1",
				ClusterIPs: []string{"2001::1"},
			},
			output: true,
		},
		{
			name: "v4, v6",
			input: core.ServiceSpec{
				ClusterIP:  "1.2.3.4",
				ClusterIPs: []string{"1.2.3.4", "2001::1"},
			},
			output: true,
		},
		{
			name: "v6, v4",
			input: core.ServiceSpec{
				ClusterIP:  "2001::1",
				ClusterIPs: []string{"2001::1", "1.2.3.4"},
			},

			output: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			s := core.Service{
				Spec: tc.input,
			}
			if IsServiceIPSet(&s) != tc.output {
				t.Errorf("case, input: %v, expected: %v, got: %v", tc.input, tc.output, !tc.output)
			}
		})
	}
}

func TestHasInvalidLabelValueInNodeSelectorTerms(t *testing.T) {
	testCases := []struct {
		name   string
		terms  []core.NodeSelectorTerm
		expect bool
	}{
		{
			name: "valid values",
			terms: []core.NodeSelectorTerm{{
				MatchExpressions: []core.NodeSelectorRequirement{{
					Key:      "foo",
					Operator: core.NodeSelectorOpIn,
					Values:   []string{"far"},
				}},
			}},
			expect: false,
		},
		{
			name:   "empty terms",
			terms:  []core.NodeSelectorTerm{},
			expect: false,
		},
		{
			name: "invalid label value",
			terms: []core.NodeSelectorTerm{{
				MatchExpressions: []core.NodeSelectorRequirement{{
					Key:      "foo",
					Operator: core.NodeSelectorOpIn,
					Values:   []string{"-1"},
				}},
			}},
			expect: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := HasInvalidLabelValueInNodeSelectorTerms(tc.terms)
			if got != tc.expect {
				t.Errorf("exepct %v, got %v", tc.expect, got)
			}
		})
	}
}

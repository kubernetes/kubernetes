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
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/kubernetes/pkg/api"
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
	}
	for i, tc := range testCases {
		if IsStandardResourceName(tc.input) != tc.output {
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
		if IsStandardContainerResourceName(tc.input) != tc.output {
			t.Errorf("case[%d], input: %s, expected: %t, got: %t", i, tc.input, tc.output, !tc.output)
		}
	}
}

func TestAddToNodeAddresses(t *testing.T) {
	testCases := []struct {
		existing []api.NodeAddress
		toAdd    []api.NodeAddress
		expected []api.NodeAddress
	}{
		{
			existing: []api.NodeAddress{},
			toAdd:    []api.NodeAddress{},
			expected: []api.NodeAddress{},
		},
		{
			existing: []api.NodeAddress{},
			toAdd: []api.NodeAddress{
				{Type: api.NodeExternalIP, Address: "1.1.1.1"},
				{Type: api.NodeHostName, Address: "localhost"},
			},
			expected: []api.NodeAddress{
				{Type: api.NodeExternalIP, Address: "1.1.1.1"},
				{Type: api.NodeHostName, Address: "localhost"},
			},
		},
		{
			existing: []api.NodeAddress{},
			toAdd: []api.NodeAddress{
				{Type: api.NodeExternalIP, Address: "1.1.1.1"},
				{Type: api.NodeExternalIP, Address: "1.1.1.1"},
			},
			expected: []api.NodeAddress{
				{Type: api.NodeExternalIP, Address: "1.1.1.1"},
			},
		},
		{
			existing: []api.NodeAddress{
				{Type: api.NodeExternalIP, Address: "1.1.1.1"},
				{Type: api.NodeInternalIP, Address: "10.1.1.1"},
			},
			toAdd: []api.NodeAddress{
				{Type: api.NodeExternalIP, Address: "1.1.1.1"},
				{Type: api.NodeHostName, Address: "localhost"},
			},
			expected: []api.NodeAddress{
				{Type: api.NodeExternalIP, Address: "1.1.1.1"},
				{Type: api.NodeInternalIP, Address: "10.1.1.1"},
				{Type: api.NodeHostName, Address: "localhost"},
			},
		},
	}

	for i, tc := range testCases {
		AddToNodeAddresses(&tc.existing, tc.toAdd...)
		if !Semantic.DeepEqual(tc.expected, tc.existing) {
			t.Errorf("case[%d], expected: %v, got: %v", i, tc.expected, tc.existing)
		}
	}
}

func TestGetAccessModesFromString(t *testing.T) {
	modes := GetAccessModesFromString("ROX")
	if !containsAccessMode(modes, api.ReadOnlyMany) {
		t.Errorf("Expected mode %s, but got %+v", api.ReadOnlyMany, modes)
	}

	modes = GetAccessModesFromString("ROX,RWX")
	if !containsAccessMode(modes, api.ReadOnlyMany) {
		t.Errorf("Expected mode %s, but got %+v", api.ReadOnlyMany, modes)
	}
	if !containsAccessMode(modes, api.ReadWriteMany) {
		t.Errorf("Expected mode %s, but got %+v", api.ReadWriteMany, modes)
	}

	modes = GetAccessModesFromString("RWO,ROX,RWX")
	if !containsAccessMode(modes, api.ReadOnlyMany) {
		t.Errorf("Expected mode %s, but got %+v", api.ReadOnlyMany, modes)
	}
	if !containsAccessMode(modes, api.ReadWriteMany) {
		t.Errorf("Expected mode %s, but got %+v", api.ReadWriteMany, modes)
	}
}

func TestRemoveDuplicateAccessModes(t *testing.T) {
	modes := []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce, api.ReadOnlyMany, api.ReadOnlyMany, api.ReadOnlyMany,
	}
	modes = removeDuplicateAccessModes(modes)
	if len(modes) != 2 {
		t.Errorf("Expected 2 distinct modes in set but found %v", len(modes))
	}
}

func TestNodeSelectorRequirementsAsSelector(t *testing.T) {
	matchExpressions := []api.NodeSelectorRequirement{{
		Key:      "foo",
		Operator: api.NodeSelectorOpIn,
		Values:   []string{"bar", "baz"},
	}}
	mustParse := func(s string) labels.Selector {
		out, e := labels.Parse(s)
		if e != nil {
			panic(e)
		}
		return out
	}
	tc := []struct {
		in        []api.NodeSelectorRequirement
		out       labels.Selector
		expectErr bool
	}{
		{in: nil, out: labels.Nothing()},
		{in: []api.NodeSelectorRequirement{}, out: labels.Nothing()},
		{
			in:  matchExpressions,
			out: mustParse("foo in (baz,bar)"),
		},
		{
			in: []api.NodeSelectorRequirement{{
				Key:      "foo",
				Operator: api.NodeSelectorOpExists,
				Values:   []string{"bar", "baz"},
			}},
			expectErr: true,
		},
		{
			in: []api.NodeSelectorRequirement{{
				Key:      "foo",
				Operator: api.NodeSelectorOpGt,
				Values:   []string{"1"},
			}},
			out: mustParse("foo>1"),
		},
		{
			in: []api.NodeSelectorRequirement{{
				Key:      "bar",
				Operator: api.NodeSelectorOpLt,
				Values:   []string{"7"},
			}},
			out: mustParse("bar<7"),
		},
	}

	for i, tc := range tc {
		out, err := NodeSelectorRequirementsAsSelector(tc.in)
		if err == nil && tc.expectErr {
			t.Errorf("[%v]expected error but got none.", i)
		}
		if err != nil && !tc.expectErr {
			t.Errorf("[%v]did not expect error but got: %v", i, err)
		}
		if !reflect.DeepEqual(out, tc.out) {
			t.Errorf("[%v]expected:\n\t%+v\nbut got:\n\t%+v", i, tc.out, out)
		}
	}
}

func TestSysctlsFromPodAnnotation(t *testing.T) {
	type Test struct {
		annotation  string
		expectValue []api.Sysctl
		expectErr   bool
	}
	for i, test := range []Test{
		{
			annotation:  "",
			expectValue: nil,
		},
		{
			annotation: "foo.bar",
			expectErr:  true,
		},
		{
			annotation: "=123",
			expectErr:  true,
		},
		{
			annotation:  "foo.bar=",
			expectValue: []api.Sysctl{{Name: "foo.bar", Value: ""}},
		},
		{
			annotation:  "foo.bar=42",
			expectValue: []api.Sysctl{{Name: "foo.bar", Value: "42"}},
		},
		{
			annotation: "foo.bar=42,",
			expectErr:  true,
		},
		{
			annotation:  "foo.bar=42,abc.def=1",
			expectValue: []api.Sysctl{{Name: "foo.bar", Value: "42"}, {Name: "abc.def", Value: "1"}},
		},
	} {
		sysctls, err := SysctlsFromPodAnnotation(test.annotation)
		if test.expectErr && err == nil {
			t.Errorf("[%v]expected error but got none", i)
		} else if !test.expectErr && err != nil {
			t.Errorf("[%v]did not expect error but got: %v", i, err)
		} else if !reflect.DeepEqual(sysctls, test.expectValue) {
			t.Errorf("[%v]expect value %v but got %v", i, test.expectValue, sysctls)
		}
	}
}

// TODO: remove when alpha support for topology constraints is removed
func TestGetNodeAffinityFromAnnotations(t *testing.T) {
	testCases := []struct {
		annotations map[string]string
		expectErr   bool
	}{
		{
			annotations: nil,
			expectErr:   false,
		},
		{
			annotations: map[string]string{},
			expectErr:   false,
		},
		{
			annotations: map[string]string{
				api.AlphaStorageNodeAffinityAnnotation: `{ 
					"requiredDuringSchedulingIgnoredDuringExecution": { 
						"nodeSelectorTerms": [ 
							{ "matchExpressions": [
								{ "key": "test-key1",
								  "operator": "In",
								  "values": ["test-value1", "test-value2"]
								},
								{ "key": "test-key2",
								  "operator": "In",
								  "values": ["test-value1", "test-value2"]
								}
							]}
						]}
					}`,
			},
			expectErr: false,
		},
		{
			annotations: map[string]string{
				api.AlphaStorageNodeAffinityAnnotation: `[{ 
					"requiredDuringSchedulingIgnoredDuringExecution": { 
						"nodeSelectorTerms": [ 
							{ "matchExpressions": [
								{ "key": "test-key1",
								  "operator": "In",
								  "values": ["test-value1", "test-value2"]
								},
								{ "key": "test-key2",
								  "operator": "In",
								  "values": ["test-value1", "test-value2"]
								}
							]}
						]}
					}]`,
			},
			expectErr: true,
		},
		{
			annotations: map[string]string{
				api.AlphaStorageNodeAffinityAnnotation: `{ 
					"requiredDuringSchedulingIgnoredDuringExecution": { 
						"nodeSelectorTerms":  
							 "matchExpressions": [
								{ "key": "test-key1",
								  "operator": "In",
								  "values": ["test-value1", "test-value2"]
								},
								{ "key": "test-key2",
								  "operator": "In",
								  "values": ["test-value1", "test-value2"]
								}
							]}
						}
					}`,
			},
			expectErr: true,
		},
	}

	for i, tc := range testCases {
		_, err := GetStorageNodeAffinityFromAnnotation(tc.annotations)
		if err == nil && tc.expectErr {
			t.Errorf("[%v]expected error but got none.", i)
		}
		if err != nil && !tc.expectErr {
			t.Errorf("[%v]did not expect error but got: %v", i, err)
		}
	}
}

func TestIsHugePageResourceName(t *testing.T) {
	testCases := []struct {
		name   api.ResourceName
		result bool
	}{
		{
			name:   api.ResourceName("hugepages-2Mi"),
			result: true,
		},
		{
			name:   api.ResourceName("hugepages-1Gi"),
			result: true,
		},
		{
			name:   api.ResourceName("cpu"),
			result: false,
		},
		{
			name:   api.ResourceName("memory"),
			result: false,
		},
	}
	for _, testCase := range testCases {
		if testCase.result != IsHugePageResourceName(testCase.name) {
			t.Errorf("resource: %v expected result: %v", testCase.name, testCase.result)
		}
	}
}

func TestHugePageResourceName(t *testing.T) {
	testCases := []struct {
		pageSize resource.Quantity
		name     api.ResourceName
	}{
		{
			pageSize: resource.MustParse("2Mi"),
			name:     api.ResourceName("hugepages-2Mi"),
		},
		{
			pageSize: resource.MustParse("1Gi"),
			name:     api.ResourceName("hugepages-1Gi"),
		},
		{
			// verify we do not regress our canonical representation
			pageSize: *resource.NewQuantity(int64(2097152), resource.BinarySI),
			name:     api.ResourceName("hugepages-2Mi"),
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
		name      api.ResourceName
		expectErr bool
		pageSize  resource.Quantity
	}{
		{
			name:      api.ResourceName("hugepages-2Mi"),
			pageSize:  resource.MustParse("2Mi"),
			expectErr: false,
		},
		{
			name:      api.ResourceName("hugepages-1Gi"),
			pageSize:  resource.MustParse("1Gi"),
			expectErr: false,
		},
		{
			name:      api.ResourceName("hugepages-bad"),
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
		name    api.ResourceName
		allowed bool
	}{
		{
			name:    api.ResourceCPU,
			allowed: true,
		},
		{
			name:    api.ResourceMemory,
			allowed: true,
		},
		{
			name:    api.ResourceNvidiaGPU,
			allowed: false,
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

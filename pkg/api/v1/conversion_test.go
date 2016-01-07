/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package v1_test

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	versioned "k8s.io/kubernetes/pkg/api/v1"
)

// TestPodSpecConversion tests that ServiceAccount is an alias for
// ServiceAccountName.
func TestPodSpecConversion(t *testing.T) {
	name, other := "foo", "bar"

	// Test internal -> v1. Should have both alias (DeprecatedServiceAccount)
	// and new field (ServiceAccountName).
	i := &api.PodSpec{
		ServiceAccountName: name,
	}
	v := versioned.PodSpec{}
	if err := api.Scheme.Convert(i, &v); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if v.ServiceAccountName != name {
		t.Fatalf("want v1.ServiceAccountName %q, got %q", name, v.ServiceAccountName)
	}
	if v.DeprecatedServiceAccount != name {
		t.Fatalf("want v1.DeprecatedServiceAccount %q, got %q", name, v.DeprecatedServiceAccount)
	}

	// Test v1 -> internal. Either DeprecatedServiceAccount, ServiceAccountName,
	// or both should translate to ServiceAccountName. ServiceAccountName wins
	// if both are set.
	testCases := []*versioned.PodSpec{
		// New
		{ServiceAccountName: name},
		// Alias
		{DeprecatedServiceAccount: name},
		// Both: same
		{ServiceAccountName: name, DeprecatedServiceAccount: name},
		// Both: different
		{ServiceAccountName: name, DeprecatedServiceAccount: other},
	}
	for k, v := range testCases {
		got := api.PodSpec{}
		err := api.Scheme.Convert(v, &got)
		if err != nil {
			t.Fatalf("unexpected error for case %d: %v", k, err)
		}
		if got.ServiceAccountName != name {
			t.Fatalf("want api.ServiceAccountName %q, got %q", name, got.ServiceAccountName)
		}
	}
}

func TestResourceListConversion(t *testing.T) {
	bigMilliQuantity := resource.NewQuantity(resource.MaxMilliValue, resource.DecimalSI)
	bigMilliQuantity.Add(resource.MustParse("12345m"))

	tests := []struct {
		input    versioned.ResourceList
		expected api.ResourceList
	}{
		{ // No changes necessary.
			input: versioned.ResourceList{
				versioned.ResourceMemory:  resource.MustParse("30M"),
				versioned.ResourceCPU:     resource.MustParse("100m"),
				versioned.ResourceStorage: resource.MustParse("1G"),
			},
			expected: api.ResourceList{
				api.ResourceMemory:  resource.MustParse("30M"),
				api.ResourceCPU:     resource.MustParse("100m"),
				api.ResourceStorage: resource.MustParse("1G"),
			},
		},
		{ // Nano-scale values should be rounded up to milli-scale.
			input: versioned.ResourceList{
				versioned.ResourceCPU:    resource.MustParse("3.000023m"),
				versioned.ResourceMemory: resource.MustParse("500.000050m"),
			},
			expected: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("4m"),
				api.ResourceMemory: resource.MustParse("501m"),
			},
		},
		{ // Large values should still be accurate.
			input: versioned.ResourceList{
				versioned.ResourceCPU:     *bigMilliQuantity.Copy(),
				versioned.ResourceStorage: *bigMilliQuantity.Copy(),
			},
			expected: api.ResourceList{
				api.ResourceCPU:     *bigMilliQuantity.Copy(),
				api.ResourceStorage: *bigMilliQuantity.Copy(),
			},
		},
	}

	output := api.ResourceList{}
	for i, test := range tests {
		err := api.Scheme.Convert(&test.input, &output)
		if err != nil {
			t.Fatalf("unexpected error for case %d: %v", i, err)
		}
		if !api.Semantic.DeepEqual(test.expected, output) {
			t.Errorf("unexpected conversion for case %d: Expected %+v; Got %+v", i, test.expected, output)
		}
	}
}

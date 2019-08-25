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

package gvk

import "testing"

var equalsTests = []struct {
	x1 Gvk
	x2 Gvk
}{
	{Gvk{Group: "a", Version: "b", Kind: "c"},
		Gvk{Group: "a", Version: "b", Kind: "c"}},
	{Gvk{Version: "b", Kind: "c"},
		Gvk{Version: "b", Kind: "c"}},
	{Gvk{Kind: "c"},
		Gvk{Kind: "c"}},
}

func TestEquals(t *testing.T) {
	for _, hey := range equalsTests {
		if !hey.x1.Equals(hey.x2) {
			t.Fatalf("%v should equal %v", hey.x1, hey.x2)
		}
	}
}

var lessThanTests = []struct {
	x1 Gvk
	x2 Gvk
}{
	{Gvk{Group: "a", Version: "b", Kind: "CustomResourceDefinition"},
		Gvk{Group: "a", Version: "b", Kind: "RoleBinding"}},
	{Gvk{Group: "a", Version: "b", Kind: "Namespace"},
		Gvk{Group: "a", Version: "b", Kind: "ClusterRole"}},
	{Gvk{Group: "a", Version: "b", Kind: "a"},
		Gvk{Group: "a", Version: "b", Kind: "b"}},
	{Gvk{Group: "a", Version: "b", Kind: "Namespace"},
		Gvk{Group: "a", Version: "c", Kind: "Namespace"}},
	{Gvk{Group: "a", Version: "c", Kind: "Namespace"},
		Gvk{Group: "b", Version: "c", Kind: "Namespace"}},
	{Gvk{Group: "b", Version: "c", Kind: "Namespace"},
		Gvk{Group: "a", Version: "c", Kind: "ClusterRole"}},
	{Gvk{Group: "a", Version: "c", Kind: "Namespace"},
		Gvk{Group: "a", Version: "b", Kind: "ClusterRole"}},
	{Gvk{Group: "a", Version: "d", Kind: "Namespace"},
		Gvk{Group: "b", Version: "c", Kind: "Namespace"}},
	{Gvk{Group: "a", Version: "b", Kind: orderFirst[len(orderFirst)-1]},
		Gvk{Group: "a", Version: "b", Kind: orderLast[0]}},
	{Gvk{Group: "a", Version: "b", Kind: orderFirst[len(orderFirst)-1]},
		Gvk{Group: "a", Version: "b", Kind: "CustomKindX"}},
	{Gvk{Group: "a", Version: "b", Kind: "CustomKindX"},
		Gvk{Group: "a", Version: "b", Kind: orderLast[0]}},
	{Gvk{Group: "a", Version: "b", Kind: "CustomKindA"},
		Gvk{Group: "a", Version: "b", Kind: "CustomKindB"}},
	{Gvk{Group: "a", Version: "b", Kind: "CustomKindX"},
		Gvk{Group: "a", Version: "b", Kind: "ValidatingWebhookConfiguration"}},
	{Gvk{Group: "a", Version: "b", Kind: "APIService"},
		Gvk{Group: "a", Version: "b", Kind: "ValidatingWebhookConfiguration"}},
	{Gvk{Group: "a", Version: "b", Kind: "Service"},
		Gvk{Group: "a", Version: "b", Kind: "APIService"}},
}

func TestIsLessThan1(t *testing.T) {
	for _, hey := range lessThanTests {
		if !hey.x1.IsLessThan(hey.x2) {
			t.Fatalf("%v should be less than %v", hey.x1, hey.x2)
		}
		if hey.x2.IsLessThan(hey.x1) {
			t.Fatalf("%v should not be less than %v", hey.x2, hey.x1)
		}
	}
}

var stringTests = []struct {
	x Gvk
	s string
}{
	{Gvk{}, "~G_~V_~K"},
	{Gvk{Kind: "k"}, "~G_~V_k"},
	{Gvk{Version: "v"}, "~G_v_~K"},
	{Gvk{Version: "v", Kind: "k"}, "~G_v_k"},
	{Gvk{Group: "g"}, "g_~V_~K"},
	{Gvk{Group: "g", Kind: "k"}, "g_~V_k"},
	{Gvk{Group: "g", Version: "v"}, "g_v_~K"},
	{Gvk{Group: "g", Version: "v", Kind: "k"}, "g_v_k"},
}

func TestString(t *testing.T) {
	for _, hey := range stringTests {
		if hey.x.String() != hey.s {
			t.Fatalf("bad string for %v '%s'", hey.x, hey.s)
		}
	}
}

func TestSelectByGVK(t *testing.T) {
	type testCase struct {
		description string
		in          Gvk
		filter      *Gvk
		expected    bool
	}
	testCases := []testCase{
		{
			description: "nil filter",
			in:          Gvk{},
			filter:      nil,
			expected:    true,
		},
		{
			description: "gvk matches",
			in: Gvk{
				Group:   "group1",
				Version: "version1",
				Kind:    "kind1",
			},
			filter: &Gvk{
				Group:   "group1",
				Version: "version1",
				Kind:    "kind1",
			},
			expected: true,
		},
		{
			description: "group doesn't matches",
			in: Gvk{
				Group:   "group1",
				Version: "version1",
				Kind:    "kind1",
			},
			filter: &Gvk{
				Group:   "group2",
				Version: "version1",
				Kind:    "kind1",
			},
			expected: false,
		},
		{
			description: "version doesn't matches",
			in: Gvk{
				Group:   "group1",
				Version: "version1",
				Kind:    "kind1",
			},
			filter: &Gvk{
				Group:   "group1",
				Version: "version2",
				Kind:    "kind1",
			},
			expected: false,
		},
		{
			description: "kind doesn't matches",
			in: Gvk{
				Group:   "group1",
				Version: "version1",
				Kind:    "kind1",
			},
			filter: &Gvk{
				Group:   "group1",
				Version: "version1",
				Kind:    "kind2",
			},
			expected: false,
		},
		{
			description: "no version in filter",
			in: Gvk{
				Group:   "group1",
				Version: "version1",
				Kind:    "kind1",
			},
			filter: &Gvk{
				Group:   "group1",
				Version: "",
				Kind:    "kind1",
			},
			expected: true,
		},
		{
			description: "only kind is set in filter",
			in: Gvk{
				Group:   "group1",
				Version: "version1",
				Kind:    "kind1",
			},
			filter: &Gvk{
				Group:   "",
				Version: "",
				Kind:    "kind1",
			},
			expected: true,
		},
	}

	for _, tc := range testCases {
		filtered := tc.in.IsSelected(tc.filter)
		if filtered != tc.expected {
			t.Fatalf("unexpected filter result for test case: %v", tc.description)
		}
	}
}

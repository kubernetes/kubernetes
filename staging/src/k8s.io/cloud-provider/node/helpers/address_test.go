/*
Copyright 2019 The Kubernetes Authors.

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

package helpers

import (
	"testing"

	"k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
)

func TestAddToNodeAddresses(t *testing.T) {
	testCases := []struct {
		name     string
		existing []v1.NodeAddress
		toAdd    []v1.NodeAddress
		expected []v1.NodeAddress
	}{
		{
			name:     "add no addresses to empty node addresses",
			existing: []v1.NodeAddress{},
			toAdd:    []v1.NodeAddress{},
			expected: []v1.NodeAddress{},
		},
		{
			name:     "add new node addresses to empty existing addresses",
			existing: []v1.NodeAddress{},
			toAdd: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "1.1.1.1"},
				{Type: v1.NodeHostName, Address: "localhost"},
			},
			expected: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "1.1.1.1"},
				{Type: v1.NodeHostName, Address: "localhost"},
			},
		},
		{
			name:     "add a duplicate node address",
			existing: []v1.NodeAddress{},
			toAdd: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "1.1.1.1"},
				{Type: v1.NodeExternalIP, Address: "1.1.1.1"},
			},
			expected: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "1.1.1.1"},
			},
		},
		{
			name: "add 1 new and 1 duplicate address",
			existing: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "1.1.1.1"},
				{Type: v1.NodeInternalIP, Address: "10.1.1.1"},
			},
			toAdd: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "1.1.1.1"},
				{Type: v1.NodeHostName, Address: "localhost"},
			},
			expected: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "1.1.1.1"},
				{Type: v1.NodeInternalIP, Address: "10.1.1.1"},
				{Type: v1.NodeHostName, Address: "localhost"},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			AddToNodeAddresses(&tc.existing, tc.toAdd...)
			if !apiequality.Semantic.DeepEqual(tc.expected, tc.existing) {
				t.Errorf("expected: %v, got: %v", tc.expected, tc.existing)
			}
		})
	}
}

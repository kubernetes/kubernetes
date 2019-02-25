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

package azure

import (
	"fmt"
	"net/http"
	"reflect"
	"testing"

	"github.com/Azure/go-autorest/autorest"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/util/sets"
)

func TestExtractNotFound(t *testing.T) {
	notFound := autorest.DetailedError{StatusCode: http.StatusNotFound}
	otherHTTP := autorest.DetailedError{StatusCode: http.StatusForbidden}
	otherErr := fmt.Errorf("other error")

	tests := []struct {
		err         error
		expectedErr error
		exists      bool
	}{
		{nil, nil, true},
		{otherErr, otherErr, false},
		{notFound, nil, false},
		{otherHTTP, otherHTTP, false},
	}

	for _, test := range tests {
		exists, _, err := checkResourceExistsFromError(test.err)
		if test.exists != exists {
			t.Errorf("expected: %v, saw: %v", test.exists, exists)
		}
		if !reflect.DeepEqual(test.expectedErr, err) {
			t.Errorf("expected err: %v, saw: %v", test.expectedErr, err)
		}
	}
}

func TestIsNodeUnmanaged(t *testing.T) {
	tests := []struct {
		name           string
		unmanagedNodes sets.String
		node           string
		expected       bool
		expectErr      bool
	}{
		{
			name:           "unmanaged node should return true",
			unmanagedNodes: sets.NewString("node1", "node2"),
			node:           "node1",
			expected:       true,
		},
		{
			name:           "managed node should return false",
			unmanagedNodes: sets.NewString("node1", "node2"),
			node:           "node3",
			expected:       false,
		},
		{
			name:           "empty unmanagedNodes should return true",
			unmanagedNodes: sets.NewString(),
			node:           "node3",
			expected:       false,
		},
		{
			name:           "no synced informer should report error",
			unmanagedNodes: sets.NewString(),
			node:           "node1",
			expectErr:      true,
		},
	}

	az := getTestCloud()
	for _, test := range tests {
		az.unmanagedNodes = test.unmanagedNodes
		if test.expectErr {
			az.nodeInformerSynced = func() bool {
				return false
			}
		}

		real, err := az.IsNodeUnmanaged(test.node)
		if test.expectErr {
			assert.Error(t, err, test.name)
			continue
		}

		assert.NoError(t, err, test.name)
		assert.Equal(t, test.expected, real, test.name)
	}
}

func TestIsNodeUnmanagedByProviderID(t *testing.T) {
	tests := []struct {
		providerID string
		expected   bool
		name       string
	}{
		{
			providerID: CloudProviderName + ":///subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/myResourceGroupName/providers/Microsoft.Compute/virtualMachines/k8s-agent-AAAAAAAA-0",
			expected:   false,
		},
		{
			providerID: CloudProviderName + "://",
			expected:   true,
		},
		{
			providerID: ":///subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/myResourceGroupName/providers/Microsoft.Compute/virtualMachines/k8s-agent-AAAAAAAA-0",
			expected:   true,
		},
		{
			providerID: "aws:///subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/myResourceGroupName/providers/Microsoft.Compute/virtualMachines/k8s-agent-AAAAAAAA-0",
			expected:   true,
		},
		{
			providerID: "k8s-agent-AAAAAAAA-0",
			expected:   true,
		},
	}

	az := getTestCloud()
	for _, test := range tests {
		isUnmanagedNode := az.IsNodeUnmanagedByProviderID(test.providerID)
		assert.Equal(t, test.expected, isUnmanagedNode, test.providerID)
	}
}

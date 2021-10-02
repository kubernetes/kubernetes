//go:build !providerless
// +build !providerless

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
	"net/http"
	"reflect"
	"testing"

	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/legacy-cloud-providers/azure/retry"
)

func TestExtractNotFound(t *testing.T) {
	notFound := &retry.Error{HTTPStatusCode: http.StatusNotFound}
	otherHTTP := &retry.Error{HTTPStatusCode: http.StatusForbidden}
	otherErr := &retry.Error{HTTPStatusCode: http.StatusTooManyRequests}

	tests := []struct {
		err         *retry.Error
		expectedErr *retry.Error
		exists      bool
	}{
		{nil, nil, true},
		{otherErr, otherErr, false},
		{notFound, nil, false},
		{otherHTTP, otherHTTP, false},
	}

	for _, test := range tests {
		exists, err := checkResourceExistsFromError(test.err)
		if test.exists != exists {
			t.Errorf("expected: %v, saw: %v", test.exists, exists)
		}
		if !reflect.DeepEqual(test.expectedErr, err) {
			t.Errorf("expected err: %v, saw: %v", test.expectedErr, err)
		}
	}
}

func TestIsNodeUnmanaged(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

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

	az := GetTestCloud(ctrl)
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
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

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

	az := GetTestCloud(ctrl)
	for _, test := range tests {
		isUnmanagedNode := az.IsNodeUnmanagedByProviderID(test.providerID)
		assert.Equal(t, test.expected, isUnmanagedNode, test.providerID)
	}
}

func TestConvertResourceGroupNameToLower(t *testing.T) {
	tests := []struct {
		desc        string
		resourceID  string
		expected    string
		expectError bool
	}{
		{
			desc:        "empty string should report error",
			resourceID:  "",
			expectError: true,
		},
		{
			desc:        "resourceID not in Azure format should report error",
			resourceID:  "invalid-id",
			expectError: true,
		},
		{
			desc:        "providerID not in Azure format should report error",
			resourceID:  "azure://invalid-id",
			expectError: true,
		},
		{
			desc:       "resource group name in VM providerID should be converted",
			resourceID: CloudProviderName + ":///subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/myResourceGroupName/providers/Microsoft.Compute/virtualMachines/k8s-agent-AAAAAAAA-0",
			expected:   CloudProviderName + ":///subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/myresourcegroupname/providers/Microsoft.Compute/virtualMachines/k8s-agent-AAAAAAAA-0",
		},
		{
			desc:       "resource group name in VM resourceID should be converted",
			resourceID: "/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/myResourceGroupName/providers/Microsoft.Compute/virtualMachines/k8s-agent-AAAAAAAA-0",
			expected:   "/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/myresourcegroupname/providers/Microsoft.Compute/virtualMachines/k8s-agent-AAAAAAAA-0",
		},
		{
			desc:       "resource group name in VMSS providerID should be converted",
			resourceID: CloudProviderName + ":///subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/myResourceGroupName/providers/Microsoft.Compute/virtualMachineScaleSets/myScaleSetName/virtualMachines/156",
			expected:   CloudProviderName + ":///subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/myresourcegroupname/providers/Microsoft.Compute/virtualMachineScaleSets/myScaleSetName/virtualMachines/156",
		},
		{
			desc:       "resource group name in VMSS resourceID should be converted",
			resourceID: "/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/myResourceGroupName/providers/Microsoft.Compute/virtualMachineScaleSets/myScaleSetName/virtualMachines/156",
			expected:   "/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/myresourcegroupname/providers/Microsoft.Compute/virtualMachineScaleSets/myScaleSetName/virtualMachines/156",
		},
	}

	for _, test := range tests {
		real, err := convertResourceGroupNameToLower(test.resourceID)
		if test.expectError {
			assert.NotNil(t, err, test.desc)
			continue
		}

		assert.Nil(t, err, test.desc)
		assert.Equal(t, test.expected, real, test.desc)
	}
}

func TestIsBackendPoolOnSameLB(t *testing.T) {
	tests := []struct {
		backendPoolID        string
		existingBackendPools []string
		expected             bool
		expectedLBName       string
		expectError          bool
	}{
		{
			backendPoolID: "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/loadBalancers/lb1/backendAddressPools/pool1",
			existingBackendPools: []string{
				"/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/loadBalancers/lb1/backendAddressPools/pool2",
			},
			expected:       true,
			expectedLBName: "",
		},
		{
			backendPoolID: "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/loadBalancers/lb1-internal/backendAddressPools/pool1",
			existingBackendPools: []string{
				"/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/loadBalancers/lb1/backendAddressPools/pool2",
			},
			expected:       true,
			expectedLBName: "",
		},
		{
			backendPoolID: "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/loadBalancers/lb1/backendAddressPools/pool1",
			existingBackendPools: []string{
				"/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/loadBalancers/lb1-internal/backendAddressPools/pool2",
			},
			expected:       true,
			expectedLBName: "",
		},
		{
			backendPoolID: "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/loadBalancers/lb1/backendAddressPools/pool1",
			existingBackendPools: []string{
				"/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/loadBalancers/lb2/backendAddressPools/pool2",
			},
			expected:       false,
			expectedLBName: "lb2",
		},
		{
			backendPoolID: "wrong-backendpool-id",
			existingBackendPools: []string{
				"/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/loadBalancers/lb1/backendAddressPools/pool2",
			},
			expectError: true,
		},
		{
			backendPoolID: "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/loadBalancers/lb1/backendAddressPools/pool1",
			existingBackendPools: []string{
				"wrong-existing-backendpool-id",
			},
			expectError: true,
		},
		{
			backendPoolID: "wrong-backendpool-id",
			existingBackendPools: []string{
				"wrong-existing-backendpool-id",
			},
			expectError: true,
		},
		{
			backendPoolID: "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/loadBalancers/malformed-lb1-internal/backendAddressPools/pool1",
			existingBackendPools: []string{
				"/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/loadBalancers/malformed-lb1-lanretni/backendAddressPools/pool2",
			},
			expected:       false,
			expectedLBName: "malformed-lb1-lanretni",
		},
	}

	for _, test := range tests {
		isSameLB, lbName, err := isBackendPoolOnSameLB(test.backendPoolID, test.existingBackendPools)
		if test.expectError {
			assert.Error(t, err)
			continue
		}

		assert.Equal(t, test.expected, isSameLB)
		assert.Equal(t, test.expectedLBName, lbName)
	}
}

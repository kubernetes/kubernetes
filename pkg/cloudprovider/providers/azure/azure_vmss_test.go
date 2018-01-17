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

package azure

import (
	"fmt"
	"testing"

	"github.com/Azure/azure-sdk-for-go/arm/compute"
	"github.com/stretchr/testify/assert"
)

func newTestScaleSet(scaleSetName string, vmList []string) *scaleSet {
	cloud := getTestCloud()
	setTestVirtualMachineCloud(cloud, scaleSetName, vmList)
	ss := newScaleSet(cloud)

	return ss.(*scaleSet)
}

func setTestVirtualMachineCloud(ss *Cloud, scaleSetName string, vmList []string) {
	virtualMachineScaleSetsClient := newFakeVirtualMachineScaleSetsClient()
	scaleSets := make(map[string]map[string]compute.VirtualMachineScaleSet)
	scaleSets["rg"] = map[string]compute.VirtualMachineScaleSet{
		scaleSetName: {
			Name: &scaleSetName,
		},
	}
	virtualMachineScaleSetsClient.setFakeStore(scaleSets)

	virtualMachineScaleSetVMsClient := newFakeVirtualMachineScaleSetVMsClient()
	ssVMs := make(map[string]map[string]compute.VirtualMachineScaleSetVM)
	ssVMs["rg"] = make(map[string]compute.VirtualMachineScaleSetVM)
	for i := range vmList {
		ID := fmt.Sprintf("azure:///subscriptions/script/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/%s/virtualMachines/%d", scaleSetName, i)
		nodeName := vmList[i]
		instanceID := fmt.Sprintf("%d", i)
		vmKey := fmt.Sprintf("%s-%s", scaleSetName, nodeName)
		networkInterfaces := []compute.NetworkInterfaceReference{
			{
				ID: &nodeName,
			},
		}
		ssVMs["rg"][vmKey] = compute.VirtualMachineScaleSetVM{
			VirtualMachineScaleSetVMProperties: &compute.VirtualMachineScaleSetVMProperties{
				OsProfile: &compute.OSProfile{
					ComputerName: &nodeName,
				},
				NetworkProfile: &compute.NetworkProfile{
					NetworkInterfaces: &networkInterfaces,
				},
			},
			ID:         &ID,
			InstanceID: &instanceID,
			Location:   &ss.Location,
		}
	}
	virtualMachineScaleSetVMsClient.setFakeStore(ssVMs)

	ss.VirtualMachineScaleSetsClient = virtualMachineScaleSetsClient
	ss.VirtualMachineScaleSetVMsClient = virtualMachineScaleSetVMsClient
}

func TestGetScaleSetVMInstanceID(t *testing.T) {
	tests := []struct {
		msg                string
		machineName        string
		expectError        bool
		expectedInstanceID string
	}{{
		msg:         "invalid vmss instance name",
		machineName: "vmvm",
		expectError: true,
	},
		{
			msg:                "valid vmss instance name",
			machineName:        "vm00000Z",
			expectError:        false,
			expectedInstanceID: "35",
		},
	}

	for i, test := range tests {
		instanceID, err := getScaleSetVMInstanceID(test.machineName)
		if test.expectError {
			assert.Error(t, err, fmt.Sprintf("TestCase[%d]: %s", i, test.msg))
		} else {
			assert.Equal(t, test.expectedInstanceID, instanceID, fmt.Sprintf("TestCase[%d]: %s", i, test.msg))
		}
	}
}

func TestGetInstanceIDByNodeName(t *testing.T) {
	testCases := []struct {
		description string
		scaleSet    string
		vmList      []string
		nodeName    string
		expected    string
		expectError bool
	}{
		{
			description: "scaleSet should get instance by node name",
			scaleSet:    "ss",
			vmList:      []string{"vm1", "vm2"},
			nodeName:    "vm1",
			expected:    "azure:///subscriptions/script/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/ss/virtualMachines/0",
		},
		{
			description: "scaleSet should get instance by node name with upper cases hostname",
			scaleSet:    "ss",
			vmList:      []string{"VM1", "vm2"},
			nodeName:    "vm1",
			expected:    "azure:///subscriptions/script/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/ss/virtualMachines/0",
		},
		{
			description: "scaleSet should not get instance for non-exist nodes",
			scaleSet:    "ss",
			vmList:      []string{"VM1", "vm2"},
			nodeName:    "vm3",
			expectError: true,
		},
	}

	for _, test := range testCases {
		ss := newTestScaleSet(test.scaleSet, test.vmList)

		real, err := ss.GetInstanceIDByNodeName(test.nodeName)
		if test.expectError {
			assert.Error(t, err, test.description)
			continue
		}

		assert.NoError(t, err, test.description)
		assert.Equal(t, test.expected, real, test.description)
	}
}

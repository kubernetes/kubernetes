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

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2018-04-01/compute"
	"github.com/stretchr/testify/assert"
)

func newTestScaleSet(scaleSetName, zone string, faultDomain int32, vmList []string) (*scaleSet, error) {
	cloud := getTestCloud()
	setTestVirtualMachineCloud(cloud, scaleSetName, zone, faultDomain, vmList)
	ss, err := newScaleSet(cloud)
	if err != nil {
		return nil, err
	}

	return ss.(*scaleSet), nil
}

func setTestVirtualMachineCloud(ss *Cloud, scaleSetName, zone string, faultDomain int32, vmList []string) {
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
		ID := fmt.Sprintf("/subscriptions/script/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/%s/virtualMachines/%d", scaleSetName, i)
		nodeName := vmList[i]
		instanceID := fmt.Sprintf("%d", i)
		vmName := fmt.Sprintf("%s_%s", scaleSetName, instanceID)
		networkInterfaces := []compute.NetworkInterfaceReference{
			{
				ID: &nodeName,
			},
		}
		vmssVM := compute.VirtualMachineScaleSetVM{
			VirtualMachineScaleSetVMProperties: &compute.VirtualMachineScaleSetVMProperties{
				OsProfile: &compute.OSProfile{
					ComputerName: &nodeName,
				},
				NetworkProfile: &compute.NetworkProfile{
					NetworkInterfaces: &networkInterfaces,
				},
				InstanceView: &compute.VirtualMachineScaleSetVMInstanceView{
					PlatformFaultDomain: &faultDomain,
				},
			},
			ID:         &ID,
			InstanceID: &instanceID,
			Name:       &vmName,
			Location:   &ss.Location,
		}

		if zone != "" {
			zones := []string{zone}
			vmssVM.Zones = &zones
		}
		ssVMs["rg"][vmName] = vmssVM
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
			vmList:      []string{"vmssee6c2000000", "vmssee6c2000001"},
			nodeName:    "vmssee6c2000001",
			expected:    "/subscriptions/script/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/ss/virtualMachines/1",
		},
		{
			description: "scaleSet should get instance by node name with upper cases hostname",
			scaleSet:    "ss",
			vmList:      []string{"VMSSEE6C2000000", "VMSSEE6C2000001"},
			nodeName:    "vmssee6c2000000",
			expected:    "/subscriptions/script/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/ss/virtualMachines/0",
		},
		{
			description: "scaleSet should not get instance for non-exist nodes",
			scaleSet:    "ss",
			vmList:      []string{"vmssee6c2000000", "vmssee6c2000001"},
			nodeName:    "agente6c2000005",
			expectError: true,
		},
	}

	for _, test := range testCases {
		ss, err := newTestScaleSet(test.scaleSet, "", 0, test.vmList)
		assert.NoError(t, err, test.description)

		real, err := ss.GetInstanceIDByNodeName(test.nodeName)
		if test.expectError {
			assert.Error(t, err, test.description)
			continue
		}

		assert.NoError(t, err, test.description)
		assert.Equal(t, test.expected, real, test.description)
	}
}

func TestGetZoneByNodeName(t *testing.T) {
	testCases := []struct {
		description string
		scaleSet    string
		vmList      []string
		nodeName    string
		zone        string
		faultDomain int32
		expected    string
		expectError bool
	}{
		{
			description: "scaleSet should get faultDomain for non-zoned nodes",
			scaleSet:    "ss",
			vmList:      []string{"vmssee6c2000000", "vmssee6c2000001"},
			nodeName:    "vmssee6c2000000",
			faultDomain: 3,
			expected:    "3",
		},
		{
			description: "scaleSet should get availability zone for zoned nodes",
			scaleSet:    "ss",
			vmList:      []string{"vmssee6c2000000", "vmssee6c2000001"},
			nodeName:    "vmssee6c2000000",
			zone:        "2",
			faultDomain: 3,
			expected:    "westus-2",
		},
		{
			description: "scaleSet should return error for non-exist nodes",
			scaleSet:    "ss",
			faultDomain: 3,
			vmList:      []string{"vmssee6c2000000", "vmssee6c2000001"},
			nodeName:    "agente6c2000005",
			expectError: true,
		},
	}

	for _, test := range testCases {
		ss, err := newTestScaleSet(test.scaleSet, test.zone, test.faultDomain, test.vmList)
		assert.NoError(t, err, test.description)

		real, err := ss.GetZoneByNodeName(test.nodeName)
		if test.expectError {
			assert.Error(t, err, test.description)
			continue
		}

		assert.NoError(t, err, test.description)
		assert.Equal(t, test.expected, real.FailureDomain, test.description)
	}
}

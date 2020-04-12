// +build !providerless

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
	"strings"
	"testing"

	cloudprovider "k8s.io/cloud-provider"
	azcache "k8s.io/legacy-cloud-providers/azure/cache"
	"k8s.io/legacy-cloud-providers/azure/clients/interfaceclient/mockinterfaceclient"
	"k8s.io/legacy-cloud-providers/azure/clients/publicipclient/mockpublicipclient"
	"k8s.io/legacy-cloud-providers/azure/clients/vmclient/mockvmclient"
	"k8s.io/legacy-cloud-providers/azure/clients/vmssclient/mockvmssclient"
	"k8s.io/legacy-cloud-providers/azure/clients/vmssvmclient/mockvmssvmclient"
	"k8s.io/legacy-cloud-providers/azure/retry"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-12-01/compute"
	"github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-06-01/network"
	"github.com/Azure/go-autorest/autorest/to"
	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"
)

const (
	fakePrivateIP = "10.240.0.10"
	fakePublicIP  = "10.10.10.10"
)

func newTestScaleSet(ctrl *gomock.Controller) (*scaleSet, error) {
	return newTestScaleSetWithState(ctrl)
}

func newTestScaleSetWithState(ctrl *gomock.Controller) (*scaleSet, error) {
	cloud := GetTestCloud(ctrl)
	ss, err := newScaleSet(cloud)
	if err != nil {
		return nil, err
	}

	return ss.(*scaleSet), nil
}

func buildTestVirtualMachineEnv(ss *Cloud, scaleSetName, zone string, faultDomain int32, vmList []string, state string) ([]compute.VirtualMachineScaleSetVM, network.Interface, network.PublicIPAddress) {
	expectedVMSSVMs := make([]compute.VirtualMachineScaleSetVM, 0)
	expectedInterface := network.Interface{}
	expectedPIP := network.PublicIPAddress{}

	for i := range vmList {
		nodeName := vmList[i]
		ID := fmt.Sprintf("/subscriptions/script/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/%s/virtualMachines/%d", scaleSetName, i)
		interfaceID := fmt.Sprintf("/subscriptions/script/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/%s/virtualMachines/%d/networkInterfaces/%s", scaleSetName, i, nodeName)
		instanceID := fmt.Sprintf("%d", i)
		vmName := fmt.Sprintf("%s_%s", scaleSetName, instanceID)
		publicAddressID := fmt.Sprintf("/subscriptions/script/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/%s/virtualMachines/%d/networkInterfaces/%s/ipConfigurations/ipconfig1/publicIPAddresses/%s", scaleSetName, i, nodeName, nodeName)

		// set vmss virtual machine.
		networkInterfaces := []compute.NetworkInterfaceReference{
			{
				ID: &interfaceID,
			},
		}
		ipConfigurations := []compute.VirtualMachineScaleSetIPConfiguration{
			{
				Name: to.StringPtr("ipconfig1"),
				VirtualMachineScaleSetIPConfigurationProperties: &compute.VirtualMachineScaleSetIPConfigurationProperties{},
			},
		}
		networkConfigurations := []compute.VirtualMachineScaleSetNetworkConfiguration{
			{
				Name: to.StringPtr("ipconfig1"),
				ID:   to.StringPtr("fakeNetworkConfiguration"),
				VirtualMachineScaleSetNetworkConfigurationProperties: &compute.VirtualMachineScaleSetNetworkConfigurationProperties{
					IPConfigurations: &ipConfigurations,
				},
			},
		}
		vmssVM := compute.VirtualMachineScaleSetVM{
			VirtualMachineScaleSetVMProperties: &compute.VirtualMachineScaleSetVMProperties{
				ProvisioningState: to.StringPtr(state),
				OsProfile: &compute.OSProfile{
					ComputerName: &nodeName,
				},
				NetworkProfile: &compute.NetworkProfile{
					NetworkInterfaces: &networkInterfaces,
				},
				NetworkProfileConfiguration: &compute.VirtualMachineScaleSetVMNetworkProfileConfiguration{
					NetworkInterfaceConfigurations: &networkConfigurations,
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

		// set interfaces.
		expectedInterface = network.Interface{
			ID: &interfaceID,
			InterfacePropertiesFormat: &network.InterfacePropertiesFormat{
				IPConfigurations: &[]network.InterfaceIPConfiguration{
					{
						InterfaceIPConfigurationPropertiesFormat: &network.InterfaceIPConfigurationPropertiesFormat{
							Primary:          to.BoolPtr(true),
							PrivateIPAddress: to.StringPtr(fakePrivateIP),
							PublicIPAddress: &network.PublicIPAddress{
								ID: to.StringPtr(publicAddressID),
							},
						},
					},
				},
			},
		}

		// set public IPs.
		expectedPIP = network.PublicIPAddress{
			ID: to.StringPtr(publicAddressID),
			PublicIPAddressPropertiesFormat: &network.PublicIPAddressPropertiesFormat{
				IPAddress: to.StringPtr(fakePublicIP),
			},
		}

		expectedVMSSVMs = append(expectedVMSSVMs, vmssVM)
	}

	return expectedVMSSVMs, expectedInterface, expectedPIP
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
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

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
		ss, err := newTestScaleSet(ctrl)
		assert.NoError(t, err, test.description)

		mockVMSSClient := mockvmssclient.NewMockInterface(ctrl)
		mockVMSSVMClient := mockvmssvmclient.NewMockInterface(ctrl)
		ss.cloud.VirtualMachineScaleSetsClient = mockVMSSClient
		ss.cloud.VirtualMachineScaleSetVMsClient = mockVMSSVMClient

		expectedScaleSet := compute.VirtualMachineScaleSet{Name: &test.scaleSet}
		mockVMSSClient.EXPECT().List(gomock.Any(), gomock.Any()).Return([]compute.VirtualMachineScaleSet{expectedScaleSet}, nil).AnyTimes()

		expectedVMs, _, _ := buildTestVirtualMachineEnv(ss.cloud, test.scaleSet, "", 0, test.vmList, "")
		mockVMSSVMClient.EXPECT().List(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Return(expectedVMs, nil).AnyTimes()

		mockVMsClient := ss.cloud.VirtualMachinesClient.(*mockvmclient.MockInterface)
		mockVMsClient.EXPECT().List(gomock.Any(), gomock.Any()).Return([]compute.VirtualMachine{}, nil).AnyTimes()

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
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	testCases := []struct {
		description string
		scaleSet    string
		vmList      []string
		nodeName    string
		location    string
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
			description: "scaleSet should get availability zone in lower cases",
			scaleSet:    "ss",
			vmList:      []string{"vmssee6c2000000", "vmssee6c2000001"},
			nodeName:    "vmssee6c2000000",
			location:    "WestUS",
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
		cloud := GetTestCloud(ctrl)
		if test.location != "" {
			cloud.Location = test.location
		}
		ss, err := newTestScaleSet(ctrl)
		assert.NoError(t, err, test.description)

		mockVMSSClient := mockvmssclient.NewMockInterface(ctrl)
		mockVMSSVMClient := mockvmssvmclient.NewMockInterface(ctrl)
		ss.cloud.VirtualMachineScaleSetsClient = mockVMSSClient
		ss.cloud.VirtualMachineScaleSetVMsClient = mockVMSSVMClient

		expectedScaleSet := compute.VirtualMachineScaleSet{Name: &test.scaleSet}
		mockVMSSClient.EXPECT().List(gomock.Any(), gomock.Any()).Return([]compute.VirtualMachineScaleSet{expectedScaleSet}, nil).AnyTimes()

		expectedVMs, _, _ := buildTestVirtualMachineEnv(ss.cloud, test.scaleSet, test.zone, test.faultDomain, test.vmList, "")
		mockVMSSVMClient.EXPECT().List(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Return(expectedVMs, nil).AnyTimes()

		mockVMsClient := ss.cloud.VirtualMachinesClient.(*mockvmclient.MockInterface)
		mockVMsClient.EXPECT().List(gomock.Any(), gomock.Any()).Return([]compute.VirtualMachine{}, nil).AnyTimes()

		real, err := ss.GetZoneByNodeName(test.nodeName)
		if test.expectError {
			assert.Error(t, err, test.description)
			continue
		}

		assert.NoError(t, err, test.description)
		assert.Equal(t, test.expected, real.FailureDomain, test.description)
		assert.Equal(t, strings.ToLower(cloud.Location), real.Region, test.description)
	}
}

func TestGetIPByNodeName(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	testCases := []struct {
		description string
		scaleSet    string
		vmList      []string
		nodeName    string
		expected    []string
		expectError bool
	}{
		{
			description: "GetIPByNodeName should get node's privateIP and publicIP",
			scaleSet:    "ss",
			vmList:      []string{"vmssee6c2000000", "vmssee6c2000001"},
			nodeName:    "vmssee6c2000000",
			expected:    []string{fakePrivateIP, fakePublicIP},
		},
		{
			description: "GetIPByNodeName should return error for non-exist nodes",
			scaleSet:    "ss",
			vmList:      []string{"vmssee6c2000000", "vmssee6c2000001"},
			nodeName:    "agente6c2000005",
			expectError: true,
		},
	}

	for _, test := range testCases {
		ss, err := newTestScaleSet(ctrl)
		assert.NoError(t, err, test.description)

		mockVMSSClient := mockvmssclient.NewMockInterface(ctrl)
		mockVMSSVMClient := mockvmssvmclient.NewMockInterface(ctrl)
		mockInterfaceClient := mockinterfaceclient.NewMockInterface(ctrl)
		mockPIPClient := mockpublicipclient.NewMockInterface(ctrl)
		ss.cloud.VirtualMachineScaleSetsClient = mockVMSSClient
		ss.cloud.VirtualMachineScaleSetVMsClient = mockVMSSVMClient
		ss.cloud.InterfacesClient = mockInterfaceClient
		ss.cloud.PublicIPAddressesClient = mockPIPClient

		expectedScaleSet := compute.VirtualMachineScaleSet{Name: &test.scaleSet}
		mockVMSSClient.EXPECT().List(gomock.Any(), gomock.Any()).Return([]compute.VirtualMachineScaleSet{expectedScaleSet}, nil).AnyTimes()

		expectedVMs, expectedInterface, expectedPIP := buildTestVirtualMachineEnv(ss.cloud, test.scaleSet, "", 0, test.vmList, "")
		mockVMSSVMClient.EXPECT().List(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Return(expectedVMs, nil).AnyTimes()
		mockInterfaceClient.EXPECT().GetVirtualMachineScaleSetNetworkInterface(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Return(expectedInterface, nil).AnyTimes()
		mockPIPClient.EXPECT().GetVirtualMachineScaleSetPublicIPAddress(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Return(expectedPIP, nil).AnyTimes()

		mockVMsClient := ss.cloud.VirtualMachinesClient.(*mockvmclient.MockInterface)
		mockVMsClient.EXPECT().List(gomock.Any(), gomock.Any()).Return([]compute.VirtualMachine{}, nil).AnyTimes()

		privateIP, publicIP, err := ss.GetIPByNodeName(test.nodeName)
		if test.expectError {
			assert.Error(t, err, test.description)
			continue
		}

		assert.NoError(t, err, test.description)
		assert.Equal(t, test.expected, []string{privateIP, publicIP}, test.description)
	}
}

func TestGetNodeNameByIPConfigurationID(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	ipConfigurationIDTemplate := "/subscriptions/script/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/%s/virtualMachines/%s/networkInterfaces/%s/ipConfigurations/ipconfig1"

	testCases := []struct {
		description       string
		scaleSet          string
		vmList            []string
		ipConfigurationID string
		expected          string
		expectError       bool
	}{
		{
			description:       "getNodeNameByIPConfigurationID should get node's Name when the node is existing",
			scaleSet:          "scaleset1",
			ipConfigurationID: fmt.Sprintf(ipConfigurationIDTemplate, "scaleset1", "0", "scaleset1"),
			vmList:            []string{"vmssee6c2000000", "vmssee6c2000001"},
			expected:          "vmssee6c2000000",
		},
		{
			description:       "getNodeNameByIPConfigurationID should return error for non-exist nodes",
			scaleSet:          "scaleset2",
			ipConfigurationID: fmt.Sprintf(ipConfigurationIDTemplate, "scaleset2", "3", "scaleset1"),
			vmList:            []string{"vmssee6c2000002", "vmssee6c2000003"},
			expectError:       true,
		},
		{
			description:       "getNodeNameByIPConfigurationID should return error for wrong ipConfigurationID",
			scaleSet:          "scaleset3",
			ipConfigurationID: "invalid-configuration-id",
			vmList:            []string{"vmssee6c2000004", "vmssee6c2000005"},
			expectError:       true,
		},
	}

	for _, test := range testCases {
		ss, err := newTestScaleSet(ctrl)
		assert.NoError(t, err, test.description)

		mockVMSSClient := mockvmssclient.NewMockInterface(ctrl)
		mockVMSSVMClient := mockvmssvmclient.NewMockInterface(ctrl)
		ss.cloud.VirtualMachineScaleSetsClient = mockVMSSClient
		ss.cloud.VirtualMachineScaleSetVMsClient = mockVMSSVMClient

		expectedScaleSet := compute.VirtualMachineScaleSet{Name: &test.scaleSet}
		mockVMSSClient.EXPECT().List(gomock.Any(), gomock.Any()).Return([]compute.VirtualMachineScaleSet{expectedScaleSet}, nil).AnyTimes()

		expectedVMs, _, _ := buildTestVirtualMachineEnv(ss.cloud, test.scaleSet, "", 0, test.vmList, "")
		mockVMSSVMClient.EXPECT().List(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Return(expectedVMs, nil).AnyTimes()

		nodeName, err := ss.getNodeNameByIPConfigurationID(test.ipConfigurationID)
		if test.expectError {
			assert.Error(t, err, test.description)
			continue
		}

		assert.NoError(t, err, test.description)
		assert.Equal(t, test.expected, nodeName, test.description)
	}
}

func TestExtractResourceGroupByVMSSNicID(t *testing.T) {
	vmssNicIDTemplate := "/subscriptions/script/resourceGroups/%s/providers/Microsoft.Compute/virtualMachineScaleSets/%s/virtualMachines/%s/networkInterfaces/nic-0"

	testCases := []struct {
		description string
		vmssNicID   string
		expected    string
		expectError bool
	}{
		{
			description: "extractResourceGroupByVMSSNicID should get resource group name for vmss nic ID",
			vmssNicID:   fmt.Sprintf(vmssNicIDTemplate, "rg1", "vmss1", "0"),
			expected:    "rg1",
		},
		{
			description: "extractResourceGroupByVMSSNicID should return error for VM nic ID",
			vmssNicID:   "/subscriptions/script/resourceGroups/rg2/providers/Microsoft.Network/networkInterfaces/nic-0",
			expectError: true,
		},
		{
			description: "extractResourceGroupByVMSSNicID should return error for wrong vmss nic ID",
			vmssNicID:   "wrong-nic-id",
			expectError: true,
		},
	}

	for _, test := range testCases {
		resourceGroup, err := extractResourceGroupByVMSSNicID(test.vmssNicID)
		if test.expectError {
			assert.Error(t, err, test.description)
			continue
		}

		assert.NoError(t, err, test.description)
		assert.Equal(t, test.expected, resourceGroup, test.description)
	}
}

func TestGetVMSS(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	testCases := []struct {
		description     string
		existedVMSSName string
		vmssName        string
		vmssListError   *retry.Error
		expectedErr     error
	}{
		{
			description:     "getVMSS should return the correct VMSS",
			existedVMSSName: "vmss-1",
			vmssName:        "vmss-1",
		},
		{
			description:     "getVMSS should return cloudprovider.InstanceNotFound if there's no matching VMSS",
			existedVMSSName: "vmss-1",
			vmssName:        "vmss-2",
			expectedErr:     cloudprovider.InstanceNotFound,
		},
		{
			description:     "getVMSS should report an error if there's something wrong during an api call",
			existedVMSSName: "vmss-1",
			vmssName:        "vmss-1",
			vmssListError:   &retry.Error{RawError: fmt.Errorf("error during vmss list")},
			expectedErr:     fmt.Errorf("Retriable: false, RetryAfter: 0s, HTTPStatusCode: 0, RawError: error during vmss list"),
		},
	}

	for _, test := range testCases {
		ss, err := newTestScaleSet(ctrl)
		assert.NoError(t, err, test.description)

		mockVMSSClient := mockvmssclient.NewMockInterface(ctrl)
		ss.cloud.VirtualMachineScaleSetsClient = mockVMSSClient

		expected := compute.VirtualMachineScaleSet{Name: to.StringPtr(test.existedVMSSName)}
		mockVMSSClient.EXPECT().List(gomock.Any(), gomock.Any()).Return([]compute.VirtualMachineScaleSet{expected}, test.vmssListError).AnyTimes()

		actual, err := ss.getVMSS(test.vmssName, azcache.CacheReadTypeDefault)
		assert.Equal(t, test.expectedErr, err, test.description)
		if actual != nil {
			assert.Equal(t, expected, *actual, test.description)
		}
	}
}

func TestGetVmssVM(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	testCases := []struct {
		description      string
		nodeName         string
		existedNodeNames []string
		existedVMSSName  string
		expectedError    error
	}{
		{
			description:      "getVmssVM should return the correct name of vmss, the instance id of the node, and the corresponding vmss instance",
			nodeName:         "vmss-vm-000000",
			existedNodeNames: []string{"vmss-vm-000000"},
			existedVMSSName:  "vmss",
		},
		{
			description:      "getVmssVM should report an error of instance not found if there's no matches",
			nodeName:         "vmss-vm-000001",
			existedNodeNames: []string{"vmss-vm-000000"},
			existedVMSSName:  "vmss",
			expectedError:    cloudprovider.InstanceNotFound,
		},
	}

	for _, test := range testCases {
		ss, err := newTestScaleSet(ctrl)
		assert.NoError(t, err, test.description)

		expectedVMSS := compute.VirtualMachineScaleSet{Name: to.StringPtr(test.existedVMSSName)}
		mockVMSSClient := ss.cloud.VirtualMachineScaleSetsClient.(*mockvmssclient.MockInterface)
		mockVMSSClient.EXPECT().List(gomock.Any(), ss.ResourceGroup).Return([]compute.VirtualMachineScaleSet{expectedVMSS}, nil).AnyTimes()

		expectedVMSSVMs, _, _ := buildTestVirtualMachineEnv(ss.cloud, test.existedVMSSName, "", 0, test.existedNodeNames, "")
		var expectedVMSSVM compute.VirtualMachineScaleSetVM
		for _, expected := range expectedVMSSVMs {
			if strings.EqualFold(*expected.OsProfile.ComputerName, test.nodeName) {
				expectedVMSSVM = expected
			}
		}

		mockVMSSVMClient := ss.cloud.VirtualMachineScaleSetVMsClient.(*mockvmssvmclient.MockInterface)
		mockVMSSVMClient.EXPECT().List(gomock.Any(), ss.ResourceGroup, test.existedVMSSName, gomock.Any()).Return(expectedVMSSVMs, nil).AnyTimes()

		_, _, vmssVM, err := ss.getVmssVM(test.nodeName, azcache.CacheReadTypeDefault)
		if vmssVM != nil {
			assert.Equal(t, expectedVMSSVM, *vmssVM, test.description)
		}
		assert.Equal(t, test.expectedError, err, test.description)
	}
}

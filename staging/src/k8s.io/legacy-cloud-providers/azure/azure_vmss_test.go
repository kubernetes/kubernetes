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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
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
	fakePrivateIP        = "10.240.0.10"
	fakePublicIP         = "10.10.10.10"
	testVMSSName         = "vmss"
	testVMPowerState     = "PowerState/Running"
	testLBBackendpoolID0 = "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/loadBalancers/lb/backendAddressPools/backendpool-0"
	testLBBackendpoolID1 = "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/loadBalancers/lb/backendAddressPools/backendpool-1"
	testLBBackendpoolID2 = "/subscriptions/sub/resourceGroups/rg1/providers/Microsoft.Network/loadBalancers/lb/backendAddressPools/backendpool-2"
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

func buildTestVMSSWithLB(name, namePrefix string, lbBackendpoolIDs []string, ipv6 bool) compute.VirtualMachineScaleSet {
	lbBackendpools := make([]compute.SubResource, 0)
	for _, id := range lbBackendpoolIDs {
		lbBackendpools = append(lbBackendpools, compute.SubResource{ID: to.StringPtr(id)})
	}
	ipConfig := []compute.VirtualMachineScaleSetIPConfiguration{
		{
			VirtualMachineScaleSetIPConfigurationProperties: &compute.VirtualMachineScaleSetIPConfigurationProperties{
				LoadBalancerBackendAddressPools: &lbBackendpools,
			},
		},
	}
	if ipv6 {
		ipConfig = append(ipConfig, compute.VirtualMachineScaleSetIPConfiguration{
			VirtualMachineScaleSetIPConfigurationProperties: &compute.VirtualMachineScaleSetIPConfigurationProperties{
				LoadBalancerBackendAddressPools: &lbBackendpools,
				PrivateIPAddressVersion:         compute.IPv6,
			},
		})
	}

	expectedVMSS := compute.VirtualMachineScaleSet{
		Name: &name,
		VirtualMachineScaleSetProperties: &compute.VirtualMachineScaleSetProperties{
			ProvisioningState: to.StringPtr("Running"),
			VirtualMachineProfile: &compute.VirtualMachineScaleSetVMProfile{
				OsProfile: &compute.VirtualMachineScaleSetOSProfile{
					ComputerNamePrefix: &namePrefix,
				},
				NetworkProfile: &compute.VirtualMachineScaleSetNetworkProfile{
					NetworkInterfaceConfigurations: &[]compute.VirtualMachineScaleSetNetworkConfiguration{
						{
							VirtualMachineScaleSetNetworkConfigurationProperties: &compute.VirtualMachineScaleSetNetworkConfigurationProperties{
								Primary:          to.BoolPtr(true),
								IPConfigurations: &ipConfig,
							},
						},
					},
				},
			},
		},
	}

	return expectedVMSS
}

func buildTestVMSS(name, computerNamePrefix string) compute.VirtualMachineScaleSet {
	return compute.VirtualMachineScaleSet{
		Name: &name,
		VirtualMachineScaleSetProperties: &compute.VirtualMachineScaleSetProperties{
			VirtualMachineProfile: &compute.VirtualMachineScaleSetVMProfile{
				OsProfile: &compute.VirtualMachineScaleSetOSProfile{
					ComputerNamePrefix: &computerNamePrefix,
				},
			},
		},
	}
}

func buildTestVirtualMachineEnv(ss *Cloud, scaleSetName, zone string, faultDomain int32, vmList []string, state string, isIPv6 bool) ([]compute.VirtualMachineScaleSetVM, network.Interface, network.PublicIPAddress) {
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
				NetworkInterfaceReferenceProperties: &compute.NetworkInterfaceReferenceProperties{
					Primary: to.BoolPtr(true),
				},
			},
		}
		ipConfigurations := []compute.VirtualMachineScaleSetIPConfiguration{
			{
				Name: to.StringPtr("ipconfig1"),
				VirtualMachineScaleSetIPConfigurationProperties: &compute.VirtualMachineScaleSetIPConfigurationProperties{
					Primary:                         to.BoolPtr(true),
					LoadBalancerBackendAddressPools: &[]compute.SubResource{{ID: to.StringPtr(testLBBackendpoolID0)}},
				},
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
		if isIPv6 {
			networkConfigurations = append(networkConfigurations, compute.VirtualMachineScaleSetNetworkConfiguration{
				Name: to.StringPtr("ipconfig1v6"),
				ID:   to.StringPtr("fakeNetworkConfigurationIPv6"),
				VirtualMachineScaleSetNetworkConfigurationProperties: &compute.VirtualMachineScaleSetNetworkConfigurationProperties{
					IPConfigurations: &[]compute.VirtualMachineScaleSetIPConfiguration{
						{
							Name: to.StringPtr("ipconfig1"),
							VirtualMachineScaleSetIPConfigurationProperties: &compute.VirtualMachineScaleSetIPConfigurationProperties{
								Primary:                         to.BoolPtr(false),
								LoadBalancerBackendAddressPools: &[]compute.SubResource{{ID: to.StringPtr(testLBBackendpoolID0)}},
								PrivateIPAddressVersion:         compute.IPv6,
							},
						},
					},
				},
			})
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
					Statuses: &[]compute.InstanceViewStatus{
						{Code: to.StringPtr(testVMPowerState)},
					},
				},
			},
			ID:         &ID,
			InstanceID: &instanceID,
			Name:       &vmName,
			Location:   &ss.Location,
			Sku:        &compute.Sku{Name: to.StringPtr("sku")},
		}
		if zone != "" {
			zones := []string{zone}
			vmssVM.Zones = &zones
		}

		// set interfaces.
		expectedInterface = network.Interface{
			Name: to.StringPtr("nic"),
			ID:   &interfaceID,
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

func TestGetNodeIdentityByNodeName(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	testCases := []struct {
		description  string
		vmList       []string
		nodeName     string
		expected     *nodeIdentity
		scaleSet     string
		computerName string
		expectError  bool
	}{
		{
			description:  "scaleSet should get node identity by node name",
			vmList:       []string{"vmssee6c2000000", "vmssee6c2000001"},
			nodeName:     "vmssee6c2000001",
			scaleSet:     "vmssee6c2",
			computerName: "vmssee6c2",
			expected:     &nodeIdentity{"rg", "vmssee6c2", "vmssee6c2000001"},
		},
		{
			description:  "scaleSet should get node identity when computerNamePrefix differs from vmss name",
			vmList:       []string{"vmssee6c2000000", "vmssee6c2000001"},
			nodeName:     "vmssee6c2000001",
			scaleSet:     "ss",
			computerName: "vmssee6c2",
			expected:     &nodeIdentity{"rg", "ss", "vmssee6c2000001"},
		},
		{
			description:  "scaleSet should get node identity by node name with upper cases hostname",
			vmList:       []string{"VMSSEE6C2000000", "VMSSEE6C2000001"},
			nodeName:     "vmssee6c2000001",
			scaleSet:     "ss",
			computerName: "vmssee6c2",
			expected:     &nodeIdentity{"rg", "ss", "vmssee6c2000001"},
		},
		{
			description:  "scaleSet should not get node identity for non-existing nodes",
			vmList:       []string{"vmssee6c2000000", "vmssee6c2000001"},
			nodeName:     "agente6c2000005",
			scaleSet:     "ss",
			computerName: "vmssee6c2",
			expectError:  true,
		},
	}

	for _, test := range testCases {
		ss, err := newTestScaleSet(ctrl)
		assert.NoError(t, err, test.description)

		mockVMSSClient := mockvmssclient.NewMockInterface(ctrl)
		mockVMSSVMClient := mockvmssvmclient.NewMockInterface(ctrl)
		ss.cloud.VirtualMachineScaleSetsClient = mockVMSSClient
		ss.cloud.VirtualMachineScaleSetVMsClient = mockVMSSVMClient

		expectedScaleSet := buildTestVMSS(test.scaleSet, test.computerName)
		mockVMSSClient.EXPECT().List(gomock.Any(), gomock.Any()).Return([]compute.VirtualMachineScaleSet{expectedScaleSet}, nil).AnyTimes()

		expectedVMs, _, _ := buildTestVirtualMachineEnv(ss.cloud, test.scaleSet, "", 0, test.vmList, "", false)
		mockVMSSVMClient.EXPECT().List(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Return(expectedVMs, nil).AnyTimes()

		mockVMsClient := ss.cloud.VirtualMachinesClient.(*mockvmclient.MockInterface)
		mockVMsClient.EXPECT().List(gomock.Any(), gomock.Any()).Return([]compute.VirtualMachine{}, nil).AnyTimes()

		nodeID, err := ss.getNodeIdentityByNodeName(test.nodeName, azcache.CacheReadTypeDefault)
		if test.expectError {
			assert.Error(t, err, test.description)
			continue
		}

		assert.NoError(t, err, test.description)
		assert.Equal(t, test.expected, nodeID, test.description)
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

		expectedScaleSet := buildTestVMSS(test.scaleSet, "vmssee6c2")
		mockVMSSClient.EXPECT().List(gomock.Any(), gomock.Any()).Return([]compute.VirtualMachineScaleSet{expectedScaleSet}, nil).AnyTimes()

		expectedVMs, _, _ := buildTestVirtualMachineEnv(ss.cloud, test.scaleSet, "", 0, test.vmList, "", false)
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

		expectedScaleSet := buildTestVMSS(test.scaleSet, "vmssee6c2")
		mockVMSSClient.EXPECT().List(gomock.Any(), gomock.Any()).Return([]compute.VirtualMachineScaleSet{expectedScaleSet}, nil).AnyTimes()

		expectedVMs, _, _ := buildTestVirtualMachineEnv(ss.cloud, test.scaleSet, test.zone, test.faultDomain, test.vmList, "", false)
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

		expectedScaleSet := buildTestVMSS(test.scaleSet, "vmssee6c2")
		mockVMSSClient.EXPECT().List(gomock.Any(), gomock.Any()).Return([]compute.VirtualMachineScaleSet{expectedScaleSet}, nil).AnyTimes()

		expectedVMs, expectedInterface, expectedPIP := buildTestVirtualMachineEnv(ss.cloud, test.scaleSet, "", 0, test.vmList, "", false)
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
		description          string
		scaleSet             string
		vmList               []string
		ipConfigurationID    string
		expectedNodeName     string
		expectedScaleSetName string
		expectError          bool
	}{
		{
			description:          "GetNodeNameByIPConfigurationID should get node's Name when the node is existing",
			scaleSet:             "scaleset1",
			ipConfigurationID:    fmt.Sprintf(ipConfigurationIDTemplate, "scaleset1", "0", "scaleset1"),
			vmList:               []string{"vmssee6c2000000", "vmssee6c2000001"},
			expectedNodeName:     "vmssee6c2000000",
			expectedScaleSetName: "scaleset1",
		},
		{
			description:       "GetNodeNameByIPConfigurationID should return error for non-exist nodes",
			scaleSet:          "scaleset2",
			ipConfigurationID: fmt.Sprintf(ipConfigurationIDTemplate, "scaleset2", "3", "scaleset1"),
			vmList:            []string{"vmssee6c2000002", "vmssee6c2000003"},
			expectError:       true,
		},
		{
			description:       "GetNodeNameByIPConfigurationID should return error for wrong ipConfigurationID",
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

		expectedScaleSet := buildTestVMSS(test.scaleSet, "vmssee6c2")
		mockVMSSClient.EXPECT().List(gomock.Any(), gomock.Any()).Return([]compute.VirtualMachineScaleSet{expectedScaleSet}, nil).AnyTimes()

		expectedVMs, _, _ := buildTestVirtualMachineEnv(ss.cloud, test.scaleSet, "", 0, test.vmList, "", false)
		mockVMSSVMClient.EXPECT().List(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Return(expectedVMs, nil).AnyTimes()

		nodeName, scalesetName, err := ss.GetNodeNameByIPConfigurationID(test.ipConfigurationID)
		if test.expectError {
			assert.Error(t, err, test.description)
			continue
		}

		assert.NoError(t, err, test.description)
		assert.Equal(t, test.expectedNodeName, nodeName, test.description)
		assert.Equal(t, test.expectedScaleSetName, scalesetName, test.description)
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

		expected := compute.VirtualMachineScaleSet{
			Name: to.StringPtr(test.existedVMSSName),
			VirtualMachineScaleSetProperties: &compute.VirtualMachineScaleSetProperties{
				VirtualMachineProfile: &compute.VirtualMachineScaleSetVMProfile{},
			},
		}
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
			existedVMSSName:  testVMSSName,
		},
		{
			description:      "getVmssVM should report an error of instance not found if there's no matches",
			nodeName:         "vmss-vm-000001",
			existedNodeNames: []string{"vmss-vm-000000"},
			existedVMSSName:  testVMSSName,
			expectedError:    cloudprovider.InstanceNotFound,
		},
	}

	for _, test := range testCases {
		ss, err := newTestScaleSet(ctrl)
		assert.NoError(t, err, test.description)

		expectedVMSS := buildTestVMSS(test.existedVMSSName, "vmss-vm-")
		mockVMSSClient := ss.cloud.VirtualMachineScaleSetsClient.(*mockvmssclient.MockInterface)
		mockVMSSClient.EXPECT().List(gomock.Any(), ss.ResourceGroup).Return([]compute.VirtualMachineScaleSet{expectedVMSS}, nil).AnyTimes()

		expectedVMSSVMs, _, _ := buildTestVirtualMachineEnv(ss.cloud, test.existedVMSSName, "", 0, test.existedNodeNames, "", false)
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

func TestGetPowerStatusByNodeName(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	testCases := []struct {
		description        string
		vmList             []string
		nilStatus          bool
		expectedPowerState string
		expectedErr        error
	}{
		{
			description:        "GetPowerStatusByNodeName should return the correct power state",
			vmList:             []string{"vmss-vm-000001"},
			expectedPowerState: "Running",
		},
		{
			description:        "GetPowerStatusByNodeName should return vmPowerStateStopped when the vm.InstanceView.Statuses is nil",
			vmList:             []string{"vmss-vm-000001"},
			nilStatus:          true,
			expectedPowerState: vmPowerStateStopped,
		},
	}

	for _, test := range testCases {
		ss, err := newTestScaleSet(ctrl)
		assert.NoError(t, err, "unexpected error when creating test VMSS")

		expectedVMSS := buildTestVMSS(testVMSSName, "vmss-vm-")
		mockVMSSClient := ss.cloud.VirtualMachineScaleSetsClient.(*mockvmssclient.MockInterface)
		mockVMSSClient.EXPECT().List(gomock.Any(), ss.ResourceGroup).Return([]compute.VirtualMachineScaleSet{expectedVMSS}, nil).AnyTimes()

		expectedVMSSVMs, _, _ := buildTestVirtualMachineEnv(ss.cloud, testVMSSName, "", 0, test.vmList, "", false)
		mockVMSSVMClient := ss.cloud.VirtualMachineScaleSetVMsClient.(*mockvmssvmclient.MockInterface)
		if test.nilStatus {
			expectedVMSSVMs[0].InstanceView.Statuses = nil
		}
		mockVMSSVMClient.EXPECT().List(gomock.Any(), ss.ResourceGroup, testVMSSName, gomock.Any()).Return(expectedVMSSVMs, nil).AnyTimes()

		mockVMsClient := ss.cloud.VirtualMachinesClient.(*mockvmclient.MockInterface)
		mockVMsClient.EXPECT().List(gomock.Any(), gomock.Any()).Return([]compute.VirtualMachine{}, nil).AnyTimes()

		powerState, err := ss.GetPowerStatusByNodeName("vmss-vm-000001")
		assert.Equal(t, test.expectedErr, err, test.description+", but an error occurs")
		assert.Equal(t, test.expectedPowerState, powerState, test.description)
	}
}

func TestGetVmssVMByInstanceID(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	testCases := []struct {
		description string
		instanceID  string
		vmList      []string
		expectedErr error
	}{
		{
			description: "GetVmssVMByInstanceID should return the correct VMSS VM",
			instanceID:  "0",
			vmList:      []string{"vmss-vm-000000"},
		},
	}

	for _, test := range testCases {
		ss, err := newTestScaleSet(ctrl)
		assert.NoError(t, err, "unexpected error when creating test VMSS")

		expectedVMSS := compute.VirtualMachineScaleSet{
			Name: to.StringPtr(testVMSSName),
			VirtualMachineScaleSetProperties: &compute.VirtualMachineScaleSetProperties{
				VirtualMachineProfile: &compute.VirtualMachineScaleSetVMProfile{},
			},
		}
		mockVMSSClient := ss.cloud.VirtualMachineScaleSetsClient.(*mockvmssclient.MockInterface)
		mockVMSSClient.EXPECT().List(gomock.Any(), ss.ResourceGroup).Return([]compute.VirtualMachineScaleSet{expectedVMSS}, nil).AnyTimes()

		expectedVMSSVMs, _, _ := buildTestVirtualMachineEnv(ss.cloud, testVMSSName, "", 0, test.vmList, "", false)
		mockVMSSVMClient := ss.cloud.VirtualMachineScaleSetVMsClient.(*mockvmssvmclient.MockInterface)
		mockVMSSVMClient.EXPECT().List(gomock.Any(), ss.ResourceGroup, testVMSSName, gomock.Any()).Return(expectedVMSSVMs, nil).AnyTimes()

		vm, err := ss.getVmssVMByInstanceID(ss.ResourceGroup, testVMSSName, test.instanceID, azcache.CacheReadTypeDefault)
		assert.Equal(t, test.expectedErr, err, test.description+", but an error occurs")
		assert.Equal(t, expectedVMSSVMs[0], *vm, test.description)
	}
}

func TestGetInstanceTypeByNodeName(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	testCases := []struct {
		description  string
		vmList       []string
		vmClientErr  *retry.Error
		expectedType string
		expectedErr  error
	}{
		{
			description:  "GetInstanceTypeByNodeName should return the correct instance type",
			vmList:       []string{"vmss-vm-000000"},
			expectedType: "sku",
		},
		{
			description:  "GetInstanceTypeByNodeName should report the error that occurs",
			vmList:       []string{"vmss-vm-000000"},
			vmClientErr:  &retry.Error{RawError: fmt.Errorf("error")},
			expectedType: "",
			expectedErr:  fmt.Errorf("Retriable: false, RetryAfter: 0s, HTTPStatusCode: 0, RawError: error"),
		},
	}

	for _, test := range testCases {
		ss, err := newTestScaleSet(ctrl)
		assert.NoError(t, err, "unexpected error when creating test VMSS")

		expectedVMSS := buildTestVMSS(testVMSSName, "vmss-vm-")
		mockVMSSClient := ss.cloud.VirtualMachineScaleSetsClient.(*mockvmssclient.MockInterface)
		mockVMSSClient.EXPECT().List(gomock.Any(), ss.ResourceGroup).Return([]compute.VirtualMachineScaleSet{expectedVMSS}, nil).AnyTimes()

		expectedVMSSVMs, _, _ := buildTestVirtualMachineEnv(ss.cloud, testVMSSName, "", 0, test.vmList, "", false)
		mockVMSSVMClient := ss.cloud.VirtualMachineScaleSetVMsClient.(*mockvmssvmclient.MockInterface)
		mockVMSSVMClient.EXPECT().List(gomock.Any(), ss.ResourceGroup, testVMSSName, gomock.Any()).Return(expectedVMSSVMs, nil).AnyTimes()

		mockVMClient := ss.cloud.VirtualMachinesClient.(*mockvmclient.MockInterface)
		mockVMClient.EXPECT().List(gomock.Any(), gomock.Any()).Return(nil, test.vmClientErr).AnyTimes()

		sku, err := ss.GetInstanceTypeByNodeName("vmss-vm-000000")
		assert.Equal(t, test.expectedErr, err, test.description+", but an error occurs")
		assert.Equal(t, test.expectedType, sku, test.description)
	}
}

func TestGetPrimaryInterfaceID(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	testCases := []struct {
		description       string
		existedInterfaces []compute.NetworkInterfaceReference
		expectedID        string
		expectedErr       error
	}{
		{
			description: "GetPrimaryInterfaceID should return the ID of the primary NIC on the VMSS VM",
			existedInterfaces: []compute.NetworkInterfaceReference{
				{
					ID: to.StringPtr("1"),
					NetworkInterfaceReferenceProperties: &compute.NetworkInterfaceReferenceProperties{
						Primary: to.BoolPtr(true),
					},
				},
				{ID: to.StringPtr("2")},
			},
			expectedID: "1",
		},
		{
			description: "GetPrimaryInterfaceID should report an error if there's no primary NIC on the VMSS VM",
			existedInterfaces: []compute.NetworkInterfaceReference{
				{
					ID: to.StringPtr("1"),
					NetworkInterfaceReferenceProperties: &compute.NetworkInterfaceReferenceProperties{
						Primary: to.BoolPtr(false),
					},
				},
				{
					ID: to.StringPtr("2"),
					NetworkInterfaceReferenceProperties: &compute.NetworkInterfaceReferenceProperties{
						Primary: to.BoolPtr(false),
					},
				},
			},
			expectedErr: fmt.Errorf("failed to find a primary nic for the vm. vmname=\"vm\""),
		},
		{
			description:       "GetPrimaryInterfaceID should report an error if there's no network interface on the VMSS VM",
			existedInterfaces: []compute.NetworkInterfaceReference{},
			expectedErr:       fmt.Errorf("failed to find the network interfaces for vm vm"),
		},
	}

	for _, test := range testCases {
		ss, err := newTestScaleSet(ctrl)
		assert.NoError(t, err, "unexpected error when creating test VMSS")

		vm := compute.VirtualMachineScaleSetVM{
			Name: to.StringPtr("vm"),
			VirtualMachineScaleSetVMProperties: &compute.VirtualMachineScaleSetVMProperties{
				NetworkProfile: &compute.NetworkProfile{
					NetworkInterfaces: &test.existedInterfaces,
				},
			},
		}
		if len(test.existedInterfaces) == 0 {
			vm.VirtualMachineScaleSetVMProperties.NetworkProfile = nil
		}

		id, err := ss.getPrimaryInterfaceID(vm)
		assert.Equal(t, test.expectedErr, err, test.description+", but an error occurs")
		assert.Equal(t, test.expectedID, id, test.description)
	}
}

func TestGetPrimaryInterface(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	testCases := []struct {
		description         string
		nodeName            string
		vmList              []string
		vmClientErr         *retry.Error
		vmssClientErr       *retry.Error
		nicClientErr        *retry.Error
		hasPrimaryInterface bool
		isInvalidNICID      bool
		expectedErr         error
	}{
		{
			description:         "GetPrimaryInterface should return the correct network interface",
			nodeName:            "vmss-vm-000000",
			vmList:              []string{"vmss-vm-000000"},
			hasPrimaryInterface: true,
		},
		{
			description:         "GetPrimaryInterface should report the error if vm client returns retry error",
			nodeName:            "vmss-vm-000000",
			vmList:              []string{"vmss-vm-000000"},
			hasPrimaryInterface: true,
			vmClientErr:         &retry.Error{RawError: fmt.Errorf("error")},
			expectedErr:         fmt.Errorf("Retriable: false, RetryAfter: 0s, HTTPStatusCode: 0, RawError: error"),
		},
		{
			description:         "GetPrimaryInterface should report the error if vmss client returns retry error",
			nodeName:            "vmss-vm-000000",
			vmList:              []string{"vmss-vm-000000"},
			hasPrimaryInterface: true,
			vmssClientErr:       &retry.Error{RawError: fmt.Errorf("error")},
			expectedErr:         fmt.Errorf("Retriable: false, RetryAfter: 0s, HTTPStatusCode: 0, RawError: error"),
		},
		{
			description:         "GetPrimaryInterface should report the error if there is no primary interface",
			nodeName:            "vmss-vm-000000",
			vmList:              []string{"vmss-vm-000000"},
			hasPrimaryInterface: false,
			expectedErr:         fmt.Errorf("failed to find a primary nic for the vm. vmname=\"vmss_0\""),
		},
		{
			description:         "GetPrimaryInterface should report the error if the id of the primary nic is not valid",
			nodeName:            "vmss-vm-000000",
			vmList:              []string{"vmss-vm-000000"},
			isInvalidNICID:      true,
			hasPrimaryInterface: true,
			expectedErr:         fmt.Errorf("resource name was missing from identifier"),
		},
		{
			description:         "GetPrimaryInterface should report the error if nic client returns retry error",
			nodeName:            "vmss-vm-000000",
			vmList:              []string{"vmss-vm-000000"},
			hasPrimaryInterface: true,
			nicClientErr:        &retry.Error{RawError: fmt.Errorf("error")},
			expectedErr:         fmt.Errorf("Retriable: false, RetryAfter: 0s, HTTPStatusCode: 0, RawError: error"),
		},
		{
			description:         "GetPrimaryInterface should report the error if the NIC instance is not found",
			nodeName:            "vmss-vm-000000",
			vmList:              []string{"vmss-vm-000000"},
			hasPrimaryInterface: true,
			nicClientErr:        &retry.Error{HTTPStatusCode: 404, RawError: fmt.Errorf("not found")},
			expectedErr:         cloudprovider.InstanceNotFound,
		},
	}

	for _, test := range testCases {
		ss, err := newTestScaleSet(ctrl)
		assert.NoError(t, err, "unexpected error when creating test VMSS")

		expectedVMSS := buildTestVMSS(testVMSSName, "vmss-vm-")
		mockVMSSClient := ss.cloud.VirtualMachineScaleSetsClient.(*mockvmssclient.MockInterface)
		mockVMSSClient.EXPECT().List(gomock.Any(), ss.ResourceGroup).Return([]compute.VirtualMachineScaleSet{expectedVMSS}, test.vmssClientErr).AnyTimes()

		expectedVMSSVMs, expectedInterface, _ := buildTestVirtualMachineEnv(ss.cloud, testVMSSName, "", 0, test.vmList, "", false)
		if !test.hasPrimaryInterface {
			networkInterfaces := *expectedVMSSVMs[0].NetworkProfile.NetworkInterfaces
			networkInterfaces[0].Primary = to.BoolPtr(false)
			networkInterfaces = append(networkInterfaces, compute.NetworkInterfaceReference{
				NetworkInterfaceReferenceProperties: &compute.NetworkInterfaceReferenceProperties{Primary: to.BoolPtr(false)},
			})
			expectedVMSSVMs[0].NetworkProfile.NetworkInterfaces = &networkInterfaces
		}
		if test.isInvalidNICID {
			networkInterfaces := *expectedVMSSVMs[0].NetworkProfile.NetworkInterfaces
			networkInterfaces[0].ID = to.StringPtr("invalid/id/")
			expectedVMSSVMs[0].NetworkProfile.NetworkInterfaces = &networkInterfaces
		}
		mockVMSSVMClient := ss.cloud.VirtualMachineScaleSetVMsClient.(*mockvmssvmclient.MockInterface)
		mockVMSSVMClient.EXPECT().List(gomock.Any(), ss.ResourceGroup, testVMSSName, gomock.Any()).Return(expectedVMSSVMs, nil).AnyTimes()

		mockVMClient := ss.cloud.VirtualMachinesClient.(*mockvmclient.MockInterface)
		mockVMClient.EXPECT().List(gomock.Any(), gomock.Any()).Return(nil, test.vmClientErr).AnyTimes()

		mockInterfaceClient := ss.cloud.InterfacesClient.(*mockinterfaceclient.MockInterface)
		mockInterfaceClient.EXPECT().GetVirtualMachineScaleSetNetworkInterface(gomock.Any(), ss.ResourceGroup, testVMSSName, "0", test.nodeName, gomock.Any()).Return(expectedInterface, test.nicClientErr).AnyTimes()
		expectedInterface.Location = &ss.Location

		if test.vmClientErr != nil || test.vmssClientErr != nil || test.nicClientErr != nil || !test.hasPrimaryInterface || test.isInvalidNICID {
			expectedInterface = network.Interface{}
		}

		nic, err := ss.GetPrimaryInterface(test.nodeName)
		assert.Equal(t, test.expectedErr, err, test.description+", but an error occurs")
		assert.Equal(t, expectedInterface, nic, test.description)
	}
}

func TestGetVMSSPublicIPAddress(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	testCases := []struct {
		description  string
		pipClientErr *retry.Error
		pipName      string
		found        bool
		expectedErr  error
	}{
		{
			description: "GetVMSSPublicIPAddress should return the correct public IP address",
			pipName:     "pip",
			found:       true,
		},
		{
			description:  "GetVMSSPublicIPAddress should report the error if the pip client returns retry.Error",
			pipName:      "pip",
			found:        false,
			pipClientErr: &retry.Error{RawError: fmt.Errorf("error")},
			expectedErr:  fmt.Errorf("Retriable: false, RetryAfter: 0s, HTTPStatusCode: 0, RawError: error"),
		},
		{
			description: "GetVMSSPublicIPAddress should not report errors if the pip cannot be found",
			pipName:     "pip-1",
			found:       false,
		},
	}

	for _, test := range testCases {
		ss, err := newTestScaleSet(ctrl)
		assert.NoError(t, err, "unexpected error when creating test VMSS")

		mockPIPClient := ss.cloud.PublicIPAddressesClient.(*mockpublicipclient.MockInterface)
		mockPIPClient.EXPECT().GetVirtualMachineScaleSetPublicIPAddress(gomock.Any(), ss.ResourceGroup, testVMSSName, "0", "nic", "ip", "pip", "").Return(network.PublicIPAddress{}, test.pipClientErr).AnyTimes()
		mockPIPClient.EXPECT().GetVirtualMachineScaleSetPublicIPAddress(gomock.Any(), ss.ResourceGroup, testVMSSName, "0", "nic", "ip", gomock.Not("pip"), "").Return(network.PublicIPAddress{}, &retry.Error{HTTPStatusCode: 404, RawError: fmt.Errorf("not found")}).AnyTimes()

		_, found, err := ss.getVMSSPublicIPAddress(ss.ResourceGroup, testVMSSName, "0", "nic", "ip", test.pipName)
		assert.Equal(t, test.expectedErr, err, test.description+", but an error occurs")
		assert.Equal(t, test.found, found, test.description)
	}
}

func TestGetPrivateIPsByNodeName(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	testCases := []struct {
		description        string
		nodeName           string
		vmList             []string
		isNilIPConfigs     bool
		vmClientErr        *retry.Error
		expectedPrivateIPs []string
		expectedErr        error
	}{
		{
			description:        "GetPrivateIPsByNodeName should return the correct private IPs",
			nodeName:           "vmss-vm-000000",
			vmList:             []string{"vmss-vm-000000"},
			expectedPrivateIPs: []string{fakePrivateIP},
		},
		{
			description:        "GetPrivateIPsByNodeName should report the error if the ipconfig of the nic is nil",
			nodeName:           "vmss-vm-000000",
			vmList:             []string{"vmss-vm-000000"},
			isNilIPConfigs:     true,
			expectedPrivateIPs: []string{},
			expectedErr:        fmt.Errorf("nic.IPConfigurations for nic (nicname=\"nic\") is nil"),
		},
		{
			description:        "GetPrivateIPsByNodeName should report the error if error happens during GetPrimaryInterface",
			nodeName:           "vmss-vm-000000",
			vmList:             []string{"vmss-vm-000000"},
			vmClientErr:        &retry.Error{RawError: fmt.Errorf("error")},
			expectedPrivateIPs: []string{},
			expectedErr:        fmt.Errorf("Retriable: false, RetryAfter: 0s, HTTPStatusCode: 0, RawError: error"),
		},
	}

	for _, test := range testCases {
		ss, err := newTestScaleSet(ctrl)
		assert.NoError(t, err, "unexpected error when creating test VMSS")

		expectedVMSS := buildTestVMSS(testVMSSName, "vmss-vm-")
		mockVMSSClient := ss.cloud.VirtualMachineScaleSetsClient.(*mockvmssclient.MockInterface)
		mockVMSSClient.EXPECT().List(gomock.Any(), ss.ResourceGroup).Return([]compute.VirtualMachineScaleSet{expectedVMSS}, nil).AnyTimes()

		expectedVMSSVMs, expectedInterface, _ := buildTestVirtualMachineEnv(ss.cloud, testVMSSName, "", 0, test.vmList, "", false)

		mockVMSSVMClient := ss.cloud.VirtualMachineScaleSetVMsClient.(*mockvmssvmclient.MockInterface)
		mockVMSSVMClient.EXPECT().List(gomock.Any(), ss.ResourceGroup, testVMSSName, gomock.Any()).Return(expectedVMSSVMs, nil).AnyTimes()

		mockVMClient := ss.cloud.VirtualMachinesClient.(*mockvmclient.MockInterface)
		mockVMClient.EXPECT().List(gomock.Any(), gomock.Any()).Return(nil, test.vmClientErr).AnyTimes()

		if test.isNilIPConfigs {
			expectedInterface.IPConfigurations = nil
		}
		mockInterfaceClient := ss.cloud.InterfacesClient.(*mockinterfaceclient.MockInterface)
		mockInterfaceClient.EXPECT().GetVirtualMachineScaleSetNetworkInterface(gomock.Any(), ss.ResourceGroup, testVMSSName, "0", test.nodeName, gomock.Any()).Return(expectedInterface, nil).AnyTimes()

		privateIPs, err := ss.GetPrivateIPsByNodeName(test.nodeName)
		assert.Equal(t, test.expectedErr, err, test.description+", but an error occurs")
		assert.Equal(t, test.expectedPrivateIPs, privateIPs, test.description)
	}
}

func TestGetVmssMachineID(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	ss, err := newTestScaleSet(ctrl)
	assert.NoError(t, err, "unexpected error when creating test VMSS")

	subscriptionID, resourceGroup, scaleSetName, instanceID := "sub", "RG", "vmss", "id"
	VMSSMachineID := ss.cloud.getVmssMachineID(subscriptionID, resourceGroup, scaleSetName, instanceID)
	expectedVMSSMachineID := fmt.Sprintf(vmssMachineIDTemplate, subscriptionID, strings.ToLower(resourceGroup), scaleSetName, instanceID)
	assert.Equal(t, expectedVMSSMachineID, VMSSMachineID, "GetVmssMachineID should return the correct VMSS machine ID")
}

func TestExtractScaleSetNameByProviderID(t *testing.T) {
	providerID := "/subscriptions/script/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss/virtualMachines/vmss-vm-000000"
	vmssName, err := extractScaleSetNameByProviderID(providerID)
	assert.Nil(t, err, fmt.Errorf("unexpected error %v happened", err))
	assert.Equal(t, "vmss", vmssName, "extractScaleSetNameByProviderID should return the correct vmss name")

	providerID = "/invalid/id"
	vmssName, err = extractScaleSetNameByProviderID(providerID)
	assert.Equal(t, ErrorNotVmssInstance, err, "extractScaleSetNameByProviderID should return the error of ErrorNotVmssInstance if the providerID is not a valid vmss ID")
	assert.Equal(t, "", vmssName, "extractScaleSetNameByProviderID should return an empty string")
}

func TestExtractResourceGroupByProviderID(t *testing.T) {
	providerID := "/subscriptions/script/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss/virtualMachines/vmss-vm-000000"
	vmssName, err := extractResourceGroupByProviderID(providerID)
	assert.Nil(t, err, fmt.Errorf("unexpected error %v happened", err))
	assert.Equal(t, "rg", vmssName, "extractScaleSetNameByProviderID should return the correct vmss name")

	providerID = "/invalid/id"
	vmssName, err = extractResourceGroupByProviderID(providerID)
	assert.Equal(t, ErrorNotVmssInstance, err, "extractScaleSetNameByProviderID should return the error of ErrorNotVmssInstance if the providerID is not a valid vmss ID")
	assert.Equal(t, "", vmssName, "extractScaleSetNameByProviderID should return an empty string")
}

func TestListScaleSets(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	testCases := []struct {
		description       string
		existedScaleSets  []compute.VirtualMachineScaleSet
		vmssClientErr     *retry.Error
		expectedVMSSNames []string
		expectedErr       error
	}{
		{
			description: "listScaleSets should return the correct scale sets",
			existedScaleSets: []compute.VirtualMachineScaleSet{
				{
					Name: to.StringPtr("vmss-0"),
					Sku:  &compute.Sku{Capacity: to.Int64Ptr(1)},
					VirtualMachineScaleSetProperties: &compute.VirtualMachineScaleSetProperties{
						VirtualMachineProfile: &compute.VirtualMachineScaleSetVMProfile{},
					},
				},
				{
					Name: to.StringPtr("vmss-1"),
					VirtualMachineScaleSetProperties: &compute.VirtualMachineScaleSetProperties{
						VirtualMachineProfile: &compute.VirtualMachineScaleSetVMProfile{},
					},
				},
				{
					Name: to.StringPtr("vmss-2"),
					Sku:  &compute.Sku{Capacity: to.Int64Ptr(0)},
					VirtualMachineScaleSetProperties: &compute.VirtualMachineScaleSetProperties{
						VirtualMachineProfile: &compute.VirtualMachineScaleSetVMProfile{},
					},
				},
				{
					Name: to.StringPtr("vmss-3"),
				},
			},
			expectedVMSSNames: []string{"vmss-0", "vmss-1"},
		},
		{
			description:   "listScaleSets should report the error if vmss client returns an retry.Error",
			vmssClientErr: &retry.Error{RawError: fmt.Errorf("error")},
			expectedErr:   fmt.Errorf("Retriable: false, RetryAfter: 0s, HTTPStatusCode: 0, RawError: error"),
		},
	}

	for _, test := range testCases {
		ss, err := newTestScaleSet(ctrl)
		assert.NoError(t, err, "unexpected error when creating test VMSS")

		mockVMSSClient := ss.cloud.VirtualMachineScaleSetsClient.(*mockvmssclient.MockInterface)
		mockVMSSClient.EXPECT().List(gomock.Any(), ss.ResourceGroup).Return(test.existedScaleSets, test.vmssClientErr).AnyTimes()

		vmssNames, err := ss.listScaleSets(ss.ResourceGroup)
		assert.Equal(t, test.expectedErr, err, test.description+", but an error occurs")
		assert.Equal(t, test.expectedVMSSNames, vmssNames, test.description)
	}
}

func TestListScaleSetVMs(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	testCases := []struct {
		description     string
		existedVMSSVMs  []compute.VirtualMachineScaleSetVM
		vmssVMClientErr *retry.Error
		expectedErr     error
	}{
		{
			description: "listScaleSetVMs should return the correct vmss vms",
			existedVMSSVMs: []compute.VirtualMachineScaleSetVM{
				{Name: to.StringPtr("vmss-vm-000000")},
				{Name: to.StringPtr("vmss-vm-000001")},
			},
		},
		{
			description:     "listScaleSetVMs should report the error that the vmss vm client hits",
			vmssVMClientErr: &retry.Error{RawError: fmt.Errorf("error")},
			expectedErr:     fmt.Errorf("Retriable: false, RetryAfter: 0s, HTTPStatusCode: 0, RawError: error"),
		},
	}

	for _, test := range testCases {
		ss, err := newTestScaleSet(ctrl)
		assert.NoError(t, err, "unexpected error when creating test VMSS")

		mockVMSSVMClient := ss.cloud.VirtualMachineScaleSetVMsClient.(*mockvmssvmclient.MockInterface)
		mockVMSSVMClient.EXPECT().List(gomock.Any(), ss.ResourceGroup, testVMSSName, gomock.Any()).Return(test.existedVMSSVMs, test.vmssVMClientErr).AnyTimes()

		expectedVMSSVMs := test.existedVMSSVMs

		vmssVMs, err := ss.listScaleSetVMs(testVMSSName, ss.ResourceGroup)
		assert.Equal(t, test.expectedErr, err, test.description+", but an error occurs")
		assert.Equal(t, expectedVMSSVMs, vmssVMs, test.description)
	}
}

func TestGetAgentPoolScaleSets(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	testCases := []struct {
		description       string
		nodes             []*v1.Node
		expectedVMSSNames *[]string
		expectedErr       error
	}{
		{
			description: "getAgentPoolScaleSets should return the correct vmss names",
			nodes: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:   "vmss-vm-000000",
						Labels: map[string]string{nodeLabelRole: "master"},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:   "vmss-vm-000001",
						Labels: map[string]string{managedByAzureLabel: "false"},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "vmss-vm-000002",
					},
				},
			},
			expectedVMSSNames: &[]string{"vmss"},
		},
	}

	for _, test := range testCases {
		ss, err := newTestScaleSet(ctrl)
		assert.NoError(t, err, "unexpected error when creating test VMSS")

		expectedVMSS := buildTestVMSS(testVMSSName, "vmss-vm-")
		mockVMSSClient := ss.cloud.VirtualMachineScaleSetsClient.(*mockvmssclient.MockInterface)
		mockVMSSClient.EXPECT().List(gomock.Any(), ss.ResourceGroup).Return([]compute.VirtualMachineScaleSet{expectedVMSS}, nil).AnyTimes()

		expectedVMSSVMs := []compute.VirtualMachineScaleSetVM{
			{
				VirtualMachineScaleSetVMProperties: &compute.VirtualMachineScaleSetVMProperties{
					OsProfile: &compute.OSProfile{ComputerName: to.StringPtr("vmss-vm-000000")},
				},
			},
			{
				VirtualMachineScaleSetVMProperties: &compute.VirtualMachineScaleSetVMProperties{
					OsProfile: &compute.OSProfile{ComputerName: to.StringPtr("vmss-vm-000001")},
				},
			},
			{
				VirtualMachineScaleSetVMProperties: &compute.VirtualMachineScaleSetVMProperties{
					OsProfile: &compute.OSProfile{ComputerName: to.StringPtr("vmss-vm-000002")},
				},
			},
		}
		mockVMSSVMClient := ss.cloud.VirtualMachineScaleSetVMsClient.(*mockvmssvmclient.MockInterface)
		mockVMSSVMClient.EXPECT().List(gomock.Any(), ss.ResourceGroup, testVMSSName, gomock.Any()).Return(expectedVMSSVMs, nil).AnyTimes()

		mockVMClient := ss.cloud.VirtualMachinesClient.(*mockvmclient.MockInterface)
		mockVMClient.EXPECT().List(gomock.Any(), gomock.Any()).Return(nil, nil).AnyTimes()

		vmssNames, err := ss.getAgentPoolScaleSets(test.nodes)
		assert.Equal(t, test.expectedErr, err, test.description+", but an error occurs")
		assert.Equal(t, test.expectedVMSSNames, vmssNames)
	}
}

func TestGetVMSetNames(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	testCases := []struct {
		description        string
		service            *v1.Service
		nodes              []*v1.Node
		useSingleSLB       bool
		expectedVMSetNames *[]string
		expectedErr        error
	}{
		{
			description:        "GetVMSetNames should return the primary vm set name if the service has no mode annotation",
			service:            &v1.Service{},
			expectedVMSetNames: &[]string{"vmss"},
		},
		{
			description: "GetVMSetNames should return the primary vm set name when using the single SLB",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{ServiceAnnotationLoadBalancerMode: ServiceAnnotationLoadBalancerAutoModeValue}},
			},
			useSingleSLB:       true,
			expectedVMSetNames: &[]string{"vmss"},
		},
		{
			description: "GetVMSetNames should return nil if the service has auto mode annotation",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{ServiceAnnotationLoadBalancerMode: ServiceAnnotationLoadBalancerAutoModeValue}},
			},
			nodes: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "vmss-vm-000002",
					},
				},
			},
		},
		{
			description: "GetVMSetNames should report the error if there's no such vmss",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{ServiceAnnotationLoadBalancerMode: "vmss-1"}},
			},
			nodes: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "vmss-vm-000002",
					},
				},
			},
			expectedErr: fmt.Errorf("scale set (vmss-1) - not found"),
		},
		{
			description: "GetVMSetNames should return the correct vmss names",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{ServiceAnnotationLoadBalancerMode: "vmss"}},
			},
			nodes: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "vmss-vm-000002",
					},
				},
			},
			expectedVMSetNames: &[]string{"vmss"},
		},
	}

	for _, test := range testCases {
		ss, err := newTestScaleSet(ctrl)
		assert.NoError(t, err, "unexpected error when creating test VMSS")

		if test.useSingleSLB {
			ss.EnableMultipleStandardLoadBalancers = false
			ss.LoadBalancerSku = loadBalancerSkuStandard
		}

		expectedVMSS := buildTestVMSS(testVMSSName, "vmss-vm-")
		mockVMSSClient := ss.cloud.VirtualMachineScaleSetsClient.(*mockvmssclient.MockInterface)
		mockVMSSClient.EXPECT().List(gomock.Any(), ss.ResourceGroup).Return([]compute.VirtualMachineScaleSet{expectedVMSS}, nil).AnyTimes()

		expectedVMSSVMs := []compute.VirtualMachineScaleSetVM{
			{
				VirtualMachineScaleSetVMProperties: &compute.VirtualMachineScaleSetVMProperties{
					OsProfile: &compute.OSProfile{ComputerName: to.StringPtr("vmss-vm-000000")},
				},
			},
			{
				VirtualMachineScaleSetVMProperties: &compute.VirtualMachineScaleSetVMProperties{
					OsProfile: &compute.OSProfile{ComputerName: to.StringPtr("vmss-vm-000001")},
				},
			},
			{
				VirtualMachineScaleSetVMProperties: &compute.VirtualMachineScaleSetVMProperties{
					OsProfile: &compute.OSProfile{ComputerName: to.StringPtr("vmss-vm-000002")},
				},
			},
		}
		mockVMSSVMClient := ss.cloud.VirtualMachineScaleSetVMsClient.(*mockvmssvmclient.MockInterface)
		mockVMSSVMClient.EXPECT().List(gomock.Any(), ss.ResourceGroup, testVMSSName, gomock.Any()).Return(expectedVMSSVMs, nil).AnyTimes()

		mockVMClient := ss.cloud.VirtualMachinesClient.(*mockvmclient.MockInterface)
		mockVMClient.EXPECT().List(gomock.Any(), gomock.Any()).Return(nil, nil).AnyTimes()

		vmSetNames, err := ss.GetVMSetNames(test.service, test.nodes)
		assert.Equal(t, test.expectedErr, err, test.description+", but an error occurs")
		assert.Equal(t, test.expectedVMSetNames, vmSetNames)
	}
}

func TestGetPrimaryNetworkInterfaceConfigurationForScaleSet(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	ss, err := newTestScaleSet(ctrl)
	assert.NoError(t, err, "unexpected error when creating test VMSS")

	networkConfigs := []compute.VirtualMachineScaleSetNetworkConfiguration{
		{Name: to.StringPtr("config-0")},
	}
	config, err := ss.getPrimaryNetworkInterfaceConfigurationForScaleSet(networkConfigs, testVMSSName)
	assert.Nil(t, err, "getPrimaryNetworkInterfaceConfigurationForScaleSet should return the correct network config")
	assert.Equal(t, &networkConfigs[0], config, "getPrimaryNetworkInterfaceConfigurationForScaleSet should return the correct network config")

	networkConfigs = []compute.VirtualMachineScaleSetNetworkConfiguration{
		{
			Name: to.StringPtr("config-0"),
			VirtualMachineScaleSetNetworkConfigurationProperties: &compute.VirtualMachineScaleSetNetworkConfigurationProperties{
				Primary: to.BoolPtr(false),
			},
		},
		{
			Name: to.StringPtr("config-1"),
			VirtualMachineScaleSetNetworkConfigurationProperties: &compute.VirtualMachineScaleSetNetworkConfigurationProperties{
				Primary: to.BoolPtr(true),
			},
		},
	}
	config, err = ss.getPrimaryNetworkInterfaceConfigurationForScaleSet(networkConfigs, testVMSSName)
	assert.Nil(t, err, "getPrimaryNetworkInterfaceConfigurationForScaleSet should return the correct network config")
	assert.Equal(t, &networkConfigs[1], config, "getPrimaryNetworkInterfaceConfigurationForScaleSet should return the correct network config")

	networkConfigs = []compute.VirtualMachineScaleSetNetworkConfiguration{
		{
			Name: to.StringPtr("config-0"),
			VirtualMachineScaleSetNetworkConfigurationProperties: &compute.VirtualMachineScaleSetNetworkConfigurationProperties{
				Primary: to.BoolPtr(false),
			},
		},
		{
			Name: to.StringPtr("config-1"),
			VirtualMachineScaleSetNetworkConfigurationProperties: &compute.VirtualMachineScaleSetNetworkConfigurationProperties{
				Primary: to.BoolPtr(false),
			},
		},
	}
	config, err = ss.getPrimaryNetworkInterfaceConfigurationForScaleSet(networkConfigs, testVMSSName)
	assert.Equal(t, fmt.Errorf("failed to find a primary network configuration for the scale set \"vmss\""), err, "getPrimaryNetworkInterfaceConfigurationForScaleSet should report an error if there is no primary nic")
	assert.Nil(t, config, "getPrimaryNetworkInterfaceConfigurationForScaleSet should report an error if there is no primary nic")
}

func TestGetPrimaryIPConfigFromVMSSNetworkConfig(t *testing.T) {
	config := &compute.VirtualMachineScaleSetNetworkConfiguration{
		VirtualMachineScaleSetNetworkConfigurationProperties: &compute.VirtualMachineScaleSetNetworkConfigurationProperties{
			IPConfigurations: &[]compute.VirtualMachineScaleSetIPConfiguration{
				{
					Name: to.StringPtr("config-0"),
				},
			},
		},
	}

	ipConfig, err := getPrimaryIPConfigFromVMSSNetworkConfig(config)
	assert.Nil(t, err, "getPrimaryIPConfigFromVMSSNetworkConfig should return the correct IP config")
	assert.Equal(t, (*config.IPConfigurations)[0], *ipConfig, "getPrimaryIPConfigFromVMSSNetworkConfig should return the correct IP config")

	config = &compute.VirtualMachineScaleSetNetworkConfiguration{
		VirtualMachineScaleSetNetworkConfigurationProperties: &compute.VirtualMachineScaleSetNetworkConfigurationProperties{
			IPConfigurations: &[]compute.VirtualMachineScaleSetIPConfiguration{
				{
					Name: to.StringPtr("config-0"),
					VirtualMachineScaleSetIPConfigurationProperties: &compute.VirtualMachineScaleSetIPConfigurationProperties{
						Primary: to.BoolPtr(false),
					},
				},
				{
					Name: to.StringPtr("config-1"),
					VirtualMachineScaleSetIPConfigurationProperties: &compute.VirtualMachineScaleSetIPConfigurationProperties{
						Primary: to.BoolPtr(true),
					},
				},
			},
		},
	}

	ipConfig, err = getPrimaryIPConfigFromVMSSNetworkConfig(config)
	assert.Nil(t, err, "getPrimaryIPConfigFromVMSSNetworkConfig should return the correct IP config")
	assert.Equal(t, (*config.IPConfigurations)[1], *ipConfig, "getPrimaryIPConfigFromVMSSNetworkConfig should return the correct IP config")

	config = &compute.VirtualMachineScaleSetNetworkConfiguration{
		VirtualMachineScaleSetNetworkConfigurationProperties: &compute.VirtualMachineScaleSetNetworkConfigurationProperties{
			IPConfigurations: &[]compute.VirtualMachineScaleSetIPConfiguration{
				{
					Name: to.StringPtr("config-0"),
					VirtualMachineScaleSetIPConfigurationProperties: &compute.VirtualMachineScaleSetIPConfigurationProperties{
						Primary: to.BoolPtr(false),
					},
				},
				{
					Name: to.StringPtr("config-1"),
					VirtualMachineScaleSetIPConfigurationProperties: &compute.VirtualMachineScaleSetIPConfigurationProperties{
						Primary: to.BoolPtr(false),
					},
				},
			},
		},
	}

	ipConfig, err = getPrimaryIPConfigFromVMSSNetworkConfig(config)
	assert.Equal(t, err, fmt.Errorf("failed to find a primary IP configuration"), "getPrimaryIPConfigFromVMSSNetworkConfig should report an error if there is no primary IP config")
	assert.Nil(t, ipConfig, "getPrimaryIPConfigFromVMSSNetworkConfig should report an error if there is no primary IP config")
}

func TestGetConfigForScaleSetByIPFamily(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	ss, err := newTestScaleSet(ctrl)
	assert.NoError(t, err, "unexpected error when creating test VMSS")

	config := &compute.VirtualMachineScaleSetNetworkConfiguration{
		VirtualMachineScaleSetNetworkConfigurationProperties: &compute.VirtualMachineScaleSetNetworkConfigurationProperties{
			IPConfigurations: &[]compute.VirtualMachineScaleSetIPConfiguration{
				{
					Name: to.StringPtr("config-0"),
					VirtualMachineScaleSetIPConfigurationProperties: &compute.VirtualMachineScaleSetIPConfigurationProperties{
						PrivateIPAddressVersion: compute.IPv4,
					},
				},
				{
					Name: to.StringPtr("config-0"),
					VirtualMachineScaleSetIPConfigurationProperties: &compute.VirtualMachineScaleSetIPConfigurationProperties{
						PrivateIPAddressVersion: compute.IPv6,
					},
				},
			},
		},
	}

	ipConfig, err := ss.getConfigForScaleSetByIPFamily(config, "vmss-vm-000000", true)
	assert.Nil(t, err, "getConfigForScaleSetByIPFamily should find the IPV6 config")
	assert.Equal(t, (*config.IPConfigurations)[1], *ipConfig, "getConfigForScaleSetByIPFamily should find the IPV6 config")

	ipConfig, err = ss.getConfigForScaleSetByIPFamily(config, "vmss-vm-000000", false)
	assert.Nil(t, err, "getConfigForScaleSetByIPFamily should find the IPV4 config")
	assert.Equal(t, (*config.IPConfigurations)[0], *ipConfig, "getConfigForScaleSetByIPFamily should find the IPV4 config")
}

func TestEnsureHostInPool(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	testCases := []struct {
		description               string
		service                   *v1.Service
		nodeName                  types.NodeName
		backendPoolID             string
		vmSetName                 string
		isBasicLB                 bool
		isNilVMNetworkConfigs     bool
		useMultipleSLBs           bool
		expectedNodeResourceGroup string
		expectedVMSSName          string
		expectedInstanceID        string
		expectedVMSSVM            *compute.VirtualMachineScaleSetVM
		expectedErr               error
	}{
		{
			description: "EnsureHostInPool should skip the current node if the vmSetName is not equal to the node's vmss name and the basic LB is used",
			nodeName:    "vmss-vm-000000",
			vmSetName:   "vmss-1",
			isBasicLB:   true,
		},
		{
			description:     "EnsureHostInPool should skip the current node if the vmSetName is not equal to the node's vmss name and multiple SLBs are used",
			nodeName:        "vmss-vm-000000",
			vmSetName:       "vmss-1",
			useMultipleSLBs: true,
		},
		{
			description:           "EnsureHostInPool should skip the current node if the network configs of the VMSS VM is nil",
			nodeName:              "vmss-vm-000000",
			vmSetName:             "vmss",
			isNilVMNetworkConfigs: true,
		},
		{
			description:   "EnsureHostInPool should skip the current node if the backend pool has existed",
			service:       &v1.Service{Spec: v1.ServiceSpec{ClusterIP: "clusterIP"}},
			nodeName:      "vmss-vm-000000",
			vmSetName:     "vmss",
			backendPoolID: testLBBackendpoolID0,
		},
		{
			description:   "EnsureHostInPool should skip the current node if it has already been added to another LB",
			service:       &v1.Service{Spec: v1.ServiceSpec{ClusterIP: "clusterIP"}},
			nodeName:      "vmss-vm-000000",
			vmSetName:     "vmss",
			backendPoolID: "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/loadBalancers/lb1-internal/backendAddressPools/backendpool-1",
			isBasicLB:     false,
		},
		{
			description:               "EnsureHostInPool should add a new backend pool to the vm",
			service:                   &v1.Service{Spec: v1.ServiceSpec{ClusterIP: "clusterIP"}},
			nodeName:                  "vmss-vm-000000",
			vmSetName:                 "vmss",
			backendPoolID:             "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/loadBalancers/lb-internal/backendAddressPools/backendpool-1",
			isBasicLB:                 false,
			expectedNodeResourceGroup: "rg",
			expectedVMSSName:          testVMSSName,
			expectedInstanceID:        "0",
			expectedVMSSVM: &compute.VirtualMachineScaleSetVM{
				Sku:      &compute.Sku{Name: to.StringPtr("sku")},
				Location: to.StringPtr("westus"),
				VirtualMachineScaleSetVMProperties: &compute.VirtualMachineScaleSetVMProperties{
					NetworkProfileConfiguration: &compute.VirtualMachineScaleSetVMNetworkProfileConfiguration{
						NetworkInterfaceConfigurations: &[]compute.VirtualMachineScaleSetNetworkConfiguration{
							{
								Name: to.StringPtr("ipconfig1"),
								ID:   to.StringPtr("fakeNetworkConfiguration"),
								VirtualMachineScaleSetNetworkConfigurationProperties: &compute.VirtualMachineScaleSetNetworkConfigurationProperties{
									IPConfigurations: &[]compute.VirtualMachineScaleSetIPConfiguration{
										{
											Name: to.StringPtr("ipconfig1"),
											VirtualMachineScaleSetIPConfigurationProperties: &compute.VirtualMachineScaleSetIPConfigurationProperties{
												Primary: to.BoolPtr(true),
												LoadBalancerBackendAddressPools: &[]compute.SubResource{
													{
														ID: to.StringPtr(testLBBackendpoolID0),
													},
													{
														ID: to.StringPtr("/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/loadBalancers/lb-internal/backendAddressPools/backendpool-1"),
													},
												},
											},
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	for _, test := range testCases {
		ss, err := newTestScaleSet(ctrl)
		assert.NoError(t, err, test.description)

		if !test.isBasicLB {
			ss.LoadBalancerSku = loadBalancerSkuStandard
		}

		if test.useMultipleSLBs {
			ss.EnableMultipleStandardLoadBalancers = true
		}

		expectedVMSS := buildTestVMSS(testVMSSName, "vmss-vm-")
		mockVMSSClient := ss.cloud.VirtualMachineScaleSetsClient.(*mockvmssclient.MockInterface)
		mockVMSSClient.EXPECT().List(gomock.Any(), ss.ResourceGroup).Return([]compute.VirtualMachineScaleSet{expectedVMSS}, nil).AnyTimes()

		expectedVMSSVMs, _, _ := buildTestVirtualMachineEnv(ss.cloud, testVMSSName, "", 0, []string{string(test.nodeName)}, "", false)
		if test.isNilVMNetworkConfigs {
			expectedVMSSVMs[0].NetworkProfileConfiguration.NetworkInterfaceConfigurations = nil
		}
		mockVMSSVMClient := ss.cloud.VirtualMachineScaleSetVMsClient.(*mockvmssvmclient.MockInterface)
		mockVMSSVMClient.EXPECT().List(gomock.Any(), ss.ResourceGroup, testVMSSName, gomock.Any()).Return(expectedVMSSVMs, nil).AnyTimes()

		nodeResourceGroup, ssName, instanceID, vm, err := ss.EnsureHostInPool(test.service, test.nodeName, test.backendPoolID, test.vmSetName, false)
		assert.Equal(t, test.expectedErr, err, test.description+", but an error occurs")
		assert.Equal(t, test.expectedNodeResourceGroup, nodeResourceGroup, test.description)
		assert.Equal(t, test.expectedVMSSName, ssName, test.description)
		assert.Equal(t, test.expectedInstanceID, instanceID, test.description)
		assert.Equal(t, test.expectedVMSSVM, vm, test.description)
	}
}

func TestGetVmssAndResourceGroupNameByVMProviderID(t *testing.T) {
	providerID := "azure:///subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss/virtualMachines/0"
	rgName, vmssName, err := getVmssAndResourceGroupNameByVMProviderID(providerID)
	assert.NoError(t, err)
	assert.Equal(t, "rg", rgName)
	assert.Equal(t, "vmss", vmssName)

	providerID = "invalid/id"
	rgName, vmssName, err = getVmssAndResourceGroupNameByVMProviderID(providerID)
	assert.Equal(t, err, ErrorNotVmssInstance)
	assert.Equal(t, "", rgName)
	assert.Equal(t, "", vmssName)
}

func TestEnsureVMSSInPool(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	testCases := []struct {
		description        string
		nodes              []*v1.Node
		backendPoolID      string
		vmSetName          string
		isBasicLB          bool
		isVMSSDeallocating bool
		isVMSSNilNICConfig bool
		expectedPutVMSS    bool
		setIPv6Config      bool
		clusterIP          string
		useMultipleSLBs    bool
		expectedErr        error
	}{
		{
			description: "ensureVMSSInPool should skip the node if it isn't managed by VMSS",
			nodes: []*v1.Node{
				{
					Spec: v1.NodeSpec{
						ProviderID: "invalid/id",
					},
				},
			},
			isBasicLB:       false,
			expectedPutVMSS: false,
		},
		{
			description: "ensureVMSSInPool should skip the node if the corresponding VMSS is deallocating",
			nodes: []*v1.Node{
				{
					Spec: v1.NodeSpec{
						ProviderID: "azure:///subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss/virtualMachines/0",
					},
				},
			},
			isBasicLB:          false,
			isVMSSDeallocating: true,
			expectedPutVMSS:    false,
		},
		{
			description: "ensureVMSSInPool should skip the node if the NIC config of the corresponding VMSS is nil",
			nodes: []*v1.Node{
				{
					Spec: v1.NodeSpec{
						ProviderID: "azure:///subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss/virtualMachines/0",
					},
				},
			},
			isBasicLB:          false,
			isVMSSNilNICConfig: true,
			expectedPutVMSS:    false,
		},
		{
			description: "ensureVMSSInPool should skip the node if the backendpool ID has been added already",
			nodes: []*v1.Node{
				{
					Spec: v1.NodeSpec{
						ProviderID: "azure:///subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss/virtualMachines/0",
					},
				},
			},
			isBasicLB:       false,
			backendPoolID:   testLBBackendpoolID0,
			expectedPutVMSS: false,
		},
		{
			description: "ensureVMSSInPool should skip the node if the VMSS has been added to another LB's backendpool already",
			nodes: []*v1.Node{
				{
					Spec: v1.NodeSpec{
						ProviderID: "azure:///subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss/virtualMachines/0",
					},
				},
			},
			isBasicLB:       false,
			backendPoolID:   "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/loadBalancers/lb1/backendAddressPools/backendpool-0",
			expectedPutVMSS: false,
		},
		{
			description: "ensureVMSSInPool should update the VMSS correctly",
			nodes: []*v1.Node{
				{
					Spec: v1.NodeSpec{
						ProviderID: "azure:///subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss/virtualMachines/0",
					},
				},
			},
			isBasicLB:       false,
			backendPoolID:   testLBBackendpoolID1,
			expectedPutVMSS: true,
		},
		{
			description: "ensureVMSSInPool should fail if no IPv6 network config",
			nodes: []*v1.Node{
				{
					Spec: v1.NodeSpec{
						ProviderID: "azure:///subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss/virtualMachines/0",
					},
				},
			},
			isBasicLB:       false,
			backendPoolID:   testLBBackendpoolID1,
			clusterIP:       "fd00::e68b",
			expectedPutVMSS: false,
			expectedErr:     fmt.Errorf("failed to find a IPconfiguration(IPv6=true) for the scale set VM \"\""),
		},
		{
			description: "ensureVMSSInPool should update the VMSS correctly for IPv6",
			nodes: []*v1.Node{
				{
					Spec: v1.NodeSpec{
						ProviderID: "azure:///subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss/virtualMachines/0",
					},
				},
			},
			isBasicLB:       false,
			backendPoolID:   testLBBackendpoolID1,
			setIPv6Config:   true,
			clusterIP:       "fd00::e68b",
			expectedPutVMSS: true,
		},
		{
			description: "ensureVMSSInPool should work for the basic load balancer",
			nodes: []*v1.Node{
				{
					Spec: v1.NodeSpec{
						ProviderID: "azure:///subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss/virtualMachines/0",
					},
				},
			},
			vmSetName:       testVMSSName,
			isBasicLB:       true,
			backendPoolID:   testLBBackendpoolID1,
			expectedPutVMSS: true,
		},
		{
			description: "ensureVMSSInPool should work for multiple standard load balancers",
			nodes: []*v1.Node{
				{
					Spec: v1.NodeSpec{
						ProviderID: "azure:///subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss/virtualMachines/0",
					},
				},
			},
			vmSetName:       testVMSSName,
			backendPoolID:   testLBBackendpoolID1,
			useMultipleSLBs: true,
			expectedPutVMSS: true,
		},
	}

	for _, test := range testCases {
		ss, err := newTestScaleSet(ctrl)
		assert.NoError(t, err, test.description)

		if !test.isBasicLB {
			ss.LoadBalancerSku = loadBalancerSkuStandard
		}

		expectedVMSS := buildTestVMSSWithLB(testVMSSName, "vmss-vm-", []string{testLBBackendpoolID0}, test.setIPv6Config)
		if test.isVMSSDeallocating {
			expectedVMSS.ProvisioningState = &virtualMachineScaleSetsDeallocating
		}
		if test.isVMSSNilNICConfig {
			expectedVMSS.VirtualMachineProfile.NetworkProfile.NetworkInterfaceConfigurations = nil
		}
		mockVMSSClient := ss.cloud.VirtualMachineScaleSetsClient.(*mockvmssclient.MockInterface)
		mockVMSSClient.EXPECT().List(gomock.Any(), ss.ResourceGroup).Return([]compute.VirtualMachineScaleSet{expectedVMSS}, nil).AnyTimes()
		vmssPutTimes := 0
		if test.expectedPutVMSS {
			vmssPutTimes = 1
			mockVMSSClient.EXPECT().Get(gomock.Any(), ss.ResourceGroup, testVMSSName).Return(expectedVMSS, nil)
		}
		mockVMSSClient.EXPECT().CreateOrUpdate(gomock.Any(), ss.ResourceGroup, testVMSSName, gomock.Any()).Return(nil).Times(vmssPutTimes)

		expectedVMSSVMs, _, _ := buildTestVirtualMachineEnv(ss.cloud, testVMSSName, "", 0, []string{"vmss-vm-000000"}, "", test.setIPv6Config)
		mockVMSSVMClient := ss.cloud.VirtualMachineScaleSetVMsClient.(*mockvmssvmclient.MockInterface)
		mockVMSSVMClient.EXPECT().List(gomock.Any(), ss.ResourceGroup, testVMSSName, gomock.Any()).Return(expectedVMSSVMs, nil).AnyTimes()

		err = ss.ensureVMSSInPool(&v1.Service{Spec: v1.ServiceSpec{ClusterIP: test.clusterIP}}, test.nodes, test.backendPoolID, test.vmSetName)
		assert.Equal(t, test.expectedErr, err, test.description+", but an error occurs")
	}
}

func TestEnsureHostsInPool(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	testCases := []struct {
		description            string
		nodes                  []*v1.Node
		backendpoolID          string
		vmSetName              string
		expectedVMSSVMPutTimes int
		expectedErr            bool
	}{
		{
			description: "EnsureHostsInPool should skip the invalid node and update the VMSS VM correctly",
			nodes: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:   "vmss-vm-000000",
						Labels: map[string]string{nodeLabelRole: "master"},
					},
					Spec: v1.NodeSpec{
						ProviderID: "azure:///subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss/virtualMachines/0",
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:   "vmss-vm-000001",
						Labels: map[string]string{managedByAzureLabel: "false"},
					},
					Spec: v1.NodeSpec{
						ProviderID: "azure:///subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss/virtualMachines/1",
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "vmss-vm-000002",
					},
					Spec: v1.NodeSpec{
						ProviderID: "azure:///subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss/virtualMachines/2",
					},
				},
			},
			backendpoolID:          testLBBackendpoolID1,
			vmSetName:              testVMSSName,
			expectedVMSSVMPutTimes: 1,
		},
		{
			description: "EnsureHostsInPool should gather report the error if something goes wrong in EnsureHostInPool",
			nodes: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "vmss-vm-000003",
					},
					Spec: v1.NodeSpec{
						ProviderID: "azure:///subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss/virtualMachines/3",
					},
				},
			},
			backendpoolID:          testLBBackendpoolID1,
			vmSetName:              testVMSSName,
			expectedVMSSVMPutTimes: 0,
			expectedErr:            true,
		},
	}

	for _, test := range testCases {
		ss, err := newTestScaleSet(ctrl)
		assert.NoError(t, err, test.description)

		ss.LoadBalancerSku = loadBalancerSkuStandard
		ss.ExcludeMasterFromStandardLB = to.BoolPtr(true)

		expectedVMSS := buildTestVMSSWithLB(testVMSSName, "vmss-vm-", []string{testLBBackendpoolID0}, false)
		mockVMSSClient := ss.cloud.VirtualMachineScaleSetsClient.(*mockvmssclient.MockInterface)
		mockVMSSClient.EXPECT().List(gomock.Any(), ss.ResourceGroup).Return([]compute.VirtualMachineScaleSet{expectedVMSS}, nil).AnyTimes()
		mockVMSSClient.EXPECT().Get(gomock.Any(), ss.ResourceGroup, testVMSSName).Return(expectedVMSS, nil).MaxTimes(1)
		mockVMSSClient.EXPECT().CreateOrUpdate(gomock.Any(), ss.ResourceGroup, testVMSSName, gomock.Any()).Return(nil).MaxTimes(1)

		expectedVMSSVMs, _, _ := buildTestVirtualMachineEnv(ss.cloud, testVMSSName, "", 0, []string{"vmss-vm-000000", "vmss-vm-000001", "vmss-vm-000002"}, "", false)
		mockVMSSVMClient := ss.cloud.VirtualMachineScaleSetVMsClient.(*mockvmssvmclient.MockInterface)
		mockVMSSVMClient.EXPECT().List(gomock.Any(), ss.ResourceGroup, testVMSSName, gomock.Any()).Return(expectedVMSSVMs, nil).AnyTimes()
		mockVMSSVMClient.EXPECT().UpdateVMs(gomock.Any(), ss.ResourceGroup, testVMSSName, gomock.Any(), gomock.Any()).Return(nil).Times(test.expectedVMSSVMPutTimes)

		mockVMClient := ss.cloud.VirtualMachinesClient.(*mockvmclient.MockInterface)
		mockVMClient.EXPECT().List(gomock.Any(), gomock.Any()).Return(nil, nil).AnyTimes()

		err = ss.EnsureHostsInPool(&v1.Service{}, test.nodes, test.backendpoolID, test.vmSetName, false)
		assert.Equal(t, test.expectedErr, err != nil, test.description+", but an error occurs")
	}
}

func TestEnsureBackendPoolDeletedFromNode(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	testCases := []struct {
		description               string
		nodeName                  string
		backendpoolID             string
		isNilVMNetworkConfigs     bool
		expectedNodeResourceGroup string
		expectedVMSSName          string
		expectedInstanceID        string
		expectedVMSSVM            *compute.VirtualMachineScaleSetVM
		expectedErr               error
	}{
		{
			description: "ensureBackendPoolDeletedFromNode should report the error that occurs during getVmssVM",
			nodeName:    "vmss-vm-000001",
			expectedErr: cloudprovider.InstanceNotFound,
		},
		{
			description:           "ensureBackendPoolDeletedFromNode skip the node if the VM's NIC config is nil",
			nodeName:              "vmss-vm-000000",
			isNilVMNetworkConfigs: true,
		},
		{
			description:   "ensureBackendPoolDeletedFromNode should skip the node if there's no wanted lb backendpool ID on that VM",
			nodeName:      "vmss-vm-000000",
			backendpoolID: testLBBackendpoolID1,
		},
		{
			description:               "ensureBackendPoolDeletedFromNode should delete the given backendpool ID",
			nodeName:                  "vmss-vm-000000",
			backendpoolID:             testLBBackendpoolID0,
			expectedNodeResourceGroup: "rg",
			expectedVMSSName:          testVMSSName,
			expectedInstanceID:        "0",
			expectedVMSSVM: &compute.VirtualMachineScaleSetVM{
				Sku:      &compute.Sku{Name: to.StringPtr("sku")},
				Location: to.StringPtr("westus"),
				VirtualMachineScaleSetVMProperties: &compute.VirtualMachineScaleSetVMProperties{
					NetworkProfileConfiguration: &compute.VirtualMachineScaleSetVMNetworkProfileConfiguration{
						NetworkInterfaceConfigurations: &[]compute.VirtualMachineScaleSetNetworkConfiguration{
							{
								Name: to.StringPtr("ipconfig1"),
								ID:   to.StringPtr("fakeNetworkConfiguration"),
								VirtualMachineScaleSetNetworkConfigurationProperties: &compute.VirtualMachineScaleSetNetworkConfigurationProperties{
									IPConfigurations: &[]compute.VirtualMachineScaleSetIPConfiguration{
										{
											Name: to.StringPtr("ipconfig1"),
											VirtualMachineScaleSetIPConfigurationProperties: &compute.VirtualMachineScaleSetIPConfigurationProperties{
												Primary:                         to.BoolPtr(true),
												LoadBalancerBackendAddressPools: &[]compute.SubResource{},
											},
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	for _, test := range testCases {
		ss, err := newTestScaleSet(ctrl)
		assert.NoError(t, err, test.description)

		expectedVMSS := buildTestVMSS(testVMSSName, "vmss-vm-")
		mockVMSSClient := ss.cloud.VirtualMachineScaleSetsClient.(*mockvmssclient.MockInterface)
		mockVMSSClient.EXPECT().List(gomock.Any(), ss.ResourceGroup).Return([]compute.VirtualMachineScaleSet{expectedVMSS}, nil).AnyTimes()

		expectedVMSSVMs, _, _ := buildTestVirtualMachineEnv(ss.cloud, testVMSSName, "", 0, []string{"vmss-vm-000000"}, "", false)
		if test.isNilVMNetworkConfigs {
			expectedVMSSVMs[0].NetworkProfileConfiguration.NetworkInterfaceConfigurations = nil
		}
		mockVMSSVMClient := ss.cloud.VirtualMachineScaleSetVMsClient.(*mockvmssvmclient.MockInterface)
		mockVMSSVMClient.EXPECT().List(gomock.Any(), ss.ResourceGroup, testVMSSName, gomock.Any()).Return(expectedVMSSVMs, nil).AnyTimes()

		nodeResourceGroup, ssName, instanceID, vm, err := ss.ensureBackendPoolDeletedFromNode(test.nodeName, test.backendpoolID)
		assert.Equal(t, test.expectedErr, err, test.description+", but an error occurs")
		assert.Equal(t, test.expectedNodeResourceGroup, nodeResourceGroup, test.description)
		assert.Equal(t, test.expectedVMSSName, ssName, test.description)
		assert.Equal(t, test.expectedInstanceID, instanceID, test.description)
		assert.Equal(t, test.expectedVMSSVM, vm, test.description)
	}
}

func TestGetScaleSetAndResourceGroupNameByIPConfigurationID(t *testing.T) {
	ipConfigID := "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss/virtualMachines/vmss-vm-000000/networkInterfaces/nic"
	scaleSetName, resourceGroup, err := getScaleSetAndResourceGroupNameByIPConfigurationID(ipConfigID)
	assert.Equal(t, "vmss", scaleSetName)
	assert.Equal(t, "rg", resourceGroup)
	assert.NoError(t, err)

	ipConfigID = "invalid/id"
	scaleSetName, resourceGroup, err = getScaleSetAndResourceGroupNameByIPConfigurationID(ipConfigID)
	assert.Equal(t, ErrorNotVmssInstance, err)
	assert.Equal(t, "", scaleSetName)
	assert.Equal(t, "", resourceGroup)
}

func TestEnsureBackendPoolDeletedFromVMSS(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	testCases := []struct {
		description        string
		backendPoolID      string
		ipConfigurationIDs []string
		isVMSSDeallocating bool
		isVMSSNilNICConfig bool
		expectedPutVMSS    bool
		vmssClientErr      *retry.Error
		expectedErr        error
	}{
		{
			description:        "ensureBackendPoolDeletedFromVMSS should skip the IP config if it's ID is invalid",
			ipConfigurationIDs: []string{"invalid/id"},
		},
		{
			description:        "ensureBackendPoolDeletedFromVMSS should skip the VMSS if it's being deleting",
			ipConfigurationIDs: []string{"/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss/virtualMachines/vmss-vm-000000/networkInterfaces/nic"},
			isVMSSDeallocating: true,
		},
		{
			description:        "ensureBackendPoolDeletedFromVMSS should skip the VMSS if it's NIC config is nil",
			ipConfigurationIDs: []string{"/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss/virtualMachines/vmss-vm-000000/networkInterfaces/nic"},
			isVMSSNilNICConfig: true,
		},
		{
			description:        "ensureBackendPoolDeletedFromVMSS should delete the corresponding LB backendpool ID",
			ipConfigurationIDs: []string{"/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss/virtualMachines/vmss-vm-000000/networkInterfaces/nic"},
			backendPoolID:      testLBBackendpoolID0,
			expectedPutVMSS:    true,
		},
		{
			description:        "ensureBackendPoolDeletedFromVMSS should skip the VMSS if there's no wanted LB backendpool ID",
			ipConfigurationIDs: []string{"/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss/virtualMachines/vmss-vm-000000/networkInterfaces/nic"},
			backendPoolID:      testLBBackendpoolID1,
		},
		{
			description:        "ensureBackendPoolDeletedFromVMSS should report the error that occurs during VMSS client's call",
			ipConfigurationIDs: []string{"/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss/virtualMachines/vmss-vm-000000/networkInterfaces/nic"},
			backendPoolID:      testLBBackendpoolID0,
			expectedPutVMSS:    true,
			vmssClientErr:      &retry.Error{RawError: fmt.Errorf("error")},
			expectedErr:        fmt.Errorf("Retriable: false, RetryAfter: 0s, HTTPStatusCode: 0, RawError: error"),
		},
	}

	for _, test := range testCases {
		ss, err := newTestScaleSet(ctrl)
		assert.NoError(t, err, test.description)

		ss.LoadBalancerSku = loadBalancerSkuStandard

		expectedVMSS := buildTestVMSSWithLB(testVMSSName, "vmss-vm-", []string{testLBBackendpoolID0}, false)
		if test.isVMSSDeallocating {
			expectedVMSS.ProvisioningState = &virtualMachineScaleSetsDeallocating
		}
		if test.isVMSSNilNICConfig {
			expectedVMSS.VirtualMachineProfile.NetworkProfile.NetworkInterfaceConfigurations = nil
		}
		mockVMSSClient := ss.cloud.VirtualMachineScaleSetsClient.(*mockvmssclient.MockInterface)
		mockVMSSClient.EXPECT().List(gomock.Any(), ss.ResourceGroup).Return([]compute.VirtualMachineScaleSet{expectedVMSS}, nil).AnyTimes()
		vmssPutTimes := 0
		if test.expectedPutVMSS {
			vmssPutTimes = 1
			mockVMSSClient.EXPECT().Get(gomock.Any(), ss.ResourceGroup, testVMSSName).Return(expectedVMSS, nil)
		}
		mockVMSSClient.EXPECT().CreateOrUpdate(gomock.Any(), ss.ResourceGroup, testVMSSName, gomock.Any()).Return(test.vmssClientErr).Times(vmssPutTimes)

		expectedVMSSVMs, _, _ := buildTestVirtualMachineEnv(ss.cloud, testVMSSName, "", 0, []string{"vmss-vm-000000"}, "", false)
		mockVMSSVMClient := ss.cloud.VirtualMachineScaleSetVMsClient.(*mockvmssvmclient.MockInterface)
		mockVMSSVMClient.EXPECT().List(gomock.Any(), ss.ResourceGroup, testVMSSName, gomock.Any()).Return(expectedVMSSVMs, nil).AnyTimes()

		err = ss.ensureBackendPoolDeletedFromVMSS(&v1.Service{}, test.backendPoolID, testVMSSName, test.ipConfigurationIDs)
		assert.Equal(t, test.expectedErr, err, test.description+", but an error occurs")
	}
}

func TestEnsureBackendPoolDeleted(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	testCases := []struct {
		description            string
		backendpoolID          string
		backendAddressPools    *[]network.BackendAddressPool
		expectedVMSSVMPutTimes int
		vmClientErr            *retry.Error
		expectedErr            bool
	}{
		{
			description:   "EnsureBackendPoolDeleted should skip the unwanted backend address pools and update the VMSS VM correctly",
			backendpoolID: testLBBackendpoolID0,
			backendAddressPools: &[]network.BackendAddressPool{
				{
					ID: to.StringPtr(testLBBackendpoolID0),
					BackendAddressPoolPropertiesFormat: &network.BackendAddressPoolPropertiesFormat{
						BackendIPConfigurations: &[]network.InterfaceIPConfiguration{
							{
								Name: to.StringPtr("ip-1"),
								ID:   to.StringPtr("/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss/virtualMachines/0/networkInterfaces/nic"),
							},
							{
								Name: to.StringPtr("ip-2"),
							},
							{
								Name: to.StringPtr("ip-3"),
								ID:   to.StringPtr("/invalid/id"),
							},
						},
					},
				},
				{
					ID: to.StringPtr(testLBBackendpoolID1),
				},
			},
			expectedVMSSVMPutTimes: 1,
		},
		{
			description:   "EnsureBackendPoolDeleted should report the error that occurs during the call of VMSS VM client",
			backendpoolID: testLBBackendpoolID0,
			backendAddressPools: &[]network.BackendAddressPool{
				{
					ID: to.StringPtr(testLBBackendpoolID0),
					BackendAddressPoolPropertiesFormat: &network.BackendAddressPoolPropertiesFormat{
						BackendIPConfigurations: &[]network.InterfaceIPConfiguration{
							{
								Name: to.StringPtr("ip-1"),
								ID:   to.StringPtr("/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss/virtualMachines/0/networkInterfaces/nic"),
							},
						},
					},
				},
				{
					ID: to.StringPtr(testLBBackendpoolID1),
				},
			},
			expectedVMSSVMPutTimes: 1,
			expectedErr:            true,
			vmClientErr:            &retry.Error{RawError: fmt.Errorf("error")},
		},
	}

	for _, test := range testCases {
		ss, err := newTestScaleSet(ctrl)
		assert.NoError(t, err, test.description)

		expectedVMSS := buildTestVMSSWithLB(testVMSSName, "vmss-vm-", []string{testLBBackendpoolID0}, false)
		mockVMSSClient := ss.cloud.VirtualMachineScaleSetsClient.(*mockvmssclient.MockInterface)
		mockVMSSClient.EXPECT().List(gomock.Any(), ss.ResourceGroup).Return([]compute.VirtualMachineScaleSet{expectedVMSS}, nil).AnyTimes()
		mockVMSSClient.EXPECT().Get(gomock.Any(), ss.ResourceGroup, testVMSSName).Return(expectedVMSS, nil).MaxTimes(1)
		mockVMSSClient.EXPECT().CreateOrUpdate(gomock.Any(), ss.ResourceGroup, testVMSSName, gomock.Any()).Return(nil).MaxTimes(1)

		expectedVMSSVMs, _, _ := buildTestVirtualMachineEnv(ss.cloud, testVMSSName, "", 0, []string{"vmss-vm-000000", "vmss-vm-000001", "vmss-vm-000002"}, "", false)
		mockVMSSVMClient := ss.cloud.VirtualMachineScaleSetVMsClient.(*mockvmssvmclient.MockInterface)
		mockVMSSVMClient.EXPECT().List(gomock.Any(), ss.ResourceGroup, testVMSSName, gomock.Any()).Return(expectedVMSSVMs, nil).AnyTimes()
		mockVMSSVMClient.EXPECT().UpdateVMs(gomock.Any(), ss.ResourceGroup, testVMSSName, gomock.Any(), gomock.Any()).Return(test.vmClientErr).Times(test.expectedVMSSVMPutTimes)

		err = ss.EnsureBackendPoolDeleted(&v1.Service{}, test.backendpoolID, testVMSSName, test.backendAddressPools)
		assert.Equal(t, test.expectedErr, err != nil, test.description+", but an error occurs")
	}
}

func TestEnsureBackendPoolDeletedConcurrently(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	ss, err := newTestScaleSet(ctrl)
	assert.NoError(t, err)

	backendAddressPools := &[]network.BackendAddressPool{
		{
			ID: to.StringPtr(testLBBackendpoolID0),
			BackendAddressPoolPropertiesFormat: &network.BackendAddressPoolPropertiesFormat{
				BackendIPConfigurations: &[]network.InterfaceIPConfiguration{
					{
						Name: to.StringPtr("ip-1"),
						ID:   to.StringPtr("/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss-0/virtualMachines/0/networkInterfaces/nic"),
					},
				},
			},
		},
		{
			ID: to.StringPtr(testLBBackendpoolID1),
			BackendAddressPoolPropertiesFormat: &network.BackendAddressPoolPropertiesFormat{
				BackendIPConfigurations: &[]network.InterfaceIPConfiguration{
					{
						Name: to.StringPtr("ip-1"),
						ID:   to.StringPtr("/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/vmss-1/virtualMachines/0/networkInterfaces/nic"),
					},
				},
			},
		},
		{
			// this would fail
			ID: to.StringPtr(testLBBackendpoolID2),
			BackendAddressPoolPropertiesFormat: &network.BackendAddressPoolPropertiesFormat{
				BackendIPConfigurations: &[]network.InterfaceIPConfiguration{
					{
						Name: to.StringPtr("ip-1"),
						ID:   to.StringPtr("/subscriptions/sub/resourceGroups/rg1/providers/Microsoft.Compute/virtualMachineScaleSets/vmss-0/virtualMachines/0/networkInterfaces/nic"),
					},
				},
			},
		},
	}

	vmss0 := buildTestVMSSWithLB("vmss-0", "vmss-0-vm-", []string{testLBBackendpoolID0, testLBBackendpoolID1}, false)
	vmss1 := buildTestVMSSWithLB("vmss-1", "vmss-1-vm-", []string{testLBBackendpoolID0, testLBBackendpoolID1}, false)

	expectedVMSSVMsOfVMSS0, _, _ := buildTestVirtualMachineEnv(ss.cloud, "vmss-0", "", 0, []string{"vmss-0-vm-000000"}, "succeeded", false)
	expectedVMSSVMsOfVMSS1, _, _ := buildTestVirtualMachineEnv(ss.cloud, "vmss-1", "", 0, []string{"vmss-1-vm-000001"}, "succeeded", false)
	for _, expectedVMSSVMs := range [][]compute.VirtualMachineScaleSetVM{expectedVMSSVMsOfVMSS0, expectedVMSSVMsOfVMSS1} {
		vmssVMNetworkConfigs := expectedVMSSVMs[0].NetworkProfileConfiguration
		vmssVMIPConfigs := (*vmssVMNetworkConfigs.NetworkInterfaceConfigurations)[0].VirtualMachineScaleSetNetworkConfigurationProperties.IPConfigurations
		lbBackendpools := (*vmssVMIPConfigs)[0].LoadBalancerBackendAddressPools
		*lbBackendpools = append(*lbBackendpools, compute.SubResource{ID: to.StringPtr(testLBBackendpoolID1)})
	}

	mockVMSSClient := ss.cloud.VirtualMachineScaleSetsClient.(*mockvmssclient.MockInterface)
	mockVMSSClient.EXPECT().List(gomock.Any(), ss.ResourceGroup).Return([]compute.VirtualMachineScaleSet{vmss0, vmss1}, nil).AnyTimes()
	mockVMSSClient.EXPECT().List(gomock.Any(), "rg1").Return(nil, nil).AnyTimes()
	mockVMSSClient.EXPECT().Get(gomock.Any(), ss.ResourceGroup, "vmss-0").Return(vmss0, nil).MaxTimes(2)
	mockVMSSClient.EXPECT().Get(gomock.Any(), ss.ResourceGroup, "vmss-1").Return(vmss1, nil).MaxTimes(2)
	mockVMSSClient.EXPECT().CreateOrUpdate(gomock.Any(), ss.ResourceGroup, gomock.Any(), gomock.Any()).Return(nil).Times(2)

	mockVMSSVMClient := ss.cloud.VirtualMachineScaleSetVMsClient.(*mockvmssvmclient.MockInterface)
	mockVMSSVMClient.EXPECT().List(gomock.Any(), "rg1", "vmss-0", gomock.Any()).Return(nil, nil).AnyTimes()
	mockVMSSVMClient.EXPECT().List(gomock.Any(), ss.ResourceGroup, "vmss-0", gomock.Any()).Return(expectedVMSSVMsOfVMSS0, nil).AnyTimes()
	mockVMSSVMClient.EXPECT().List(gomock.Any(), ss.ResourceGroup, "vmss-1", gomock.Any()).Return(expectedVMSSVMsOfVMSS1, nil).AnyTimes()
	mockVMSSVMClient.EXPECT().UpdateVMs(gomock.Any(), ss.ResourceGroup, gomock.Any(), gomock.Any(), gomock.Any()).Return(nil).Times(2)

	backendpoolAddressIDs := []string{testLBBackendpoolID0, testLBBackendpoolID1, testLBBackendpoolID2}
	testVMSSNames := []string{"vmss-0", "vmss-1", "vmss-0"}
	testFunc := make([]func() error, 0)
	for i, id := range backendpoolAddressIDs {
		i := i
		id := id
		testFunc = append(testFunc, func() error {
			return ss.EnsureBackendPoolDeleted(&v1.Service{}, id, testVMSSNames[i], backendAddressPools)
		})
	}
	errs := utilerrors.AggregateGoroutines(testFunc...)
	assert.Equal(t, 1, len(errs.Errors()))
	assert.Equal(t, "instance not found", errs.Error())
}

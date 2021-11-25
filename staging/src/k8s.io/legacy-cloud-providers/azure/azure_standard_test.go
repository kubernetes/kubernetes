//go:build !providerless
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
	"net/http"
	"strconv"
	"testing"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-12-01/compute"
	"github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-06-01/network"
	"github.com/Azure/go-autorest/autorest/to"
	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/legacy-cloud-providers/azure/clients/interfaceclient/mockinterfaceclient"
	"k8s.io/legacy-cloud-providers/azure/clients/publicipclient/mockpublicipclient"
	"k8s.io/legacy-cloud-providers/azure/clients/vmclient/mockvmclient"
	"k8s.io/legacy-cloud-providers/azure/retry"
)

const (
	networkResourceTenantID       = "networkResourceTenantID"
	networkResourceSubscriptionID = "networkResourceSubscriptionID"
)

func TestIsMasterNode(t *testing.T) {
	if isMasterNode(&v1.Node{}) {
		t.Errorf("Empty node should not be master!")
	}
	if isMasterNode(&v1.Node{
		ObjectMeta: meta.ObjectMeta{
			Labels: map[string]string{
				nodeLabelRole: "worker",
			},
		},
	}) {
		t.Errorf("Node labelled 'worker' should not be master!")
	}
	if !isMasterNode(&v1.Node{
		ObjectMeta: meta.ObjectMeta{
			Labels: map[string]string{
				nodeLabelRole: "master",
			},
		},
	}) {
		t.Errorf("Node should be master!")
	}
}

func TestGetLastSegment(t *testing.T) {
	tests := []struct {
		ID        string
		separator string
		expected  string
		expectErr bool
	}{
		{
			ID:        "",
			separator: "/",
			expected:  "",
			expectErr: true,
		},
		{
			ID:        "foo/",
			separator: "/",
			expected:  "",
			expectErr: true,
		},
		{
			ID:        "foo/bar",
			separator: "/",
			expected:  "bar",
			expectErr: false,
		},
		{
			ID:        "foo/bar/baz",
			separator: "/",
			expected:  "baz",
			expectErr: false,
		},
		{
			ID:        "k8s-agentpool-36841236-vmss_1",
			separator: "_",
			expected:  "1",
			expectErr: false,
		},
	}

	for _, test := range tests {
		s, e := getLastSegment(test.ID, test.separator)
		if test.expectErr && e == nil {
			t.Errorf("Expected err, but it was nil")
			continue
		}
		if !test.expectErr && e != nil {
			t.Errorf("Unexpected error: %v", e)
			continue
		}
		if s != test.expected {
			t.Errorf("expected: %s, got %s", test.expected, s)
		}
	}
}

func TestGenerateStorageAccountName(t *testing.T) {
	tests := []struct {
		prefix string
	}{
		{
			prefix: "",
		},
		{
			prefix: "pvc",
		},
		{
			prefix: "1234512345123451234512345",
		},
	}

	for _, test := range tests {
		accountName := generateStorageAccountName(test.prefix)
		if len(accountName) > storageAccountNameMaxLength || len(accountName) < 3 {
			t.Errorf("input prefix: %s, output account name: %s, length not in [3,%d]", test.prefix, accountName, storageAccountNameMaxLength)
		}

		for _, char := range accountName {
			if (char < 'a' || char > 'z') && (char < '0' || char > '9') {
				t.Errorf("input prefix: %s, output account name: %s, there is non-digit or non-letter(%q)", test.prefix, accountName, char)
				break
			}
		}
	}
}

func TestMapLoadBalancerNameToVMSet(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	az := GetTestCloud(ctrl)
	az.PrimaryAvailabilitySetName = "primary"

	cases := []struct {
		description   string
		lbName        string
		useStandardLB bool
		clusterName   string
		expectedVMSet string
	}{
		{
			description:   "default external LB should map to primary vmset",
			lbName:        "azure",
			clusterName:   "azure",
			expectedVMSet: "primary",
		},
		{
			description:   "default internal LB should map to primary vmset",
			lbName:        "azure-internal",
			clusterName:   "azure",
			expectedVMSet: "primary",
		},
		{
			description:   "non-default external LB should map to its own vmset",
			lbName:        "azuretest-internal",
			clusterName:   "azure",
			expectedVMSet: "azuretest",
		},
		{
			description:   "non-default internal LB should map to its own vmset",
			lbName:        "azuretest-internal",
			clusterName:   "azure",
			expectedVMSet: "azuretest",
		},
	}

	for _, c := range cases {
		if c.useStandardLB {
			az.Config.LoadBalancerSku = loadBalancerSkuStandard
		} else {
			az.Config.LoadBalancerSku = loadBalancerSkuBasic
		}
		vmset := az.mapLoadBalancerNameToVMSet(c.lbName, c.clusterName)
		assert.Equal(t, c.expectedVMSet, vmset, c.description)
	}
}

func TestGetAzureLoadBalancerName(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	az := GetTestCloud(ctrl)
	az.PrimaryAvailabilitySetName = "primary"

	cases := []struct {
		description   string
		vmSet         string
		isInternal    bool
		useStandardLB bool
		clusterName   string
		lbName        string
		expected      string
	}{
		{
			description: "prefix of loadBalancerName should be az.LoadBalancerName if az.LoadBalancerName is not nil",
			vmSet:       "primary",
			clusterName: "azure",
			lbName:      "azurelb",
			expected:    "azurelb",
		},
		{
			description: "default external LB should get primary vmset",
			vmSet:       "primary",
			clusterName: "azure",
			expected:    "azure",
		},
		{
			description: "default internal LB should get primary vmset",
			vmSet:       "primary",
			clusterName: "azure",
			isInternal:  true,
			expected:    "azure-internal",
		},
		{
			description: "non-default external LB should get its own vmset",
			vmSet:       "as",
			clusterName: "azure",
			expected:    "as",
		},
		{
			description: "non-default internal LB should get its own vmset",
			vmSet:       "as",
			clusterName: "azure",
			isInternal:  true,
			expected:    "as-internal",
		},
		{
			description:   "default standard external LB should get cluster name",
			vmSet:         "primary",
			useStandardLB: true,
			clusterName:   "azure",
			expected:      "azure",
		},
		{
			description:   "default standard internal LB should get cluster name",
			vmSet:         "primary",
			useStandardLB: true,
			isInternal:    true,
			clusterName:   "azure",
			expected:      "azure-internal",
		},
		{
			description:   "non-default standard external LB should get cluster-name",
			vmSet:         "as",
			useStandardLB: true,
			clusterName:   "azure",
			expected:      "azure",
		},
		{
			description:   "non-default standard internal LB should get cluster-name",
			vmSet:         "as",
			useStandardLB: true,
			isInternal:    true,
			clusterName:   "azure",
			expected:      "azure-internal",
		},
	}

	for _, c := range cases {
		if c.useStandardLB {
			az.Config.LoadBalancerSku = loadBalancerSkuStandard
		} else {
			az.Config.LoadBalancerSku = loadBalancerSkuBasic
		}
		az.Config.LoadBalancerName = c.lbName
		loadbalancerName := az.getAzureLoadBalancerName(c.clusterName, c.vmSet, c.isInternal)
		assert.Equal(t, c.expected, loadbalancerName, c.description)
	}
}

func TestGetLoadBalancingRuleName(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	az := GetTestCloud(ctrl)
	az.PrimaryAvailabilitySetName = "primary"

	svc := &v1.Service{
		ObjectMeta: meta.ObjectMeta{
			Annotations: map[string]string{},
			UID:         "257b9655-5137-4ad2-b091-ef3f07043ad3",
		},
	}

	cases := []struct {
		description   string
		subnetName    string
		isInternal    bool
		useStandardLB bool
		protocol      v1.Protocol
		port          int32
		expected      string
	}{
		{
			description:   "internal lb should have subnet name on the rule name",
			subnetName:    "shortsubnet",
			isInternal:    true,
			useStandardLB: true,
			protocol:      v1.ProtocolTCP,
			port:          9000,
			expected:      "a257b965551374ad2b091ef3f07043ad-shortsubnet-TCP-9000",
		},
		{
			description:   "internal standard lb should have subnet name on the rule name but truncated to 80 characters",
			subnetName:    "averylonnnngggnnnnnnnnnnnnnnnnnnnnnngggggggggggggggggggggggggggggggggggggsubet",
			isInternal:    true,
			useStandardLB: true,
			protocol:      v1.ProtocolTCP,
			port:          9000,
			expected:      "a257b965551374ad2b091ef3f07043ad-averylonnnngggnnnnnnnnnnnnnnnnnnnnnngg-TCP-9000",
		},
		{
			description:   "internal basic lb should have subnet name on the rule name but truncated to 80 characters",
			subnetName:    "averylonnnngggnnnnnnnnnnnnnnnnnnnnnngggggggggggggggggggggggggggggggggggggsubet",
			isInternal:    true,
			useStandardLB: false,
			protocol:      v1.ProtocolTCP,
			port:          9000,
			expected:      "a257b965551374ad2b091ef3f07043ad-averylonnnngggnnnnnnnnnnnnnnnnnnnnnngg-TCP-9000",
		},
		{
			description:   "external standard lb should not have subnet name on the rule name",
			subnetName:    "shortsubnet",
			isInternal:    false,
			useStandardLB: true,
			protocol:      v1.ProtocolTCP,
			port:          9000,
			expected:      "a257b965551374ad2b091ef3f07043ad-TCP-9000",
		},
		{
			description:   "external basic lb should not have subnet name on the rule name",
			subnetName:    "shortsubnet",
			isInternal:    false,
			useStandardLB: false,
			protocol:      v1.ProtocolTCP,
			port:          9000,
			expected:      "a257b965551374ad2b091ef3f07043ad-TCP-9000",
		},
	}

	for _, c := range cases {
		if c.useStandardLB {
			az.Config.LoadBalancerSku = loadBalancerSkuStandard
		} else {
			az.Config.LoadBalancerSku = loadBalancerSkuBasic
		}
		svc.Annotations[ServiceAnnotationLoadBalancerInternalSubnet] = c.subnetName
		svc.Annotations[ServiceAnnotationLoadBalancerInternal] = strconv.FormatBool(c.isInternal)

		loadbalancerRuleName := az.getLoadBalancerRuleName(svc, c.protocol, c.port)
		assert.Equal(t, c.expected, loadbalancerRuleName, c.description)
	}
}

func TestGetFrontendIPConfigName(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	az := GetTestCloud(ctrl)
	az.PrimaryAvailabilitySetName = "primary"

	svc := &v1.Service{
		ObjectMeta: meta.ObjectMeta{
			Annotations: map[string]string{
				ServiceAnnotationLoadBalancerInternalSubnet: "subnet",
				ServiceAnnotationLoadBalancerInternal:       "true",
			},
			UID: "257b9655-5137-4ad2-b091-ef3f07043ad3",
		},
	}

	cases := []struct {
		description   string
		subnetName    string
		isInternal    bool
		useStandardLB bool
		expected      string
	}{
		{
			description:   "internal lb should have subnet name on the frontend ip configuration name",
			subnetName:    "shortsubnet",
			isInternal:    true,
			useStandardLB: true,
			expected:      "a257b965551374ad2b091ef3f07043ad-shortsubnet",
		},
		{
			description:   "internal standard lb should have subnet name on the frontend ip configuration name but truncated to 80 characters",
			subnetName:    "averylonnnngggnnnnnnnnnnnnnnnnnnnnnngggggggggggggggggggggggggggggggggggggsubet",
			isInternal:    true,
			useStandardLB: true,
			expected:      "a257b965551374ad2b091ef3f07043ad-averylonnnngggnnnnnnnnnnnnnnnnnnnnnnggggggggggg",
		},
		{
			description:   "internal basic lb should have subnet name on the frontend ip configuration name but truncated to 80 characters",
			subnetName:    "averylonnnngggnnnnnnnnnnnnnnnnnnnnnngggggggggggggggggggggggggggggggggggggsubet",
			isInternal:    true,
			useStandardLB: false,
			expected:      "a257b965551374ad2b091ef3f07043ad-averylonnnngggnnnnnnnnnnnnnnnnnnnnnnggggggggggg",
		},
		{
			description:   "external standard lb should not have subnet name on the frontend ip configuration name",
			subnetName:    "shortsubnet",
			isInternal:    false,
			useStandardLB: true,
			expected:      "a257b965551374ad2b091ef3f07043ad",
		},
		{
			description:   "external basic lb should not have subnet name on the frontend ip configuration name",
			subnetName:    "shortsubnet",
			isInternal:    false,
			useStandardLB: false,
			expected:      "a257b965551374ad2b091ef3f07043ad",
		},
	}

	for _, c := range cases {
		if c.useStandardLB {
			az.Config.LoadBalancerSku = loadBalancerSkuStandard
		} else {
			az.Config.LoadBalancerSku = loadBalancerSkuBasic
		}
		svc.Annotations[ServiceAnnotationLoadBalancerInternalSubnet] = c.subnetName
		svc.Annotations[ServiceAnnotationLoadBalancerInternal] = strconv.FormatBool(c.isInternal)

		ipconfigName := az.getDefaultFrontendIPConfigName(svc)
		assert.Equal(t, c.expected, ipconfigName, c.description)
	}
}

func TestGetFrontendIPConfigID(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	az := GetTestCloud(ctrl)

	testGetLoadBalancerSubResourceID(t, az, az.getFrontendIPConfigID, frontendIPConfigIDTemplate)
}

func TestGetBackendPoolID(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	az := GetTestCloud(ctrl)

	testGetLoadBalancerSubResourceID(t, az, az.getBackendPoolID, backendPoolIDTemplate)
}

func TestGetLoadBalancerProbeID(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	az := GetTestCloud(ctrl)

	testGetLoadBalancerSubResourceID(t, az, az.getLoadBalancerProbeID, loadBalancerProbeIDTemplate)
}

func testGetLoadBalancerSubResourceID(
	t *testing.T,
	az *Cloud,
	getLoadBalancerSubResourceID func(string, string, string) string,
	expectedResourceIDTemplate string) {
	cases := []struct {
		description                         string
		loadBalancerName                    string
		resourceGroupName                   string
		subResourceName                     string
		useNetworkResourceInDifferentTenant bool
		expected                            string
	}{
		{
			description:                         "resource id should contain NetworkResourceSubscriptionID when using network resources in different subscription",
			loadBalancerName:                    "lbName",
			resourceGroupName:                   "rgName",
			subResourceName:                     "subResourceName",
			useNetworkResourceInDifferentTenant: true,
		},
		{
			description:                         "resource id should contain SubscriptionID when not using network resources in different subscription",
			loadBalancerName:                    "lbName",
			resourceGroupName:                   "rgName",
			subResourceName:                     "subResourceName",
			useNetworkResourceInDifferentTenant: false,
		},
	}

	for _, c := range cases {
		if c.useNetworkResourceInDifferentTenant {
			az.NetworkResourceTenantID = networkResourceTenantID
			az.NetworkResourceSubscriptionID = networkResourceSubscriptionID
			c.expected = fmt.Sprintf(
				expectedResourceIDTemplate,
				az.NetworkResourceSubscriptionID,
				c.resourceGroupName,
				c.loadBalancerName,
				c.subResourceName)
		} else {
			az.NetworkResourceTenantID = ""
			az.NetworkResourceSubscriptionID = ""
			c.expected = fmt.Sprintf(
				expectedResourceIDTemplate,
				az.SubscriptionID,
				c.resourceGroupName,
				c.loadBalancerName,
				c.subResourceName)
		}
		subResourceID := getLoadBalancerSubResourceID(c.loadBalancerName, c.resourceGroupName, c.subResourceName)
		assert.Equal(t, c.expected, subResourceID, c.description)
	}
}

func TestGetProtocolsFromKubernetesProtocol(t *testing.T) {
	testcases := []struct {
		Name                       string
		protocol                   v1.Protocol
		expectedTransportProto     network.TransportProtocol
		expectedSecurityGroupProto network.SecurityRuleProtocol
		expectedProbeProto         network.ProbeProtocol
		nilProbeProto              bool
		expectedErrMsg             error
	}{
		{
			Name:                       "getProtocolsFromKubernetesProtocol should get TCP protocol",
			protocol:                   v1.ProtocolTCP,
			expectedTransportProto:     network.TransportProtocolTCP,
			expectedSecurityGroupProto: network.SecurityRuleProtocolTCP,
			expectedProbeProto:         network.ProbeProtocolTCP,
		},
		{
			Name:                       "getProtocolsFromKubernetesProtocol should get UDP protocol",
			protocol:                   v1.ProtocolUDP,
			expectedTransportProto:     network.TransportProtocolUDP,
			expectedSecurityGroupProto: network.SecurityRuleProtocolUDP,
			nilProbeProto:              true,
		},
		{
			Name:           "getProtocolsFromKubernetesProtocol should report error",
			protocol:       v1.ProtocolSCTP,
			expectedErrMsg: fmt.Errorf("only TCP and UDP are supported for Azure LoadBalancers"),
		},
	}

	for _, test := range testcases {
		transportProto, securityGroupProto, probeProto, err := getProtocolsFromKubernetesProtocol(test.protocol)
		assert.Equal(t, test.expectedTransportProto, *transportProto, test.Name)
		assert.Equal(t, test.expectedSecurityGroupProto, *securityGroupProto, test.Name)
		if test.nilProbeProto {
			assert.Nil(t, probeProto, test.Name)
		} else {
			assert.Equal(t, test.expectedProbeProto, *probeProto, test.Name)
		}
		assert.Equal(t, test.expectedErrMsg, err, test.Name)
	}
}

func TestGetStandardVMPrimaryInterfaceID(t *testing.T) {
	testcases := []struct {
		name           string
		vm             compute.VirtualMachine
		expectedNicID  string
		expectedErrMsg error
	}{
		{
			name:          "GetPrimaryInterfaceID should get the only NIC ID",
			vm:            buildDefaultTestVirtualMachine("", []string{"/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/networkInterfaces/nic"}),
			expectedNicID: "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/networkInterfaces/nic",
		},
		{
			name: "GetPrimaryInterfaceID should get primary NIC ID",
			vm: compute.VirtualMachine{
				Name: to.StringPtr("vm2"),
				VirtualMachineProperties: &compute.VirtualMachineProperties{
					NetworkProfile: &compute.NetworkProfile{
						NetworkInterfaces: &[]compute.NetworkInterfaceReference{
							{
								NetworkInterfaceReferenceProperties: &compute.NetworkInterfaceReferenceProperties{
									Primary: to.BoolPtr(true),
								},
								ID: to.StringPtr("/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/networkInterfaces/nic1"),
							},
							{
								NetworkInterfaceReferenceProperties: &compute.NetworkInterfaceReferenceProperties{
									Primary: to.BoolPtr(false),
								},
								ID: to.StringPtr("/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/networkInterfaces/nic2"),
							},
						},
					},
				},
			},
			expectedNicID: "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/networkInterfaces/nic1",
		},
		{
			name: "GetPrimaryInterfaceID should report error if node don't have primary NIC",
			vm: compute.VirtualMachine{
				Name: to.StringPtr("vm3"),
				VirtualMachineProperties: &compute.VirtualMachineProperties{
					NetworkProfile: &compute.NetworkProfile{
						NetworkInterfaces: &[]compute.NetworkInterfaceReference{
							{
								NetworkInterfaceReferenceProperties: &compute.NetworkInterfaceReferenceProperties{
									Primary: to.BoolPtr(false),
								},
								ID: to.StringPtr("/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/networkInterfaces/nic1"),
							},
							{
								NetworkInterfaceReferenceProperties: &compute.NetworkInterfaceReferenceProperties{
									Primary: to.BoolPtr(false),
								},
								ID: to.StringPtr("/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/networkInterfaces/nic2"),
							},
						},
					},
				},
			},
			expectedErrMsg: fmt.Errorf("failed to find a primary nic for the vm. vmname=%q", "vm3"),
		},
	}

	for _, test := range testcases {
		primaryNicID, err := getPrimaryInterfaceID(test.vm)
		assert.Equal(t, test.expectedNicID, primaryNicID, test.name)
		assert.Equal(t, test.expectedErrMsg, err, test.name)
	}
}

func TestGetPrimaryIPConfig(t *testing.T) {
	testcases := []struct {
		name             string
		nic              network.Interface
		expectedIPConfig *network.InterfaceIPConfiguration
		expectedErrMsg   error
	}{

		{
			name: "GetPrimaryIPConfig should get the only IP configuration",
			nic: network.Interface{
				Name: to.StringPtr("nic"),
				InterfacePropertiesFormat: &network.InterfacePropertiesFormat{
					IPConfigurations: &[]network.InterfaceIPConfiguration{
						{
							Name: to.StringPtr("ipconfig1"),
						},
					},
				},
			},
			expectedIPConfig: &network.InterfaceIPConfiguration{
				Name: to.StringPtr("ipconfig1"),
			},
		},
		{
			name: "GetPrimaryIPConfig should get the primary IP configuration",
			nic: network.Interface{
				Name: to.StringPtr("nic"),
				InterfacePropertiesFormat: &network.InterfacePropertiesFormat{
					IPConfigurations: &[]network.InterfaceIPConfiguration{
						{
							Name: to.StringPtr("ipconfig1"),
							InterfaceIPConfigurationPropertiesFormat: &network.InterfaceIPConfigurationPropertiesFormat{
								Primary: to.BoolPtr(true),
							},
						},
						{
							Name: to.StringPtr("ipconfig2"),
							InterfaceIPConfigurationPropertiesFormat: &network.InterfaceIPConfigurationPropertiesFormat{
								Primary: to.BoolPtr(false),
							},
						},
					},
				},
			},
			expectedIPConfig: &network.InterfaceIPConfiguration{
				Name: to.StringPtr("ipconfig1"),
				InterfaceIPConfigurationPropertiesFormat: &network.InterfaceIPConfigurationPropertiesFormat{
					Primary: to.BoolPtr(true),
				},
			},
		},
		{
			name: "GetPrimaryIPConfig should report error if nic don't have IP configuration",
			nic: network.Interface{
				Name:                      to.StringPtr("nic"),
				InterfacePropertiesFormat: &network.InterfacePropertiesFormat{},
			},
			expectedErrMsg: fmt.Errorf("nic.IPConfigurations for nic (nicname=%q) is nil", "nic"),
		},
		{
			name: "GetPrimaryIPConfig should report error if node has more than one IP configuration and don't have primary IP configuration",
			nic: network.Interface{
				Name: to.StringPtr("nic"),
				InterfacePropertiesFormat: &network.InterfacePropertiesFormat{
					IPConfigurations: &[]network.InterfaceIPConfiguration{
						{
							Name: to.StringPtr("ipconfig1"),
							InterfaceIPConfigurationPropertiesFormat: &network.InterfaceIPConfigurationPropertiesFormat{
								Primary: to.BoolPtr(false),
							},
						},
						{
							Name: to.StringPtr("ipconfig2"),
							InterfaceIPConfigurationPropertiesFormat: &network.InterfaceIPConfigurationPropertiesFormat{
								Primary: to.BoolPtr(false),
							},
						},
					},
				},
			},
			expectedErrMsg: fmt.Errorf("failed to determine the primary ipconfig. nicname=%q", "nic"),
		},
	}

	for _, test := range testcases {
		primaryIPConfig, err := getPrimaryIPConfig(test.nic)
		assert.Equal(t, test.expectedIPConfig, primaryIPConfig, test.name)
		assert.Equal(t, test.expectedErrMsg, err, test.name)
	}
}

func TestGetIPConfigByIPFamily(t *testing.T) {
	ipv4IPconfig := network.InterfaceIPConfiguration{
		Name: to.StringPtr("ipconfig1"),
		InterfaceIPConfigurationPropertiesFormat: &network.InterfaceIPConfigurationPropertiesFormat{
			PrivateIPAddressVersion: network.IPv4,
			PrivateIPAddress:        to.StringPtr("10.10.0.12"),
		},
	}
	ipv6IPconfig := network.InterfaceIPConfiguration{
		Name: to.StringPtr("ipconfig2"),
		InterfaceIPConfigurationPropertiesFormat: &network.InterfaceIPConfigurationPropertiesFormat{
			PrivateIPAddressVersion: network.IPv6,
			PrivateIPAddress:        to.StringPtr("1111:11111:00:00:1111:1111:000:111"),
		},
	}
	testNic := network.Interface{
		Name: to.StringPtr("nic"),
		InterfacePropertiesFormat: &network.InterfacePropertiesFormat{
			IPConfigurations: &[]network.InterfaceIPConfiguration{ipv4IPconfig, ipv6IPconfig},
		},
	}
	testcases := []struct {
		name             string
		nic              network.Interface
		expectedIPConfig *network.InterfaceIPConfiguration
		IPv6             bool
		expectedErrMsg   error
	}{
		{
			name:             "GetIPConfigByIPFamily should get the IPv6 IP configuration if IPv6 is false",
			nic:              testNic,
			expectedIPConfig: &ipv4IPconfig,
		},
		{
			name:             "GetIPConfigByIPFamily should get the IPv4 IP configuration if IPv6 is true",
			nic:              testNic,
			IPv6:             true,
			expectedIPConfig: &ipv6IPconfig,
		},
		{
			name: "GetIPConfigByIPFamily should report error if nic don't have IP configuration",
			nic: network.Interface{
				Name:                      to.StringPtr("nic"),
				InterfacePropertiesFormat: &network.InterfacePropertiesFormat{},
			},
			expectedErrMsg: fmt.Errorf("nic.IPConfigurations for nic (nicname=%q) is nil", "nic"),
		},
		{
			name: "GetIPConfigByIPFamily should report error if nic don't have IPv6 configuration when IPv6 is true",
			nic: network.Interface{
				Name: to.StringPtr("nic"),
				InterfacePropertiesFormat: &network.InterfacePropertiesFormat{
					IPConfigurations: &[]network.InterfaceIPConfiguration{ipv4IPconfig},
				},
			},
			IPv6:           true,
			expectedErrMsg: fmt.Errorf("failed to determine the ipconfig(IPv6=%v). nicname=%q", true, "nic"),
		},
		{
			name: "GetIPConfigByIPFamily should report error if nic don't have PrivateIPAddress",
			nic: network.Interface{
				Name: to.StringPtr("nic"),
				InterfacePropertiesFormat: &network.InterfacePropertiesFormat{
					IPConfigurations: &[]network.InterfaceIPConfiguration{
						{
							Name: to.StringPtr("ipconfig1"),
							InterfaceIPConfigurationPropertiesFormat: &network.InterfaceIPConfigurationPropertiesFormat{
								PrivateIPAddressVersion: network.IPv4,
							},
						},
					},
				},
			},
			expectedErrMsg: fmt.Errorf("failed to determine the ipconfig(IPv6=%v). nicname=%q", false, "nic"),
		},
	}

	for _, test := range testcases {
		ipConfig, err := getIPConfigByIPFamily(test.nic, test.IPv6)
		assert.Equal(t, test.expectedIPConfig, ipConfig, test.name)
		assert.Equal(t, test.expectedErrMsg, err, test.name)
	}
}

func TestGetBackendPoolName(t *testing.T) {
	testcases := []struct {
		name             string
		service          v1.Service
		clusterName      string
		expectedPoolName string
	}{
		{
			name:             "GetBackendPoolName should return <clusterName>-IPv6",
			service:          getTestService("test1", v1.ProtocolTCP, nil, true, 80),
			clusterName:      "azure",
			expectedPoolName: "azure-IPv6",
		},
		{
			name:             "GetBackendPoolName should return <clusterName>",
			service:          getTestService("test1", v1.ProtocolTCP, nil, false, 80),
			clusterName:      "azure",
			expectedPoolName: "azure",
		},
	}
	for _, test := range testcases {
		backPoolName := getBackendPoolName(test.clusterName, &test.service)
		assert.Equal(t, test.expectedPoolName, backPoolName, test.name)
	}
}

func TestGetStandardInstanceIDByNodeName(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	cloud := GetTestCloud(ctrl)

	expectedVM := compute.VirtualMachine{
		Name: to.StringPtr("vm1"),
		ID:   to.StringPtr("/subscriptions/subscription/resourceGroups/rg/providers/Microsoft.Compute/virtualMachines/vm1"),
	}
	invalidResouceID := "/subscriptions/subscription/resourceGroups/rg/Microsoft.Compute/virtualMachines/vm4"
	testcases := []struct {
		name           string
		nodeName       string
		expectedID     string
		expectedErrMsg error
	}{
		{
			name:       "GetInstanceIDByNodeName should get instanceID as expected",
			nodeName:   "vm1",
			expectedID: "/subscriptions/subscription/resourceGroups/rg/providers/Microsoft.Compute/virtualMachines/vm1",
		},
		{
			name:           "GetInstanceIDByNodeName should report error if node don't exist",
			nodeName:       "vm2",
			expectedErrMsg: fmt.Errorf("instance not found"),
		},
		{
			name:           "GetInstanceIDByNodeName should report error if Error encountered when invoke mockVMClient.Get",
			nodeName:       "vm3",
			expectedErrMsg: fmt.Errorf("Retriable: false, RetryAfter: 0s, HTTPStatusCode: 500, RawError: %w", fmt.Errorf("VMGet error")),
		},
		{
			name:           "GetInstanceIDByNodeName should report error if ResourceID is invalid",
			nodeName:       "vm4",
			expectedErrMsg: fmt.Errorf("%q isn't in Azure resource ID format %q", invalidResouceID, azureResourceGroupNameRE.String()),
		},
	}
	for _, test := range testcases {
		mockVMClient := cloud.VirtualMachinesClient.(*mockvmclient.MockInterface)
		mockVMClient.EXPECT().Get(gomock.Any(), cloud.ResourceGroup, "vm1", gomock.Any()).Return(expectedVM, nil).AnyTimes()
		mockVMClient.EXPECT().Get(gomock.Any(), cloud.ResourceGroup, "vm2", gomock.Any()).Return(compute.VirtualMachine{}, &retry.Error{HTTPStatusCode: http.StatusNotFound, RawError: cloudprovider.InstanceNotFound}).AnyTimes()
		mockVMClient.EXPECT().Get(gomock.Any(), cloud.ResourceGroup, "vm3", gomock.Any()).Return(compute.VirtualMachine{}, &retry.Error{
			HTTPStatusCode: http.StatusInternalServerError,
			RawError:       fmt.Errorf("VMGet error"),
		}).AnyTimes()
		mockVMClient.EXPECT().Get(gomock.Any(), cloud.ResourceGroup, "vm4", gomock.Any()).Return(compute.VirtualMachine{
			Name: to.StringPtr("vm4"),
			ID:   to.StringPtr(invalidResouceID),
		}, nil).AnyTimes()

		instanceID, err := cloud.VMSet.GetInstanceIDByNodeName(test.nodeName)
		assert.Equal(t, test.expectedErrMsg, err, test.name)
		assert.Equal(t, test.expectedID, instanceID, test.name)
	}
}

func TestGetStandardVMPowerStatusByNodeName(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	cloud := GetTestCloud(ctrl)

	testcases := []struct {
		name           string
		nodeName       string
		vm             compute.VirtualMachine
		expectedStatus string
		getErr         *retry.Error
		expectedErrMsg error
	}{
		{
			name:     "GetPowerStatusByNodeName should report error if node don't exist",
			nodeName: "vm1",
			vm:       compute.VirtualMachine{},
			getErr: &retry.Error{
				HTTPStatusCode: http.StatusNotFound,
				RawError:       cloudprovider.InstanceNotFound,
			},
			expectedErrMsg: fmt.Errorf("instance not found"),
		},
		{
			name:     "GetPowerStatusByNodeName should get power status as expected",
			nodeName: "vm2",
			vm: compute.VirtualMachine{
				Name: to.StringPtr("vm2"),
				VirtualMachineProperties: &compute.VirtualMachineProperties{
					InstanceView: &compute.VirtualMachineInstanceView{
						Statuses: &[]compute.InstanceViewStatus{
							{
								Code: to.StringPtr("PowerState/Running"),
							},
						},
					},
				},
			},
			expectedStatus: "Running",
		},
		{
			name:     "GetPowerStatusByNodeName should get vmPowerStateStopped if vm.InstanceView is nil",
			nodeName: "vm3",
			vm: compute.VirtualMachine{
				Name:                     to.StringPtr("vm3"),
				VirtualMachineProperties: &compute.VirtualMachineProperties{},
			},
			expectedStatus: vmPowerStateStopped,
		},
		{
			name:     "GetPowerStatusByNodeName should get vmPowerStateStopped if vm.InstanceView.statuses is nil",
			nodeName: "vm4",
			vm: compute.VirtualMachine{
				Name: to.StringPtr("vm4"),
				VirtualMachineProperties: &compute.VirtualMachineProperties{
					InstanceView: &compute.VirtualMachineInstanceView{},
				},
			},
			expectedStatus: vmPowerStateStopped,
		},
	}
	for _, test := range testcases {
		mockVMClient := cloud.VirtualMachinesClient.(*mockvmclient.MockInterface)
		mockVMClient.EXPECT().Get(gomock.Any(), cloud.ResourceGroup, test.nodeName, gomock.Any()).Return(test.vm, test.getErr).AnyTimes()

		powerState, err := cloud.VMSet.GetPowerStatusByNodeName(test.nodeName)
		assert.Equal(t, test.expectedErrMsg, err, test.name)
		assert.Equal(t, test.expectedStatus, powerState, test.name)
	}
}

func TestGetStandardVMProvisioningStateByNodeName(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	cloud := GetTestCloud(ctrl)

	testcases := []struct {
		name                      string
		nodeName                  string
		vm                        compute.VirtualMachine
		expectedProvisioningState string
		getErr                    *retry.Error
		expectedErrMsg            error
	}{
		{
			name:     "GetProvisioningStateByNodeName should report error if node don't exist",
			nodeName: "vm1",
			vm:       compute.VirtualMachine{},
			getErr: &retry.Error{
				HTTPStatusCode: http.StatusNotFound,
				RawError:       cloudprovider.InstanceNotFound,
			},
			expectedErrMsg: fmt.Errorf("instance not found"),
		},
		{
			name:     "GetProvisioningStateByNodeName should return Succeeded for running VM",
			nodeName: "vm2",
			vm: compute.VirtualMachine{
				Name: to.StringPtr("vm2"),
				VirtualMachineProperties: &compute.VirtualMachineProperties{
					ProvisioningState: to.StringPtr("Succeeded"),
					InstanceView: &compute.VirtualMachineInstanceView{
						Statuses: &[]compute.InstanceViewStatus{
							{
								Code: to.StringPtr("PowerState/Running"),
							},
						},
					},
				},
			},
			expectedProvisioningState: "Succeeded",
		},
		{
			name:     "GetProvisioningStateByNodeName should return empty string when vm.ProvisioningState is nil",
			nodeName: "vm3",
			vm: compute.VirtualMachine{
				Name: to.StringPtr("vm3"),
				VirtualMachineProperties: &compute.VirtualMachineProperties{
					ProvisioningState: nil,
				},
			},
			expectedProvisioningState: "",
		},
	}
	for _, test := range testcases {
		mockVMClient := cloud.VirtualMachinesClient.(*mockvmclient.MockInterface)
		mockVMClient.EXPECT().Get(gomock.Any(), cloud.ResourceGroup, test.nodeName, gomock.Any()).Return(test.vm, test.getErr).AnyTimes()

		provisioningState, err := cloud.VMSet.GetProvisioningStateByNodeName(test.nodeName)
		assert.Equal(t, test.expectedErrMsg, err, test.name)
		assert.Equal(t, test.expectedProvisioningState, provisioningState, test.name)
	}
}

func TestGetStandardVMZoneByNodeName(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	cloud := GetTestCloud(ctrl)

	var faultDomain int32 = 3
	testcases := []struct {
		name           string
		nodeName       string
		vm             compute.VirtualMachine
		expectedZone   cloudprovider.Zone
		getErr         *retry.Error
		expectedErrMsg error
	}{
		{
			name:     "GetZoneByNodeName should report error if node don't exist",
			nodeName: "vm1",
			vm:       compute.VirtualMachine{},
			getErr: &retry.Error{
				HTTPStatusCode: http.StatusNotFound,
				RawError:       cloudprovider.InstanceNotFound,
			},
			expectedErrMsg: fmt.Errorf("instance not found"),
		},
		{
			name:     "GetZoneByNodeName should get zone as expected",
			nodeName: "vm2",
			vm: compute.VirtualMachine{
				Name:     to.StringPtr("vm2"),
				Location: to.StringPtr("EASTUS"),
				Zones:    &[]string{"2"},
				VirtualMachineProperties: &compute.VirtualMachineProperties{
					InstanceView: &compute.VirtualMachineInstanceView{
						PlatformFaultDomain: &faultDomain,
					},
				},
			},
			expectedZone: cloudprovider.Zone{
				FailureDomain: "eastus-2",
				Region:        "eastus",
			},
		},
		{
			name:     "GetZoneByNodeName should get FailureDomain as zone if zone is not used for node",
			nodeName: "vm3",
			vm: compute.VirtualMachine{
				Name:     to.StringPtr("vm3"),
				Location: to.StringPtr("EASTUS"),
				VirtualMachineProperties: &compute.VirtualMachineProperties{
					InstanceView: &compute.VirtualMachineInstanceView{
						PlatformFaultDomain: &faultDomain,
					},
				},
			},
			expectedZone: cloudprovider.Zone{
				FailureDomain: "3",
				Region:        "eastus",
			},
		},
		{
			name:     "GetZoneByNodeName should report error if zones is invalid",
			nodeName: "vm4",
			vm: compute.VirtualMachine{
				Name:     to.StringPtr("vm4"),
				Location: to.StringPtr("EASTUS"),
				Zones:    &[]string{"a"},
				VirtualMachineProperties: &compute.VirtualMachineProperties{
					InstanceView: &compute.VirtualMachineInstanceView{
						PlatformFaultDomain: &faultDomain,
					},
				},
			},
			expectedErrMsg: fmt.Errorf("failed to parse zone %q: strconv.Atoi: parsing %q: invalid syntax", []string{"a"}, "a"),
		},
	}
	for _, test := range testcases {
		mockVMClient := cloud.VirtualMachinesClient.(*mockvmclient.MockInterface)
		mockVMClient.EXPECT().Get(gomock.Any(), cloud.ResourceGroup, test.nodeName, gomock.Any()).Return(test.vm, test.getErr).AnyTimes()

		zone, err := cloud.VMSet.GetZoneByNodeName(test.nodeName)
		assert.Equal(t, test.expectedErrMsg, err, test.name)
		assert.Equal(t, test.expectedZone, zone, test.name)
	}
}

func TestGetStandardVMSetNames(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	asID := "/subscriptions/subscription/resourceGroups/rg/providers/Microsoft.Compute/availabilitySets/myAvailabilitySet"
	testVM := compute.VirtualMachine{
		Name: to.StringPtr("vm1"),
		VirtualMachineProperties: &compute.VirtualMachineProperties{
			AvailabilitySet: &compute.SubResource{ID: to.StringPtr(asID)},
		},
	}
	testVMWithoutAS := compute.VirtualMachine{
		Name:                     to.StringPtr("vm2"),
		VirtualMachineProperties: &compute.VirtualMachineProperties{},
	}
	testCases := []struct {
		name               string
		vm                 []compute.VirtualMachine
		service            *v1.Service
		nodes              []*v1.Node
		usingSingleSLBS    bool
		expectedVMSetNames *[]string
		expectedErrMsg     error
	}{
		{
			name:               "GetVMSetNames should return the primary vm set name if the service has no mode annotation",
			vm:                 []compute.VirtualMachine{testVM},
			service:            &v1.Service{},
			expectedVMSetNames: &[]string{"as"},
		},
		{
			name: "GetVMSetNames should return the primary vm set name when using the single SLB",
			vm:   []compute.VirtualMachine{testVM},
			service: &v1.Service{
				ObjectMeta: meta.ObjectMeta{Annotations: map[string]string{ServiceAnnotationLoadBalancerMode: ServiceAnnotationLoadBalancerAutoModeValue}},
			},
			usingSingleSLBS:    true,
			expectedVMSetNames: &[]string{"as"},
		},
		{
			name: "GetVMSetNames should return the correct as names if the service has auto mode annotation",
			vm:   []compute.VirtualMachine{testVM},
			service: &v1.Service{
				ObjectMeta: meta.ObjectMeta{Annotations: map[string]string{ServiceAnnotationLoadBalancerMode: ServiceAnnotationLoadBalancerAutoModeValue}},
			},
			nodes: []*v1.Node{
				{
					ObjectMeta: meta.ObjectMeta{
						Name: "vm1",
					},
				},
			},
			expectedVMSetNames: &[]string{"myavailabilityset"},
		},
		{
			name: "GetVMSetNames should return the correct as names if node don't have availability set",
			vm:   []compute.VirtualMachine{testVMWithoutAS},
			service: &v1.Service{
				ObjectMeta: meta.ObjectMeta{Annotations: map[string]string{ServiceAnnotationLoadBalancerMode: ServiceAnnotationLoadBalancerAutoModeValue}},
			},
			nodes: []*v1.Node{
				{
					ObjectMeta: meta.ObjectMeta{
						Name: "vm2",
					},
				},
			},
			expectedErrMsg: fmt.Errorf("node (vm2) - has no availability sets"),
		},
		{
			name: "GetVMSetNames should report the error if there's no such availability set",
			vm:   []compute.VirtualMachine{testVM},
			service: &v1.Service{
				ObjectMeta: meta.ObjectMeta{Annotations: map[string]string{ServiceAnnotationLoadBalancerMode: "vm2"}},
			},
			nodes: []*v1.Node{
				{
					ObjectMeta: meta.ObjectMeta{
						Name: "vm1",
					},
				},
			},
			expectedErrMsg: fmt.Errorf("availability set (vm2) - not found"),
		},
		{
			name: "GetVMSetNames should return the correct node name",
			vm:   []compute.VirtualMachine{testVM},
			service: &v1.Service{
				ObjectMeta: meta.ObjectMeta{Annotations: map[string]string{ServiceAnnotationLoadBalancerMode: "myAvailabilitySet"}},
			},
			nodes: []*v1.Node{
				{
					ObjectMeta: meta.ObjectMeta{
						Name: "vm1",
					},
				},
			},
			expectedVMSetNames: &[]string{"myavailabilityset"},
		},
	}

	for _, test := range testCases {
		cloud := GetTestCloud(ctrl)
		if test.usingSingleSLBS {
			cloud.EnableMultipleStandardLoadBalancers = false
			cloud.LoadBalancerSku = loadBalancerSkuStandard
		}
		mockVMClient := cloud.VirtualMachinesClient.(*mockvmclient.MockInterface)
		mockVMClient.EXPECT().List(gomock.Any(), cloud.ResourceGroup).Return(test.vm, nil).AnyTimes()

		vmSetNames, err := cloud.VMSet.GetVMSetNames(test.service, test.nodes)
		assert.Equal(t, test.expectedErrMsg, err, test.name)
		assert.Equal(t, test.expectedVMSetNames, vmSetNames, test.name)
	}
}

func TestExtractResourceGroupByNicID(t *testing.T) {
	testCases := []struct {
		name           string
		nicID          string
		expectedRG     string
		expectedErrMsg error
	}{
		{
			name:       "ExtractResourceGroupByNicID should return correct resource group",
			nicID:      "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/networkInterfaces/nic",
			expectedRG: "rg",
		},
		{
			name:           "ExtractResourceGroupByNicID should report error if nicID is invalid",
			nicID:          "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/networkInterfaces/nic",
			expectedErrMsg: fmt.Errorf("error of extracting resourceGroup from nicID %q", "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/networkInterfaces/nic"),
		},
	}

	for _, test := range testCases {
		rgName, err := extractResourceGroupByNicID(test.nicID)
		assert.Equal(t, test.expectedErrMsg, err, test.name)
		assert.Equal(t, test.expectedRG, rgName, test.name)
	}
}

func TestStandardEnsureHostInPool(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	cloud := GetTestCloud(ctrl)

	availabilitySetID := "/subscriptions/subscription/resourceGroups/rg/providers/Microsoft.Compute/availabilitySets/myAvailabilitySet"
	backendAddressPoolID := "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/loadBalancers/lb1-internal/backendAddressPools/backendpool-1"

	testCases := []struct {
		name              string
		service           *v1.Service
		nodeName          types.NodeName
		backendPoolID     string
		nicName           string
		nicID             string
		vmSetName         string
		nicProvisionState string
		isStandardLB      bool
		useMultipleSLBs   bool
		expectedErrMsg    error
	}{
		{
			name:      "EnsureHostInPool should return nil if node is not in VMSet",
			service:   &v1.Service{},
			nodeName:  "vm1",
			nicName:   "nic1",
			nicID:     "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/networkInterfaces/nic1",
			vmSetName: "availabilityset-1",
		},
		{
			name:            "EnsureHostInPool should return nil if node is not in VMSet when using multiple SLBs",
			service:         &v1.Service{},
			nodeName:        "vm1",
			nicName:         "nic1",
			nicID:           "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/networkInterfaces/nic1",
			vmSetName:       "availabilityset-1",
			isStandardLB:    true,
			useMultipleSLBs: true,
		},
		{
			name:           "EnsureHostInPool should report error if last segment of nicID is nil",
			service:        &v1.Service{},
			nodeName:       "vm2",
			nicName:        "nic2",
			nicID:          "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/networkInterfaces/",
			vmSetName:      "availabilityset-1",
			expectedErrMsg: fmt.Errorf("resource name was missing from identifier"),
		},
		{
			name:              "EnsureHostInPool should return nil if node's provisioning state is Failed",
			service:           &v1.Service{},
			nodeName:          "vm3",
			nicName:           "nic3",
			nicID:             "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/networkInterfaces/nic3",
			nicProvisionState: nicFailedState,
			vmSetName:         "myAvailabilitySet",
		},
		{
			name:           "EnsureHostInPool should report error if service.Spec.ClusterIP is ipv6 but node don't have IPv6 address",
			service:        &v1.Service{Spec: v1.ServiceSpec{ClusterIP: "2001:0db8:85a3:0000:0000:8a2e:0370:7334"}},
			nodeName:       "vm4",
			nicName:        "nic4",
			nicID:          "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/networkInterfaces/nic4",
			vmSetName:      "myAvailabilitySet",
			expectedErrMsg: fmt.Errorf("failed to determine the ipconfig(IPv6=true). nicname=%q", "nic4"),
		},
		{
			name:          "EnsureHostInPool should return nil if there is matched backend pool",
			service:       &v1.Service{},
			backendPoolID: backendAddressPoolID,
			nodeName:      "vm5",
			nicName:       "nic5",
			nicID:         "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/networkInterfaces/nic5",
			vmSetName:     "myAvailabilitySet",
		},
		{
			name:          "EnsureHostInPool should return nil if there isn't matched backend pool",
			service:       &v1.Service{},
			backendPoolID: "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/loadBalancers/lb1-internal/backendAddressPools/backendpool-2",
			nodeName:      "vm6",
			nicName:       "nic6",
			nicID:         "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/networkInterfaces/nic6",
			vmSetName:     "myAvailabilitySet",
		},
		{
			name:          "EnsureHostInPool should return nil if BackendPool is not on same LB",
			service:       &v1.Service{},
			isStandardLB:  true,
			backendPoolID: "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/loadBalancers/lb2-internal/backendAddressPools/backendpool-3",
			nodeName:      "vm7",
			nicName:       "nic7",
			nicID:         "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/networkInterfaces/nic7",
			vmSetName:     "myAvailabilitySet",
		},
		{
			name:           "EnsureHostInPool should report error if the format of backendPoolID is invalid",
			service:        &v1.Service{},
			isStandardLB:   true,
			backendPoolID:  "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/loadBalancers/lb2-internal/backendAddressPool/backendpool-3",
			nodeName:       "vm8",
			nicName:        "nic8",
			nicID:          "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/networkInterfaces/nic7",
			vmSetName:      "myAvailabilitySet",
			expectedErrMsg: fmt.Errorf("new backendPoolID %q is in wrong format", "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/loadBalancers/lb2-internal/backendAddressPool/backendpool-3"),
		},
	}

	for _, test := range testCases {
		if test.isStandardLB {
			cloud.Config.LoadBalancerSku = loadBalancerSkuStandard
		}

		if test.useMultipleSLBs {
			cloud.EnableMultipleStandardLoadBalancers = true
		}

		testVM := buildDefaultTestVirtualMachine(availabilitySetID, []string{test.nicID})
		testVM.Name = to.StringPtr(string(test.nodeName))
		testNIC := buildDefaultTestInterface(false, []string{backendAddressPoolID})
		testNIC.Name = to.StringPtr(test.nicName)
		testNIC.ID = to.StringPtr(test.nicID)
		testNIC.ProvisioningState = to.StringPtr(test.nicProvisionState)

		mockVMClient := cloud.VirtualMachinesClient.(*mockvmclient.MockInterface)
		mockVMClient.EXPECT().Get(gomock.Any(), cloud.ResourceGroup, string(test.nodeName), gomock.Any()).Return(testVM, nil).AnyTimes()

		mockInterfaceClient := cloud.InterfacesClient.(*mockinterfaceclient.MockInterface)
		mockInterfaceClient.EXPECT().Get(gomock.Any(), cloud.ResourceGroup, test.nicName, gomock.Any()).Return(testNIC, nil).AnyTimes()
		mockInterfaceClient.EXPECT().CreateOrUpdate(gomock.Any(), cloud.ResourceGroup, gomock.Any(), gomock.Any()).Return(nil).AnyTimes()

		_, _, _, vm, err := cloud.VMSet.EnsureHostInPool(test.service, test.nodeName, test.backendPoolID, test.vmSetName, false)
		assert.Equal(t, test.expectedErrMsg, err, test.name)
		assert.Nil(t, vm, test.name)
	}
}

func TestStandardEnsureHostsInPool(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	cloud := GetTestCloud(ctrl)

	availabilitySetID := "/subscriptions/subscription/resourceGroups/rg/providers/Microsoft.Compute/availabilitySets/myAvailabilitySet"
	backendAddressPoolID := "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/loadBalancers/lb1-internal/backendAddressPools/backendpool-1"

	testCases := []struct {
		name           string
		service        *v1.Service
		nodes          []*v1.Node
		excludeLBNodes []string
		nodeName       string
		backendPoolID  string
		nicName        string
		nicID          string
		vmSetName      string
		expectedErr    bool
		expectedErrMsg string
	}{
		{
			name:     "EnsureHostsInPool should return nil if there's no error when invoke EnsureHostInPool",
			service:  &v1.Service{},
			nodeName: "vm1",
			nodes: []*v1.Node{
				{
					ObjectMeta: meta.ObjectMeta{
						Name: "vm1",
					},
				},
			},
			nicName:       "nic1",
			nicID:         "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/networkInterfaces/nic1",
			backendPoolID: backendAddressPoolID,
			vmSetName:     "availabilityset-1",
		},
		{
			name:           "EnsureHostsInPool should skip if node is master node",
			service:        &v1.Service{},
			nodeName:       "vm2",
			excludeLBNodes: []string{"vm2"},
			nodes: []*v1.Node{
				{
					ObjectMeta: meta.ObjectMeta{
						Name:   "vm2",
						Labels: map[string]string{nodeLabelRole: "master"},
					},
				},
			},
			nicName:   "nic2",
			nicID:     "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/networkInterfaces/",
			vmSetName: "availabilityset-1",
		},
		{
			name:           "EnsureHostsInPool should skip if node is in external resource group",
			service:        &v1.Service{},
			nodeName:       "vm3",
			excludeLBNodes: []string{"vm3"},
			nodes: []*v1.Node{
				{
					ObjectMeta: meta.ObjectMeta{
						Name:   "vm3",
						Labels: map[string]string{externalResourceGroupLabel: "rg-external"},
					},
				},
			},
			nicName:   "nic3",
			nicID:     "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/networkInterfaces/nic3",
			vmSetName: "availabilityset-1",
		},
		{
			name:           "EnsureHostsInPool should skip if node is unmanaged",
			service:        &v1.Service{},
			nodeName:       "vm4",
			excludeLBNodes: []string{"vm4"},
			nodes: []*v1.Node{
				{
					ObjectMeta: meta.ObjectMeta{
						Name:   "vm4",
						Labels: map[string]string{managedByAzureLabel: "false"},
					},
				},
			},

			nicName:   "nic4",
			nicID:     "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/networkInterfaces/nic4",
			vmSetName: "availabilityset-1",
		},
		{
			name: "EnsureHostsInPool should report error if service.Spec.ClusterIP is ipv6 but node don't have IPv6 address",
			service: &v1.Service{
				ObjectMeta: meta.ObjectMeta{
					Name:      "svc",
					Namespace: "default",
				},
				Spec: v1.ServiceSpec{
					ClusterIP: "2001:0db8:85a3:0000:0000:8a2e:0370:7334",
				},
			},
			nodeName: "vm5",
			nodes: []*v1.Node{
				{
					ObjectMeta: meta.ObjectMeta{
						Name: "vm5",
					},
				},
			},
			nicName:        "nic5",
			backendPoolID:  backendAddressPoolID,
			nicID:          "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/networkInterfaces/nic5",
			vmSetName:      "myAvailabilitySet",
			expectedErr:    true,
			expectedErrMsg: fmt.Sprintf("ensure(default/svc): backendPoolID(%s) - failed to ensure host in pool: %q", backendAddressPoolID, fmt.Errorf("failed to determine the ipconfig(IPv6=true). nicname=%q", "nic5")),
		},
	}

	for _, test := range testCases {
		cloud.Config.LoadBalancerSku = loadBalancerSkuStandard
		cloud.Config.ExcludeMasterFromStandardLB = to.BoolPtr(true)
		cloud.excludeLoadBalancerNodes = sets.NewString(test.excludeLBNodes...)

		testVM := buildDefaultTestVirtualMachine(availabilitySetID, []string{test.nicID})
		testNIC := buildDefaultTestInterface(false, []string{backendAddressPoolID})
		testNIC.Name = to.StringPtr(test.nicName)
		testNIC.ID = to.StringPtr(test.nicID)

		mockVMClient := cloud.VirtualMachinesClient.(*mockvmclient.MockInterface)
		mockVMClient.EXPECT().Get(gomock.Any(), cloud.ResourceGroup, test.nodeName, gomock.Any()).Return(testVM, nil).AnyTimes()

		mockInterfaceClient := cloud.InterfacesClient.(*mockinterfaceclient.MockInterface)
		mockInterfaceClient.EXPECT().Get(gomock.Any(), cloud.ResourceGroup, test.nicName, gomock.Any()).Return(testNIC, nil).AnyTimes()
		mockInterfaceClient.EXPECT().CreateOrUpdate(gomock.Any(), cloud.ResourceGroup, gomock.Any(), gomock.Any()).Return(nil).AnyTimes()

		err := cloud.VMSet.EnsureHostsInPool(test.service, test.nodes, test.backendPoolID, test.vmSetName, false)
		if test.expectedErr {
			assert.Equal(t, test.expectedErrMsg, err.Error(), test.name)
		} else {
			assert.Nil(t, err, test.name)
		}
	}
}

func TestServiceOwnsFrontendIP(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	cloud := GetTestCloud(ctrl)

	testCases := []struct {
		desc         string
		existingPIPs []network.PublicIPAddress
		fip          network.FrontendIPConfiguration
		service      *v1.Service
		isOwned      bool
		isPrimary    bool
		expectedErr  error
	}{
		{
			desc: "serviceOwnsFrontendIP should detect the primary service",
			fip: network.FrontendIPConfiguration{
				Name: to.StringPtr("auid"),
			},
			service: &v1.Service{
				ObjectMeta: meta.ObjectMeta{
					UID: types.UID("uid"),
				},
			},
			isOwned:   true,
			isPrimary: true,
		},
		{
			desc: "serviceOwnsFrontendIP should return false if the secondary external service doesn't set it's loadBalancer IP",
			fip: network.FrontendIPConfiguration{
				Name: to.StringPtr("auid"),
			},
			service: &v1.Service{
				ObjectMeta: meta.ObjectMeta{
					UID: types.UID("secondary"),
				},
			},
		},
		{
			desc: "serviceOwnsFrontendIP should report a not found error if there is no public IP " +
				"found according to the external service's loadBalancer IP but do not return the error",
			existingPIPs: []network.PublicIPAddress{
				{
					ID: to.StringPtr("pip"),
					PublicIPAddressPropertiesFormat: &network.PublicIPAddressPropertiesFormat{
						IPAddress: to.StringPtr("4.3.2.1"),
					},
				},
			},
			fip: network.FrontendIPConfiguration{
				Name: to.StringPtr("auid"),
				FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
					PublicIPAddress: &network.PublicIPAddress{
						ID: to.StringPtr("pip"),
					},
				},
			},
			service: &v1.Service{
				ObjectMeta: meta.ObjectMeta{
					UID: types.UID("secondary"),
				},
				Spec: v1.ServiceSpec{
					LoadBalancerIP: "1.2.3.4",
				},
			},
		},
		{
			desc: "serviceOwnsFrontendIP should return false if there is a mismatch between the PIP's ID and " +
				"the counterpart on the frontend IP config",
			existingPIPs: []network.PublicIPAddress{
				{
					ID: to.StringPtr("pip"),
					PublicIPAddressPropertiesFormat: &network.PublicIPAddressPropertiesFormat{
						IPAddress: to.StringPtr("4.3.2.1"),
					},
				},
			},
			fip: network.FrontendIPConfiguration{
				Name: to.StringPtr("auid"),
				FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
					PublicIPAddress: &network.PublicIPAddress{
						ID: to.StringPtr("pip1"),
					},
				},
			},
			service: &v1.Service{
				ObjectMeta: meta.ObjectMeta{
					UID: types.UID("secondary"),
				},
				Spec: v1.ServiceSpec{
					LoadBalancerIP: "4.3.2.1",
				},
			},
		},
		{
			desc: "serviceOwnsFrontendIP should detect the secondary external service",
			existingPIPs: []network.PublicIPAddress{
				{
					ID: to.StringPtr("pip"),
					PublicIPAddressPropertiesFormat: &network.PublicIPAddressPropertiesFormat{
						IPAddress: to.StringPtr("4.3.2.1"),
					},
				},
			},
			fip: network.FrontendIPConfiguration{
				Name: to.StringPtr("auid"),
				FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
					PublicIPAddress: &network.PublicIPAddress{
						ID: to.StringPtr("pip"),
					},
				},
			},
			service: &v1.Service{
				ObjectMeta: meta.ObjectMeta{
					UID: types.UID("secondary"),
				},
				Spec: v1.ServiceSpec{
					LoadBalancerIP: "4.3.2.1",
				},
			},
			isOwned: true,
		},
		{
			desc: "serviceOwnsFrontendIP should detect the secondary internal service",
			fip: network.FrontendIPConfiguration{
				Name: to.StringPtr("auid"),
				FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
					PrivateIPAddress: to.StringPtr("4.3.2.1"),
				},
			},
			service: &v1.Service{
				ObjectMeta: meta.ObjectMeta{
					UID:         types.UID("secondary"),
					Annotations: map[string]string{ServiceAnnotationLoadBalancerInternal: "true"},
				},
				Spec: v1.ServiceSpec{
					LoadBalancerIP: "4.3.2.1",
				},
			},
			isOwned: true,
		},
	}

	for _, test := range testCases {
		mockPIPClient := mockpublicipclient.NewMockInterface(ctrl)
		cloud.PublicIPAddressesClient = mockPIPClient
		mockPIPClient.EXPECT().List(gomock.Any(), gomock.Any()).Return(test.existingPIPs, nil).MaxTimes(1)

		isOwned, isPrimary, err := cloud.serviceOwnsFrontendIP(test.fip, test.service)
		assert.Equal(t, test.expectedErr, err, test.desc)
		assert.Equal(t, test.isOwned, isOwned, test.desc)
		assert.Equal(t, test.isPrimary, isPrimary, test.desc)
	}
}

func TestStandardEnsureBackendPoolDeleted(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	cloud := GetTestCloud(ctrl)
	service := getTestService("test", v1.ProtocolTCP, nil, false, 80)
	backendPoolID := "backendPoolID"
	vmSetName := "AS"

	tests := []struct {
		desc                string
		backendAddressPools *[]network.BackendAddressPool
		loadBalancerSKU     string
		existingVM          compute.VirtualMachine
		existingNIC         network.Interface
	}{
		{
			desc: "EnsureBackendPoolDeleted should decouple the nic and the load balancer properly",
			backendAddressPools: &[]network.BackendAddressPool{
				{
					ID: to.StringPtr(backendPoolID),
					BackendAddressPoolPropertiesFormat: &network.BackendAddressPoolPropertiesFormat{
						BackendIPConfigurations: &[]network.InterfaceIPConfiguration{
							{
								ID: to.StringPtr("/subscriptions/subscription/resourceGroups/rg/providers/Microsoft.Network/networkInterfaces/k8s-agentpool1-00000000-nic-1/ipConfigurations/ipconfig1"),
							},
						},
					},
				},
			},
			existingVM: buildDefaultTestVirtualMachine("/subscriptions/subscription/resourceGroups/rg/providers/Microsoft.Compute/availabilitySets/as", []string{
				"/subscriptions/subscription/resourceGroups/rg/providers/Microsoft.Network/networkInterfaces/k8s-agentpool1-00000000-nic-1",
			}),
			existingNIC: buildDefaultTestInterface(true, []string{"/subscriptions/sub/resourceGroups/gh/providers/Microsoft.Network/loadBalancers/testCluster/backendAddressPools/testCluster"}),
		},
	}

	for _, test := range tests {
		cloud.LoadBalancerSku = test.loadBalancerSKU
		mockVMClient := mockvmclient.NewMockInterface(ctrl)
		mockVMClient.EXPECT().Get(gomock.Any(), cloud.ResourceGroup, "k8s-agentpool1-00000000-1", gomock.Any()).Return(test.existingVM, nil)
		cloud.VirtualMachinesClient = mockVMClient
		mockNICClient := mockinterfaceclient.NewMockInterface(ctrl)
		test.existingNIC.VirtualMachine = &network.SubResource{
			ID: to.StringPtr("/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachines/k8s-agentpool1-00000000-1"),
		}
		mockNICClient.EXPECT().Get(gomock.Any(), "rg", "k8s-agentpool1-00000000-nic-1", gomock.Any()).Return(test.existingNIC, nil).Times(2)
		mockNICClient.EXPECT().CreateOrUpdate(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Return(nil)
		cloud.InterfacesClient = mockNICClient

		err := cloud.VMSet.EnsureBackendPoolDeleted(&service, backendPoolID, vmSetName, test.backendAddressPools, true)
		assert.NoError(t, err, test.desc)
	}
}

func buildDefaultTestInterface(isPrimary bool, lbBackendpoolIDs []string) network.Interface {
	expectedNIC := network.Interface{
		InterfacePropertiesFormat: &network.InterfacePropertiesFormat{
			ProvisioningState: to.StringPtr("Succeeded"),
			IPConfigurations: &[]network.InterfaceIPConfiguration{
				{
					InterfaceIPConfigurationPropertiesFormat: &network.InterfaceIPConfigurationPropertiesFormat{
						Primary: to.BoolPtr(isPrimary),
					},
				},
			},
		},
	}
	backendAddressPool := make([]network.BackendAddressPool, 0)
	for _, id := range lbBackendpoolIDs {
		backendAddressPool = append(backendAddressPool, network.BackendAddressPool{
			ID: to.StringPtr(id),
		})
	}
	(*expectedNIC.IPConfigurations)[0].LoadBalancerBackendAddressPools = &backendAddressPool
	return expectedNIC
}

func buildDefaultTestVirtualMachine(asID string, nicIDs []string) compute.VirtualMachine {
	expectedVM := compute.VirtualMachine{
		VirtualMachineProperties: &compute.VirtualMachineProperties{
			AvailabilitySet: &compute.SubResource{
				ID: to.StringPtr(asID),
			},
			NetworkProfile: &compute.NetworkProfile{},
		},
	}
	networkInterfaces := make([]compute.NetworkInterfaceReference, 0)
	for _, nicID := range nicIDs {
		networkInterfaces = append(networkInterfaces, compute.NetworkInterfaceReference{
			ID: to.StringPtr(nicID),
		})
	}
	expectedVM.VirtualMachineProperties.NetworkProfile.NetworkInterfaces = &networkInterfaces
	return expectedVM
}

func TestStandardGetNodeNameByIPConfigurationID(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	cloud := GetTestCloud(ctrl)
	expectedVM := buildDefaultTestVirtualMachine("/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/availabilitySets/AGENTPOOL1-AVAILABILITYSET-00000000", []string{})
	expectedVM.Name = to.StringPtr("name")
	mockVMClient := cloud.VirtualMachinesClient.(*mockvmclient.MockInterface)
	mockVMClient.EXPECT().Get(gomock.Any(), "rg", "k8s-agentpool1-00000000-0", gomock.Any()).Return(expectedVM, nil)
	expectedNIC := buildDefaultTestInterface(true, []string{})
	expectedNIC.VirtualMachine = &network.SubResource{
		ID: to.StringPtr("/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachines/k8s-agentpool1-00000000-0"),
	}
	mockNICClient := cloud.InterfacesClient.(*mockinterfaceclient.MockInterface)
	mockNICClient.EXPECT().Get(gomock.Any(), "rg", "k8s-agentpool1-00000000-nic-0", gomock.Any()).Return(expectedNIC, nil)
	ipConfigurationID := `/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/networkInterfaces/k8s-agentpool1-00000000-nic-0/ipConfigurations/ipconfig1`
	nodeName, asName, err := cloud.VMSet.GetNodeNameByIPConfigurationID(ipConfigurationID)
	assert.NoError(t, err)
	assert.Equal(t, "k8s-agentpool1-00000000-0", nodeName)
	assert.Equal(t, "agentpool1-availabilityset-00000000", asName)
}

func TestGetAvailabilitySetNameByID(t *testing.T) {
	t.Run("getAvailabilitySetNameByID should return empty string if the given ID is empty", func(t *testing.T) {
		vmasName, err := getAvailabilitySetNameByID("")
		assert.Nil(t, err)
		assert.Empty(t, vmasName)
	})

	t.Run("getAvailabilitySetNameByID should report an error if the format of the given ID is wrong", func(t *testing.T) {
		asID := "illegal-id"
		vmasName, err := getAvailabilitySetNameByID(asID)
		assert.Equal(t, fmt.Errorf("getAvailabilitySetNameByID: failed to parse the VMAS ID illegal-id"), err)
		assert.Empty(t, vmasName)
	})

	t.Run("getAvailabilitySetNameByID should extract the VMAS name from the given ID", func(t *testing.T) {
		asID := "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/availabilitySets/as"
		vmasName, err := getAvailabilitySetNameByID(asID)
		assert.Nil(t, err)
		assert.Equal(t, "as", vmasName)
	})
}

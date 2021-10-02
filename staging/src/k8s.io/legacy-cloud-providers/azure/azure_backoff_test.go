//go:build !providerless
// +build !providerless

/*
Copyright 2020 The Kubernetes Authors.

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
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/legacy-cloud-providers/azure/cache"
	"k8s.io/legacy-cloud-providers/azure/clients/interfaceclient/mockinterfaceclient"
	"k8s.io/legacy-cloud-providers/azure/clients/loadbalancerclient/mockloadbalancerclient"
	"k8s.io/legacy-cloud-providers/azure/clients/publicipclient/mockpublicipclient"
	"k8s.io/legacy-cloud-providers/azure/clients/routeclient/mockrouteclient"
	"k8s.io/legacy-cloud-providers/azure/clients/routetableclient/mockroutetableclient"
	"k8s.io/legacy-cloud-providers/azure/clients/securitygroupclient/mocksecuritygroupclient"
	"k8s.io/legacy-cloud-providers/azure/clients/vmclient/mockvmclient"
	"k8s.io/legacy-cloud-providers/azure/clients/vmssclient/mockvmssclient"
	"k8s.io/legacy-cloud-providers/azure/retry"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-12-01/compute"
	"github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-06-01/network"
	"github.com/Azure/go-autorest/autorest/to"
	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"
)

func TestGetVirtualMachineWithRetry(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	tests := []struct {
		vmClientErr *retry.Error
		expectedErr error
	}{
		{
			vmClientErr: &retry.Error{HTTPStatusCode: http.StatusNotFound},
			expectedErr: cloudprovider.InstanceNotFound,
		},
		{
			vmClientErr: &retry.Error{HTTPStatusCode: http.StatusInternalServerError},
			expectedErr: fmt.Errorf("Retriable: false, RetryAfter: 0s, HTTPStatusCode: 500, RawError: %w", error(nil)),
		},
	}

	for _, test := range tests {
		az := GetTestCloud(ctrl)
		mockVMClient := az.VirtualMachinesClient.(*mockvmclient.MockInterface)
		mockVMClient.EXPECT().Get(gomock.Any(), az.ResourceGroup, "vm", gomock.Any()).Return(compute.VirtualMachine{}, test.vmClientErr)

		vm, err := az.GetVirtualMachineWithRetry("vm", cache.CacheReadTypeDefault)
		assert.Empty(t, vm)
		assert.Equal(t, test.expectedErr, err)
	}
}

func TestGetPrivateIPsForMachine(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	tests := []struct {
		vmClientErr        *retry.Error
		expectedPrivateIPs []string
		expectedErr        error
	}{
		{
			expectedPrivateIPs: []string{"1.2.3.4"},
		},
		{
			vmClientErr:        &retry.Error{HTTPStatusCode: http.StatusNotFound},
			expectedErr:        cloudprovider.InstanceNotFound,
			expectedPrivateIPs: []string{},
		},
		{
			vmClientErr:        &retry.Error{HTTPStatusCode: http.StatusInternalServerError},
			expectedErr:        wait.ErrWaitTimeout,
			expectedPrivateIPs: []string{},
		},
	}

	expectedVM := compute.VirtualMachine{
		VirtualMachineProperties: &compute.VirtualMachineProperties{
			AvailabilitySet: &compute.SubResource{ID: to.StringPtr("availability-set")},
			NetworkProfile: &compute.NetworkProfile{
				NetworkInterfaces: &[]compute.NetworkInterfaceReference{
					{
						NetworkInterfaceReferenceProperties: &compute.NetworkInterfaceReferenceProperties{
							Primary: to.BoolPtr(true),
						},
						ID: to.StringPtr("/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/networkInterfaces/nic"),
					},
				},
			},
		},
	}

	expectedInterface := network.Interface{
		InterfacePropertiesFormat: &network.InterfacePropertiesFormat{
			IPConfigurations: &[]network.InterfaceIPConfiguration{
				{
					InterfaceIPConfigurationPropertiesFormat: &network.InterfaceIPConfigurationPropertiesFormat{
						PrivateIPAddress: to.StringPtr("1.2.3.4"),
					},
				},
			},
		},
	}

	for _, test := range tests {
		az := GetTestCloud(ctrl)
		mockVMClient := az.VirtualMachinesClient.(*mockvmclient.MockInterface)
		mockVMClient.EXPECT().Get(gomock.Any(), az.ResourceGroup, "vm", gomock.Any()).Return(expectedVM, test.vmClientErr)

		mockInterfaceClient := az.InterfacesClient.(*mockinterfaceclient.MockInterface)
		mockInterfaceClient.EXPECT().Get(gomock.Any(), az.ResourceGroup, "nic", gomock.Any()).Return(expectedInterface, nil).MaxTimes(1)

		privateIPs, err := az.getPrivateIPsForMachine("vm")
		assert.Equal(t, test.expectedErr, err)
		assert.Equal(t, test.expectedPrivateIPs, privateIPs)
	}
}

func TestGetIPForMachineWithRetry(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	tests := []struct {
		clientErr         *retry.Error
		expectedPrivateIP string
		expectedPublicIP  string
		expectedErr       error
	}{
		{
			expectedPrivateIP: "1.2.3.4",
			expectedPublicIP:  "5.6.7.8",
		},
		{
			clientErr:   &retry.Error{HTTPStatusCode: http.StatusNotFound},
			expectedErr: wait.ErrWaitTimeout,
		},
	}

	expectedVM := compute.VirtualMachine{
		VirtualMachineProperties: &compute.VirtualMachineProperties{
			AvailabilitySet: &compute.SubResource{ID: to.StringPtr("availability-set")},
			NetworkProfile: &compute.NetworkProfile{
				NetworkInterfaces: &[]compute.NetworkInterfaceReference{
					{
						NetworkInterfaceReferenceProperties: &compute.NetworkInterfaceReferenceProperties{
							Primary: to.BoolPtr(true),
						},
						ID: to.StringPtr("/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/networkInterfaces/nic"),
					},
				},
			},
		},
	}

	expectedInterface := network.Interface{
		InterfacePropertiesFormat: &network.InterfacePropertiesFormat{
			IPConfigurations: &[]network.InterfaceIPConfiguration{
				{
					InterfaceIPConfigurationPropertiesFormat: &network.InterfaceIPConfigurationPropertiesFormat{
						PrivateIPAddress: to.StringPtr("1.2.3.4"),
						PublicIPAddress: &network.PublicIPAddress{
							ID: to.StringPtr("test/pip"),
						},
					},
				},
			},
		},
	}

	expectedPIP := network.PublicIPAddress{
		PublicIPAddressPropertiesFormat: &network.PublicIPAddressPropertiesFormat{
			IPAddress: to.StringPtr("5.6.7.8"),
		},
	}

	for _, test := range tests {
		az := GetTestCloud(ctrl)
		mockVMClient := az.VirtualMachinesClient.(*mockvmclient.MockInterface)
		mockVMClient.EXPECT().Get(gomock.Any(), az.ResourceGroup, "vm", gomock.Any()).Return(expectedVM, test.clientErr)

		mockInterfaceClient := az.InterfacesClient.(*mockinterfaceclient.MockInterface)
		mockInterfaceClient.EXPECT().Get(gomock.Any(), az.ResourceGroup, "nic", gomock.Any()).Return(expectedInterface, nil).MaxTimes(1)

		mockPIPClient := az.PublicIPAddressesClient.(*mockpublicipclient.MockInterface)
		mockPIPClient.EXPECT().Get(gomock.Any(), az.ResourceGroup, "pip", gomock.Any()).Return(expectedPIP, nil).MaxTimes(1)

		privateIP, publicIP, err := az.GetIPForMachineWithRetry("vm")
		assert.Equal(t, test.expectedErr, err)
		assert.Equal(t, test.expectedPrivateIP, privateIP)
		assert.Equal(t, test.expectedPublicIP, publicIP)
	}
}

func TestCreateOrUpdateSecurityGroupCanceled(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	az := GetTestCloud(ctrl)
	az.nsgCache.Set("sg", "test")

	mockSGClient := az.SecurityGroupsClient.(*mocksecuritygroupclient.MockInterface)
	mockSGClient.EXPECT().CreateOrUpdate(gomock.Any(), az.ResourceGroup, gomock.Any(), gomock.Any(), gomock.Any()).Return(&retry.Error{
		RawError: fmt.Errorf(operationCanceledErrorMessage),
	})
	mockSGClient.EXPECT().Get(gomock.Any(), az.ResourceGroup, "sg", gomock.Any()).Return(network.SecurityGroup{}, nil)

	err := az.CreateOrUpdateSecurityGroup(network.SecurityGroup{Name: to.StringPtr("sg")})
	assert.EqualError(t, fmt.Errorf("Retriable: false, RetryAfter: 0s, HTTPStatusCode: 0, RawError: %w", fmt.Errorf("canceledandsupersededduetoanotheroperation")), err.Error())

	// security group should be removed from cache if the operation is canceled
	shouldBeEmpty, err := az.nsgCache.Get("sg", cache.CacheReadTypeDefault)
	assert.NoError(t, err)
	assert.Empty(t, shouldBeEmpty)
}

func TestCreateOrUpdateLB(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	referencedResourceNotProvisionedRawErrorString := `Code="ReferencedResourceNotProvisioned" Message="Cannot proceed with operation because resource /subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/publicIPAddresses/pip used by resource /subscriptions/sub/resourceGroups/rg/providers/Microsoft.Network/loadBalancers/lb is not in Succeeded state. Resource is in Failed state and the last operation that updated/is updating the resource is PutPublicIpAddressOperation."`

	tests := []struct {
		clientErr   *retry.Error
		expectedErr error
	}{
		{
			clientErr:   &retry.Error{HTTPStatusCode: http.StatusPreconditionFailed},
			expectedErr: fmt.Errorf("Retriable: false, RetryAfter: 0s, HTTPStatusCode: 412, RawError: %w", error(nil)),
		},
		{
			clientErr:   &retry.Error{RawError: fmt.Errorf("canceledandsupersededduetoanotheroperation")},
			expectedErr: fmt.Errorf("Retriable: false, RetryAfter: 0s, HTTPStatusCode: 0, RawError: %w", fmt.Errorf("canceledandsupersededduetoanotheroperation")),
		},
		{
			clientErr:   &retry.Error{RawError: fmt.Errorf(referencedResourceNotProvisionedRawErrorString)},
			expectedErr: fmt.Errorf("Retriable: false, RetryAfter: 0s, HTTPStatusCode: 0, RawError: %w", fmt.Errorf(referencedResourceNotProvisionedRawErrorString)),
		},
	}

	for _, test := range tests {
		az := GetTestCloud(ctrl)
		az.lbCache.Set("lb", "test")

		mockLBClient := az.LoadBalancerClient.(*mockloadbalancerclient.MockInterface)
		mockLBClient.EXPECT().CreateOrUpdate(gomock.Any(), az.ResourceGroup, gomock.Any(), gomock.Any(), gomock.Any()).Return(test.clientErr)
		mockLBClient.EXPECT().Get(gomock.Any(), az.ResourceGroup, "lb", gomock.Any()).Return(network.LoadBalancer{}, nil)

		mockPIPClient := az.PublicIPAddressesClient.(*mockpublicipclient.MockInterface)
		mockPIPClient.EXPECT().CreateOrUpdate(gomock.Any(), az.ResourceGroup, "pip", gomock.Any()).Return(nil).AnyTimes()
		mockPIPClient.EXPECT().Get(gomock.Any(), az.ResourceGroup, "pip", gomock.Any()).Return(network.PublicIPAddress{
			Name: to.StringPtr("pip"),
			PublicIPAddressPropertiesFormat: &network.PublicIPAddressPropertiesFormat{
				ProvisioningState: to.StringPtr("Succeeded"),
			},
		}, nil).AnyTimes()

		err := az.CreateOrUpdateLB(&v1.Service{}, network.LoadBalancer{
			Name: to.StringPtr("lb"),
			Etag: to.StringPtr("etag"),
		})
		assert.Equal(t, test.expectedErr, err)

		// loadbalancer should be removed from cache if the etag is mismatch or the operation is canceled
		shouldBeEmpty, err := az.lbCache.Get("lb", cache.CacheReadTypeDefault)
		assert.NoError(t, err)
		assert.Empty(t, shouldBeEmpty)
	}
}

func TestListLB(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	tests := []struct {
		clientErr   *retry.Error
		expectedErr error
	}{
		{
			clientErr:   &retry.Error{HTTPStatusCode: http.StatusInternalServerError},
			expectedErr: fmt.Errorf("Retriable: false, RetryAfter: 0s, HTTPStatusCode: 500, RawError: %w", error(nil)),
		},
		{
			clientErr:   &retry.Error{HTTPStatusCode: http.StatusNotFound},
			expectedErr: nil,
		},
	}
	for _, test := range tests {
		az := GetTestCloud(ctrl)
		mockLBClient := az.LoadBalancerClient.(*mockloadbalancerclient.MockInterface)
		mockLBClient.EXPECT().List(gomock.Any(), az.ResourceGroup).Return(nil, test.clientErr)

		pips, err := az.ListLB(&v1.Service{})
		assert.Equal(t, test.expectedErr, err)
		assert.Empty(t, pips)
	}

}

func TestListPIP(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	tests := []struct {
		clientErr   *retry.Error
		expectedErr error
	}{
		{
			clientErr:   &retry.Error{HTTPStatusCode: http.StatusInternalServerError},
			expectedErr: fmt.Errorf("Retriable: false, RetryAfter: 0s, HTTPStatusCode: 500, RawError: %w", error(nil)),
		},
		{
			clientErr:   &retry.Error{HTTPStatusCode: http.StatusNotFound},
			expectedErr: nil,
		},
	}
	for _, test := range tests {
		az := GetTestCloud(ctrl)
		mockPIPClient := az.PublicIPAddressesClient.(*mockpublicipclient.MockInterface)
		mockPIPClient.EXPECT().List(gomock.Any(), az.ResourceGroup).Return(nil, test.clientErr)

		pips, err := az.ListPIP(&v1.Service{}, az.ResourceGroup)
		assert.Equal(t, test.expectedErr, err)
		assert.Empty(t, pips)
	}
}

func TestCreateOrUpdatePIP(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	az := GetTestCloud(ctrl)
	mockPIPClient := az.PublicIPAddressesClient.(*mockpublicipclient.MockInterface)
	mockPIPClient.EXPECT().CreateOrUpdate(gomock.Any(), az.ResourceGroup, "nic", gomock.Any()).Return(&retry.Error{HTTPStatusCode: http.StatusInternalServerError})

	err := az.CreateOrUpdatePIP(&v1.Service{}, az.ResourceGroup, network.PublicIPAddress{Name: to.StringPtr("nic")})
	assert.Equal(t, fmt.Errorf("Retriable: false, RetryAfter: 0s, HTTPStatusCode: 500, RawError: %w", error(nil)), err)
}

func TestCreateOrUpdateInterface(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	az := GetTestCloud(ctrl)
	mockInterfaceClient := az.InterfacesClient.(*mockinterfaceclient.MockInterface)
	mockInterfaceClient.EXPECT().CreateOrUpdate(gomock.Any(), az.ResourceGroup, "nic", gomock.Any()).Return(&retry.Error{HTTPStatusCode: http.StatusInternalServerError})

	err := az.CreateOrUpdateInterface(&v1.Service{}, network.Interface{Name: to.StringPtr("nic")})
	assert.Equal(t, fmt.Errorf("Retriable: false, RetryAfter: 0s, HTTPStatusCode: 500, RawError: %w", error(nil)), err)
}

func TestDeletePublicIP(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	az := GetTestCloud(ctrl)
	mockPIPClient := az.PublicIPAddressesClient.(*mockpublicipclient.MockInterface)
	mockPIPClient.EXPECT().Delete(gomock.Any(), az.ResourceGroup, "pip").Return(&retry.Error{HTTPStatusCode: http.StatusInternalServerError})

	err := az.DeletePublicIP(&v1.Service{}, az.ResourceGroup, "pip")
	assert.Equal(t, fmt.Errorf("Retriable: false, RetryAfter: 0s, HTTPStatusCode: 500, RawError: %w", error(nil)), err)
}

func TestDeleteLB(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	az := GetTestCloud(ctrl)
	mockLBClient := az.LoadBalancerClient.(*mockloadbalancerclient.MockInterface)
	mockLBClient.EXPECT().Delete(gomock.Any(), az.ResourceGroup, "lb").Return(&retry.Error{HTTPStatusCode: http.StatusInternalServerError})

	err := az.DeleteLB(&v1.Service{}, "lb")
	assert.Equal(t, fmt.Errorf("Retriable: false, RetryAfter: 0s, HTTPStatusCode: 500, RawError: %w", error(nil)), err)
}

func TestCreateOrUpdateRouteTable(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	tests := []struct {
		clientErr   *retry.Error
		expectedErr error
	}{
		{
			clientErr:   &retry.Error{HTTPStatusCode: http.StatusPreconditionFailed},
			expectedErr: fmt.Errorf("Retriable: false, RetryAfter: 0s, HTTPStatusCode: 412, RawError: %w", error(nil)),
		},
		{
			clientErr:   &retry.Error{RawError: fmt.Errorf("canceledandsupersededduetoanotheroperation")},
			expectedErr: fmt.Errorf("Retriable: false, RetryAfter: 0s, HTTPStatusCode: 0, RawError: %w", fmt.Errorf("canceledandsupersededduetoanotheroperation")),
		},
	}

	for _, test := range tests {
		az := GetTestCloud(ctrl)
		az.rtCache.Set("rt", "test")

		mockRTClient := az.RouteTablesClient.(*mockroutetableclient.MockInterface)
		mockRTClient.EXPECT().CreateOrUpdate(gomock.Any(), az.ResourceGroup, gomock.Any(), gomock.Any(), gomock.Any()).Return(test.clientErr)
		mockRTClient.EXPECT().Get(gomock.Any(), az.ResourceGroup, "rt", gomock.Any()).Return(network.RouteTable{}, nil)

		err := az.CreateOrUpdateRouteTable(network.RouteTable{
			Name: to.StringPtr("rt"),
			Etag: to.StringPtr("etag"),
		})
		assert.Equal(t, test.expectedErr, err)

		// route table should be removed from cache if the etag is mismatch or the operation is canceled
		shouldBeEmpty, err := az.rtCache.Get("rt", cache.CacheReadTypeDefault)
		assert.NoError(t, err)
		assert.Empty(t, shouldBeEmpty)
	}
}

func TestCreateOrUpdateRoute(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	tests := []struct {
		clientErr   *retry.Error
		expectedErr error
	}{
		{
			clientErr:   &retry.Error{HTTPStatusCode: http.StatusPreconditionFailed},
			expectedErr: fmt.Errorf("Retriable: false, RetryAfter: 0s, HTTPStatusCode: 412, RawError: %w", error(nil)),
		},
		{
			clientErr:   &retry.Error{RawError: fmt.Errorf("canceledandsupersededduetoanotheroperation")},
			expectedErr: fmt.Errorf("Retriable: false, RetryAfter: 0s, HTTPStatusCode: 0, RawError: %w", fmt.Errorf("canceledandsupersededduetoanotheroperation")),
		},
		{
			clientErr:   nil,
			expectedErr: nil,
		},
	}

	for _, test := range tests {
		az := GetTestCloud(ctrl)
		az.rtCache.Set("rt", "test")

		mockRTClient := az.RoutesClient.(*mockrouteclient.MockInterface)
		mockRTClient.EXPECT().CreateOrUpdate(gomock.Any(), az.ResourceGroup, "rt", gomock.Any(), gomock.Any(), gomock.Any()).Return(test.clientErr)

		mockRTableClient := az.RouteTablesClient.(*mockroutetableclient.MockInterface)
		mockRTableClient.EXPECT().Get(gomock.Any(), az.ResourceGroup, "rt", gomock.Any()).Return(network.RouteTable{}, nil)

		err := az.CreateOrUpdateRoute(network.Route{
			Name: to.StringPtr("rt"),
			Etag: to.StringPtr("etag"),
		})
		assert.Equal(t, test.expectedErr, err)

		shouldBeEmpty, err := az.rtCache.Get("rt", cache.CacheReadTypeDefault)
		assert.NoError(t, err)
		assert.Empty(t, shouldBeEmpty)
	}
}

func TestDeleteRouteWithName(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	tests := []struct {
		clientErr   *retry.Error
		expectedErr error
	}{
		{
			clientErr:   &retry.Error{HTTPStatusCode: http.StatusInternalServerError},
			expectedErr: fmt.Errorf("Retriable: false, RetryAfter: 0s, HTTPStatusCode: 500, RawError: %w", error(nil)),
		},
		{
			clientErr:   nil,
			expectedErr: nil,
		},
	}

	for _, test := range tests {
		az := GetTestCloud(ctrl)

		mockRTClient := az.RoutesClient.(*mockrouteclient.MockInterface)
		mockRTClient.EXPECT().Delete(gomock.Any(), az.ResourceGroup, "rt", "rt").Return(test.clientErr)

		err := az.DeleteRouteWithName("rt")
		assert.Equal(t, test.expectedErr, err)
	}
}

func TestCreateOrUpdateVMSS(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	tests := []struct {
		vmss        compute.VirtualMachineScaleSet
		clientErr   *retry.Error
		expectedErr *retry.Error
	}{
		{
			clientErr:   &retry.Error{HTTPStatusCode: http.StatusInternalServerError},
			expectedErr: &retry.Error{HTTPStatusCode: http.StatusInternalServerError},
		},
		{
			clientErr:   &retry.Error{HTTPStatusCode: http.StatusTooManyRequests},
			expectedErr: &retry.Error{HTTPStatusCode: http.StatusTooManyRequests},
		},
		{
			clientErr:   &retry.Error{RawError: fmt.Errorf("azure cloud provider rate limited(write) for operation CreateOrUpdate")},
			expectedErr: &retry.Error{RawError: fmt.Errorf("azure cloud provider rate limited(write) for operation CreateOrUpdate")},
		},
		{
			vmss: compute.VirtualMachineScaleSet{
				VirtualMachineScaleSetProperties: &compute.VirtualMachineScaleSetProperties{
					ProvisioningState: &virtualMachineScaleSetsDeallocating,
				},
			},
		},
	}

	for _, test := range tests {
		az := GetTestCloud(ctrl)

		mockVMSSClient := az.VirtualMachineScaleSetsClient.(*mockvmssclient.MockInterface)
		mockVMSSClient.EXPECT().Get(gomock.Any(), az.ResourceGroup, testVMSSName).Return(test.vmss, test.clientErr)

		err := az.CreateOrUpdateVMSS(az.ResourceGroup, testVMSSName, compute.VirtualMachineScaleSet{})
		assert.Equal(t, test.expectedErr, err)
	}
}

func TestRequestBackoff(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	az := GetTestCloud(ctrl)
	az.CloudProviderBackoff = true
	az.ResourceRequestBackoff = wait.Backoff{Steps: 3}

	backoff := az.RequestBackoff()
	assert.Equal(t, wait.Backoff{Steps: 3}, backoff)

}

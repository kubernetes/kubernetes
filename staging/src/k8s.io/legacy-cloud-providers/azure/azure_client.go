// +build !providerless

/*
Copyright 2017 The Kubernetes Authors.

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
	"context"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-07-01/compute"
	"github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-06-01/network"
	"github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2019-06-01/storage"

	"k8s.io/legacy-cloud-providers/azure/retry"
)

const (
	// The version number is taken from "github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-06-01/network".
	azureNetworkAPIVersion              = "2019-06-01"
	virtualMachineScaleSetsDeallocating = "Deallocating"
)

// VirtualMachinesClient defines needed functions for azure compute.VirtualMachinesClient
type VirtualMachinesClient interface {
	CreateOrUpdate(ctx context.Context, resourceGroupName string, VMName string, parameters compute.VirtualMachine, source string) *retry.Error
	Update(ctx context.Context, resourceGroupName string, VMName string, parameters compute.VirtualMachineUpdate, source string) *retry.Error
	Get(ctx context.Context, resourceGroupName string, VMName string, expand compute.InstanceViewTypes) (result compute.VirtualMachine, rerr *retry.Error)
	List(ctx context.Context, resourceGroupName string) (result []compute.VirtualMachine, rerr *retry.Error)
}

// InterfacesClient defines needed functions for azure network.InterfacesClient
type InterfacesClient interface {
	CreateOrUpdate(ctx context.Context, resourceGroupName string, networkInterfaceName string, parameters network.Interface) *retry.Error
	Get(ctx context.Context, resourceGroupName string, networkInterfaceName string, expand string) (result network.Interface, rerr *retry.Error)
	GetVirtualMachineScaleSetNetworkInterface(ctx context.Context, resourceGroupName string, virtualMachineScaleSetName string, virtualmachineIndex string, networkInterfaceName string, expand string) (result network.Interface, rerr *retry.Error)
}

// LoadBalancersClient defines needed functions for azure network.LoadBalancersClient
type LoadBalancersClient interface {
	CreateOrUpdate(ctx context.Context, resourceGroupName string, loadBalancerName string, parameters network.LoadBalancer, etag string) *retry.Error
	Delete(ctx context.Context, resourceGroupName string, loadBalancerName string) *retry.Error
	Get(ctx context.Context, resourceGroupName string, loadBalancerName string, expand string) (result network.LoadBalancer, rerr *retry.Error)
	List(ctx context.Context, resourceGroupName string) (result []network.LoadBalancer, rerr *retry.Error)
}

// PublicIPAddressesClient defines needed functions for azure network.PublicIPAddressesClient
type PublicIPAddressesClient interface {
	CreateOrUpdate(ctx context.Context, resourceGroupName string, publicIPAddressName string, parameters network.PublicIPAddress) *retry.Error
	Delete(ctx context.Context, resourceGroupName string, publicIPAddressName string) *retry.Error
	Get(ctx context.Context, resourceGroupName string, publicIPAddressName string, expand string) (result network.PublicIPAddress, rerr *retry.Error)
	GetVirtualMachineScaleSetPublicIPAddress(ctx context.Context, resourceGroupName string, virtualMachineScaleSetName string, virtualmachineIndex string, networkInterfaceName string, IPConfigurationName string, publicIPAddressName string, expand string) (result network.PublicIPAddress, rerr *retry.Error)
	List(ctx context.Context, resourceGroupName string) (result []network.PublicIPAddress, rerr *retry.Error)
}

// SubnetsClient defines needed functions for azure network.SubnetsClient
type SubnetsClient interface {
	CreateOrUpdate(ctx context.Context, resourceGroupName string, virtualNetworkName string, subnetName string, subnetParameters network.Subnet) *retry.Error
	Delete(ctx context.Context, resourceGroupName string, virtualNetworkName string, subnetName string) *retry.Error
	Get(ctx context.Context, resourceGroupName string, virtualNetworkName string, subnetName string, expand string) (result network.Subnet, rerr *retry.Error)
	List(ctx context.Context, resourceGroupName string, virtualNetworkName string) (result []network.Subnet, rerr *retry.Error)
}

// SecurityGroupsClient defines needed functions for azure network.SecurityGroupsClient
type SecurityGroupsClient interface {
	CreateOrUpdate(ctx context.Context, resourceGroupName string, networkSecurityGroupName string, parameters network.SecurityGroup, etag string) *retry.Error
	Delete(ctx context.Context, resourceGroupName string, networkSecurityGroupName string) *retry.Error
	Get(ctx context.Context, resourceGroupName string, networkSecurityGroupName string, expand string) (result network.SecurityGroup, rerr *retry.Error)
	List(ctx context.Context, resourceGroupName string) (result []network.SecurityGroup, rerr *retry.Error)
}

// VirtualMachineScaleSetsClient defines needed functions for azure compute.VirtualMachineScaleSetsClient
type VirtualMachineScaleSetsClient interface {
	Get(ctx context.Context, resourceGroupName string, VMScaleSetName string) (result compute.VirtualMachineScaleSet, rerr *retry.Error)
	List(ctx context.Context, resourceGroupName string) (result []compute.VirtualMachineScaleSet, rerr *retry.Error)
	CreateOrUpdate(ctx context.Context, resourceGroupName string, VMScaleSetName string, parameters compute.VirtualMachineScaleSet) *retry.Error
}

// VirtualMachineScaleSetVMsClient defines needed functions for azure compute.VirtualMachineScaleSetVMsClient
type VirtualMachineScaleSetVMsClient interface {
	Get(ctx context.Context, resourceGroupName string, VMScaleSetName string, instanceID string, expand compute.InstanceViewTypes) (result compute.VirtualMachineScaleSetVM, rerr *retry.Error)
	List(ctx context.Context, resourceGroupName string, virtualMachineScaleSetName string, expand string) (result []compute.VirtualMachineScaleSetVM, rerr *retry.Error)
	Update(ctx context.Context, resourceGroupName string, VMScaleSetName string, instanceID string, parameters compute.VirtualMachineScaleSetVM, source string) *retry.Error
}

// RoutesClient defines needed functions for azure network.RoutesClient
type RoutesClient interface {
	CreateOrUpdate(ctx context.Context, resourceGroupName string, routeTableName string, routeName string, routeParameters network.Route, etag string) *retry.Error
	Delete(ctx context.Context, resourceGroupName string, routeTableName string, routeName string) *retry.Error
}

// RouteTablesClient defines needed functions for azure network.RouteTablesClient
type RouteTablesClient interface {
	CreateOrUpdate(ctx context.Context, resourceGroupName string, routeTableName string, parameters network.RouteTable, etag string) *retry.Error
	Get(ctx context.Context, resourceGroupName string, routeTableName string, expand string) (result network.RouteTable, rerr *retry.Error)
}

// StorageAccountClient defines needed functions for azure storage.AccountsClient
type StorageAccountClient interface {
	Create(ctx context.Context, resourceGroupName string, accountName string, parameters storage.AccountCreateParameters) *retry.Error
	Delete(ctx context.Context, resourceGroupName string, accountName string) *retry.Error
	ListKeys(ctx context.Context, resourceGroupName string, accountName string) (result storage.AccountListKeysResult, rerr *retry.Error)
	ListByResourceGroup(ctx context.Context, resourceGroupName string) (result []storage.Account, rerr *retry.Error)
	GetProperties(ctx context.Context, resourceGroupName string, accountName string) (result storage.Account, rerr *retry.Error)
}

// SnapshotsClient defines needed functions for azure compute.SnapshotsClient
type SnapshotsClient interface {
	Get(ctx context.Context, resourceGroupName string, snapshotName string) (compute.Snapshot, *retry.Error)
	Delete(ctx context.Context, resourceGroupName string, snapshotName string) *retry.Error
	ListByResourceGroup(ctx context.Context, resourceGroupName string) ([]compute.Snapshot, *retry.Error)
	CreateOrUpdate(ctx context.Context, resourceGroupName string, snapshotName string, snapshot compute.Snapshot) *retry.Error
}

// DisksClient defines needed functions for azure compute.DisksClient
type DisksClient interface {
	CreateOrUpdate(ctx context.Context, resourceGroupName string, diskName string, diskParameter compute.Disk) *retry.Error
	Delete(ctx context.Context, resourceGroupName string, diskName string) *retry.Error
	Get(ctx context.Context, resourceGroupName string, diskName string) (result compute.Disk, rerr *retry.Error)
}

// VirtualMachineSizesClient defines needed functions for azure compute.VirtualMachineSizesClient
type VirtualMachineSizesClient interface {
	List(ctx context.Context, location string) (result compute.VirtualMachineSizeListResult, rerr *retry.Error)
}

func getContextWithCancel() (context.Context, context.CancelFunc) {
	return context.WithCancel(context.Background())
}

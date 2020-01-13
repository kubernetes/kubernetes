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
	"fmt"
	"time"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-07-01/compute"
	"github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-06-01/network"
	"github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2019-06-01/storage"
	"github.com/Azure/go-autorest/autorest"

	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/klog"
	azclients "k8s.io/legacy-cloud-providers/azure/clients"
	"k8s.io/legacy-cloud-providers/azure/metrics"
	"k8s.io/legacy-cloud-providers/azure/retry"
)

const (
	// The version number is taken from "github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-06-01/network".
	azureNetworkAPIVersion              = "2019-06-01"
	virtualMachineScaleSetsDeallocating = "Deallocating"
)

// Helpers for rate limiting error/error channel creation
func createRateLimitErr(isWrite bool, opName string) *retry.Error {
	opType := "read"
	if isWrite {
		opType = "write"
	}
	return retry.GetRetriableError(fmt.Errorf("azure - cloud provider rate limited(%s) for operation:%s", opType, opName))
}

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
	ListByResourceGroup(ctx context.Context, resourceGroupName string) (result storage.AccountListResult, rerr *retry.Error)
	GetProperties(ctx context.Context, resourceGroupName string, accountName string) (result storage.Account, rerr *retry.Error)
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

// azStorageAccountClient implements StorageAccountClient.
type azStorageAccountClient struct {
	client            storage.AccountsClient
	rateLimiterReader flowcontrol.RateLimiter
	rateLimiterWriter flowcontrol.RateLimiter
}

func newAzStorageAccountClient(config *azclients.ClientConfig) *azStorageAccountClient {
	storageAccountClient := storage.NewAccountsClientWithBaseURI(config.ResourceManagerEndpoint, config.SubscriptionID)
	storageAccountClient.Authorizer = autorest.NewBearerAuthorizer(config.ServicePrincipalToken)
	storageAccountClient.PollingDelay = 5 * time.Second
	if config.ShouldOmitCloudProviderBackoff {
		storageAccountClient.RetryAttempts = config.CloudProviderBackoffRetries
		storageAccountClient.RetryDuration = time.Duration(config.CloudProviderBackoffDuration) * time.Second
	}
	configureUserAgent(&storageAccountClient.Client)

	klog.V(2).Infof("Azure StorageAccountClient (read ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPS,
		config.RateLimitConfig.CloudProviderRateLimitBucket)
	klog.V(2).Infof("Azure StorageAccountClient (write ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPSWrite,
		config.RateLimitConfig.CloudProviderRateLimitBucketWrite)
	rateLimiterReader, rateLimiterWriter := azclients.NewRateLimiter(config.RateLimitConfig)
	return &azStorageAccountClient{
		client:            storageAccountClient,
		rateLimiterReader: rateLimiterReader,
		rateLimiterWriter: rateLimiterWriter,
	}
}

func (az *azStorageAccountClient) Create(ctx context.Context, resourceGroupName string, accountName string, parameters storage.AccountCreateParameters) *retry.Error {
	mc := metrics.NewMetricContext("storage_account", "create", resourceGroupName, az.client.SubscriptionID, "")
	/* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return createRateLimitErr(true, "StorageAccountCreate")
	}

	klog.V(10).Infof("azStorageAccountClient.Create(%q,%q): start", resourceGroupName, accountName)
	defer func() {
		klog.V(10).Infof("azStorageAccountClient.Create(%q,%q): end", resourceGroupName, accountName)
	}()

	future, err := az.client.Create(ctx, resourceGroupName, accountName, parameters)
	if err != nil {
		return retry.GetError(future.Response(), err)
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	mc.Observe(err)
	return retry.GetError(future.Response(), err)
}

func (az *azStorageAccountClient) Delete(ctx context.Context, resourceGroupName string, accountName string) *retry.Error {
	mc := metrics.NewMetricContext("storage_account", "delete", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		return createRateLimitErr(false, "DeleteStorageAccount")
	}

	klog.V(10).Infof("azStorageAccountClient.Delete(%q,%q): start", resourceGroupName, accountName)
	defer func() {
		klog.V(10).Infof("azStorageAccountClient.Delete(%q,%q): end", resourceGroupName, accountName)
	}()

	result, err := az.client.Delete(ctx, resourceGroupName, accountName)
	mc.Observe(err)
	return retry.GetStatusNotFoundAndForbiddenIgnoredError(result.Response, err)
}

func (az *azStorageAccountClient) ListKeys(ctx context.Context, resourceGroupName string, accountName string) (result storage.AccountListKeysResult, rerr *retry.Error) {
	mc := metrics.NewMetricContext("storage_account", "list_keys", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		rerr = createRateLimitErr(false, "ListStorageAccountKeys")
		return
	}

	klog.V(10).Infof("azStorageAccountClient.ListKeys(%q,%q): start", resourceGroupName, accountName)
	defer func() {
		klog.V(10).Infof("azStorageAccountClient.ListKeys(%q,%q): end", resourceGroupName, accountName)
	}()

	var err error
	result, err = az.client.ListKeys(ctx, resourceGroupName, accountName, storage.Kerb)
	mc.Observe(err)
	return result, retry.GetError(result.Response.Response, err)
}

func (az *azStorageAccountClient) ListByResourceGroup(ctx context.Context, resourceGroupName string) (result storage.AccountListResult, rerr *retry.Error) {
	mc := metrics.NewMetricContext("storage_account", "list_by_resource_group", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		rerr = createRateLimitErr(false, "ListStorageAccountsByResourceGroup")
		return
	}

	klog.V(10).Infof("azStorageAccountClient.ListByResourceGroup(%q): start", resourceGroupName)
	defer func() {
		klog.V(10).Infof("azStorageAccountClient.ListByResourceGroup(%q): end", resourceGroupName)
	}()

	var err error
	result, err = az.client.ListByResourceGroup(ctx, resourceGroupName)
	mc.Observe(err)
	return result, retry.GetError(result.Response.Response, err)
}

func (az *azStorageAccountClient) GetProperties(ctx context.Context, resourceGroupName string, accountName string) (result storage.Account, rerr *retry.Error) {
	mc := metrics.NewMetricContext("storage_account", "get_properties", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		rerr = createRateLimitErr(false, "GetStorageAccount/Properties")
		return
	}

	klog.V(10).Infof("azStorageAccountClient.GetProperties(%q,%q): start", resourceGroupName, accountName)
	defer func() {
		klog.V(10).Infof("azStorageAccountClient.GetProperties(%q,%q): end", resourceGroupName, accountName)
	}()

	var err error
	result, err = az.client.GetProperties(ctx, resourceGroupName, accountName, "")
	mc.Observe(err)
	return result, retry.GetError(result.Response.Response, err)
}

// azDisksClient implements DisksClient.
type azDisksClient struct {
	client            compute.DisksClient
	rateLimiterReader flowcontrol.RateLimiter
	rateLimiterWriter flowcontrol.RateLimiter
}

func newAzDisksClient(config *azclients.ClientConfig) *azDisksClient {
	disksClient := compute.NewDisksClientWithBaseURI(config.ResourceManagerEndpoint, config.SubscriptionID)
	disksClient.Authorizer = autorest.NewBearerAuthorizer(config.ServicePrincipalToken)
	disksClient.PollingDelay = 5 * time.Second
	if config.ShouldOmitCloudProviderBackoff {
		disksClient.RetryAttempts = config.CloudProviderBackoffRetries
		disksClient.RetryDuration = time.Duration(config.CloudProviderBackoffDuration) * time.Second
	}
	configureUserAgent(&disksClient.Client)

	klog.V(2).Infof("Azure DisksClient (read ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPS,
		config.RateLimitConfig.CloudProviderRateLimitBucket)
	klog.V(2).Infof("Azure DisksClient (write ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPSWrite,
		config.RateLimitConfig.CloudProviderRateLimitBucketWrite)
	rateLimiterReader, rateLimiterWriter := azclients.NewRateLimiter(config.RateLimitConfig)
	return &azDisksClient{
		client:            disksClient,
		rateLimiterReader: rateLimiterReader,
		rateLimiterWriter: rateLimiterWriter,
	}
}

func (az *azDisksClient) CreateOrUpdate(ctx context.Context, resourceGroupName string, diskName string, diskParameter compute.Disk) *retry.Error {
	mc := metrics.NewMetricContext("disks", "create_or_update", resourceGroupName, az.client.SubscriptionID, "")
	/* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return createRateLimitErr(true, "DiskCreateOrUpdate")
	}

	klog.V(10).Infof("azDisksClient.CreateOrUpdate(%q,%q): start", resourceGroupName, diskName)
	defer func() {
		klog.V(10).Infof("azDisksClient.CreateOrUpdate(%q,%q): end", resourceGroupName, diskName)
	}()

	future, err := az.client.CreateOrUpdate(ctx, resourceGroupName, diskName, diskParameter)
	if err != nil {
		return retry.GetError(future.Response(), mc.Observe(err))
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	return retry.GetError(future.Response(), mc.Observe(err))
}

func (az *azDisksClient) Delete(ctx context.Context, resourceGroupName string, diskName string) *retry.Error {
	mc := metrics.NewMetricContext("disks", "delete", resourceGroupName, az.client.SubscriptionID, "")
	/* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return createRateLimitErr(true, "DiskDelete")
	}

	klog.V(10).Infof("azDisksClient.Delete(%q,%q): start", resourceGroupName, diskName)
	defer func() {
		klog.V(10).Infof("azDisksClient.Delete(%q,%q): end", resourceGroupName, diskName)
	}()

	future, err := az.client.Delete(ctx, resourceGroupName, diskName)
	if err != nil {
		return retry.GetStatusNotFoundAndForbiddenIgnoredError(future.Response(), mc.Observe(err))
	}
	err = future.WaitForCompletionRef(ctx, az.client.Client)
	return retry.GetStatusNotFoundAndForbiddenIgnoredError(future.Response(), mc.Observe(err))
}

func (az *azDisksClient) Get(ctx context.Context, resourceGroupName string, diskName string) (result compute.Disk, rerr *retry.Error) {
	mc := metrics.NewMetricContext("disks", "get", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		rerr = createRateLimitErr(false, "GetDisk")
		return
	}

	klog.V(10).Infof("azDisksClient.Get(%q,%q): start", resourceGroupName, diskName)
	defer func() {
		klog.V(10).Infof("azDisksClient.Get(%q,%q): end", resourceGroupName, diskName)
	}()

	var err error
	result, err = az.client.Get(ctx, resourceGroupName, diskName)
	mc.Observe(err)
	return result, retry.GetError(result.Response.Response, err)
}

// TODO(feiskyer): refactor compute.SnapshotsClient to Interface.
func newSnapshotsClient(config *azclients.ClientConfig) *compute.SnapshotsClient {
	snapshotsClient := compute.NewSnapshotsClientWithBaseURI(config.ResourceManagerEndpoint, config.SubscriptionID)
	snapshotsClient.Authorizer = autorest.NewBearerAuthorizer(config.ServicePrincipalToken)
	snapshotsClient.PollingDelay = 5 * time.Second
	if config.ShouldOmitCloudProviderBackoff {
		snapshotsClient.RetryAttempts = config.CloudProviderBackoffRetries
		snapshotsClient.RetryDuration = time.Duration(config.CloudProviderBackoffDuration) * time.Second
	}
	configureUserAgent(&snapshotsClient.Client)
	return &snapshotsClient
}

// azVirtualMachineSizesClient implements VirtualMachineSizesClient.
type azVirtualMachineSizesClient struct {
	client            compute.VirtualMachineSizesClient
	rateLimiterReader flowcontrol.RateLimiter
	rateLimiterWriter flowcontrol.RateLimiter
}

func newAzVirtualMachineSizesClient(config *azclients.ClientConfig) *azVirtualMachineSizesClient {
	VirtualMachineSizesClient := compute.NewVirtualMachineSizesClient(config.SubscriptionID)
	VirtualMachineSizesClient.BaseURI = config.ResourceManagerEndpoint
	VirtualMachineSizesClient.Authorizer = autorest.NewBearerAuthorizer(config.ServicePrincipalToken)
	VirtualMachineSizesClient.PollingDelay = 5 * time.Second
	if config.ShouldOmitCloudProviderBackoff {
		VirtualMachineSizesClient.RetryAttempts = config.CloudProviderBackoffRetries
		VirtualMachineSizesClient.RetryDuration = time.Duration(config.CloudProviderBackoffDuration) * time.Second
	}
	configureUserAgent(&VirtualMachineSizesClient.Client)

	klog.V(2).Infof("Azure VirtualMachineSizesClient (read ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPS,
		config.RateLimitConfig.CloudProviderRateLimitBucket)
	klog.V(2).Infof("Azure VirtualMachineSizesClient (write ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPSWrite,
		config.RateLimitConfig.CloudProviderRateLimitBucketWrite)
	rateLimiterReader, rateLimiterWriter := azclients.NewRateLimiter(config.RateLimitConfig)
	return &azVirtualMachineSizesClient{
		rateLimiterReader: rateLimiterReader,
		rateLimiterWriter: rateLimiterWriter,
		client:            VirtualMachineSizesClient,
	}
}

func (az *azVirtualMachineSizesClient) List(ctx context.Context, location string) (result compute.VirtualMachineSizeListResult, rerr *retry.Error) {
	mc := metrics.NewMetricContext("vmsizes", "list", "", az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		rerr = createRateLimitErr(false, "VMSizesList")
		return
	}

	klog.V(10).Infof("azVirtualMachineSizesClient.List(%q): start", location)
	defer func() {
		klog.V(10).Infof("azVirtualMachineSizesClient.List(%q): end", location)
	}()

	var err error
	result, err = az.client.List(ctx, location)
	mc.Observe(err)
	return result, retry.GetError(result.Response.Response, err)
}

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
	"net/http"
	"time"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-07-01/compute"
	"github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-06-01/network"
	"github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2019-06-01/storage"
	"github.com/Azure/go-autorest/autorest"

	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/klog"
	azclients "k8s.io/legacy-cloud-providers/azure/clients"
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
	List(ctx context.Context, resourceGroupName string, virtualMachineScaleSetName string, filter string, selectParameter string, expand string) (result []compute.VirtualMachineScaleSetVM, rerr *retry.Error)
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

// azVirtualMachinesClient implements VirtualMachinesClient.
type azVirtualMachinesClient struct {
	client            compute.VirtualMachinesClient
	rateLimiterReader flowcontrol.RateLimiter
	rateLimiterWriter flowcontrol.RateLimiter
}

func getContextWithCancel() (context.Context, context.CancelFunc) {
	return context.WithCancel(context.Background())
}

func newAzVirtualMachinesClient(config *azclients.ClientConfig) *azVirtualMachinesClient {
	virtualMachinesClient := compute.NewVirtualMachinesClient(config.SubscriptionID)
	virtualMachinesClient.BaseURI = config.ResourceManagerEndpoint
	virtualMachinesClient.Authorizer = autorest.NewBearerAuthorizer(config.ServicePrincipalToken)
	virtualMachinesClient.PollingDelay = 5 * time.Second
	if config.ShouldOmitCloudProviderBackoff {
		virtualMachinesClient.RetryAttempts = config.CloudProviderBackoffRetries
		virtualMachinesClient.RetryDuration = time.Duration(config.CloudProviderBackoffDuration) * time.Second
	}
	configureUserAgent(&virtualMachinesClient.Client)

	klog.V(2).Infof("Azure VirtualMachinesClient (read ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPS,
		config.RateLimitConfig.CloudProviderRateLimitBucket)
	klog.V(2).Infof("Azure VirtualMachinesClient (write ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPSWrite,
		config.RateLimitConfig.CloudProviderRateLimitBucketWrite)
	rateLimiterReader, rateLimiterWriter := azclients.NewRateLimiter(config.RateLimitConfig)
	return &azVirtualMachinesClient{
		rateLimiterReader: rateLimiterReader,
		rateLimiterWriter: rateLimiterWriter,
		client:            virtualMachinesClient,
	}
}

func (az *azVirtualMachinesClient) CreateOrUpdate(ctx context.Context, resourceGroupName string, VMName string, parameters compute.VirtualMachine, source string) *retry.Error {
	// /* Write rate limiting */
	mc := newMetricContext("vm", "create_or_update", resourceGroupName, az.client.SubscriptionID, source)
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return createRateLimitErr(true, "VMCreateOrUpdate")
	}

	klog.V(10).Infof("azVirtualMachinesClient.CreateOrUpdate(%q, %q): start", resourceGroupName, VMName)
	defer func() {
		klog.V(10).Infof("azVirtualMachinesClient.CreateOrUpdate(%q, %q): end", resourceGroupName, VMName)
	}()

	future, err := az.client.CreateOrUpdate(ctx, resourceGroupName, VMName, parameters)
	if err != nil {
		return retry.GetError(future.Response(), err)
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	mc.Observe(err)
	return retry.GetError(future.Response(), err)
}

func (az *azVirtualMachinesClient) Update(ctx context.Context, resourceGroupName string, VMName string, parameters compute.VirtualMachineUpdate, source string) *retry.Error {
	mc := newMetricContext("vm", "update", resourceGroupName, az.client.SubscriptionID, source)
	// /* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return createRateLimitErr(true, "VMUpdate")
	}

	klog.V(10).Infof("azVirtualMachinesClient.Update(%q, %q): start", resourceGroupName, VMName)
	defer func() {
		klog.V(10).Infof("azVirtualMachinesClient.Update(%q, %q): end", resourceGroupName, VMName)
	}()

	future, err := az.client.Update(ctx, resourceGroupName, VMName, parameters)
	if err != nil {
		return retry.GetError(future.Response(), err)
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	mc.Observe(err)
	return retry.GetError(future.Response(), err)
}

func (az *azVirtualMachinesClient) Get(ctx context.Context, resourceGroupName string, VMName string, expand compute.InstanceViewTypes) (result compute.VirtualMachine, rerr *retry.Error) {
	mc := newMetricContext("vm", "get", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		rerr = createRateLimitErr(false, "VMGet")
		return
	}

	klog.V(10).Infof("azVirtualMachinesClient.Get(%q, %q): start", resourceGroupName, VMName)
	defer func() {
		klog.V(10).Infof("azVirtualMachinesClient.Get(%q, %q): end", resourceGroupName, VMName)
	}()

	var err error
	result, err = az.client.Get(ctx, resourceGroupName, VMName, expand)
	mc.Observe(err)
	return result, retry.GetError(result.Response.Response, err)
}

func (az *azVirtualMachinesClient) List(ctx context.Context, resourceGroupName string) (result []compute.VirtualMachine, rerr *retry.Error) {
	mc := newMetricContext("vm", "list", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		rerr = createRateLimitErr(false, "VMList")
		return
	}

	klog.V(10).Infof("azVirtualMachinesClient.List(%q): start", resourceGroupName)
	defer func() {
		klog.V(10).Infof("azVirtualMachinesClient.List(%q): end", resourceGroupName)
	}()

	iterator, err := az.client.ListComplete(ctx, resourceGroupName)
	mc.Observe(err)
	if err != nil {
		return nil, retry.GetRetriableError(err)
	}

	result = make([]compute.VirtualMachine, 0)
	for ; iterator.NotDone(); err = iterator.Next() {
		if err != nil {
			return nil, retry.GetRetriableError(err)
		}

		result = append(result, iterator.Value())
	}

	return result, nil
}

// azInterfacesClient implements InterfacesClient.
type azInterfacesClient struct {
	client            network.InterfacesClient
	rateLimiterReader flowcontrol.RateLimiter
	rateLimiterWriter flowcontrol.RateLimiter
}

func newAzInterfacesClient(config *azclients.ClientConfig) *azInterfacesClient {
	interfacesClient := network.NewInterfacesClient(config.SubscriptionID)
	interfacesClient.BaseURI = config.ResourceManagerEndpoint
	interfacesClient.Authorizer = autorest.NewBearerAuthorizer(config.ServicePrincipalToken)
	interfacesClient.PollingDelay = 5 * time.Second
	if config.ShouldOmitCloudProviderBackoff {
		interfacesClient.RetryAttempts = config.CloudProviderBackoffRetries
		interfacesClient.RetryDuration = time.Duration(config.CloudProviderBackoffDuration) * time.Second
	}
	configureUserAgent(&interfacesClient.Client)

	klog.V(2).Infof("Azure InterfacesClient (read ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPS,
		config.RateLimitConfig.CloudProviderRateLimitBucket)
	klog.V(2).Infof("Azure InterfacesClient (write ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPSWrite,
		config.RateLimitConfig.CloudProviderRateLimitBucketWrite)
	rateLimiterReader, rateLimiterWriter := azclients.NewRateLimiter(config.RateLimitConfig)
	return &azInterfacesClient{
		rateLimiterReader: rateLimiterReader,
		rateLimiterWriter: rateLimiterWriter,
		client:            interfacesClient,
	}
}

func (az *azInterfacesClient) CreateOrUpdate(ctx context.Context, resourceGroupName string, networkInterfaceName string, parameters network.Interface) *retry.Error {
	mc := newMetricContext("interfaces", "create_or_update", resourceGroupName, az.client.SubscriptionID, "")
	/* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return createRateLimitErr(true, "NiCreateOrUpdate")
	}

	klog.V(10).Infof("azInterfacesClient.CreateOrUpdate(%q,%q): start", resourceGroupName, networkInterfaceName)
	defer func() {
		klog.V(10).Infof("azInterfacesClient.CreateOrUpdate(%q,%q): end", resourceGroupName, networkInterfaceName)
	}()

	future, err := az.client.CreateOrUpdate(ctx, resourceGroupName, networkInterfaceName, parameters)
	if err != nil {
		return retry.GetError(future.Response(), mc.Observe(err))
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	return retry.GetError(future.Response(), mc.Observe(err))
}

func (az *azInterfacesClient) Get(ctx context.Context, resourceGroupName string, networkInterfaceName string, expand string) (result network.Interface, rerr *retry.Error) {
	mc := newMetricContext("interfaces", "get", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		rerr = createRateLimitErr(false, "NicGet")
		return
	}

	klog.V(10).Infof("azInterfacesClient.Get(%q,%q): start", resourceGroupName, networkInterfaceName)
	defer func() {
		klog.V(10).Infof("azInterfacesClient.Get(%q,%q): end", resourceGroupName, networkInterfaceName)
	}()

	var err error
	result, err = az.client.Get(ctx, resourceGroupName, networkInterfaceName, expand)
	mc.Observe(err)
	return result, retry.GetError(result.Response.Response, err)
}

func (az *azInterfacesClient) GetVirtualMachineScaleSetNetworkInterface(ctx context.Context, resourceGroupName string, virtualMachineScaleSetName string, virtualmachineIndex string, networkInterfaceName string, expand string) (result network.Interface, rerr *retry.Error) {
	mc := newMetricContext("interfaces", "get_vmss_ni", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		rerr = createRateLimitErr(false, "NicGetVirtualMachineScaleSetNetworkInterface")
		return
	}

	klog.V(10).Infof("azInterfacesClient.GetVirtualMachineScaleSetNetworkInterface(%q,%q,%q,%q): start", resourceGroupName, virtualMachineScaleSetName, virtualmachineIndex, networkInterfaceName)
	defer func() {
		klog.V(10).Infof("azInterfacesClient.GetVirtualMachineScaleSetNetworkInterface(%q,%q,%q,%q): end", resourceGroupName, virtualMachineScaleSetName, virtualmachineIndex, networkInterfaceName)
	}()

	var err error
	result, err = az.client.GetVirtualMachineScaleSetNetworkInterface(ctx, resourceGroupName, virtualMachineScaleSetName, virtualmachineIndex, networkInterfaceName, expand)
	mc.Observe(err)
	return result, retry.GetError(result.Response.Response, err)
}

// azLoadBalancersClient implements LoadBalancersClient.
type azLoadBalancersClient struct {
	client            network.LoadBalancersClient
	rateLimiterReader flowcontrol.RateLimiter
	rateLimiterWriter flowcontrol.RateLimiter
}

func newAzLoadBalancersClient(config *azclients.ClientConfig) *azLoadBalancersClient {
	loadBalancerClient := network.NewLoadBalancersClient(config.SubscriptionID)
	loadBalancerClient.BaseURI = config.ResourceManagerEndpoint
	loadBalancerClient.Authorizer = autorest.NewBearerAuthorizer(config.ServicePrincipalToken)
	loadBalancerClient.PollingDelay = 5 * time.Second
	if config.ShouldOmitCloudProviderBackoff {
		loadBalancerClient.RetryAttempts = config.CloudProviderBackoffRetries
		loadBalancerClient.RetryDuration = time.Duration(config.CloudProviderBackoffDuration) * time.Second
	}
	configureUserAgent(&loadBalancerClient.Client)

	klog.V(2).Infof("Azure LoadBalancersClient (read ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPS,
		config.RateLimitConfig.CloudProviderRateLimitBucket)
	klog.V(2).Infof("Azure LoadBalancersClient (write ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPSWrite,
		config.RateLimitConfig.CloudProviderRateLimitBucketWrite)
	rateLimiterReader, rateLimiterWriter := azclients.NewRateLimiter(config.RateLimitConfig)
	return &azLoadBalancersClient{
		rateLimiterReader: rateLimiterReader,
		rateLimiterWriter: rateLimiterWriter,
		client:            loadBalancerClient,
	}
}

func (az *azLoadBalancersClient) CreateOrUpdate(ctx context.Context, resourceGroupName string, loadBalancerName string, parameters network.LoadBalancer, etag string) *retry.Error {
	mc := newMetricContext("load_balancers", "create_or_update", resourceGroupName, az.client.SubscriptionID, "")
	/* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return createRateLimitErr(true, "LBCreateOrUpdate")
	}

	klog.V(10).Infof("azLoadBalancersClient.CreateOrUpdate(%q,%q): start", resourceGroupName, loadBalancerName)
	defer func() {
		klog.V(10).Infof("azLoadBalancersClient.CreateOrUpdate(%q,%q): end", resourceGroupName, loadBalancerName)
	}()

	req, err := az.createOrUpdatePreparer(ctx, resourceGroupName, loadBalancerName, parameters, etag)
	if err != nil {
		return retry.NewError(false, mc.Observe(err))
	}

	future, err := az.client.CreateOrUpdateSender(req)
	if err != nil {
		return retry.GetError(future.Response(), mc.Observe(err))
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	return retry.GetError(future.Response(), mc.Observe(err))
}

// createOrUpdatePreparer prepares the CreateOrUpdate request.
func (az *azLoadBalancersClient) createOrUpdatePreparer(ctx context.Context, resourceGroupName string, loadBalancerName string, parameters network.LoadBalancer, etag string) (*http.Request, error) {
	pathParameters := map[string]interface{}{
		"loadBalancerName":  autorest.Encode("path", loadBalancerName),
		"resourceGroupName": autorest.Encode("path", resourceGroupName),
		"subscriptionId":    autorest.Encode("path", az.client.SubscriptionID),
	}

	queryParameters := map[string]interface{}{
		"api-version": azureNetworkAPIVersion,
	}

	preparerDecorators := []autorest.PrepareDecorator{
		autorest.AsContentType("application/json; charset=utf-8"),
		autorest.AsPut(),
		autorest.WithBaseURL(az.client.BaseURI),
		autorest.WithPathParameters("/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Network/loadBalancers/{loadBalancerName}", pathParameters),
		autorest.WithJSON(parameters),
		autorest.WithQueryParameters(queryParameters),
	}
	if etag != "" {
		preparerDecorators = append(preparerDecorators, autorest.WithHeader("If-Match", autorest.String(etag)))
	}
	preparer := autorest.CreatePreparer(preparerDecorators...)
	return preparer.Prepare((&http.Request{}).WithContext(ctx))
}

func (az *azLoadBalancersClient) Delete(ctx context.Context, resourceGroupName string, loadBalancerName string) *retry.Error {
	mc := newMetricContext("load_balancers", "delete", resourceGroupName, az.client.SubscriptionID, "")
	/* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return createRateLimitErr(true, "LBDelete")
	}

	klog.V(10).Infof("azLoadBalancersClient.Delete(%q,%q): start", resourceGroupName, loadBalancerName)
	defer func() {
		klog.V(10).Infof("azLoadBalancersClient.Delete(%q,%q): end", resourceGroupName, loadBalancerName)
	}()

	future, err := az.client.Delete(ctx, resourceGroupName, loadBalancerName)
	if err != nil {
		return retry.GetStatusNotFoundAndForbiddenIgnoredError(future.Response(), mc.Observe(err))
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	return retry.GetStatusNotFoundAndForbiddenIgnoredError(future.Response(), mc.Observe(err))
}

func (az *azLoadBalancersClient) Get(ctx context.Context, resourceGroupName string, loadBalancerName string, expand string) (result network.LoadBalancer, rerr *retry.Error) {
	mc := newMetricContext("load_balancers", "get", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		rerr = createRateLimitErr(false, "LBGet")
		return
	}

	klog.V(10).Infof("azLoadBalancersClient.Get(%q,%q): start", resourceGroupName, loadBalancerName)
	defer func() {
		klog.V(10).Infof("azLoadBalancersClient.Get(%q,%q): end", resourceGroupName, loadBalancerName)
	}()

	var err error
	result, err = az.client.Get(ctx, resourceGroupName, loadBalancerName, expand)
	mc.Observe(err)
	return result, retry.GetError(result.Response.Response, err)
}

func (az *azLoadBalancersClient) List(ctx context.Context, resourceGroupName string) ([]network.LoadBalancer, *retry.Error) {
	mc := newMetricContext("load_balancers", "list", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		rerr := createRateLimitErr(false, "LBList")
		return nil, rerr
	}

	klog.V(10).Infof("azLoadBalancersClient.List(%q): start", resourceGroupName)
	defer func() {
		klog.V(10).Infof("azLoadBalancersClient.List(%q): end", resourceGroupName)
	}()

	iterator, err := az.client.ListComplete(ctx, resourceGroupName)
	mc.Observe(err)
	if err != nil {
		return nil, retry.GetRetriableError(err)
	}

	result := make([]network.LoadBalancer, 0)
	for ; iterator.NotDone(); err = iterator.Next() {
		if err != nil {
			return nil, retry.GetRetriableError(err)
		}

		result = append(result, iterator.Value())
	}

	return result, nil
}

// azPublicIPAddressesClient implements PublicIPAddressesClient.
type azPublicIPAddressesClient struct {
	client            network.PublicIPAddressesClient
	rateLimiterReader flowcontrol.RateLimiter
	rateLimiterWriter flowcontrol.RateLimiter
}

func newAzPublicIPAddressesClient(config *azclients.ClientConfig) *azPublicIPAddressesClient {
	publicIPAddressClient := network.NewPublicIPAddressesClient(config.SubscriptionID)
	publicIPAddressClient.BaseURI = config.ResourceManagerEndpoint
	publicIPAddressClient.Authorizer = autorest.NewBearerAuthorizer(config.ServicePrincipalToken)
	publicIPAddressClient.PollingDelay = 5 * time.Second
	if config.ShouldOmitCloudProviderBackoff {
		publicIPAddressClient.RetryAttempts = config.CloudProviderBackoffRetries
		publicIPAddressClient.RetryDuration = time.Duration(config.CloudProviderBackoffDuration) * time.Second
	}
	configureUserAgent(&publicIPAddressClient.Client)

	klog.V(2).Infof("Azure PublicIPAddressesClient (read ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPS,
		config.RateLimitConfig.CloudProviderRateLimitBucket)
	klog.V(2).Infof("Azure PublicIPAddressesClient (write ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPSWrite,
		config.RateLimitConfig.CloudProviderRateLimitBucketWrite)
	rateLimiterReader, rateLimiterWriter := azclients.NewRateLimiter(config.RateLimitConfig)
	return &azPublicIPAddressesClient{
		rateLimiterReader: rateLimiterReader,
		rateLimiterWriter: rateLimiterWriter,
		client:            publicIPAddressClient,
	}
}

func (az *azPublicIPAddressesClient) CreateOrUpdate(ctx context.Context, resourceGroupName string, publicIPAddressName string, parameters network.PublicIPAddress) *retry.Error {
	mc := newMetricContext("public_ip_addresses", "create_or_update", resourceGroupName, az.client.SubscriptionID, "")
	/* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return createRateLimitErr(true, "PublicIPCreateOrUpdate")
	}

	klog.V(10).Infof("azPublicIPAddressesClient.CreateOrUpdate(%q,%q): start", resourceGroupName, publicIPAddressName)
	defer func() {
		klog.V(10).Infof("azPublicIPAddressesClient.CreateOrUpdate(%q,%q): end", resourceGroupName, publicIPAddressName)
	}()

	future, err := az.client.CreateOrUpdate(ctx, resourceGroupName, publicIPAddressName, parameters)
	if err != nil {
		return retry.GetError(future.Response(), mc.Observe(err))
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	return retry.GetError(future.Response(), mc.Observe(err))
}

func (az *azPublicIPAddressesClient) Delete(ctx context.Context, resourceGroupName string, publicIPAddressName string) *retry.Error {
	mc := newMetricContext("public_ip_addresses", "delete", resourceGroupName, az.client.SubscriptionID, "")
	/* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return createRateLimitErr(true, "PublicIPDelete")
	}

	klog.V(10).Infof("azPublicIPAddressesClient.Delete(%q,%q): start", resourceGroupName, publicIPAddressName)
	defer func() {
		klog.V(10).Infof("azPublicIPAddressesClient.Delete(%q,%q): end", resourceGroupName, publicIPAddressName)
	}()

	future, err := az.client.Delete(ctx, resourceGroupName, publicIPAddressName)
	if err != nil {
		return retry.GetStatusNotFoundAndForbiddenIgnoredError(future.Response(), mc.Observe(err))
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	return retry.GetStatusNotFoundAndForbiddenIgnoredError(future.Response(), mc.Observe(err))
}

func (az *azPublicIPAddressesClient) Get(ctx context.Context, resourceGroupName string, publicIPAddressName string, expand string) (result network.PublicIPAddress, rerr *retry.Error) {
	mc := newMetricContext("public_ip_addresses", "get", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		rerr = createRateLimitErr(false, "PublicIPGet")
		return
	}

	klog.V(10).Infof("azPublicIPAddressesClient.Get(%q,%q): start", resourceGroupName, publicIPAddressName)
	defer func() {
		klog.V(10).Infof("azPublicIPAddressesClient.Get(%q,%q): end", resourceGroupName, publicIPAddressName)
	}()

	var err error
	result, err = az.client.Get(ctx, resourceGroupName, publicIPAddressName, expand)
	mc.Observe(err)
	return result, retry.GetError(result.Response.Response, err)
}

func (az *azPublicIPAddressesClient) GetVirtualMachineScaleSetPublicIPAddress(ctx context.Context, resourceGroupName string, virtualMachineScaleSetName string, virtualmachineIndex string, networkInterfaceName string, IPConfigurationName string, publicIPAddressName string, expand string) (result network.PublicIPAddress, rerr *retry.Error) {
	mc := newMetricContext("vmss_public_ip_addresses", "get", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		rerr = createRateLimitErr(false, "VMSSPublicIPGet")
		return
	}

	klog.V(10).Infof("azPublicIPAddressesClient.GetVirtualMachineScaleSetPublicIPAddress(%q,%q): start", resourceGroupName, publicIPAddressName)
	defer func() {
		klog.V(10).Infof("azPublicIPAddressesClient.GetVirtualMachineScaleSetPublicIPAddress(%q,%q): end", resourceGroupName, publicIPAddressName)
	}()

	var err error
	result, err = az.client.GetVirtualMachineScaleSetPublicIPAddress(ctx, resourceGroupName, virtualMachineScaleSetName, virtualmachineIndex, networkInterfaceName, IPConfigurationName, publicIPAddressName, expand)
	mc.Observe(err)
	return result, retry.GetError(result.Response.Response, err)
}

func (az *azPublicIPAddressesClient) List(ctx context.Context, resourceGroupName string) ([]network.PublicIPAddress, *retry.Error) {
	mc := newMetricContext("public_ip_addresses", "list", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		return nil, createRateLimitErr(false, "PublicIPList")
	}

	klog.V(10).Infof("azPublicIPAddressesClient.List(%q): start", resourceGroupName)
	defer func() {
		klog.V(10).Infof("azPublicIPAddressesClient.List(%q): end", resourceGroupName)
	}()

	iterator, err := az.client.ListComplete(ctx, resourceGroupName)
	mc.Observe(err)
	if err != nil {
		return nil, retry.GetRetriableError(err)
	}

	result := make([]network.PublicIPAddress, 0)
	for ; iterator.NotDone(); err = iterator.Next() {
		if err != nil {
			return nil, retry.GetRetriableError(err)
		}

		result = append(result, iterator.Value())
	}

	return result, nil
}

// azSubnetsClient implements SubnetsClient.
type azSubnetsClient struct {
	client            network.SubnetsClient
	rateLimiterReader flowcontrol.RateLimiter
	rateLimiterWriter flowcontrol.RateLimiter
}

func newAzSubnetsClient(config *azclients.ClientConfig) *azSubnetsClient {
	subnetsClient := network.NewSubnetsClient(config.SubscriptionID)
	subnetsClient.BaseURI = config.ResourceManagerEndpoint
	subnetsClient.Authorizer = autorest.NewBearerAuthorizer(config.ServicePrincipalToken)
	subnetsClient.PollingDelay = 5 * time.Second
	if config.ShouldOmitCloudProviderBackoff {
		subnetsClient.RetryAttempts = config.CloudProviderBackoffRetries
		subnetsClient.RetryDuration = time.Duration(config.CloudProviderBackoffDuration) * time.Second
	}
	configureUserAgent(&subnetsClient.Client)

	klog.V(2).Infof("Azure SubnetsClient (read ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPS,
		config.RateLimitConfig.CloudProviderRateLimitBucket)
	klog.V(2).Infof("Azure SubnetsClient (write ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPSWrite,
		config.RateLimitConfig.CloudProviderRateLimitBucketWrite)
	rateLimiterReader, rateLimiterWriter := azclients.NewRateLimiter(config.RateLimitConfig)
	return &azSubnetsClient{
		client:            subnetsClient,
		rateLimiterReader: rateLimiterReader,
		rateLimiterWriter: rateLimiterWriter,
	}
}

func (az *azSubnetsClient) CreateOrUpdate(ctx context.Context, resourceGroupName string, virtualNetworkName string, subnetName string, subnetParameters network.Subnet) *retry.Error {
	mc := newMetricContext("subnets", "create_or_update", resourceGroupName, az.client.SubscriptionID, "")
	/* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return createRateLimitErr(true, "SubnetCreateOrUpdate")
	}

	klog.V(10).Infof("azSubnetsClient.CreateOrUpdate(%q,%q,%q): start", resourceGroupName, virtualNetworkName, subnetName)
	defer func() {
		klog.V(10).Infof("azSubnetsClient.CreateOrUpdate(%q,%q,%q): end", resourceGroupName, virtualNetworkName, subnetName)
	}()

	future, err := az.client.CreateOrUpdate(ctx, resourceGroupName, virtualNetworkName, subnetName, subnetParameters)
	if err != nil {
		return retry.GetError(future.Response(), mc.Observe(err))
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	return retry.GetError(future.Response(), mc.Observe(err))
}

func (az *azSubnetsClient) Delete(ctx context.Context, resourceGroupName string, virtualNetworkName string, subnetName string) *retry.Error {
	mc := newMetricContext("subnets", "delete", resourceGroupName, az.client.SubscriptionID, "")
	/* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return createRateLimitErr(true, "SubnetDelete")
	}

	klog.V(10).Infof("azSubnetsClient.Delete(%q,%q,%q): start", resourceGroupName, virtualNetworkName, subnetName)
	defer func() {
		klog.V(10).Infof("azSubnetsClient.Delete(%q,%q,%q): end", resourceGroupName, virtualNetworkName, subnetName)
	}()

	future, err := az.client.Delete(ctx, resourceGroupName, virtualNetworkName, subnetName)
	if err != nil {
		return retry.GetStatusNotFoundAndForbiddenIgnoredError(future.Response(), mc.Observe(err))
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	return retry.GetStatusNotFoundAndForbiddenIgnoredError(future.Response(), mc.Observe(err))
}

func (az *azSubnetsClient) Get(ctx context.Context, resourceGroupName string, virtualNetworkName string, subnetName string, expand string) (result network.Subnet, rerr *retry.Error) {
	mc := newMetricContext("subnets", "get", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		rerr = createRateLimitErr(false, "SubnetGet")
		return
	}

	klog.V(10).Infof("azSubnetsClient.Get(%q,%q,%q): start", resourceGroupName, virtualNetworkName, subnetName)
	defer func() {
		klog.V(10).Infof("azSubnetsClient.Get(%q,%q,%q): end", resourceGroupName, virtualNetworkName, subnetName)
	}()

	var err error
	result, err = az.client.Get(ctx, resourceGroupName, virtualNetworkName, subnetName, expand)
	mc.Observe(err)
	return result, retry.GetError(result.Response.Response, err)
}

func (az *azSubnetsClient) List(ctx context.Context, resourceGroupName string, virtualNetworkName string) ([]network.Subnet, *retry.Error) {
	mc := newMetricContext("subnets", "list", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		return nil, createRateLimitErr(false, "SubnetList")
	}

	klog.V(10).Infof("azSubnetsClient.List(%q,%q): start", resourceGroupName, virtualNetworkName)
	defer func() {
		klog.V(10).Infof("azSubnetsClient.List(%q,%q): end", resourceGroupName, virtualNetworkName)
	}()

	iterator, err := az.client.ListComplete(ctx, resourceGroupName, virtualNetworkName)
	mc.Observe(err)
	if err != nil {
		return nil, retry.GetRetriableError(err)
	}

	result := make([]network.Subnet, 0)
	for ; iterator.NotDone(); err = iterator.Next() {
		if err != nil {
			return nil, retry.GetRetriableError(err)
		}

		result = append(result, iterator.Value())
	}

	return result, nil
}

// azSecurityGroupsClient implements SecurityGroupsClient.
type azSecurityGroupsClient struct {
	client            network.SecurityGroupsClient
	rateLimiterReader flowcontrol.RateLimiter
	rateLimiterWriter flowcontrol.RateLimiter
}

func newAzSecurityGroupsClient(config *azclients.ClientConfig) *azSecurityGroupsClient {
	securityGroupsClient := network.NewSecurityGroupsClient(config.SubscriptionID)
	securityGroupsClient.BaseURI = config.ResourceManagerEndpoint
	securityGroupsClient.Authorizer = autorest.NewBearerAuthorizer(config.ServicePrincipalToken)
	securityGroupsClient.PollingDelay = 5 * time.Second
	if config.ShouldOmitCloudProviderBackoff {
		securityGroupsClient.RetryAttempts = config.CloudProviderBackoffRetries
		securityGroupsClient.RetryDuration = time.Duration(config.CloudProviderBackoffDuration) * time.Second
	}
	configureUserAgent(&securityGroupsClient.Client)

	klog.V(2).Infof("Azure SecurityGroupsClient (read ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPS,
		config.RateLimitConfig.CloudProviderRateLimitBucket)
	klog.V(2).Infof("Azure SecurityGroupsClient (write ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPSWrite,
		config.RateLimitConfig.CloudProviderRateLimitBucketWrite)
	rateLimiterReader, rateLimiterWriter := azclients.NewRateLimiter(config.RateLimitConfig)
	return &azSecurityGroupsClient{
		client:            securityGroupsClient,
		rateLimiterReader: rateLimiterReader,
		rateLimiterWriter: rateLimiterWriter,
	}
}

func (az *azSecurityGroupsClient) CreateOrUpdate(ctx context.Context, resourceGroupName string, networkSecurityGroupName string, parameters network.SecurityGroup, etag string) *retry.Error {
	mc := newMetricContext("security_groups", "create_or_update", resourceGroupName, az.client.SubscriptionID, "")
	/* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return createRateLimitErr(true, "NSGCreateOrUpdate")
	}

	klog.V(10).Infof("azSecurityGroupsClient.CreateOrUpdate(%q,%q): start", resourceGroupName, networkSecurityGroupName)
	defer func() {
		klog.V(10).Infof("azSecurityGroupsClient.CreateOrUpdate(%q,%q): end", resourceGroupName, networkSecurityGroupName)
	}()

	req, err := az.createOrUpdatePreparer(ctx, resourceGroupName, networkSecurityGroupName, parameters, etag)
	if err != nil {
		return retry.NewError(false, mc.Observe(err))
	}

	future, err := az.client.CreateOrUpdateSender(req)
	if err != nil {
		return retry.GetError(future.Response(), mc.Observe(err))
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	return retry.GetError(future.Response(), mc.Observe(err))
}

// createOrUpdatePreparer prepares the CreateOrUpdate request.
func (az *azSecurityGroupsClient) createOrUpdatePreparer(ctx context.Context, resourceGroupName string, networkSecurityGroupName string, parameters network.SecurityGroup, etag string) (*http.Request, error) {
	pathParameters := map[string]interface{}{
		"networkSecurityGroupName": autorest.Encode("path", networkSecurityGroupName),
		"resourceGroupName":        autorest.Encode("path", resourceGroupName),
		"subscriptionId":           autorest.Encode("path", az.client.SubscriptionID),
	}

	queryParameters := map[string]interface{}{
		"api-version": azureNetworkAPIVersion,
	}

	preparerDecorators := []autorest.PrepareDecorator{
		autorest.AsContentType("application/json; charset=utf-8"),
		autorest.AsPut(),
		autorest.WithBaseURL(az.client.BaseURI),
		autorest.WithPathParameters("/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Network/networkSecurityGroups/{networkSecurityGroupName}", pathParameters),
		autorest.WithJSON(parameters),
		autorest.WithQueryParameters(queryParameters),
	}
	if etag != "" {
		preparerDecorators = append(preparerDecorators, autorest.WithHeader("If-Match", autorest.String(etag)))
	}
	preparer := autorest.CreatePreparer(preparerDecorators...)
	return preparer.Prepare((&http.Request{}).WithContext(ctx))
}

func (az *azSecurityGroupsClient) Delete(ctx context.Context, resourceGroupName string, networkSecurityGroupName string) *retry.Error {
	mc := newMetricContext("security_groups", "delete", resourceGroupName, az.client.SubscriptionID, "")
	/* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return createRateLimitErr(true, "NSGDelete")
	}

	klog.V(10).Infof("azSecurityGroupsClient.Delete(%q,%q): start", resourceGroupName, networkSecurityGroupName)
	defer func() {
		klog.V(10).Infof("azSecurityGroupsClient.Delete(%q,%q): end", resourceGroupName, networkSecurityGroupName)
	}()

	future, err := az.client.Delete(ctx, resourceGroupName, networkSecurityGroupName)
	if err != nil {
		return retry.GetStatusNotFoundAndForbiddenIgnoredError(future.Response(), mc.Observe(err))
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	return retry.GetStatusNotFoundAndForbiddenIgnoredError(future.Response(), mc.Observe(err))
}

func (az *azSecurityGroupsClient) Get(ctx context.Context, resourceGroupName string, networkSecurityGroupName string, expand string) (result network.SecurityGroup, rerr *retry.Error) {
	mc := newMetricContext("security_groups", "get", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		rerr = createRateLimitErr(false, "NSGGet")
		return
	}

	klog.V(10).Infof("azSecurityGroupsClient.Get(%q,%q): start", resourceGroupName, networkSecurityGroupName)
	defer func() {
		klog.V(10).Infof("azSecurityGroupsClient.Get(%q,%q): end", resourceGroupName, networkSecurityGroupName)
	}()

	var err error
	result, err = az.client.Get(ctx, resourceGroupName, networkSecurityGroupName, expand)
	mc.Observe(err)
	return result, retry.GetError(result.Response.Response, err)
}

func (az *azSecurityGroupsClient) List(ctx context.Context, resourceGroupName string) ([]network.SecurityGroup, *retry.Error) {
	mc := newMetricContext("security_groups", "list", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		return nil, createRateLimitErr(false, "NSGList")
	}

	klog.V(10).Infof("azSecurityGroupsClient.List(%q): start", resourceGroupName)
	defer func() {
		klog.V(10).Infof("azSecurityGroupsClient.List(%q): end", resourceGroupName)
	}()

	iterator, err := az.client.ListComplete(ctx, resourceGroupName)
	mc.Observe(err)
	if err != nil {
		return nil, retry.GetRetriableError(err)
	}

	result := make([]network.SecurityGroup, 0)
	for ; iterator.NotDone(); err = iterator.Next() {
		if err != nil {
			return nil, retry.GetRetriableError(err)
		}

		result = append(result, iterator.Value())
	}

	return result, nil
}

// azVirtualMachineScaleSetsClient implements VirtualMachineScaleSetsClient.
type azVirtualMachineScaleSetsClient struct {
	client            compute.VirtualMachineScaleSetsClient
	rateLimiterReader flowcontrol.RateLimiter
	rateLimiterWriter flowcontrol.RateLimiter
}

func newAzVirtualMachineScaleSetsClient(config *azclients.ClientConfig) *azVirtualMachineScaleSetsClient {
	virtualMachineScaleSetsClient := compute.NewVirtualMachineScaleSetsClient(config.SubscriptionID)
	virtualMachineScaleSetsClient.BaseURI = config.ResourceManagerEndpoint
	virtualMachineScaleSetsClient.Authorizer = autorest.NewBearerAuthorizer(config.ServicePrincipalToken)
	virtualMachineScaleSetsClient.PollingDelay = 5 * time.Second
	if config.ShouldOmitCloudProviderBackoff {
		virtualMachineScaleSetsClient.RetryAttempts = config.CloudProviderBackoffRetries
		virtualMachineScaleSetsClient.RetryDuration = time.Duration(config.CloudProviderBackoffDuration) * time.Second
	}
	configureUserAgent(&virtualMachineScaleSetsClient.Client)

	klog.V(2).Infof("Azure VirtualMachineScaleSetsClient (read ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPS,
		config.RateLimitConfig.CloudProviderRateLimitBucket)
	klog.V(2).Infof("Azure VirtualMachineScaleSetsClient (write ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPSWrite,
		config.RateLimitConfig.CloudProviderRateLimitBucketWrite)
	rateLimiterReader, rateLimiterWriter := azclients.NewRateLimiter(config.RateLimitConfig)
	return &azVirtualMachineScaleSetsClient{
		client:            virtualMachineScaleSetsClient,
		rateLimiterReader: rateLimiterReader,
		rateLimiterWriter: rateLimiterWriter,
	}
}

func (az *azVirtualMachineScaleSetsClient) Get(ctx context.Context, resourceGroupName string, VMScaleSetName string) (result compute.VirtualMachineScaleSet, rerr *retry.Error) {
	mc := newMetricContext("vmss", "get", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		rerr = createRateLimitErr(false, "VMSSGet")
		return
	}

	klog.V(10).Infof("azVirtualMachineScaleSetsClient.Get(%q,%q): start", resourceGroupName, VMScaleSetName)
	defer func() {
		klog.V(10).Infof("azVirtualMachineScaleSetsClient.Get(%q,%q): end", resourceGroupName, VMScaleSetName)
	}()

	var err error
	result, err = az.client.Get(ctx, resourceGroupName, VMScaleSetName)
	mc.Observe(err)
	return result, retry.GetError(result.Response.Response, err)
}

func (az *azVirtualMachineScaleSetsClient) List(ctx context.Context, resourceGroupName string) (result []compute.VirtualMachineScaleSet, rerr *retry.Error) {
	mc := newMetricContext("vmss", "list", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		rerr = createRateLimitErr(false, "VMSSList")
		return
	}

	klog.V(10).Infof("azVirtualMachineScaleSetsClient.List(%q): start", resourceGroupName)
	defer func() {
		klog.V(10).Infof("azVirtualMachineScaleSetsClient.List(%q): end", resourceGroupName)
	}()

	iterator, err := az.client.ListComplete(ctx, resourceGroupName)
	mc.Observe(err)
	if err != nil {
		return nil, retry.GetRetriableError(err)
	}

	result = make([]compute.VirtualMachineScaleSet, 0)
	for ; iterator.NotDone(); err = iterator.Next() {
		if err != nil {
			return nil, retry.GetRetriableError(err)
		}

		result = append(result, iterator.Value())
	}

	return result, nil
}

func (az *azVirtualMachineScaleSetsClient) CreateOrUpdate(ctx context.Context, resourceGroupName string, vmScaleSetName string, parameters compute.VirtualMachineScaleSet) *retry.Error {
	mc := newMetricContext("vmss", "create_or_update", resourceGroupName, az.client.SubscriptionID, "")
	/* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return createRateLimitErr(true, "NiCreateOrUpdate")
	}

	klog.V(10).Infof("azVirtualMachineScaleSetsClient.CreateOrUpdate(%q,%q): start", resourceGroupName, vmScaleSetName)
	defer func() {
		klog.V(10).Infof("azVirtualMachineScaleSetsClient.CreateOrUpdate(%q,%q): end", resourceGroupName, vmScaleSetName)
	}()

	future, err := az.client.CreateOrUpdate(ctx, resourceGroupName, vmScaleSetName, parameters)
	if err != nil {
		return retry.GetError(future.Response(), mc.Observe(err))
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	return retry.GetError(future.Response(), mc.Observe(err))
}

// azVirtualMachineScaleSetVMsClient implements VirtualMachineScaleSetVMsClient.
type azVirtualMachineScaleSetVMsClient struct {
	client            compute.VirtualMachineScaleSetVMsClient
	rateLimiterReader flowcontrol.RateLimiter
	rateLimiterWriter flowcontrol.RateLimiter
}

func newAzVirtualMachineScaleSetVMsClient(config *azclients.ClientConfig) *azVirtualMachineScaleSetVMsClient {
	virtualMachineScaleSetVMsClient := compute.NewVirtualMachineScaleSetVMsClient(config.SubscriptionID)
	virtualMachineScaleSetVMsClient.BaseURI = config.ResourceManagerEndpoint
	virtualMachineScaleSetVMsClient.Authorizer = autorest.NewBearerAuthorizer(config.ServicePrincipalToken)
	virtualMachineScaleSetVMsClient.PollingDelay = 5 * time.Second
	if config.ShouldOmitCloudProviderBackoff {
		virtualMachineScaleSetVMsClient.RetryAttempts = config.CloudProviderBackoffRetries
		virtualMachineScaleSetVMsClient.RetryDuration = time.Duration(config.CloudProviderBackoffDuration) * time.Second
	}
	configureUserAgent(&virtualMachineScaleSetVMsClient.Client)

	klog.V(2).Infof("Azure VirtualMachineScaleSetVMsClient (read ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPS,
		config.RateLimitConfig.CloudProviderRateLimitBucket)
	klog.V(2).Infof("Azure VirtualMachineScaleSetVMsClient (write ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPSWrite,
		config.RateLimitConfig.CloudProviderRateLimitBucketWrite)
	rateLimiterReader, rateLimiterWriter := azclients.NewRateLimiter(config.RateLimitConfig)
	return &azVirtualMachineScaleSetVMsClient{
		client:            virtualMachineScaleSetVMsClient,
		rateLimiterReader: rateLimiterReader,
		rateLimiterWriter: rateLimiterWriter,
	}
}

func (az *azVirtualMachineScaleSetVMsClient) Get(ctx context.Context, resourceGroupName string, VMScaleSetName string, instanceID string, expand compute.InstanceViewTypes) (result compute.VirtualMachineScaleSetVM, rerr *retry.Error) {
	mc := newMetricContext("vmssvm", "get", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		rerr = createRateLimitErr(false, "VMSSGet")
		return
	}

	klog.V(10).Infof("azVirtualMachineScaleSetVMsClient.Get(%q,%q,%q): start", resourceGroupName, VMScaleSetName, instanceID)
	defer func() {
		klog.V(10).Infof("azVirtualMachineScaleSetVMsClient.Get(%q,%q,%q): end", resourceGroupName, VMScaleSetName, instanceID)
	}()

	var err error
	result, err = az.client.Get(ctx, resourceGroupName, VMScaleSetName, instanceID, expand)
	mc.Observe(err)
	return result, retry.GetError(result.Response.Response, err)
}

func (az *azVirtualMachineScaleSetVMsClient) List(ctx context.Context, resourceGroupName string, virtualMachineScaleSetName string, filter string, selectParameter string, expand string) (result []compute.VirtualMachineScaleSetVM, rerr *retry.Error) {
	mc := newMetricContext("vmssvm", "list", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		rerr = createRateLimitErr(false, "VMSSList")
		return
	}

	klog.V(10).Infof("azVirtualMachineScaleSetVMsClient.List(%q,%q,%q): start", resourceGroupName, virtualMachineScaleSetName, filter)
	defer func() {
		klog.V(10).Infof("azVirtualMachineScaleSetVMsClient.List(%q,%q,%q): end", resourceGroupName, virtualMachineScaleSetName, filter)
	}()

	iterator, err := az.client.ListComplete(ctx, resourceGroupName, virtualMachineScaleSetName, filter, selectParameter, expand)
	mc.Observe(err)
	if err != nil {
		return nil, retry.GetRetriableError(err)
	}

	result = make([]compute.VirtualMachineScaleSetVM, 0)
	for ; iterator.NotDone(); err = iterator.Next() {
		if err != nil {
			return nil, retry.GetRetriableError(err)
		}

		result = append(result, iterator.Value())
	}

	return result, nil
}

func (az *azVirtualMachineScaleSetVMsClient) Update(ctx context.Context, resourceGroupName string, VMScaleSetName string, instanceID string, parameters compute.VirtualMachineScaleSetVM, source string) *retry.Error {
	mc := newMetricContext("vmssvm", "create_or_update", resourceGroupName, az.client.SubscriptionID, source)
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return createRateLimitErr(true, "VMSSVMUpdate")
	}

	klog.V(10).Infof("azVirtualMachineScaleSetVMsClient.Update(%q,%q,%q): start", resourceGroupName, VMScaleSetName, instanceID)
	defer func() {
		klog.V(10).Infof("azVirtualMachineScaleSetVMsClient.Update(%q,%q,%q): end", resourceGroupName, VMScaleSetName, instanceID)
	}()

	future, err := az.client.Update(ctx, resourceGroupName, VMScaleSetName, instanceID, parameters)
	if err != nil {
		return retry.GetError(future.Response(), mc.Observe(err))
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	return retry.GetError(future.Response(), mc.Observe(err))
}

// azRoutesClient implements RoutesClient.
type azRoutesClient struct {
	client            network.RoutesClient
	rateLimiterReader flowcontrol.RateLimiter
	rateLimiterWriter flowcontrol.RateLimiter
}

func newAzRoutesClient(config *azclients.ClientConfig) *azRoutesClient {
	routesClient := network.NewRoutesClient(config.SubscriptionID)
	routesClient.BaseURI = config.ResourceManagerEndpoint
	routesClient.Authorizer = autorest.NewBearerAuthorizer(config.ServicePrincipalToken)
	routesClient.PollingDelay = 5 * time.Second
	if config.ShouldOmitCloudProviderBackoff {
		routesClient.RetryAttempts = config.CloudProviderBackoffRetries
		routesClient.RetryDuration = time.Duration(config.CloudProviderBackoffDuration) * time.Second
	}
	configureUserAgent(&routesClient.Client)

	klog.V(2).Infof("Azure RoutesClient (read ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPS,
		config.RateLimitConfig.CloudProviderRateLimitBucket)
	klog.V(2).Infof("Azure RoutesClient (write ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPSWrite,
		config.RateLimitConfig.CloudProviderRateLimitBucketWrite)
	rateLimiterReader, rateLimiterWriter := azclients.NewRateLimiter(config.RateLimitConfig)
	return &azRoutesClient{
		client:            routesClient,
		rateLimiterReader: rateLimiterReader,
		rateLimiterWriter: rateLimiterWriter,
	}
}

func (az *azRoutesClient) CreateOrUpdate(ctx context.Context, resourceGroupName string, routeTableName string, routeName string, routeParameters network.Route, etag string) *retry.Error {
	mc := newMetricContext("routes", "create_or_update", resourceGroupName, az.client.SubscriptionID, "")
	/* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return createRateLimitErr(true, "RouteCreateOrUpdate")
	}

	klog.V(10).Infof("azRoutesClient.CreateOrUpdate(%q,%q,%q): start", resourceGroupName, routeTableName, routeName)
	defer func() {
		klog.V(10).Infof("azRoutesClient.CreateOrUpdate(%q,%q,%q): end", resourceGroupName, routeTableName, routeName)
	}()

	req, err := az.createOrUpdatePreparer(ctx, resourceGroupName, routeTableName, routeName, routeParameters, etag)
	if err != nil {
		return retry.NewError(false, mc.Observe(err))
	}

	future, err := az.client.CreateOrUpdateSender(req)
	if err != nil {
		return retry.GetError(future.Response(), mc.Observe(err))
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	return retry.GetError(future.Response(), mc.Observe(err))
}

// createOrUpdatePreparer prepares the CreateOrUpdate request.
func (az *azRoutesClient) createOrUpdatePreparer(ctx context.Context, resourceGroupName string, routeTableName string, routeName string, routeParameters network.Route, etag string) (*http.Request, error) {
	pathParameters := map[string]interface{}{
		"resourceGroupName": autorest.Encode("path", resourceGroupName),
		"routeName":         autorest.Encode("path", routeName),
		"routeTableName":    autorest.Encode("path", routeTableName),
		"subscriptionId":    autorest.Encode("path", az.client.SubscriptionID),
	}

	queryParameters := map[string]interface{}{
		"api-version": azureNetworkAPIVersion,
	}

	preparerDecorators := []autorest.PrepareDecorator{
		autorest.AsContentType("application/json; charset=utf-8"),
		autorest.AsPut(),
		autorest.WithBaseURL(az.client.BaseURI),
		autorest.WithPathParameters("/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Network/routeTables/{routeTableName}/routes/{routeName}", pathParameters),
		autorest.WithJSON(routeParameters),
		autorest.WithQueryParameters(queryParameters),
	}
	if etag != "" {
		preparerDecorators = append(preparerDecorators, autorest.WithHeader("If-Match", autorest.String(etag)))
	}
	preparer := autorest.CreatePreparer(preparerDecorators...)

	return preparer.Prepare((&http.Request{}).WithContext(ctx))
}

func (az *azRoutesClient) Delete(ctx context.Context, resourceGroupName string, routeTableName string, routeName string) *retry.Error {
	mc := newMetricContext("routes", "delete", resourceGroupName, az.client.SubscriptionID, "")
	/* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return createRateLimitErr(true, "RouteDelete")
	}

	klog.V(10).Infof("azRoutesClient.Delete(%q,%q,%q): start", resourceGroupName, routeTableName, routeName)
	defer func() {
		klog.V(10).Infof("azRoutesClient.Delete(%q,%q,%q): end", resourceGroupName, routeTableName, routeName)
	}()

	future, err := az.client.Delete(ctx, resourceGroupName, routeTableName, routeName)
	if err != nil {
		return retry.GetStatusNotFoundAndForbiddenIgnoredError(future.Response(), mc.Observe(err))
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	return retry.GetStatusNotFoundAndForbiddenIgnoredError(future.Response(), mc.Observe(err))
}

// azRouteTablesClient implements RouteTablesClient.
type azRouteTablesClient struct {
	client            network.RouteTablesClient
	rateLimiterReader flowcontrol.RateLimiter
	rateLimiterWriter flowcontrol.RateLimiter
}

func newAzRouteTablesClient(config *azclients.ClientConfig) *azRouteTablesClient {
	routeTablesClient := network.NewRouteTablesClient(config.SubscriptionID)
	routeTablesClient.BaseURI = config.ResourceManagerEndpoint
	routeTablesClient.Authorizer = autorest.NewBearerAuthorizer(config.ServicePrincipalToken)
	routeTablesClient.PollingDelay = 5 * time.Second
	if config.ShouldOmitCloudProviderBackoff {
		routeTablesClient.RetryAttempts = config.CloudProviderBackoffRetries
		routeTablesClient.RetryDuration = time.Duration(config.CloudProviderBackoffDuration) * time.Second
	}
	configureUserAgent(&routeTablesClient.Client)

	klog.V(2).Infof("Azure RouteTablesClient (read ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPS,
		config.RateLimitConfig.CloudProviderRateLimitBucket)
	klog.V(2).Infof("Azure RouteTablesClient (write ops) using rate limit config: QPS=%g, bucket=%d",
		config.RateLimitConfig.CloudProviderRateLimitQPSWrite,
		config.RateLimitConfig.CloudProviderRateLimitBucketWrite)
	rateLimiterReader, rateLimiterWriter := azclients.NewRateLimiter(config.RateLimitConfig)
	return &azRouteTablesClient{
		client:            routeTablesClient,
		rateLimiterReader: rateLimiterReader,
		rateLimiterWriter: rateLimiterWriter,
	}
}

func (az *azRouteTablesClient) CreateOrUpdate(ctx context.Context, resourceGroupName string, routeTableName string, parameters network.RouteTable, etag string) *retry.Error {
	mc := newMetricContext("route_tables", "create_or_update", resourceGroupName, az.client.SubscriptionID, "")
	/* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		return createRateLimitErr(true, "RouteTableCreateOrUpdate")
	}

	klog.V(10).Infof("azRouteTablesClient.CreateOrUpdate(%q,%q): start", resourceGroupName, routeTableName)
	defer func() {
		klog.V(10).Infof("azRouteTablesClient.CreateOrUpdate(%q,%q): end", resourceGroupName, routeTableName)
	}()

	req, err := az.createOrUpdatePreparer(ctx, resourceGroupName, routeTableName, parameters, etag)
	if err != nil {
		return retry.NewError(false, mc.Observe(err))
	}

	future, err := az.client.CreateOrUpdateSender(req)
	if err != nil {
		return retry.GetError(future.Response(), mc.Observe(err))
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	return retry.GetError(future.Response(), mc.Observe(err))
}

// createOrUpdatePreparer prepares the CreateOrUpdate request.
func (az *azRouteTablesClient) createOrUpdatePreparer(ctx context.Context, resourceGroupName string, routeTableName string, parameters network.RouteTable, etag string) (*http.Request, error) {
	pathParameters := map[string]interface{}{
		"resourceGroupName": autorest.Encode("path", resourceGroupName),
		"routeTableName":    autorest.Encode("path", routeTableName),
		"subscriptionId":    autorest.Encode("path", az.client.SubscriptionID),
	}

	queryParameters := map[string]interface{}{
		"api-version": azureNetworkAPIVersion,
	}
	preparerDecorators := []autorest.PrepareDecorator{
		autorest.AsContentType("application/json; charset=utf-8"),
		autorest.AsPut(),
		autorest.WithBaseURL(az.client.BaseURI),
		autorest.WithPathParameters("/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Network/routeTables/{routeTableName}", pathParameters),
		autorest.WithJSON(parameters),
		autorest.WithQueryParameters(queryParameters),
	}
	if etag != "" {
		preparerDecorators = append(preparerDecorators, autorest.WithHeader("If-Match", autorest.String(etag)))
	}
	preparer := autorest.CreatePreparer(preparerDecorators...)

	return preparer.Prepare((&http.Request{}).WithContext(ctx))
}

func (az *azRouteTablesClient) Get(ctx context.Context, resourceGroupName string, routeTableName string, expand string) (result network.RouteTable, rerr *retry.Error) {
	mc := newMetricContext("route_tables", "get", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		rerr = createRateLimitErr(false, "GetRouteTable")
		return
	}

	klog.V(10).Infof("azRouteTablesClient.Get(%q,%q): start", resourceGroupName, routeTableName)
	defer func() {
		klog.V(10).Infof("azRouteTablesClient.Get(%q,%q): end", resourceGroupName, routeTableName)
	}()

	var err error
	result, err = az.client.Get(ctx, resourceGroupName, routeTableName, expand)
	mc.Observe(err)
	return result, retry.GetError(result.Response.Response, err)
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
	mc := newMetricContext("storage_account", "create", resourceGroupName, az.client.SubscriptionID, "")
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
	mc := newMetricContext("storage_account", "delete", resourceGroupName, az.client.SubscriptionID, "")
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
	mc := newMetricContext("storage_account", "list_keys", resourceGroupName, az.client.SubscriptionID, "")
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
	mc := newMetricContext("storage_account", "list_by_resource_group", resourceGroupName, az.client.SubscriptionID, "")
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
	mc := newMetricContext("storage_account", "get_properties", resourceGroupName, az.client.SubscriptionID, "")
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
	mc := newMetricContext("disks", "create_or_update", resourceGroupName, az.client.SubscriptionID, "")
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
	mc := newMetricContext("disks", "delete", resourceGroupName, az.client.SubscriptionID, "")
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
	mc := newMetricContext("disks", "get", resourceGroupName, az.client.SubscriptionID, "")
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
	mc := newMetricContext("vmsizes", "list", "", az.client.SubscriptionID, "")
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

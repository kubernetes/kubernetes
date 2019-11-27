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
	"github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2019-04-01/storage"
	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/adal"

	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/klog"
)

const (
	// The version number is taken from "github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-06-01/network".
	azureNetworkAPIVersion              = "2019-06-01"
	virtualMachineScaleSetsDeallocating = "Deallocating"
)

// Helpers for rate limiting error/error channel creation
func createRateLimitErr(isWrite bool, opName string) error {
	opType := "read"
	if isWrite {
		opType = "write"
	}
	return fmt.Errorf("azure - cloud provider rate limited(%s) for operation:%s", opType, opName)
}

// VirtualMachinesClient defines needed functions for azure compute.VirtualMachinesClient
type VirtualMachinesClient interface {
	CreateOrUpdate(ctx context.Context, resourceGroupName string, VMName string, parameters compute.VirtualMachine, source string) (resp *http.Response, err error)
	Update(ctx context.Context, resourceGroupName string, VMName string, parameters compute.VirtualMachineUpdate, source string) (resp *http.Response, err error)
	Get(ctx context.Context, resourceGroupName string, VMName string, expand compute.InstanceViewTypes) (result compute.VirtualMachine, err error)
	List(ctx context.Context, resourceGroupName string) (result []compute.VirtualMachine, err error)
}

// InterfacesClient defines needed functions for azure network.InterfacesClient
type InterfacesClient interface {
	CreateOrUpdate(ctx context.Context, resourceGroupName string, networkInterfaceName string, parameters network.Interface) (resp *http.Response, err error)
	Get(ctx context.Context, resourceGroupName string, networkInterfaceName string, expand string) (result network.Interface, err error)
	GetVirtualMachineScaleSetNetworkInterface(ctx context.Context, resourceGroupName string, virtualMachineScaleSetName string, virtualmachineIndex string, networkInterfaceName string, expand string) (result network.Interface, err error)
}

// LoadBalancersClient defines needed functions for azure network.LoadBalancersClient
type LoadBalancersClient interface {
	CreateOrUpdate(ctx context.Context, resourceGroupName string, loadBalancerName string, parameters network.LoadBalancer, etag string) (resp *http.Response, err error)
	Delete(ctx context.Context, resourceGroupName string, loadBalancerName string) (resp *http.Response, err error)
	Get(ctx context.Context, resourceGroupName string, loadBalancerName string, expand string) (result network.LoadBalancer, err error)
	List(ctx context.Context, resourceGroupName string) (result []network.LoadBalancer, err error)
}

// PublicIPAddressesClient defines needed functions for azure network.PublicIPAddressesClient
type PublicIPAddressesClient interface {
	CreateOrUpdate(ctx context.Context, resourceGroupName string, publicIPAddressName string, parameters network.PublicIPAddress) (resp *http.Response, err error)
	Delete(ctx context.Context, resourceGroupName string, publicIPAddressName string) (resp *http.Response, err error)
	Get(ctx context.Context, resourceGroupName string, publicIPAddressName string, expand string) (result network.PublicIPAddress, err error)
	GetVirtualMachineScaleSetPublicIPAddress(ctx context.Context, resourceGroupName string, virtualMachineScaleSetName string, virtualmachineIndex string, networkInterfaceName string, IPConfigurationName string, publicIPAddressName string, expand string) (result network.PublicIPAddress, err error)
	List(ctx context.Context, resourceGroupName string) (result []network.PublicIPAddress, err error)
}

// SubnetsClient defines needed functions for azure network.SubnetsClient
type SubnetsClient interface {
	CreateOrUpdate(ctx context.Context, resourceGroupName string, virtualNetworkName string, subnetName string, subnetParameters network.Subnet) (resp *http.Response, err error)
	Delete(ctx context.Context, resourceGroupName string, virtualNetworkName string, subnetName string) (resp *http.Response, err error)
	Get(ctx context.Context, resourceGroupName string, virtualNetworkName string, subnetName string, expand string) (result network.Subnet, err error)
	List(ctx context.Context, resourceGroupName string, virtualNetworkName string) (result []network.Subnet, err error)
}

// SecurityGroupsClient defines needed functions for azure network.SecurityGroupsClient
type SecurityGroupsClient interface {
	CreateOrUpdate(ctx context.Context, resourceGroupName string, networkSecurityGroupName string, parameters network.SecurityGroup, etag string) (resp *http.Response, err error)
	Delete(ctx context.Context, resourceGroupName string, networkSecurityGroupName string) (resp *http.Response, err error)
	Get(ctx context.Context, resourceGroupName string, networkSecurityGroupName string, expand string) (result network.SecurityGroup, err error)
	List(ctx context.Context, resourceGroupName string) (result []network.SecurityGroup, err error)
}

// VirtualMachineScaleSetsClient defines needed functions for azure compute.VirtualMachineScaleSetsClient
type VirtualMachineScaleSetsClient interface {
	Get(ctx context.Context, resourceGroupName string, VMScaleSetName string) (result compute.VirtualMachineScaleSet, err error)
	List(ctx context.Context, resourceGroupName string) (result []compute.VirtualMachineScaleSet, err error)
	CreateOrUpdate(ctx context.Context, resourceGroupName string, VMScaleSetName string, parameters compute.VirtualMachineScaleSet) (resp *http.Response, err error)
}

// VirtualMachineScaleSetVMsClient defines needed functions for azure compute.VirtualMachineScaleSetVMsClient
type VirtualMachineScaleSetVMsClient interface {
	Get(ctx context.Context, resourceGroupName string, VMScaleSetName string, instanceID string, expand compute.InstanceViewTypes) (result compute.VirtualMachineScaleSetVM, err error)
	List(ctx context.Context, resourceGroupName string, virtualMachineScaleSetName string, filter string, selectParameter string, expand string) (result []compute.VirtualMachineScaleSetVM, err error)
	Update(ctx context.Context, resourceGroupName string, VMScaleSetName string, instanceID string, parameters compute.VirtualMachineScaleSetVM, source string) (resp *http.Response, err error)
}

// RoutesClient defines needed functions for azure network.RoutesClient
type RoutesClient interface {
	CreateOrUpdate(ctx context.Context, resourceGroupName string, routeTableName string, routeName string, routeParameters network.Route, etag string) (resp *http.Response, err error)
	Delete(ctx context.Context, resourceGroupName string, routeTableName string, routeName string) (resp *http.Response, err error)
}

// RouteTablesClient defines needed functions for azure network.RouteTablesClient
type RouteTablesClient interface {
	CreateOrUpdate(ctx context.Context, resourceGroupName string, routeTableName string, parameters network.RouteTable, etag string) (resp *http.Response, err error)
	Get(ctx context.Context, resourceGroupName string, routeTableName string, expand string) (result network.RouteTable, err error)
}

// StorageAccountClient defines needed functions for azure storage.AccountsClient
type StorageAccountClient interface {
	Create(ctx context.Context, resourceGroupName string, accountName string, parameters storage.AccountCreateParameters) (result *http.Response, err error)
	Delete(ctx context.Context, resourceGroupName string, accountName string) (result autorest.Response, err error)
	ListKeys(ctx context.Context, resourceGroupName string, accountName string) (result storage.AccountListKeysResult, err error)
	ListByResourceGroup(ctx context.Context, resourceGroupName string) (result storage.AccountListResult, err error)
	GetProperties(ctx context.Context, resourceGroupName string, accountName string) (result storage.Account, err error)
}

// DisksClient defines needed functions for azure compute.DisksClient
type DisksClient interface {
	CreateOrUpdate(ctx context.Context, resourceGroupName string, diskName string, diskParameter compute.Disk) (resp *http.Response, err error)
	Delete(ctx context.Context, resourceGroupName string, diskName string) (resp *http.Response, err error)
	Get(ctx context.Context, resourceGroupName string, diskName string) (result compute.Disk, err error)
}

// VirtualMachineSizesClient defines needed functions for azure compute.VirtualMachineSizesClient
type VirtualMachineSizesClient interface {
	List(ctx context.Context, location string) (result compute.VirtualMachineSizeListResult, err error)
}

// azClientConfig contains all essential information to create an Azure client.
type azClientConfig struct {
	subscriptionID          string
	resourceManagerEndpoint string
	servicePrincipalToken   *adal.ServicePrincipalToken
	// ARM Rate limiting for GET vs PUT/POST
	//Details: https://docs.microsoft.com/en-us/azure/azure-resource-manager/resource-manager-request-limits
	rateLimiterReader flowcontrol.RateLimiter
	rateLimiterWriter flowcontrol.RateLimiter

	CloudProviderBackoffRetries    int
	CloudProviderBackoffDuration   int
	ShouldOmitCloudProviderBackoff bool
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

func newAzVirtualMachinesClient(config *azClientConfig) *azVirtualMachinesClient {
	virtualMachinesClient := compute.NewVirtualMachinesClient(config.subscriptionID)
	virtualMachinesClient.BaseURI = config.resourceManagerEndpoint
	virtualMachinesClient.Authorizer = autorest.NewBearerAuthorizer(config.servicePrincipalToken)
	virtualMachinesClient.PollingDelay = 5 * time.Second
	if config.ShouldOmitCloudProviderBackoff {
		virtualMachinesClient.RetryAttempts = config.CloudProviderBackoffRetries
		virtualMachinesClient.RetryDuration = time.Duration(config.CloudProviderBackoffDuration) * time.Second
	}
	configureUserAgent(&virtualMachinesClient.Client)

	return &azVirtualMachinesClient{
		rateLimiterReader: config.rateLimiterReader,
		rateLimiterWriter: config.rateLimiterWriter,
		client:            virtualMachinesClient,
	}
}

func (az *azVirtualMachinesClient) CreateOrUpdate(ctx context.Context, resourceGroupName string, VMName string, parameters compute.VirtualMachine, source string) (resp *http.Response, err error) {
	// /* Write rate limiting */
	mc := newMetricContext("vm", "create_or_update", resourceGroupName, az.client.SubscriptionID, source)
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(true, "VMCreateOrUpdate")
		return
	}

	klog.V(10).Infof("azVirtualMachinesClient.CreateOrUpdate(%q, %q): start", resourceGroupName, VMName)
	defer func() {
		klog.V(10).Infof("azVirtualMachinesClient.CreateOrUpdate(%q, %q): end", resourceGroupName, VMName)
	}()

	future, err := az.client.CreateOrUpdate(ctx, resourceGroupName, VMName, parameters)
	if err != nil {
		return future.Response(), err
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	mc.Observe(err)
	return future.Response(), err
}

func (az *azVirtualMachinesClient) Update(ctx context.Context, resourceGroupName string, VMName string, parameters compute.VirtualMachineUpdate, source string) (resp *http.Response, err error) {
	mc := newMetricContext("vm", "update", resourceGroupName, az.client.SubscriptionID, source)
	// /* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(true, "VMUpdate")
		return
	}

	klog.V(10).Infof("azVirtualMachinesClient.Update(%q, %q): start", resourceGroupName, VMName)
	defer func() {
		klog.V(10).Infof("azVirtualMachinesClient.Update(%q, %q): end", resourceGroupName, VMName)
	}()

	future, err := az.client.Update(ctx, resourceGroupName, VMName, parameters)
	if err != nil {
		return future.Response(), err
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	mc.Observe(err)
	return future.Response(), err
}

func (az *azVirtualMachinesClient) Get(ctx context.Context, resourceGroupName string, VMName string, expand compute.InstanceViewTypes) (result compute.VirtualMachine, err error) {
	mc := newMetricContext("vm", "get", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(false, "VMGet")
		return
	}

	klog.V(10).Infof("azVirtualMachinesClient.Get(%q, %q): start", resourceGroupName, VMName)
	defer func() {
		klog.V(10).Infof("azVirtualMachinesClient.Get(%q, %q): end", resourceGroupName, VMName)
	}()

	result, err = az.client.Get(ctx, resourceGroupName, VMName, expand)
	mc.Observe(err)
	return
}

func (az *azVirtualMachinesClient) List(ctx context.Context, resourceGroupName string) (result []compute.VirtualMachine, err error) {
	mc := newMetricContext("vm", "list", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(false, "VMList")
		return
	}

	klog.V(10).Infof("azVirtualMachinesClient.List(%q): start", resourceGroupName)
	defer func() {
		klog.V(10).Infof("azVirtualMachinesClient.List(%q): end", resourceGroupName)
	}()

	iterator, err := az.client.ListComplete(ctx, resourceGroupName)
	mc.Observe(err)
	if err != nil {
		return nil, err
	}

	result = make([]compute.VirtualMachine, 0)
	for ; iterator.NotDone(); err = iterator.Next() {
		if err != nil {
			return nil, err
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

func newAzInterfacesClient(config *azClientConfig) *azInterfacesClient {
	interfacesClient := network.NewInterfacesClient(config.subscriptionID)
	interfacesClient.BaseURI = config.resourceManagerEndpoint
	interfacesClient.Authorizer = autorest.NewBearerAuthorizer(config.servicePrincipalToken)
	interfacesClient.PollingDelay = 5 * time.Second
	if config.ShouldOmitCloudProviderBackoff {
		interfacesClient.RetryAttempts = config.CloudProviderBackoffRetries
		interfacesClient.RetryDuration = time.Duration(config.CloudProviderBackoffDuration) * time.Second
	}
	configureUserAgent(&interfacesClient.Client)

	return &azInterfacesClient{
		rateLimiterReader: config.rateLimiterReader,
		rateLimiterWriter: config.rateLimiterWriter,
		client:            interfacesClient,
	}
}

func (az *azInterfacesClient) CreateOrUpdate(ctx context.Context, resourceGroupName string, networkInterfaceName string, parameters network.Interface) (resp *http.Response, err error) {
	mc := newMetricContext("interfaces", "create_or_update", resourceGroupName, az.client.SubscriptionID, "")
	/* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(true, "NiCreateOrUpdate")
		return
	}

	klog.V(10).Infof("azInterfacesClient.CreateOrUpdate(%q,%q): start", resourceGroupName, networkInterfaceName)
	defer func() {
		klog.V(10).Infof("azInterfacesClient.CreateOrUpdate(%q,%q): end", resourceGroupName, networkInterfaceName)
	}()

	future, err := az.client.CreateOrUpdate(ctx, resourceGroupName, networkInterfaceName, parameters)
	if err != nil {
		return future.Response(), mc.Observe(err)
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	return future.Response(), mc.Observe(err)
}

func (az *azInterfacesClient) Get(ctx context.Context, resourceGroupName string, networkInterfaceName string, expand string) (result network.Interface, err error) {
	mc := newMetricContext("interfaces", "get", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(false, "NicGet")
		return
	}

	klog.V(10).Infof("azInterfacesClient.Get(%q,%q): start", resourceGroupName, networkInterfaceName)
	defer func() {
		klog.V(10).Infof("azInterfacesClient.Get(%q,%q): end", resourceGroupName, networkInterfaceName)
	}()

	result, err = az.client.Get(ctx, resourceGroupName, networkInterfaceName, expand)
	mc.Observe(err)
	return
}

func (az *azInterfacesClient) GetVirtualMachineScaleSetNetworkInterface(ctx context.Context, resourceGroupName string, virtualMachineScaleSetName string, virtualmachineIndex string, networkInterfaceName string, expand string) (result network.Interface, err error) {
	mc := newMetricContext("interfaces", "get_vmss_ni", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(false, "NicGetVirtualMachineScaleSetNetworkInterface")
		return
	}

	klog.V(10).Infof("azInterfacesClient.GetVirtualMachineScaleSetNetworkInterface(%q,%q,%q,%q): start", resourceGroupName, virtualMachineScaleSetName, virtualmachineIndex, networkInterfaceName)
	defer func() {
		klog.V(10).Infof("azInterfacesClient.GetVirtualMachineScaleSetNetworkInterface(%q,%q,%q,%q): end", resourceGroupName, virtualMachineScaleSetName, virtualmachineIndex, networkInterfaceName)
	}()

	result, err = az.client.GetVirtualMachineScaleSetNetworkInterface(ctx, resourceGroupName, virtualMachineScaleSetName, virtualmachineIndex, networkInterfaceName, expand)
	mc.Observe(err)
	return
}

// azLoadBalancersClient implements LoadBalancersClient.
type azLoadBalancersClient struct {
	client            network.LoadBalancersClient
	rateLimiterReader flowcontrol.RateLimiter
	rateLimiterWriter flowcontrol.RateLimiter
}

func newAzLoadBalancersClient(config *azClientConfig) *azLoadBalancersClient {
	loadBalancerClient := network.NewLoadBalancersClient(config.subscriptionID)
	loadBalancerClient.BaseURI = config.resourceManagerEndpoint
	loadBalancerClient.Authorizer = autorest.NewBearerAuthorizer(config.servicePrincipalToken)
	loadBalancerClient.PollingDelay = 5 * time.Second
	if config.ShouldOmitCloudProviderBackoff {
		loadBalancerClient.RetryAttempts = config.CloudProviderBackoffRetries
		loadBalancerClient.RetryDuration = time.Duration(config.CloudProviderBackoffDuration) * time.Second
	}
	configureUserAgent(&loadBalancerClient.Client)

	return &azLoadBalancersClient{
		rateLimiterReader: config.rateLimiterReader,
		rateLimiterWriter: config.rateLimiterWriter,
		client:            loadBalancerClient,
	}
}

func (az *azLoadBalancersClient) CreateOrUpdate(ctx context.Context, resourceGroupName string, loadBalancerName string, parameters network.LoadBalancer, etag string) (resp *http.Response, err error) {
	mc := newMetricContext("load_balancers", "create_or_update", resourceGroupName, az.client.SubscriptionID, "")
	/* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(true, "LBCreateOrUpdate")
		return nil, err
	}

	klog.V(10).Infof("azLoadBalancersClient.CreateOrUpdate(%q,%q): start", resourceGroupName, loadBalancerName)
	defer func() {
		klog.V(10).Infof("azLoadBalancersClient.CreateOrUpdate(%q,%q): end", resourceGroupName, loadBalancerName)
	}()

	req, err := az.createOrUpdatePreparer(ctx, resourceGroupName, loadBalancerName, parameters, etag)
	if err != nil {
		return nil, mc.Observe(err)
	}

	future, err := az.client.CreateOrUpdateSender(req)
	if err != nil {
		return future.Response(), mc.Observe(err)
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	return future.Response(), mc.Observe(err)
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

func (az *azLoadBalancersClient) Delete(ctx context.Context, resourceGroupName string, loadBalancerName string) (resp *http.Response, err error) {
	mc := newMetricContext("load_balancers", "delete", resourceGroupName, az.client.SubscriptionID, "")
	/* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(true, "LBDelete")
		return nil, err
	}

	klog.V(10).Infof("azLoadBalancersClient.Delete(%q,%q): start", resourceGroupName, loadBalancerName)
	defer func() {
		klog.V(10).Infof("azLoadBalancersClient.Delete(%q,%q): end", resourceGroupName, loadBalancerName)
	}()

	future, err := az.client.Delete(ctx, resourceGroupName, loadBalancerName)
	if err != nil {
		return future.Response(), mc.Observe(err)
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	return future.Response(), mc.Observe(err)
}

func (az *azLoadBalancersClient) Get(ctx context.Context, resourceGroupName string, loadBalancerName string, expand string) (result network.LoadBalancer, err error) {
	mc := newMetricContext("load_balancers", "get", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(false, "LBGet")
		return
	}

	klog.V(10).Infof("azLoadBalancersClient.Get(%q,%q): start", resourceGroupName, loadBalancerName)
	defer func() {
		klog.V(10).Infof("azLoadBalancersClient.Get(%q,%q): end", resourceGroupName, loadBalancerName)
	}()

	result, err = az.client.Get(ctx, resourceGroupName, loadBalancerName, expand)
	mc.Observe(err)
	return
}

func (az *azLoadBalancersClient) List(ctx context.Context, resourceGroupName string) ([]network.LoadBalancer, error) {
	mc := newMetricContext("load_balancers", "list", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		err := createRateLimitErr(false, "LBList")
		return nil, err
	}

	klog.V(10).Infof("azLoadBalancersClient.List(%q): start", resourceGroupName)
	defer func() {
		klog.V(10).Infof("azLoadBalancersClient.List(%q): end", resourceGroupName)
	}()

	iterator, err := az.client.ListComplete(ctx, resourceGroupName)
	mc.Observe(err)
	if err != nil {
		return nil, err
	}

	result := make([]network.LoadBalancer, 0)
	for ; iterator.NotDone(); err = iterator.Next() {
		if err != nil {
			return nil, err
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

func newAzPublicIPAddressesClient(config *azClientConfig) *azPublicIPAddressesClient {
	publicIPAddressClient := network.NewPublicIPAddressesClient(config.subscriptionID)
	publicIPAddressClient.BaseURI = config.resourceManagerEndpoint
	publicIPAddressClient.Authorizer = autorest.NewBearerAuthorizer(config.servicePrincipalToken)
	publicIPAddressClient.PollingDelay = 5 * time.Second
	if config.ShouldOmitCloudProviderBackoff {
		publicIPAddressClient.RetryAttempts = config.CloudProviderBackoffRetries
		publicIPAddressClient.RetryDuration = time.Duration(config.CloudProviderBackoffDuration) * time.Second
	}
	configureUserAgent(&publicIPAddressClient.Client)

	return &azPublicIPAddressesClient{
		rateLimiterReader: config.rateLimiterReader,
		rateLimiterWriter: config.rateLimiterWriter,
		client:            publicIPAddressClient,
	}
}

func (az *azPublicIPAddressesClient) CreateOrUpdate(ctx context.Context, resourceGroupName string, publicIPAddressName string, parameters network.PublicIPAddress) (resp *http.Response, err error) {
	mc := newMetricContext("public_ip_addresses", "create_or_update", resourceGroupName, az.client.SubscriptionID, "")
	/* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(true, "PublicIPCreateOrUpdate")
		return nil, err
	}

	klog.V(10).Infof("azPublicIPAddressesClient.CreateOrUpdate(%q,%q): start", resourceGroupName, publicIPAddressName)
	defer func() {
		klog.V(10).Infof("azPublicIPAddressesClient.CreateOrUpdate(%q,%q): end", resourceGroupName, publicIPAddressName)
	}()

	future, err := az.client.CreateOrUpdate(ctx, resourceGroupName, publicIPAddressName, parameters)
	if err != nil {
		return future.Response(), mc.Observe(err)
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	return future.Response(), mc.Observe(err)
}

func (az *azPublicIPAddressesClient) Delete(ctx context.Context, resourceGroupName string, publicIPAddressName string) (resp *http.Response, err error) {
	mc := newMetricContext("public_ip_addresses", "delete", resourceGroupName, az.client.SubscriptionID, "")
	/* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(true, "PublicIPDelete")
		return nil, err
	}

	klog.V(10).Infof("azPublicIPAddressesClient.Delete(%q,%q): start", resourceGroupName, publicIPAddressName)
	defer func() {
		klog.V(10).Infof("azPublicIPAddressesClient.Delete(%q,%q): end", resourceGroupName, publicIPAddressName)
	}()

	future, err := az.client.Delete(ctx, resourceGroupName, publicIPAddressName)
	if err != nil {
		return future.Response(), mc.Observe(err)
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	return future.Response(), mc.Observe(err)
}

func (az *azPublicIPAddressesClient) Get(ctx context.Context, resourceGroupName string, publicIPAddressName string, expand string) (result network.PublicIPAddress, err error) {
	mc := newMetricContext("public_ip_addresses", "get", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(false, "PublicIPGet")
		return
	}

	klog.V(10).Infof("azPublicIPAddressesClient.Get(%q,%q): start", resourceGroupName, publicIPAddressName)
	defer func() {
		klog.V(10).Infof("azPublicIPAddressesClient.Get(%q,%q): end", resourceGroupName, publicIPAddressName)
	}()

	result, err = az.client.Get(ctx, resourceGroupName, publicIPAddressName, expand)
	mc.Observe(err)
	return
}

func (az *azPublicIPAddressesClient) GetVirtualMachineScaleSetPublicIPAddress(ctx context.Context, resourceGroupName string, virtualMachineScaleSetName string, virtualmachineIndex string, networkInterfaceName string, IPConfigurationName string, publicIPAddressName string, expand string) (result network.PublicIPAddress, err error) {
	mc := newMetricContext("vmss_public_ip_addresses", "get", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(false, "VMSSPublicIPGet")
		return
	}

	klog.V(10).Infof("azPublicIPAddressesClient.GetVirtualMachineScaleSetPublicIPAddress(%q,%q): start", resourceGroupName, publicIPAddressName)
	defer func() {
		klog.V(10).Infof("azPublicIPAddressesClient.GetVirtualMachineScaleSetPublicIPAddress(%q,%q): end", resourceGroupName, publicIPAddressName)
	}()

	result, err = az.client.GetVirtualMachineScaleSetPublicIPAddress(ctx, resourceGroupName, virtualMachineScaleSetName, virtualmachineIndex, networkInterfaceName, IPConfigurationName, publicIPAddressName, expand)
	mc.Observe(err)
	return
}

func (az *azPublicIPAddressesClient) List(ctx context.Context, resourceGroupName string) ([]network.PublicIPAddress, error) {
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
		return nil, err
	}

	result := make([]network.PublicIPAddress, 0)
	for ; iterator.NotDone(); err = iterator.Next() {
		if err != nil {
			return nil, err
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

func newAzSubnetsClient(config *azClientConfig) *azSubnetsClient {
	subnetsClient := network.NewSubnetsClient(config.subscriptionID)
	subnetsClient.BaseURI = config.resourceManagerEndpoint
	subnetsClient.Authorizer = autorest.NewBearerAuthorizer(config.servicePrincipalToken)
	subnetsClient.PollingDelay = 5 * time.Second
	if config.ShouldOmitCloudProviderBackoff {
		subnetsClient.RetryAttempts = config.CloudProviderBackoffRetries
		subnetsClient.RetryDuration = time.Duration(config.CloudProviderBackoffDuration) * time.Second
	}
	configureUserAgent(&subnetsClient.Client)

	return &azSubnetsClient{
		client:            subnetsClient,
		rateLimiterReader: config.rateLimiterReader,
		rateLimiterWriter: config.rateLimiterWriter,
	}
}

func (az *azSubnetsClient) CreateOrUpdate(ctx context.Context, resourceGroupName string, virtualNetworkName string, subnetName string, subnetParameters network.Subnet) (resp *http.Response, err error) {
	mc := newMetricContext("subnets", "create_or_update", resourceGroupName, az.client.SubscriptionID, "")
	/* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(true, "SubnetCreateOrUpdate")
		return
	}

	klog.V(10).Infof("azSubnetsClient.CreateOrUpdate(%q,%q,%q): start", resourceGroupName, virtualNetworkName, subnetName)
	defer func() {
		klog.V(10).Infof("azSubnetsClient.CreateOrUpdate(%q,%q,%q): end", resourceGroupName, virtualNetworkName, subnetName)
	}()

	future, err := az.client.CreateOrUpdate(ctx, resourceGroupName, virtualNetworkName, subnetName, subnetParameters)
	if err != nil {
		return future.Response(), mc.Observe(err)
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	return future.Response(), mc.Observe(err)
}

func (az *azSubnetsClient) Delete(ctx context.Context, resourceGroupName string, virtualNetworkName string, subnetName string) (resp *http.Response, err error) {
	mc := newMetricContext("subnets", "delete", resourceGroupName, az.client.SubscriptionID, "")
	/* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(true, "SubnetDelete")
		return
	}

	klog.V(10).Infof("azSubnetsClient.Delete(%q,%q,%q): start", resourceGroupName, virtualNetworkName, subnetName)
	defer func() {
		klog.V(10).Infof("azSubnetsClient.Delete(%q,%q,%q): end", resourceGroupName, virtualNetworkName, subnetName)
	}()

	future, err := az.client.Delete(ctx, resourceGroupName, virtualNetworkName, subnetName)
	if err != nil {
		return future.Response(), mc.Observe(err)
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	return future.Response(), mc.Observe(err)
}

func (az *azSubnetsClient) Get(ctx context.Context, resourceGroupName string, virtualNetworkName string, subnetName string, expand string) (result network.Subnet, err error) {
	mc := newMetricContext("subnets", "get", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(false, "SubnetGet")
		return
	}

	klog.V(10).Infof("azSubnetsClient.Get(%q,%q,%q): start", resourceGroupName, virtualNetworkName, subnetName)
	defer func() {
		klog.V(10).Infof("azSubnetsClient.Get(%q,%q,%q): end", resourceGroupName, virtualNetworkName, subnetName)
	}()

	result, err = az.client.Get(ctx, resourceGroupName, virtualNetworkName, subnetName, expand)
	mc.Observe(err)
	return
}

func (az *azSubnetsClient) List(ctx context.Context, resourceGroupName string, virtualNetworkName string) ([]network.Subnet, error) {
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
		return nil, err
	}

	result := make([]network.Subnet, 0)
	for ; iterator.NotDone(); err = iterator.Next() {
		if err != nil {
			return nil, err
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

func newAzSecurityGroupsClient(config *azClientConfig) *azSecurityGroupsClient {
	securityGroupsClient := network.NewSecurityGroupsClient(config.subscriptionID)
	securityGroupsClient.BaseURI = config.resourceManagerEndpoint
	securityGroupsClient.Authorizer = autorest.NewBearerAuthorizer(config.servicePrincipalToken)
	securityGroupsClient.PollingDelay = 5 * time.Second
	if config.ShouldOmitCloudProviderBackoff {
		securityGroupsClient.RetryAttempts = config.CloudProviderBackoffRetries
		securityGroupsClient.RetryDuration = time.Duration(config.CloudProviderBackoffDuration) * time.Second
	}
	configureUserAgent(&securityGroupsClient.Client)

	return &azSecurityGroupsClient{
		client:            securityGroupsClient,
		rateLimiterReader: config.rateLimiterReader,
		rateLimiterWriter: config.rateLimiterWriter,
	}
}

func (az *azSecurityGroupsClient) CreateOrUpdate(ctx context.Context, resourceGroupName string, networkSecurityGroupName string, parameters network.SecurityGroup, etag string) (resp *http.Response, err error) {
	mc := newMetricContext("security_groups", "create_or_update", resourceGroupName, az.client.SubscriptionID, "")
	/* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(true, "NSGCreateOrUpdate")
		return
	}

	klog.V(10).Infof("azSecurityGroupsClient.CreateOrUpdate(%q,%q): start", resourceGroupName, networkSecurityGroupName)
	defer func() {
		klog.V(10).Infof("azSecurityGroupsClient.CreateOrUpdate(%q,%q): end", resourceGroupName, networkSecurityGroupName)
	}()

	req, err := az.createOrUpdatePreparer(ctx, resourceGroupName, networkSecurityGroupName, parameters, etag)
	if err != nil {
		return nil, mc.Observe(err)
	}

	future, err := az.client.CreateOrUpdateSender(req)
	if err != nil {
		return future.Response(), mc.Observe(err)
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	return future.Response(), mc.Observe(err)
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

func (az *azSecurityGroupsClient) Delete(ctx context.Context, resourceGroupName string, networkSecurityGroupName string) (resp *http.Response, err error) {
	mc := newMetricContext("security_groups", "delete", resourceGroupName, az.client.SubscriptionID, "")
	/* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(true, "NSGDelete")
		return
	}

	klog.V(10).Infof("azSecurityGroupsClient.Delete(%q,%q): start", resourceGroupName, networkSecurityGroupName)
	defer func() {
		klog.V(10).Infof("azSecurityGroupsClient.Delete(%q,%q): end", resourceGroupName, networkSecurityGroupName)
	}()

	future, err := az.client.Delete(ctx, resourceGroupName, networkSecurityGroupName)
	if err != nil {
		return future.Response(), mc.Observe(err)
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	return future.Response(), mc.Observe(err)
}

func (az *azSecurityGroupsClient) Get(ctx context.Context, resourceGroupName string, networkSecurityGroupName string, expand string) (result network.SecurityGroup, err error) {
	mc := newMetricContext("security_groups", "get", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(false, "NSGGet")
		return
	}

	klog.V(10).Infof("azSecurityGroupsClient.Get(%q,%q): start", resourceGroupName, networkSecurityGroupName)
	defer func() {
		klog.V(10).Infof("azSecurityGroupsClient.Get(%q,%q): end", resourceGroupName, networkSecurityGroupName)
	}()

	result, err = az.client.Get(ctx, resourceGroupName, networkSecurityGroupName, expand)
	mc.Observe(err)
	return
}

func (az *azSecurityGroupsClient) List(ctx context.Context, resourceGroupName string) ([]network.SecurityGroup, error) {
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
		return nil, err
	}

	result := make([]network.SecurityGroup, 0)
	for ; iterator.NotDone(); err = iterator.Next() {
		if err != nil {
			return nil, err
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

func newAzVirtualMachineScaleSetsClient(config *azClientConfig) *azVirtualMachineScaleSetsClient {
	virtualMachineScaleSetsClient := compute.NewVirtualMachineScaleSetsClient(config.subscriptionID)
	virtualMachineScaleSetsClient.BaseURI = config.resourceManagerEndpoint
	virtualMachineScaleSetsClient.Authorizer = autorest.NewBearerAuthorizer(config.servicePrincipalToken)
	virtualMachineScaleSetsClient.PollingDelay = 5 * time.Second
	if config.ShouldOmitCloudProviderBackoff {
		virtualMachineScaleSetsClient.RetryAttempts = config.CloudProviderBackoffRetries
		virtualMachineScaleSetsClient.RetryDuration = time.Duration(config.CloudProviderBackoffDuration) * time.Second
	}
	configureUserAgent(&virtualMachineScaleSetsClient.Client)

	return &azVirtualMachineScaleSetsClient{
		client:            virtualMachineScaleSetsClient,
		rateLimiterReader: config.rateLimiterReader,
		rateLimiterWriter: config.rateLimiterWriter,
	}
}

func (az *azVirtualMachineScaleSetsClient) Get(ctx context.Context, resourceGroupName string, VMScaleSetName string) (result compute.VirtualMachineScaleSet, err error) {
	mc := newMetricContext("vmss", "get", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(false, "VMSSGet")
		return
	}

	klog.V(10).Infof("azVirtualMachineScaleSetsClient.Get(%q,%q): start", resourceGroupName, VMScaleSetName)
	defer func() {
		klog.V(10).Infof("azVirtualMachineScaleSetsClient.Get(%q,%q): end", resourceGroupName, VMScaleSetName)
	}()

	result, err = az.client.Get(ctx, resourceGroupName, VMScaleSetName)
	mc.Observe(err)
	return
}

func (az *azVirtualMachineScaleSetsClient) List(ctx context.Context, resourceGroupName string) (result []compute.VirtualMachineScaleSet, err error) {
	mc := newMetricContext("vmss", "list", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(false, "VMSSList")
		return
	}

	klog.V(10).Infof("azVirtualMachineScaleSetsClient.List(%q): start", resourceGroupName)
	defer func() {
		klog.V(10).Infof("azVirtualMachineScaleSetsClient.List(%q): end", resourceGroupName)
	}()

	iterator, err := az.client.ListComplete(ctx, resourceGroupName)
	mc.Observe(err)
	if err != nil {
		return nil, err
	}

	result = make([]compute.VirtualMachineScaleSet, 0)
	for ; iterator.NotDone(); err = iterator.Next() {
		if err != nil {
			return nil, err
		}

		result = append(result, iterator.Value())
	}

	return result, nil
}

func (az *azVirtualMachineScaleSetsClient) CreateOrUpdate(ctx context.Context, resourceGroupName string, vmScaleSetName string, parameters compute.VirtualMachineScaleSet) (resp *http.Response, err error) {
	mc := newMetricContext("vmss", "create_or_update", resourceGroupName, az.client.SubscriptionID, "")
	/* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(true, "NiCreateOrUpdate")
		return
	}

	klog.V(10).Infof("azVirtualMachineScaleSetsClient.CreateOrUpdate(%q,%q): start", resourceGroupName, vmScaleSetName)
	defer func() {
		klog.V(10).Infof("azVirtualMachineScaleSetsClient.CreateOrUpdate(%q,%q): end", resourceGroupName, vmScaleSetName)
	}()

	future, err := az.client.CreateOrUpdate(ctx, resourceGroupName, vmScaleSetName, parameters)
	if err != nil {
		return future.Response(), mc.Observe(err)
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	return future.Response(), mc.Observe(err)
}

// azVirtualMachineScaleSetVMsClient implements VirtualMachineScaleSetVMsClient.
type azVirtualMachineScaleSetVMsClient struct {
	client            compute.VirtualMachineScaleSetVMsClient
	rateLimiterReader flowcontrol.RateLimiter
	rateLimiterWriter flowcontrol.RateLimiter
}

func newAzVirtualMachineScaleSetVMsClient(config *azClientConfig) *azVirtualMachineScaleSetVMsClient {
	virtualMachineScaleSetVMsClient := compute.NewVirtualMachineScaleSetVMsClient(config.subscriptionID)
	virtualMachineScaleSetVMsClient.BaseURI = config.resourceManagerEndpoint
	virtualMachineScaleSetVMsClient.Authorizer = autorest.NewBearerAuthorizer(config.servicePrincipalToken)
	virtualMachineScaleSetVMsClient.PollingDelay = 5 * time.Second
	if config.ShouldOmitCloudProviderBackoff {
		virtualMachineScaleSetVMsClient.RetryAttempts = config.CloudProviderBackoffRetries
		virtualMachineScaleSetVMsClient.RetryDuration = time.Duration(config.CloudProviderBackoffDuration) * time.Second
	}
	configureUserAgent(&virtualMachineScaleSetVMsClient.Client)

	return &azVirtualMachineScaleSetVMsClient{
		client:            virtualMachineScaleSetVMsClient,
		rateLimiterReader: config.rateLimiterReader,
		rateLimiterWriter: config.rateLimiterWriter,
	}
}

func (az *azVirtualMachineScaleSetVMsClient) Get(ctx context.Context, resourceGroupName string, VMScaleSetName string, instanceID string, expand compute.InstanceViewTypes) (result compute.VirtualMachineScaleSetVM, err error) {
	mc := newMetricContext("vmssvm", "get", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(false, "VMSSGet")
		return
	}

	klog.V(10).Infof("azVirtualMachineScaleSetVMsClient.Get(%q,%q,%q): start", resourceGroupName, VMScaleSetName, instanceID)
	defer func() {
		klog.V(10).Infof("azVirtualMachineScaleSetVMsClient.Get(%q,%q,%q): end", resourceGroupName, VMScaleSetName, instanceID)
	}()

	result, err = az.client.Get(ctx, resourceGroupName, VMScaleSetName, instanceID, expand)
	mc.Observe(err)
	return
}

func (az *azVirtualMachineScaleSetVMsClient) List(ctx context.Context, resourceGroupName string, virtualMachineScaleSetName string, filter string, selectParameter string, expand string) (result []compute.VirtualMachineScaleSetVM, err error) {
	mc := newMetricContext("vmssvm", "list", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(false, "VMSSList")
		return
	}

	klog.V(10).Infof("azVirtualMachineScaleSetVMsClient.List(%q,%q,%q): start", resourceGroupName, virtualMachineScaleSetName, filter)
	defer func() {
		klog.V(10).Infof("azVirtualMachineScaleSetVMsClient.List(%q,%q,%q): end", resourceGroupName, virtualMachineScaleSetName, filter)
	}()

	iterator, err := az.client.ListComplete(ctx, resourceGroupName, virtualMachineScaleSetName, filter, selectParameter, expand)
	mc.Observe(err)
	if err != nil {
		return nil, err
	}

	result = make([]compute.VirtualMachineScaleSetVM, 0)
	for ; iterator.NotDone(); err = iterator.Next() {
		if err != nil {
			return nil, err
		}

		result = append(result, iterator.Value())
	}

	return result, nil
}

func (az *azVirtualMachineScaleSetVMsClient) Update(ctx context.Context, resourceGroupName string, VMScaleSetName string, instanceID string, parameters compute.VirtualMachineScaleSetVM, source string) (resp *http.Response, err error) {
	mc := newMetricContext("vmssvm", "create_or_update", resourceGroupName, az.client.SubscriptionID, source)
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(true, "VMSSVMUpdate")
		return
	}

	klog.V(10).Infof("azVirtualMachineScaleSetVMsClient.Update(%q,%q,%q): start", resourceGroupName, VMScaleSetName, instanceID)
	defer func() {
		klog.V(10).Infof("azVirtualMachineScaleSetVMsClient.Update(%q,%q,%q): end", resourceGroupName, VMScaleSetName, instanceID)
	}()

	future, err := az.client.Update(ctx, resourceGroupName, VMScaleSetName, instanceID, parameters)
	if err != nil {
		return future.Response(), mc.Observe(err)
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	return future.Response(), mc.Observe(err)
}

// azRoutesClient implements RoutesClient.
type azRoutesClient struct {
	client            network.RoutesClient
	rateLimiterReader flowcontrol.RateLimiter
	rateLimiterWriter flowcontrol.RateLimiter
}

func newAzRoutesClient(config *azClientConfig) *azRoutesClient {
	routesClient := network.NewRoutesClient(config.subscriptionID)
	routesClient.BaseURI = config.resourceManagerEndpoint
	routesClient.Authorizer = autorest.NewBearerAuthorizer(config.servicePrincipalToken)
	routesClient.PollingDelay = 5 * time.Second
	if config.ShouldOmitCloudProviderBackoff {
		routesClient.RetryAttempts = config.CloudProviderBackoffRetries
		routesClient.RetryDuration = time.Duration(config.CloudProviderBackoffDuration) * time.Second
	}
	configureUserAgent(&routesClient.Client)

	return &azRoutesClient{
		client:            routesClient,
		rateLimiterReader: config.rateLimiterReader,
		rateLimiterWriter: config.rateLimiterWriter,
	}
}

func (az *azRoutesClient) CreateOrUpdate(ctx context.Context, resourceGroupName string, routeTableName string, routeName string, routeParameters network.Route, etag string) (resp *http.Response, err error) {
	mc := newMetricContext("routes", "create_or_update", resourceGroupName, az.client.SubscriptionID, "")
	/* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(true, "RouteCreateOrUpdate")
		return
	}

	klog.V(10).Infof("azRoutesClient.CreateOrUpdate(%q,%q,%q): start", resourceGroupName, routeTableName, routeName)
	defer func() {
		klog.V(10).Infof("azRoutesClient.CreateOrUpdate(%q,%q,%q): end", resourceGroupName, routeTableName, routeName)
	}()

	req, err := az.createOrUpdatePreparer(ctx, resourceGroupName, routeTableName, routeName, routeParameters, etag)
	if err != nil {
		mc.Observe(err)
		return nil, err
	}

	future, err := az.client.CreateOrUpdateSender(req)
	if err != nil {
		return future.Response(), mc.Observe(err)
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	return future.Response(), mc.Observe(err)
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

func (az *azRoutesClient) Delete(ctx context.Context, resourceGroupName string, routeTableName string, routeName string) (resp *http.Response, err error) {
	mc := newMetricContext("routes", "delete", resourceGroupName, az.client.SubscriptionID, "")
	/* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(true, "RouteDelete")
		return
	}

	klog.V(10).Infof("azRoutesClient.Delete(%q,%q,%q): start", resourceGroupName, routeTableName, routeName)
	defer func() {
		klog.V(10).Infof("azRoutesClient.Delete(%q,%q,%q): end", resourceGroupName, routeTableName, routeName)
	}()

	future, err := az.client.Delete(ctx, resourceGroupName, routeTableName, routeName)
	if err != nil {
		return future.Response(), mc.Observe(err)
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	return future.Response(), mc.Observe(err)
}

// azRouteTablesClient implements RouteTablesClient.
type azRouteTablesClient struct {
	client            network.RouteTablesClient
	rateLimiterReader flowcontrol.RateLimiter
	rateLimiterWriter flowcontrol.RateLimiter
}

func newAzRouteTablesClient(config *azClientConfig) *azRouteTablesClient {
	routeTablesClient := network.NewRouteTablesClient(config.subscriptionID)
	routeTablesClient.BaseURI = config.resourceManagerEndpoint
	routeTablesClient.Authorizer = autorest.NewBearerAuthorizer(config.servicePrincipalToken)
	routeTablesClient.PollingDelay = 5 * time.Second
	if config.ShouldOmitCloudProviderBackoff {
		routeTablesClient.RetryAttempts = config.CloudProviderBackoffRetries
		routeTablesClient.RetryDuration = time.Duration(config.CloudProviderBackoffDuration) * time.Second
	}
	configureUserAgent(&routeTablesClient.Client)

	return &azRouteTablesClient{
		client:            routeTablesClient,
		rateLimiterReader: config.rateLimiterReader,
		rateLimiterWriter: config.rateLimiterWriter,
	}
}

func (az *azRouteTablesClient) CreateOrUpdate(ctx context.Context, resourceGroupName string, routeTableName string, parameters network.RouteTable, etag string) (resp *http.Response, err error) {
	mc := newMetricContext("route_tables", "create_or_update", resourceGroupName, az.client.SubscriptionID, "")
	/* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(true, "RouteTableCreateOrUpdate")
		return
	}

	klog.V(10).Infof("azRouteTablesClient.CreateOrUpdate(%q,%q): start", resourceGroupName, routeTableName)
	defer func() {
		klog.V(10).Infof("azRouteTablesClient.CreateOrUpdate(%q,%q): end", resourceGroupName, routeTableName)
	}()

	req, err := az.createOrUpdatePreparer(ctx, resourceGroupName, routeTableName, parameters, etag)
	if err != nil {
		return nil, mc.Observe(err)
	}

	future, err := az.client.CreateOrUpdateSender(req)
	if err != nil {
		return future.Response(), mc.Observe(err)
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	return future.Response(), mc.Observe(err)
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

func (az *azRouteTablesClient) Get(ctx context.Context, resourceGroupName string, routeTableName string, expand string) (result network.RouteTable, err error) {
	mc := newMetricContext("route_tables", "get", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(false, "GetRouteTable")
		return
	}

	klog.V(10).Infof("azRouteTablesClient.Get(%q,%q): start", resourceGroupName, routeTableName)
	defer func() {
		klog.V(10).Infof("azRouteTablesClient.Get(%q,%q): end", resourceGroupName, routeTableName)
	}()

	result, err = az.client.Get(ctx, resourceGroupName, routeTableName, expand)
	mc.Observe(err)
	return
}

// azStorageAccountClient implements StorageAccountClient.
type azStorageAccountClient struct {
	client            storage.AccountsClient
	rateLimiterReader flowcontrol.RateLimiter
	rateLimiterWriter flowcontrol.RateLimiter
}

func newAzStorageAccountClient(config *azClientConfig) *azStorageAccountClient {
	storageAccountClient := storage.NewAccountsClientWithBaseURI(config.resourceManagerEndpoint, config.subscriptionID)
	storageAccountClient.Authorizer = autorest.NewBearerAuthorizer(config.servicePrincipalToken)
	storageAccountClient.PollingDelay = 5 * time.Second
	if config.ShouldOmitCloudProviderBackoff {
		storageAccountClient.RetryAttempts = config.CloudProviderBackoffRetries
		storageAccountClient.RetryDuration = time.Duration(config.CloudProviderBackoffDuration) * time.Second
	}
	configureUserAgent(&storageAccountClient.Client)

	return &azStorageAccountClient{
		client:            storageAccountClient,
		rateLimiterReader: config.rateLimiterReader,
		rateLimiterWriter: config.rateLimiterWriter,
	}
}

func (az *azStorageAccountClient) Create(ctx context.Context, resourceGroupName string, accountName string, parameters storage.AccountCreateParameters) (result *http.Response, err error) {
	mc := newMetricContext("storage_account", "create", resourceGroupName, az.client.SubscriptionID, "")
	/* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(true, "StorageAccountCreate")
		return
	}

	klog.V(10).Infof("azStorageAccountClient.Create(%q,%q): start", resourceGroupName, accountName)
	defer func() {
		klog.V(10).Infof("azStorageAccountClient.Create(%q,%q): end", resourceGroupName, accountName)
	}()

	future, err := az.client.Create(ctx, resourceGroupName, accountName, parameters)
	if err != nil {
		return future.Response(), err
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	mc.Observe(err)
	return future.Response(), err
}

func (az *azStorageAccountClient) Delete(ctx context.Context, resourceGroupName string, accountName string) (result autorest.Response, err error) {
	mc := newMetricContext("storage_account", "delete", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(false, "DeleteStorageAccount")
		return
	}

	klog.V(10).Infof("azStorageAccountClient.Delete(%q,%q): start", resourceGroupName, accountName)
	defer func() {
		klog.V(10).Infof("azStorageAccountClient.Delete(%q,%q): end", resourceGroupName, accountName)
	}()

	result, err = az.client.Delete(ctx, resourceGroupName, accountName)
	mc.Observe(err)
	return
}

func (az *azStorageAccountClient) ListKeys(ctx context.Context, resourceGroupName string, accountName string) (result storage.AccountListKeysResult, err error) {
	mc := newMetricContext("storage_account", "list_keys", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(false, "ListStorageAccountKeys")
		return
	}

	klog.V(10).Infof("azStorageAccountClient.ListKeys(%q,%q): start", resourceGroupName, accountName)
	defer func() {
		klog.V(10).Infof("azStorageAccountClient.ListKeys(%q,%q): end", resourceGroupName, accountName)
	}()

	result, err = az.client.ListKeys(ctx, resourceGroupName, accountName, storage.Kerb)
	mc.Observe(err)
	return
}

func (az *azStorageAccountClient) ListByResourceGroup(ctx context.Context, resourceGroupName string) (result storage.AccountListResult, err error) {
	mc := newMetricContext("storage_account", "list_by_resource_group", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(false, "ListStorageAccountsByResourceGroup")
		return
	}

	klog.V(10).Infof("azStorageAccountClient.ListByResourceGroup(%q): start", resourceGroupName)
	defer func() {
		klog.V(10).Infof("azStorageAccountClient.ListByResourceGroup(%q): end", resourceGroupName)
	}()

	result, err = az.client.ListByResourceGroup(ctx, resourceGroupName)
	mc.Observe(err)
	return
}

func (az *azStorageAccountClient) GetProperties(ctx context.Context, resourceGroupName string, accountName string) (result storage.Account, err error) {
	mc := newMetricContext("storage_account", "get_properties", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(false, "GetStorageAccount/Properties")
		return
	}

	klog.V(10).Infof("azStorageAccountClient.GetProperties(%q,%q): start", resourceGroupName, accountName)
	defer func() {
		klog.V(10).Infof("azStorageAccountClient.GetProperties(%q,%q): end", resourceGroupName, accountName)
	}()

	result, err = az.client.GetProperties(ctx, resourceGroupName, accountName, "")
	mc.Observe(err)
	return
}

// azDisksClient implements DisksClient.
type azDisksClient struct {
	client            compute.DisksClient
	rateLimiterReader flowcontrol.RateLimiter
	rateLimiterWriter flowcontrol.RateLimiter
}

func newAzDisksClient(config *azClientConfig) *azDisksClient {
	disksClient := compute.NewDisksClientWithBaseURI(config.resourceManagerEndpoint, config.subscriptionID)
	disksClient.Authorizer = autorest.NewBearerAuthorizer(config.servicePrincipalToken)
	disksClient.PollingDelay = 5 * time.Second
	if config.ShouldOmitCloudProviderBackoff {
		disksClient.RetryAttempts = config.CloudProviderBackoffRetries
		disksClient.RetryDuration = time.Duration(config.CloudProviderBackoffDuration) * time.Second
	}
	configureUserAgent(&disksClient.Client)

	return &azDisksClient{
		client:            disksClient,
		rateLimiterReader: config.rateLimiterReader,
		rateLimiterWriter: config.rateLimiterWriter,
	}
}

func (az *azDisksClient) CreateOrUpdate(ctx context.Context, resourceGroupName string, diskName string, diskParameter compute.Disk) (resp *http.Response, err error) {
	mc := newMetricContext("disks", "create_or_update", resourceGroupName, az.client.SubscriptionID, "")
	/* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(true, "DiskCreateOrUpdate")
		return
	}

	klog.V(10).Infof("azDisksClient.CreateOrUpdate(%q,%q): start", resourceGroupName, diskName)
	defer func() {
		klog.V(10).Infof("azDisksClient.CreateOrUpdate(%q,%q): end", resourceGroupName, diskName)
	}()

	future, err := az.client.CreateOrUpdate(ctx, resourceGroupName, diskName, diskParameter)
	if err != nil {
		return future.Response(), mc.Observe(err)
	}

	err = future.WaitForCompletionRef(ctx, az.client.Client)
	return future.Response(), mc.Observe(err)
}

func (az *azDisksClient) Delete(ctx context.Context, resourceGroupName string, diskName string) (resp *http.Response, err error) {
	mc := newMetricContext("disks", "delete", resourceGroupName, az.client.SubscriptionID, "")
	/* Write rate limiting */
	if !az.rateLimiterWriter.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(true, "DiskDelete")
		return
	}

	klog.V(10).Infof("azDisksClient.Delete(%q,%q): start", resourceGroupName, diskName)
	defer func() {
		klog.V(10).Infof("azDisksClient.Delete(%q,%q): end", resourceGroupName, diskName)
	}()

	future, err := az.client.Delete(ctx, resourceGroupName, diskName)
	if err != nil {
		return future.Response(), mc.Observe(err)
	}
	err = future.WaitForCompletionRef(ctx, az.client.Client)
	return future.Response(), mc.Observe(err)
}

func (az *azDisksClient) Get(ctx context.Context, resourceGroupName string, diskName string) (result compute.Disk, err error) {
	mc := newMetricContext("disks", "get", resourceGroupName, az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(false, "GetDisk")
		return
	}

	klog.V(10).Infof("azDisksClient.Get(%q,%q): start", resourceGroupName, diskName)
	defer func() {
		klog.V(10).Infof("azDisksClient.Get(%q,%q): end", resourceGroupName, diskName)
	}()

	result, err = az.client.Get(ctx, resourceGroupName, diskName)
	mc.Observe(err)
	return
}

func newSnapshotsClient(config *azClientConfig) *compute.SnapshotsClient {
	snapshotsClient := compute.NewSnapshotsClientWithBaseURI(config.resourceManagerEndpoint, config.subscriptionID)
	snapshotsClient.Authorizer = autorest.NewBearerAuthorizer(config.servicePrincipalToken)
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

func newAzVirtualMachineSizesClient(config *azClientConfig) *azVirtualMachineSizesClient {
	VirtualMachineSizesClient := compute.NewVirtualMachineSizesClient(config.subscriptionID)
	VirtualMachineSizesClient.BaseURI = config.resourceManagerEndpoint
	VirtualMachineSizesClient.Authorizer = autorest.NewBearerAuthorizer(config.servicePrincipalToken)
	VirtualMachineSizesClient.PollingDelay = 5 * time.Second
	if config.ShouldOmitCloudProviderBackoff {
		VirtualMachineSizesClient.RetryAttempts = config.CloudProviderBackoffRetries
		VirtualMachineSizesClient.RetryDuration = time.Duration(config.CloudProviderBackoffDuration) * time.Second
	}
	configureUserAgent(&VirtualMachineSizesClient.Client)

	return &azVirtualMachineSizesClient{
		rateLimiterReader: config.rateLimiterReader,
		rateLimiterWriter: config.rateLimiterWriter,
		client:            VirtualMachineSizesClient,
	}
}

func (az *azVirtualMachineSizesClient) List(ctx context.Context, location string) (result compute.VirtualMachineSizeListResult, err error) {
	mc := newMetricContext("vmsizes", "list", "", az.client.SubscriptionID, "")
	if !az.rateLimiterReader.TryAccept() {
		mc.RateLimitedCount()
		err = createRateLimitErr(false, "VMSizesList")
		return
	}

	klog.V(10).Infof("azVirtualMachineSizesClient.List(%q): start", location)
	defer func() {
		klog.V(10).Infof("azVirtualMachineSizesClient.List(%q): end", location)
	}()

	result, err = az.client.List(ctx, location)
	mc.Observe(err)
	return
}

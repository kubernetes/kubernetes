/*
Copyright 2016 The Kubernetes Authors.

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
	"regexp"
	"strings"
	"time"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2018-10-01/compute"
	"github.com/Azure/azure-sdk-for-go/services/network/mgmt/2017-09-01/network"
	"github.com/Azure/go-autorest/autorest"
	"k8s.io/apimachinery/pkg/types"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/klog"
)

var (
	vmCacheTTL  = time.Minute
	lbCacheTTL  = 2 * time.Minute
	nsgCacheTTL = 2 * time.Minute
	rtCacheTTL  = 2 * time.Minute

	azureNodeProviderIDRE    = regexp.MustCompile(`^azure:///subscriptions/(?:.*)/resourceGroups/(?:.*)/providers/Microsoft.Compute/(?:.*)`)
	azureResourceGroupNameRE = regexp.MustCompile(`.*/subscriptions/(?:.*)/resourceGroups/(.+)/providers/(?:.*)`)
)

// checkExistsFromError inspects an error and returns a true if err is nil,
// false if error is an autorest.Error with StatusCode=404 and will return the
// error back if error is another status code or another type of error.
func checkResourceExistsFromError(err error) (bool, string, error) {
	if err == nil {
		return true, "", nil
	}
	v, ok := err.(autorest.DetailedError)
	if !ok {
		return false, "", err
	}
	if v.StatusCode == http.StatusNotFound {
		return false, err.Error(), nil
	}
	return false, "", v
}

// If it is StatusNotFound return nil,
// Otherwise, return what it is
func ignoreStatusNotFoundFromError(err error) error {
	if err == nil {
		return nil
	}
	v, ok := err.(autorest.DetailedError)
	if ok && v.StatusCode == http.StatusNotFound {
		return nil
	}
	return err
}

// ignoreStatusForbiddenFromError returns nil if the status code is StatusForbidden.
// This happens when AuthorizationFailed is reported from Azure API.
func ignoreStatusForbiddenFromError(err error) error {
	if err == nil {
		return nil
	}
	v, ok := err.(autorest.DetailedError)
	if ok && v.StatusCode == http.StatusForbidden {
		return nil
	}
	return err
}

/// getVirtualMachine calls 'VirtualMachinesClient.Get' with a timed cache
/// The service side has throttling control that delays responses if there're multiple requests onto certain vm
/// resource request in short period.
func (az *Cloud) getVirtualMachine(nodeName types.NodeName) (vm compute.VirtualMachine, err error) {
	vmName := string(nodeName)
	cachedVM, err := az.vmCache.Get(vmName)
	if err != nil {
		return vm, err
	}

	if cachedVM == nil {
		return vm, cloudprovider.InstanceNotFound
	}

	return *(cachedVM.(*compute.VirtualMachine)), nil
}

func (az *Cloud) getRouteTable() (routeTable network.RouteTable, exists bool, err error) {
	cachedRt, err := az.rtCache.Get(az.RouteTableName)
	if err != nil {
		return routeTable, false, err
	}

	if cachedRt == nil {
		return routeTable, false, nil
	}

	return *(cachedRt.(*network.RouteTable)), true, nil
}

func (az *Cloud) getPublicIPAddress(pipResourceGroup string, pipName string) (pip network.PublicIPAddress, exists bool, err error) {
	resourceGroup := az.ResourceGroup
	if pipResourceGroup != "" {
		resourceGroup = pipResourceGroup
	}

	var realErr error
	var message string
	ctx, cancel := getContextWithCancel()
	defer cancel()
	pip, err = az.PublicIPAddressesClient.Get(ctx, resourceGroup, pipName, "")
	exists, message, realErr = checkResourceExistsFromError(err)
	if realErr != nil {
		return pip, false, realErr
	}

	if !exists {
		klog.V(2).Infof("Public IP %q not found with message: %q", pipName, message)
		return pip, false, nil
	}

	return pip, exists, err
}

func (az *Cloud) getSubnet(virtualNetworkName string, subnetName string) (subnet network.Subnet, exists bool, err error) {
	var realErr error
	var message string
	var rg string

	if len(az.VnetResourceGroup) > 0 {
		rg = az.VnetResourceGroup
	} else {
		rg = az.ResourceGroup
	}

	ctx, cancel := getContextWithCancel()
	defer cancel()
	subnet, err = az.SubnetsClient.Get(ctx, rg, virtualNetworkName, subnetName, "")
	exists, message, realErr = checkResourceExistsFromError(err)
	if realErr != nil {
		return subnet, false, realErr
	}

	if !exists {
		klog.V(2).Infof("Subnet %q not found with message: %q", subnetName, message)
		return subnet, false, nil
	}

	return subnet, exists, err
}

func (az *Cloud) getAzureLoadBalancer(name string) (lb network.LoadBalancer, exists bool, err error) {
	cachedLB, err := az.lbCache.Get(name)
	if err != nil {
		return lb, false, err
	}

	if cachedLB == nil {
		return lb, false, nil
	}

	return *(cachedLB.(*network.LoadBalancer)), true, nil
}

func (az *Cloud) getSecurityGroup() (nsg network.SecurityGroup, err error) {
	if az.SecurityGroupName == "" {
		return nsg, fmt.Errorf("securityGroupName is not configured")
	}

	securityGroup, err := az.nsgCache.Get(az.SecurityGroupName)
	if err != nil {
		return nsg, err
	}

	if securityGroup == nil {
		return nsg, fmt.Errorf("nsg %q not found", az.SecurityGroupName)
	}

	return *(securityGroup.(*network.SecurityGroup)), nil
}

func (az *Cloud) newVMCache() (*timedCache, error) {
	getter := func(key string) (interface{}, error) {
		// Currently InstanceView request are used by azure_zones, while the calls come after non-InstanceView
		// request. If we first send an InstanceView request and then a non InstanceView request, the second
		// request will still hit throttling. This is what happens now for cloud controller manager: In this
		// case we do get instance view every time to fulfill the azure_zones requirement without hitting
		// throttling.
		// Consider adding separate parameter for controlling 'InstanceView' once node update issue #56276 is fixed
		ctx, cancel := getContextWithCancel()
		defer cancel()

		resourceGroup, err := az.GetNodeResourceGroup(key)
		if err != nil {
			return nil, err
		}

		vm, err := az.VirtualMachinesClient.Get(ctx, resourceGroup, key, compute.InstanceView)
		exists, message, realErr := checkResourceExistsFromError(err)
		if realErr != nil {
			return nil, realErr
		}

		if !exists {
			klog.V(2).Infof("Virtual machine %q not found with message: %q", key, message)
			return nil, nil
		}

		return &vm, nil
	}

	return newTimedcache(vmCacheTTL, getter)
}

func (az *Cloud) newLBCache() (*timedCache, error) {
	getter := func(key string) (interface{}, error) {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		lb, err := az.LoadBalancerClient.Get(ctx, az.ResourceGroup, key, "")
		exists, message, realErr := checkResourceExistsFromError(err)
		if realErr != nil {
			return nil, realErr
		}

		if !exists {
			klog.V(2).Infof("Load balancer %q not found with message: %q", key, message)
			return nil, nil
		}

		return &lb, nil
	}

	return newTimedcache(lbCacheTTL, getter)
}

func (az *Cloud) newNSGCache() (*timedCache, error) {
	getter := func(key string) (interface{}, error) {
		ctx, cancel := getContextWithCancel()
		defer cancel()
		nsg, err := az.SecurityGroupsClient.Get(ctx, az.ResourceGroup, key, "")
		exists, message, realErr := checkResourceExistsFromError(err)
		if realErr != nil {
			return nil, realErr
		}

		if !exists {
			klog.V(2).Infof("Security group %q not found with message: %q", key, message)
			return nil, nil
		}

		return &nsg, nil
	}

	return newTimedcache(nsgCacheTTL, getter)
}

func (az *Cloud) newRouteTableCache() (*timedCache, error) {
	getter := func(key string) (interface{}, error) {
		ctx, cancel := getContextWithCancel()
		defer cancel()
		rt, err := az.RouteTablesClient.Get(ctx, az.ResourceGroup, key, "")
		exists, message, realErr := checkResourceExistsFromError(err)
		if realErr != nil {
			return nil, realErr
		}

		if !exists {
			klog.V(2).Infof("Route table %q not found with message: %q", key, message)
			return nil, nil
		}

		return &rt, nil
	}

	return newTimedcache(rtCacheTTL, getter)
}

func (az *Cloud) useStandardLoadBalancer() bool {
	return strings.EqualFold(az.LoadBalancerSku, loadBalancerSkuStandard)
}

func (az *Cloud) excludeMasterNodesFromStandardLB() bool {
	return az.ExcludeMasterFromStandardLB != nil && *az.ExcludeMasterFromStandardLB
}

func (az *Cloud) disableLoadBalancerOutboundSNAT() bool {
	if !az.useStandardLoadBalancer() || az.DisableOutboundSNAT == nil {
		return false
	}

	return *az.DisableOutboundSNAT
}

// IsNodeUnmanaged returns true if the node is not managed by Azure cloud provider.
// Those nodes includes on-prem or VMs from other clouds. They will not be added to load balancer
// backends. Azure routes and managed disks are also not supported for them.
func (az *Cloud) IsNodeUnmanaged(nodeName string) (bool, error) {
	unmanagedNodes, err := az.GetUnmanagedNodes()
	if err != nil {
		return false, err
	}

	return unmanagedNodes.Has(nodeName), nil
}

// IsNodeUnmanagedByProviderID returns true if the node is not managed by Azure cloud provider.
// All managed node's providerIDs are in format 'azure:///subscriptions/<id>/resourceGroups/<rg>/providers/Microsoft.Compute/.*'
func (az *Cloud) IsNodeUnmanagedByProviderID(providerID string) bool {
	return !azureNodeProviderIDRE.Match([]byte(providerID))
}

// convertResourceGroupNameToLower converts the resource group name in the resource ID to be lowered.
func convertResourceGroupNameToLower(resourceID string) (string, error) {
	matches := azureResourceGroupNameRE.FindStringSubmatch(resourceID)
	if len(matches) != 2 {
		return "", fmt.Errorf("%q isn't in Azure resource ID format %q", resourceID, azureResourceGroupNameRE.String())
	}

	resourceGroup := matches[1]
	return strings.Replace(resourceID, resourceGroup, strings.ToLower(resourceGroup), 1), nil
}

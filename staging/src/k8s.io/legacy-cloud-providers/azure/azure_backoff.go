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
	"net/http"
	"strings"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-12-01/compute"
	"github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-06-01/network"
	"github.com/Azure/go-autorest/autorest/to"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/klog/v2"
	azcache "k8s.io/legacy-cloud-providers/azure/cache"
	"k8s.io/legacy-cloud-providers/azure/retry"
)

const (
	// not active means the instance is under deleting from Azure VMSS.
	vmssVMNotActiveErrorMessage = "not an active Virtual Machine Scale Set VM instanceId"

	// operationCanceledErrorMessage means the operation is canceled by another new operation.
	operationCanceledErrorMessage = "canceledandsupersededduetoanotheroperation"
)

// RequestBackoff if backoff is disabled in cloud provider it
// returns a new Backoff object steps = 1
// This is to make sure that the requested command executes
// at least once
func (az *Cloud) RequestBackoff() (resourceRequestBackoff wait.Backoff) {
	if az.CloudProviderBackoff {
		return az.ResourceRequestBackoff
	}
	resourceRequestBackoff = wait.Backoff{
		Steps: 1,
	}
	return resourceRequestBackoff
}

// Event creates a event for the specified object.
func (az *Cloud) Event(obj runtime.Object, eventtype, reason, message string) {
	if obj != nil && reason != "" {
		az.eventRecorder.Event(obj, eventtype, reason, message)
	}
}

// GetVirtualMachineWithRetry invokes az.getVirtualMachine with exponential backoff retry
func (az *Cloud) GetVirtualMachineWithRetry(name types.NodeName, crt azcache.AzureCacheReadType) (compute.VirtualMachine, error) {
	var machine compute.VirtualMachine
	var retryErr error
	err := wait.ExponentialBackoff(az.RequestBackoff(), func() (bool, error) {
		machine, retryErr = az.getVirtualMachine(name, crt)
		if retryErr == cloudprovider.InstanceNotFound {
			return true, cloudprovider.InstanceNotFound
		}
		if retryErr != nil {
			klog.Errorf("GetVirtualMachineWithRetry(%s): backoff failure, will retry, err=%v", name, retryErr)
			return false, nil
		}
		klog.V(2).Infof("GetVirtualMachineWithRetry(%s): backoff success", name)
		return true, nil
	})
	if err == wait.ErrWaitTimeout {
		err = retryErr
	}
	return machine, err
}

// ListVirtualMachines invokes az.VirtualMachinesClient.List with exponential backoff retry
func (az *Cloud) ListVirtualMachines(resourceGroup string) ([]compute.VirtualMachine, error) {
	ctx, cancel := getContextWithCancel()
	defer cancel()

	allNodes, rerr := az.VirtualMachinesClient.List(ctx, resourceGroup)
	if rerr != nil {
		klog.Errorf("VirtualMachinesClient.List(%v) failure with err=%v", resourceGroup, rerr)
		return nil, rerr.Error()
	}
	klog.V(2).Infof("VirtualMachinesClient.List(%v) success", resourceGroup)
	return allNodes, nil
}

// getPrivateIPsForMachine is wrapper for optional backoff getting private ips
// list of a node by name
func (az *Cloud) getPrivateIPsForMachine(nodeName types.NodeName) ([]string, error) {
	return az.getPrivateIPsForMachineWithRetry(nodeName)
}

func (az *Cloud) getPrivateIPsForMachineWithRetry(nodeName types.NodeName) ([]string, error) {
	var privateIPs []string
	err := wait.ExponentialBackoff(az.RequestBackoff(), func() (bool, error) {
		var retryErr error
		privateIPs, retryErr = az.VMSet.GetPrivateIPsByNodeName(string(nodeName))
		if retryErr != nil {
			// won't retry since the instance doesn't exist on Azure.
			if retryErr == cloudprovider.InstanceNotFound {
				return true, retryErr
			}
			klog.Errorf("GetPrivateIPsByNodeName(%s): backoff failure, will retry,err=%v", nodeName, retryErr)
			return false, nil
		}
		klog.V(3).Infof("GetPrivateIPsByNodeName(%s): backoff success", nodeName)
		return true, nil
	})
	return privateIPs, err
}

func (az *Cloud) getIPForMachine(nodeName types.NodeName) (string, string, error) {
	return az.GetIPForMachineWithRetry(nodeName)
}

// GetIPForMachineWithRetry invokes az.getIPForMachine with exponential backoff retry
func (az *Cloud) GetIPForMachineWithRetry(name types.NodeName) (string, string, error) {
	var ip, publicIP string
	err := wait.ExponentialBackoff(az.RequestBackoff(), func() (bool, error) {
		var retryErr error
		ip, publicIP, retryErr = az.VMSet.GetIPByNodeName(string(name))
		if retryErr != nil {
			klog.Errorf("GetIPForMachineWithRetry(%s): backoff failure, will retry,err=%v", name, retryErr)
			return false, nil
		}
		klog.V(3).Infof("GetIPForMachineWithRetry(%s): backoff success", name)
		return true, nil
	})
	return ip, publicIP, err
}

// CreateOrUpdateSecurityGroup invokes az.SecurityGroupsClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) CreateOrUpdateSecurityGroup(service *v1.Service, sg network.SecurityGroup) error {
	ctx, cancel := getContextWithCancel()
	defer cancel()

	rerr := az.SecurityGroupsClient.CreateOrUpdate(ctx, az.SecurityGroupResourceGroup, *sg.Name, sg, to.String(sg.Etag))
	klog.V(10).Infof("SecurityGroupsClient.CreateOrUpdate(%s): end", *sg.Name)
	if rerr == nil {
		// Invalidate the cache right after updating
		az.nsgCache.Delete(*sg.Name)
		return nil
	}

	// Invalidate the cache because ETAG precondition mismatch.
	if rerr.HTTPStatusCode == http.StatusPreconditionFailed {
		klog.V(3).Infof("SecurityGroup cache for %s is cleanup because of http.StatusPreconditionFailed", *sg.Name)
		az.nsgCache.Delete(*sg.Name)
	}

	// Invalidate the cache because another new operation has canceled the current request.
	if strings.Contains(strings.ToLower(rerr.Error().Error()), operationCanceledErrorMessage) {
		klog.V(3).Infof("SecurityGroup cache for %s is cleanup because CreateOrUpdateSecurityGroup is canceled by another operation", *sg.Name)
		az.nsgCache.Delete(*sg.Name)
	}

	return rerr.Error()
}

// CreateOrUpdateLB invokes az.LoadBalancerClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) CreateOrUpdateLB(service *v1.Service, lb network.LoadBalancer) error {
	ctx, cancel := getContextWithCancel()
	defer cancel()

	rgName := az.getLoadBalancerResourceGroup()
	rerr := az.LoadBalancerClient.CreateOrUpdate(ctx, rgName, *lb.Name, lb, to.String(lb.Etag))
	klog.V(10).Infof("LoadBalancerClient.CreateOrUpdate(%s): end", *lb.Name)
	if rerr == nil {
		// Invalidate the cache right after updating
		az.lbCache.Delete(*lb.Name)
		return nil
	}

	// Invalidate the cache because ETAG precondition mismatch.
	if rerr.HTTPStatusCode == http.StatusPreconditionFailed {
		klog.V(3).Infof("LoadBalancer cache for %s is cleanup because of http.StatusPreconditionFailed", *lb.Name)
		az.lbCache.Delete(*lb.Name)
	}
	// Invalidate the cache because another new operation has canceled the current request.
	if strings.Contains(strings.ToLower(rerr.Error().Error()), operationCanceledErrorMessage) {
		klog.V(3).Infof("LoadBalancer cache for %s is cleanup because CreateOrUpdate is canceled by another operation", *lb.Name)
		az.lbCache.Delete(*lb.Name)
	}

	return rerr.Error()
}

// ListLB invokes az.LoadBalancerClient.List with exponential backoff retry
func (az *Cloud) ListLB(service *v1.Service) ([]network.LoadBalancer, error) {
	ctx, cancel := getContextWithCancel()
	defer cancel()

	rgName := az.getLoadBalancerResourceGroup()
	allLBs, rerr := az.LoadBalancerClient.List(ctx, rgName)
	if rerr != nil {
		az.Event(service, v1.EventTypeWarning, "ListLoadBalancers", rerr.Error().Error())
		klog.Errorf("LoadBalancerClient.List(%v) failure with err=%v", rgName, rerr)
		return nil, rerr.Error()
	}
	klog.V(2).Infof("LoadBalancerClient.List(%v) success", rgName)
	return allLBs, nil
}

// ListPIP list the PIP resources in the given resource group
func (az *Cloud) ListPIP(service *v1.Service, pipResourceGroup string) ([]network.PublicIPAddress, error) {
	ctx, cancel := getContextWithCancel()
	defer cancel()

	allPIPs, rerr := az.PublicIPAddressesClient.List(ctx, pipResourceGroup)
	if rerr != nil {
		az.Event(service, v1.EventTypeWarning, "ListPublicIPs", rerr.Error().Error())
		klog.Errorf("PublicIPAddressesClient.List(%v) failure with err=%v", pipResourceGroup, rerr)
		return nil, rerr.Error()
	}

	klog.V(2).Infof("PublicIPAddressesClient.List(%v) success", pipResourceGroup)
	return allPIPs, nil
}

// CreateOrUpdatePIP invokes az.PublicIPAddressesClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) CreateOrUpdatePIP(service *v1.Service, pipResourceGroup string, pip network.PublicIPAddress) error {
	ctx, cancel := getContextWithCancel()
	defer cancel()

	rerr := az.PublicIPAddressesClient.CreateOrUpdate(ctx, pipResourceGroup, *pip.Name, pip)
	klog.V(10).Infof("PublicIPAddressesClient.CreateOrUpdate(%s, %s): end", pipResourceGroup, *pip.Name)
	if rerr != nil {
		klog.Errorf("PublicIPAddressesClient.CreateOrUpdate(%s, %s) failed: %s", pipResourceGroup, *pip.Name, rerr.Error().Error())
		az.Event(service, v1.EventTypeWarning, "CreateOrUpdatePublicIPAddress", rerr.Error().Error())
		return rerr.Error()
	}

	return nil
}

// CreateOrUpdateInterface invokes az.PublicIPAddressesClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) CreateOrUpdateInterface(service *v1.Service, nic network.Interface) error {
	ctx, cancel := getContextWithCancel()
	defer cancel()

	rerr := az.InterfacesClient.CreateOrUpdate(ctx, az.ResourceGroup, *nic.Name, nic)
	klog.V(10).Infof("InterfacesClient.CreateOrUpdate(%s): end", *nic.Name)
	if rerr != nil {
		klog.Errorf("InterfacesClient.CreateOrUpdate(%s) failed: %s", *nic.Name, rerr.Error().Error())
		az.Event(service, v1.EventTypeWarning, "CreateOrUpdateInterface", rerr.Error().Error())
		return rerr.Error()
	}

	return nil
}

// DeletePublicIP invokes az.PublicIPAddressesClient.Delete with exponential backoff retry
func (az *Cloud) DeletePublicIP(service *v1.Service, pipResourceGroup string, pipName string) error {
	ctx, cancel := getContextWithCancel()
	defer cancel()

	rerr := az.PublicIPAddressesClient.Delete(ctx, pipResourceGroup, pipName)
	if rerr != nil {
		klog.Errorf("PublicIPAddressesClient.Delete(%s) failed: %s", pipName, rerr.Error().Error())
		az.Event(service, v1.EventTypeWarning, "DeletePublicIPAddress", rerr.Error().Error())
		return rerr.Error()
	}

	return nil
}

// DeleteLB invokes az.LoadBalancerClient.Delete with exponential backoff retry
func (az *Cloud) DeleteLB(service *v1.Service, lbName string) error {
	ctx, cancel := getContextWithCancel()
	defer cancel()

	rgName := az.getLoadBalancerResourceGroup()
	rerr := az.LoadBalancerClient.Delete(ctx, rgName, lbName)
	if rerr == nil {
		// Invalidate the cache right after updating
		az.lbCache.Delete(lbName)
		return nil
	}

	klog.Errorf("LoadBalancerClient.Delete(%s) failed: %s", lbName, rerr.Error().Error())
	az.Event(service, v1.EventTypeWarning, "DeleteLoadBalancer", rerr.Error().Error())
	return rerr.Error()
}

// CreateOrUpdateRouteTable invokes az.RouteTablesClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) CreateOrUpdateRouteTable(routeTable network.RouteTable) error {
	ctx, cancel := getContextWithCancel()
	defer cancel()

	rerr := az.RouteTablesClient.CreateOrUpdate(ctx, az.RouteTableResourceGroup, az.RouteTableName, routeTable, to.String(routeTable.Etag))
	if rerr == nil {
		// Invalidate the cache right after updating
		az.rtCache.Delete(*routeTable.Name)
		return nil
	}

	// Invalidate the cache because etag mismatch.
	if rerr.HTTPStatusCode == http.StatusPreconditionFailed {
		klog.V(3).Infof("Route table cache for %s is cleanup because of http.StatusPreconditionFailed", *routeTable.Name)
		az.rtCache.Delete(*routeTable.Name)
	}
	// Invalidate the cache because another new operation has canceled the current request.
	if strings.Contains(strings.ToLower(rerr.Error().Error()), operationCanceledErrorMessage) {
		klog.V(3).Infof("Route table cache for %s is cleanup because CreateOrUpdateRouteTable is canceld by another operation", *routeTable.Name)
		az.rtCache.Delete(*routeTable.Name)
	}
	klog.Errorf("RouteTablesClient.CreateOrUpdate(%s) failed: %v", az.RouteTableName, rerr.Error())
	return rerr.Error()
}

// CreateOrUpdateRoute invokes az.RoutesClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) CreateOrUpdateRoute(route network.Route) error {
	ctx, cancel := getContextWithCancel()
	defer cancel()

	rerr := az.RoutesClient.CreateOrUpdate(ctx, az.RouteTableResourceGroup, az.RouteTableName, *route.Name, route, to.String(route.Etag))
	klog.V(10).Infof("RoutesClient.CreateOrUpdate(%s): end", *route.Name)
	if rerr == nil {
		az.rtCache.Delete(az.RouteTableName)
		return nil
	}

	if rerr.HTTPStatusCode == http.StatusPreconditionFailed {
		klog.V(3).Infof("Route cache for %s is cleanup because of http.StatusPreconditionFailed", *route.Name)
		az.rtCache.Delete(az.RouteTableName)
	}
	// Invalidate the cache because another new operation has canceled the current request.
	if strings.Contains(strings.ToLower(rerr.Error().Error()), operationCanceledErrorMessage) {
		klog.V(3).Infof("Route cache for %s is cleanup because CreateOrUpdateRouteTable is canceld by another operation", *route.Name)
		az.rtCache.Delete(az.RouteTableName)
	}
	return rerr.Error()
}

// DeleteRouteWithName invokes az.RoutesClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) DeleteRouteWithName(routeName string) error {
	ctx, cancel := getContextWithCancel()
	defer cancel()

	rerr := az.RoutesClient.Delete(ctx, az.RouteTableResourceGroup, az.RouteTableName, routeName)
	klog.V(10).Infof("RoutesClient.Delete(%s,%s): end", az.RouteTableName, routeName)
	if rerr == nil {
		return nil
	}

	klog.Errorf("RoutesClient.Delete(%s, %s) failed: %v", az.RouteTableName, routeName, rerr.Error())
	return rerr.Error()
}

// CreateOrUpdateVMSS invokes az.VirtualMachineScaleSetsClient.Update().
func (az *Cloud) CreateOrUpdateVMSS(resourceGroupName string, VMScaleSetName string, parameters compute.VirtualMachineScaleSet) *retry.Error {
	ctx, cancel := getContextWithCancel()
	defer cancel()

	// When vmss is being deleted, CreateOrUpdate API would report "the vmss is being deleted" error.
	// Since it is being deleted, we shouldn't send more CreateOrUpdate requests for it.
	klog.V(3).Infof("CreateOrUpdateVMSS: verify the status of the vmss being created or updated")
	vmss, rerr := az.VirtualMachineScaleSetsClient.Get(ctx, resourceGroupName, VMScaleSetName)
	if rerr != nil {
		klog.Errorf("CreateOrUpdateVMSS: error getting vmss(%s): %v", VMScaleSetName, rerr)
		return rerr
	}
	if vmss.ProvisioningState != nil && strings.EqualFold(*vmss.ProvisioningState, virtualMachineScaleSetsDeallocating) {
		klog.V(3).Infof("CreateOrUpdateVMSS: found vmss %s being deleted, skipping", VMScaleSetName)
		return nil
	}

	rerr = az.VirtualMachineScaleSetsClient.CreateOrUpdate(ctx, resourceGroupName, VMScaleSetName, parameters)
	klog.V(10).Infof("UpdateVmssVMWithRetry: VirtualMachineScaleSetsClient.CreateOrUpdate(%s): end", VMScaleSetName)
	if rerr != nil {
		klog.Errorf("CreateOrUpdateVMSS: error CreateOrUpdate vmss(%s): %v", VMScaleSetName, rerr)
		return rerr
	}

	return nil
}

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
	"fmt"
	"net/http"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-03-01/compute"
	"github.com/Azure/azure-sdk-for-go/services/network/mgmt/2017-09-01/network"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/klog"
)

// requestBackoff if backoff is disabled in cloud provider it
// returns a new Backoff object steps = 1
// This is to make sure that the requested command executes
// at least once
func (az *Cloud) requestBackoff() (resourceRequestBackoff wait.Backoff) {
	if az.CloudProviderBackoff {
		return az.resourceRequestBackoff
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
func (az *Cloud) GetVirtualMachineWithRetry(name types.NodeName) (compute.VirtualMachine, error) {
	var machine compute.VirtualMachine
	var retryErr error
	err := wait.ExponentialBackoff(az.requestBackoff(), func() (bool, error) {
		machine, retryErr = az.getVirtualMachine(name)
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

// ListVirtualMachinesWithRetry invokes az.VirtualMachinesClient.List with exponential backoff retry
func (az *Cloud) ListVirtualMachinesWithRetry(resourceGroup string) ([]compute.VirtualMachine, error) {
	allNodes := []compute.VirtualMachine{}
	err := wait.ExponentialBackoff(az.requestBackoff(), func() (bool, error) {
		var retryErr error
		ctx, cancel := getContextWithCancel()
		defer cancel()
		allNodes, retryErr = az.VirtualMachinesClient.List(ctx, resourceGroup)
		if retryErr != nil {
			klog.Errorf("VirtualMachinesClient.List(%v) - backoff: failure, will retry,err=%v",
				resourceGroup,
				retryErr)
			return false, retryErr
		}
		klog.V(2).Infof("VirtualMachinesClient.List(%v) - backoff: success", resourceGroup)
		return true, nil
	})
	if err != nil {
		return nil, err
	}

	return allNodes, err
}

// ListVirtualMachines invokes az.VirtualMachinesClient.List with exponential backoff retry
func (az *Cloud) ListVirtualMachines(resourceGroup string) ([]compute.VirtualMachine, error) {
	if az.Config.shouldOmitCloudProviderBackoff() {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		allNodes, err := az.VirtualMachinesClient.List(ctx, resourceGroup)
		if err != nil {
			klog.Errorf("VirtualMachinesClient.List(%v) failure with err=%v", resourceGroup, err)
			return nil, err
		}
		klog.V(2).Infof("VirtualMachinesClient.List(%v) success", resourceGroup)
		return allNodes, nil
	}

	return az.ListVirtualMachinesWithRetry(resourceGroup)
}

func (az *Cloud) getIPForMachine(nodeName types.NodeName) (string, string, error) {
	if az.Config.shouldOmitCloudProviderBackoff() {
		return az.vmSet.GetIPByNodeName(string(nodeName))
	}

	return az.GetIPForMachineWithRetry(nodeName)
}

// GetIPForMachineWithRetry invokes az.getIPForMachine with exponential backoff retry
func (az *Cloud) GetIPForMachineWithRetry(name types.NodeName) (string, string, error) {
	var ip, publicIP string
	err := wait.ExponentialBackoff(az.requestBackoff(), func() (bool, error) {
		var retryErr error
		ip, publicIP, retryErr = az.vmSet.GetIPByNodeName(string(name))
		if retryErr != nil {
			klog.Errorf("GetIPForMachineWithRetry(%s): backoff failure, will retry,err=%v", name, retryErr)
			return false, nil
		}
		klog.V(2).Infof("GetIPForMachineWithRetry(%s): backoff success", name)
		return true, nil
	})
	return ip, publicIP, err
}

// CreateOrUpdateSecurityGroup invokes az.SecurityGroupsClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) CreateOrUpdateSecurityGroup(service *v1.Service, sg network.SecurityGroup) error {
	if az.Config.shouldOmitCloudProviderBackoff() {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		resp, err := az.SecurityGroupsClient.CreateOrUpdate(ctx, az.ResourceGroup, *sg.Name, sg)
		klog.V(10).Infof("SecurityGroupsClient.CreateOrUpdate(%s): end", *sg.Name)
		if err == nil {
			if isSuccessHTTPResponse(resp) {
				// Invalidate the cache right after updating
				az.nsgCache.Delete(*sg.Name)
			} else if resp != nil {
				return fmt.Errorf("HTTP response %q", resp.Status)
			}
		}
		return err
	}

	return az.CreateOrUpdateSGWithRetry(service, sg)
}

// CreateOrUpdateSGWithRetry invokes az.SecurityGroupsClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) CreateOrUpdateSGWithRetry(service *v1.Service, sg network.SecurityGroup) error {
	return wait.ExponentialBackoff(az.requestBackoff(), func() (bool, error) {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		resp, err := az.SecurityGroupsClient.CreateOrUpdate(ctx, az.ResourceGroup, *sg.Name, sg)
		klog.V(10).Infof("SecurityGroupsClient.CreateOrUpdate(%s): end", *sg.Name)
		done, err := az.processHTTPRetryResponse(service, "CreateOrUpdateSecurityGroup", resp, err)
		if done && err == nil {
			// Invalidate the cache right after updating
			az.nsgCache.Delete(*sg.Name)
		}
		return done, err
	})
}

// CreateOrUpdateLB invokes az.LoadBalancerClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) CreateOrUpdateLB(service *v1.Service, lb network.LoadBalancer) error {
	if az.Config.shouldOmitCloudProviderBackoff() {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		resp, err := az.LoadBalancerClient.CreateOrUpdate(ctx, az.ResourceGroup, *lb.Name, lb)
		klog.V(10).Infof("LoadBalancerClient.CreateOrUpdate(%s): end", *lb.Name)
		if err == nil {
			if isSuccessHTTPResponse(resp) {
				// Invalidate the cache right after updating
				az.lbCache.Delete(*lb.Name)
			} else if resp != nil {
				return fmt.Errorf("HTTP response %q", resp.Status)
			}
		}
		return err
	}

	return az.createOrUpdateLBWithRetry(service, lb)
}

// createOrUpdateLBWithRetry invokes az.LoadBalancerClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) createOrUpdateLBWithRetry(service *v1.Service, lb network.LoadBalancer) error {
	return wait.ExponentialBackoff(az.requestBackoff(), func() (bool, error) {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		resp, err := az.LoadBalancerClient.CreateOrUpdate(ctx, az.ResourceGroup, *lb.Name, lb)
		klog.V(10).Infof("LoadBalancerClient.CreateOrUpdate(%s): end", *lb.Name)
		done, err := az.processHTTPRetryResponse(service, "CreateOrUpdateLoadBalancer", resp, err)
		if done && err == nil {
			// Invalidate the cache right after updating
			az.lbCache.Delete(*lb.Name)
		}
		return done, err
	})
}

// ListLB invokes az.LoadBalancerClient.List with exponential backoff retry
func (az *Cloud) ListLB(service *v1.Service) ([]network.LoadBalancer, error) {
	if az.Config.shouldOmitCloudProviderBackoff() {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		allLBs, err := az.LoadBalancerClient.List(ctx, az.ResourceGroup)
		if err != nil {
			az.Event(service, v1.EventTypeWarning, "ListLoadBalancers", err.Error())
			klog.Errorf("LoadBalancerClient.List(%v) failure with err=%v", az.ResourceGroup, err)
			return nil, err
		}
		klog.V(2).Infof("LoadBalancerClient.List(%v) success", az.ResourceGroup)
		return allLBs, nil
	}

	return az.listLBWithRetry(service)
}

// listLBWithRetry invokes az.LoadBalancerClient.List with exponential backoff retry
func (az *Cloud) listLBWithRetry(service *v1.Service) ([]network.LoadBalancer, error) {
	var allLBs []network.LoadBalancer

	err := wait.ExponentialBackoff(az.requestBackoff(), func() (bool, error) {
		var retryErr error
		ctx, cancel := getContextWithCancel()
		defer cancel()

		allLBs, retryErr = az.LoadBalancerClient.List(ctx, az.ResourceGroup)
		if retryErr != nil {
			az.Event(service, v1.EventTypeWarning, "ListLoadBalancers", retryErr.Error())
			klog.Errorf("LoadBalancerClient.List(%v) - backoff: failure, will retry,err=%v",
				az.ResourceGroup,
				retryErr)
			return false, retryErr
		}
		klog.V(2).Infof("LoadBalancerClient.List(%v) - backoff: success", az.ResourceGroup)
		return true, nil
	})
	if err != nil {
		return nil, err
	}

	return allLBs, nil
}

// ListPIP list the PIP resources in the given resource group
func (az *Cloud) ListPIP(service *v1.Service, pipResourceGroup string) ([]network.PublicIPAddress, error) {
	if az.Config.shouldOmitCloudProviderBackoff() {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		allPIPs, err := az.PublicIPAddressesClient.List(ctx, pipResourceGroup)
		if err != nil {
			az.Event(service, v1.EventTypeWarning, "ListPublicIPs", err.Error())
			klog.Errorf("PublicIPAddressesClient.List(%v) failure with err=%v", pipResourceGroup, err)
			return nil, err
		}
		klog.V(2).Infof("PublicIPAddressesClient.List(%v) success", pipResourceGroup)
		return allPIPs, nil
	}

	return az.listPIPWithRetry(service, pipResourceGroup)
}

// listPIPWithRetry list the PIP resources in the given resource group
func (az *Cloud) listPIPWithRetry(service *v1.Service, pipResourceGroup string) ([]network.PublicIPAddress, error) {
	var allPIPs []network.PublicIPAddress

	err := wait.ExponentialBackoff(az.requestBackoff(), func() (bool, error) {
		var retryErr error
		ctx, cancel := getContextWithCancel()
		defer cancel()

		allPIPs, retryErr = az.PublicIPAddressesClient.List(ctx, pipResourceGroup)
		if retryErr != nil {
			az.Event(service, v1.EventTypeWarning, "ListPublicIPs", retryErr.Error())
			klog.Errorf("PublicIPAddressesClient.List(%v) - backoff: failure, will retry,err=%v",
				pipResourceGroup,
				retryErr)
			return false, retryErr
		}
		klog.V(2).Infof("PublicIPAddressesClient.List(%v) - backoff: success", pipResourceGroup)
		return true, nil
	})
	if err != nil {
		return nil, err
	}

	return allPIPs, nil
}

// CreateOrUpdatePIP invokes az.PublicIPAddressesClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) CreateOrUpdatePIP(service *v1.Service, pipResourceGroup string, pip network.PublicIPAddress) error {
	if az.Config.shouldOmitCloudProviderBackoff() {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		resp, err := az.PublicIPAddressesClient.CreateOrUpdate(ctx, pipResourceGroup, *pip.Name, pip)
		klog.V(10).Infof("PublicIPAddressesClient.CreateOrUpdate(%s, %s): end", pipResourceGroup, *pip.Name)
		return az.processHTTPResponse(service, "CreateOrUpdatePublicIPAddress", resp, err)
	}

	return az.createOrUpdatePIPWithRetry(service, pipResourceGroup, pip)
}

// createOrUpdatePIPWithRetry invokes az.PublicIPAddressesClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) createOrUpdatePIPWithRetry(service *v1.Service, pipResourceGroup string, pip network.PublicIPAddress) error {
	return wait.ExponentialBackoff(az.requestBackoff(), func() (bool, error) {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		resp, err := az.PublicIPAddressesClient.CreateOrUpdate(ctx, pipResourceGroup, *pip.Name, pip)
		klog.V(10).Infof("PublicIPAddressesClient.CreateOrUpdate(%s, %s): end", pipResourceGroup, *pip.Name)
		return az.processHTTPRetryResponse(service, "CreateOrUpdatePublicIPAddress", resp, err)
	})
}

// CreateOrUpdateInterface invokes az.PublicIPAddressesClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) CreateOrUpdateInterface(service *v1.Service, nic network.Interface) error {
	if az.Config.shouldOmitCloudProviderBackoff() {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		resp, err := az.InterfacesClient.CreateOrUpdate(ctx, az.ResourceGroup, *nic.Name, nic)
		klog.V(10).Infof("InterfacesClient.CreateOrUpdate(%s): end", *nic.Name)
		return az.processHTTPResponse(service, "CreateOrUpdateInterface", resp, err)
	}

	return az.createOrUpdateInterfaceWithRetry(service, nic)
}

// createOrUpdateInterfaceWithRetry invokes az.PublicIPAddressesClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) createOrUpdateInterfaceWithRetry(service *v1.Service, nic network.Interface) error {
	return wait.ExponentialBackoff(az.requestBackoff(), func() (bool, error) {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		resp, err := az.InterfacesClient.CreateOrUpdate(ctx, az.ResourceGroup, *nic.Name, nic)
		klog.V(10).Infof("InterfacesClient.CreateOrUpdate(%s): end", *nic.Name)
		return az.processHTTPRetryResponse(service, "CreateOrUpdateInterface", resp, err)
	})
}

// DeletePublicIP invokes az.PublicIPAddressesClient.Delete with exponential backoff retry
func (az *Cloud) DeletePublicIP(service *v1.Service, pipResourceGroup string, pipName string) error {
	if az.Config.shouldOmitCloudProviderBackoff() {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		resp, err := az.PublicIPAddressesClient.Delete(ctx, pipResourceGroup, pipName)
		return az.processHTTPResponse(service, "DeletePublicIPAddress", resp, err)
	}

	return az.deletePublicIPWithRetry(service, pipResourceGroup, pipName)
}

// deletePublicIPWithRetry invokes az.PublicIPAddressesClient.Delete with exponential backoff retry
func (az *Cloud) deletePublicIPWithRetry(service *v1.Service, pipResourceGroup string, pipName string) error {
	return wait.ExponentialBackoff(az.requestBackoff(), func() (bool, error) {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		resp, err := az.PublicIPAddressesClient.Delete(ctx, pipResourceGroup, pipName)
		return az.processHTTPRetryResponse(service, "DeletePublicIPAddress", resp, err)
	})
}

// DeleteLB invokes az.LoadBalancerClient.Delete with exponential backoff retry
func (az *Cloud) DeleteLB(service *v1.Service, lbName string) error {
	if az.Config.shouldOmitCloudProviderBackoff() {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		resp, err := az.LoadBalancerClient.Delete(ctx, az.ResourceGroup, lbName)
		if err == nil {
			if isSuccessHTTPResponse(resp) {
				// Invalidate the cache right after updating
				az.lbCache.Delete(lbName)
			} else if resp != nil {
				return fmt.Errorf("HTTP response %q", resp.Status)
			}
		}
		return err
	}

	return az.deleteLBWithRetry(service, lbName)
}

// deleteLBWithRetry invokes az.LoadBalancerClient.Delete with exponential backoff retry
func (az *Cloud) deleteLBWithRetry(service *v1.Service, lbName string) error {
	return wait.ExponentialBackoff(az.requestBackoff(), func() (bool, error) {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		resp, err := az.LoadBalancerClient.Delete(ctx, az.ResourceGroup, lbName)
		done, err := az.processHTTPRetryResponse(service, "DeleteLoadBalancer", resp, err)
		if done && err == nil {
			// Invalidate the cache right after deleting
			az.lbCache.Delete(lbName)
		}
		return done, err
	})
}

// CreateOrUpdateRouteTable invokes az.RouteTablesClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) CreateOrUpdateRouteTable(routeTable network.RouteTable) error {
	if az.Config.shouldOmitCloudProviderBackoff() {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		resp, err := az.RouteTablesClient.CreateOrUpdate(ctx, az.ResourceGroup, az.RouteTableName, routeTable)
		return az.processHTTPResponse(nil, "", resp, err)
	}

	return az.createOrUpdateRouteTableWithRetry(routeTable)
}

// createOrUpdateRouteTableWithRetry invokes az.RouteTablesClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) createOrUpdateRouteTableWithRetry(routeTable network.RouteTable) error {
	return wait.ExponentialBackoff(az.requestBackoff(), func() (bool, error) {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		resp, err := az.RouteTablesClient.CreateOrUpdate(ctx, az.ResourceGroup, az.RouteTableName, routeTable)
		return az.processHTTPRetryResponse(nil, "", resp, err)
	})
}

// CreateOrUpdateRoute invokes az.RoutesClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) CreateOrUpdateRoute(route network.Route) error {
	if az.Config.shouldOmitCloudProviderBackoff() {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		resp, err := az.RoutesClient.CreateOrUpdate(ctx, az.ResourceGroup, az.RouteTableName, *route.Name, route)
		klog.V(10).Infof("RoutesClient.CreateOrUpdate(%s): end", *route.Name)
		return az.processHTTPResponse(nil, "", resp, err)
	}

	return az.createOrUpdateRouteWithRetry(route)
}

// createOrUpdateRouteWithRetry invokes az.RoutesClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) createOrUpdateRouteWithRetry(route network.Route) error {
	return wait.ExponentialBackoff(az.requestBackoff(), func() (bool, error) {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		resp, err := az.RoutesClient.CreateOrUpdate(ctx, az.ResourceGroup, az.RouteTableName, *route.Name, route)
		klog.V(10).Infof("RoutesClient.CreateOrUpdate(%s): end", *route.Name)
		return az.processHTTPRetryResponse(nil, "", resp, err)
	})
}

// DeleteRouteWithName invokes az.RoutesClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) DeleteRouteWithName(routeName string) error {
	if az.Config.shouldOmitCloudProviderBackoff() {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		resp, err := az.RoutesClient.Delete(ctx, az.ResourceGroup, az.RouteTableName, routeName)
		klog.V(10).Infof("RoutesClient.Delete(%s,%s): end", az.RouteTableName, routeName)
		return az.processHTTPResponse(nil, "", resp, err)
	}

	return az.deleteRouteWithRetry(routeName)
}

// deleteRouteWithRetry invokes az.RoutesClient.Delete with exponential backoff retry
func (az *Cloud) deleteRouteWithRetry(routeName string) error {
	return wait.ExponentialBackoff(az.requestBackoff(), func() (bool, error) {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		resp, err := az.RoutesClient.Delete(ctx, az.ResourceGroup, az.RouteTableName, routeName)
		klog.V(10).Infof("RoutesClient.Delete(%s,%s): end", az.RouteTableName, routeName)
		return az.processHTTPRetryResponse(nil, "", resp, err)
	})
}

// CreateOrUpdateVMWithRetry invokes az.VirtualMachinesClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) CreateOrUpdateVMWithRetry(resourceGroup, vmName string, newVM compute.VirtualMachine) error {
	return wait.ExponentialBackoff(az.requestBackoff(), func() (bool, error) {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		resp, err := az.VirtualMachinesClient.CreateOrUpdate(ctx, resourceGroup, vmName, newVM)
		klog.V(10).Infof("VirtualMachinesClient.CreateOrUpdate(%s): end", vmName)
		return az.processHTTPRetryResponse(nil, "", resp, err)
	})
}

// UpdateVmssVMWithRetry invokes az.VirtualMachineScaleSetVMsClient.Update with exponential backoff retry
func (az *Cloud) UpdateVmssVMWithRetry(resourceGroupName string, VMScaleSetName string, instanceID string, parameters compute.VirtualMachineScaleSetVM) error {
	return wait.ExponentialBackoff(az.requestBackoff(), func() (bool, error) {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		resp, err := az.VirtualMachineScaleSetVMsClient.Update(ctx, resourceGroupName, VMScaleSetName, instanceID, parameters)
		klog.V(10).Infof("VirtualMachinesClient.CreateOrUpdate(%s,%s): end", VMScaleSetName, instanceID)
		return az.processHTTPRetryResponse(nil, "", resp, err)
	})
}

// isSuccessHTTPResponse determines if the response from an HTTP request suggests success
func isSuccessHTTPResponse(resp *http.Response) bool {
	if resp == nil {
		return false
	}

	// HTTP 2xx suggests a successful response
	if 199 < resp.StatusCode && resp.StatusCode < 300 {
		return true
	}

	return false
}

func shouldRetryHTTPRequest(resp *http.Response, err error) bool {
	if err != nil {
		return true
	}

	if resp != nil {
		// HTTP 4xx or 5xx suggests we should retry
		if 399 < resp.StatusCode && resp.StatusCode < 600 {
			return true
		}
	}

	return false
}

func (az *Cloud) processHTTPRetryResponse(service *v1.Service, reason string, resp *http.Response, err error) (bool, error) {
	if resp != nil && isSuccessHTTPResponse(resp) {
		// HTTP 2xx suggests a successful response
		return true, nil
	}

	if shouldRetryHTTPRequest(resp, err) {
		if err != nil {
			az.Event(service, v1.EventTypeWarning, reason, err.Error())
			klog.Errorf("processHTTPRetryResponse: backoff failure, will retry, err=%v", err)
		} else {
			az.Event(service, v1.EventTypeWarning, reason, fmt.Sprintf("Azure HTTP response %d", resp.StatusCode))
			klog.Errorf("processHTTPRetryResponse: backoff failure, will retry, HTTP response=%d", resp.StatusCode)
		}

		// suppress the error object so that backoff process continues
		return false, nil
	}

	// Fall-through: stop periodic backoff
	return true, nil
}

func (az *Cloud) processHTTPResponse(service *v1.Service, reason string, resp *http.Response, err error) error {
	if isSuccessHTTPResponse(resp) {
		// HTTP 2xx suggests a successful response
		return nil
	}

	if err != nil {
		az.Event(service, v1.EventTypeWarning, reason, err.Error())
		klog.Errorf("processHTTPRetryResponse failure with err: %v", err)
	} else if resp != nil {
		az.Event(service, v1.EventTypeWarning, reason, fmt.Sprintf("Azure HTTP response %d", resp.StatusCode))
		klog.Errorf("processHTTPRetryResponse failure with HTTP response %q", resp.Status)
	}

	return err
}

func (cfg *Config) shouldOmitCloudProviderBackoff() bool {
	return cfg.CloudProviderBackoffMode == backoffModeV2
}

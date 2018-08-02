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
	"net/http"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2018-04-01/compute"
	"github.com/Azure/azure-sdk-for-go/services/network/mgmt/2017-09-01/network"
	"github.com/Azure/go-autorest/autorest"
	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/cloudprovider"
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
			glog.Errorf("GetVirtualMachineWithRetry(%s): backoff failure, will retry, err=%v", name, retryErr)
			return false, nil
		}
		glog.V(2).Infof("GetVirtualMachineWithRetry(%s): backoff success", name)
		return true, nil
	})
	if err == wait.ErrWaitTimeout {
		err = retryErr
	}

	return machine, err
}

// VirtualMachineClientListWithRetry invokes az.VirtualMachinesClient.List with exponential backoff retry
func (az *Cloud) VirtualMachineClientListWithRetry() ([]compute.VirtualMachine, error) {
	allNodes := []compute.VirtualMachine{}
	err := wait.ExponentialBackoff(az.requestBackoff(), func() (bool, error) {
		var retryErr error
		ctx, cancel := getContextWithCancel()
		defer cancel()
		allNodes, retryErr = az.VirtualMachinesClient.List(ctx, az.ResourceGroup)
		if retryErr != nil {
			glog.Errorf("VirtualMachinesClient.List(%v) - backoff: failure, will retry,err=%v",
				az.ResourceGroup,
				retryErr)
			return false, retryErr
		}
		glog.V(2).Infof("VirtualMachinesClient.List(%v) - backoff: success", az.ResourceGroup)
		return true, nil
	})
	if err != nil {
		return nil, err
	}

	return allNodes, err
}

// GetIPForMachineWithRetry invokes az.getIPForMachine with exponential backoff retry
func (az *Cloud) GetIPForMachineWithRetry(name types.NodeName) (string, string, error) {
	var ip, publicIP string
	err := wait.ExponentialBackoff(az.requestBackoff(), func() (bool, error) {
		var retryErr error
		ip, publicIP, retryErr = az.getIPForMachine(name)
		if retryErr != nil {
			glog.Errorf("GetIPForMachineWithRetry(%s): backoff failure, will retry,err=%v", name, retryErr)
			return false, nil
		}
		glog.V(2).Infof("GetIPForMachineWithRetry(%s): backoff success", name)
		return true, nil
	})
	return ip, publicIP, err
}

// CreateOrUpdateSGWithRetry invokes az.SecurityGroupsClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) CreateOrUpdateSGWithRetry(sg network.SecurityGroup) error {
	return wait.ExponentialBackoff(az.requestBackoff(), func() (bool, error) {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		resp, err := az.SecurityGroupsClient.CreateOrUpdate(ctx, az.ResourceGroup, *sg.Name, sg)
		glog.V(10).Infof("SecurityGroupsClient.CreateOrUpdate(%s): end", *sg.Name)
		done, err := processHTTPRetryResponse(resp, err)
		if done && err == nil {
			// Invalidate the cache right after updating
			az.nsgCache.Delete(*sg.Name)
		}
		return done, err
	})
}

// CreateOrUpdateLBWithRetry invokes az.LoadBalancerClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) CreateOrUpdateLBWithRetry(lb network.LoadBalancer) error {
	return wait.ExponentialBackoff(az.requestBackoff(), func() (bool, error) {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		resp, err := az.LoadBalancerClient.CreateOrUpdate(ctx, az.ResourceGroup, *lb.Name, lb)
		glog.V(10).Infof("LoadBalancerClient.CreateOrUpdate(%s): end", *lb.Name)
		done, err := processHTTPRetryResponse(resp, err)
		if done && err == nil {
			// Invalidate the cache right after updating
			az.lbCache.Delete(*lb.Name)
		}
		return done, err
	})
}

// ListLBWithRetry invokes az.LoadBalancerClient.List with exponential backoff retry
func (az *Cloud) ListLBWithRetry() ([]network.LoadBalancer, error) {
	var allLBs []network.LoadBalancer

	err := wait.ExponentialBackoff(az.requestBackoff(), func() (bool, error) {
		var retryErr error
		ctx, cancel := getContextWithCancel()
		defer cancel()

		allLBs, retryErr = az.LoadBalancerClient.List(ctx, az.ResourceGroup)
		if retryErr != nil {
			glog.Errorf("LoadBalancerClient.List(%v) - backoff: failure, will retry,err=%v",
				az.ResourceGroup,
				retryErr)
			return false, retryErr
		}
		glog.V(2).Infof("LoadBalancerClient.List(%v) - backoff: success", az.ResourceGroup)
		return true, nil
	})
	if err != nil {
		return nil, err
	}

	return allLBs, nil
}

// ListPIPWithRetry list the PIP resources in the given resource group
func (az *Cloud) ListPIPWithRetry(pipResourceGroup string) ([]network.PublicIPAddress, error) {
	var allPIPs []network.PublicIPAddress

	err := wait.ExponentialBackoff(az.requestBackoff(), func() (bool, error) {
		var retryErr error
		ctx, cancel := getContextWithCancel()
		defer cancel()

		allPIPs, retryErr = az.PublicIPAddressesClient.List(ctx, pipResourceGroup)
		if retryErr != nil {
			glog.Errorf("PublicIPAddressesClient.List(%v) - backoff: failure, will retry,err=%v",
				pipResourceGroup,
				retryErr)
			return false, retryErr
		}
		glog.V(2).Infof("PublicIPAddressesClient.List(%v) - backoff: success", pipResourceGroup)
		return true, nil
	})
	if err != nil {
		return nil, err
	}

	return allPIPs, nil
}

// CreateOrUpdatePIPWithRetry invokes az.PublicIPAddressesClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) CreateOrUpdatePIPWithRetry(pipResourceGroup string, pip network.PublicIPAddress) error {
	return wait.ExponentialBackoff(az.requestBackoff(), func() (bool, error) {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		resp, err := az.PublicIPAddressesClient.CreateOrUpdate(ctx, pipResourceGroup, *pip.Name, pip)
		glog.V(10).Infof("PublicIPAddressesClient.CreateOrUpdate(%s, %s): end", pipResourceGroup, *pip.Name)
		return processHTTPRetryResponse(resp, err)
	})
}

// CreateOrUpdateInterfaceWithRetry invokes az.PublicIPAddressesClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) CreateOrUpdateInterfaceWithRetry(nic network.Interface) error {
	return wait.ExponentialBackoff(az.requestBackoff(), func() (bool, error) {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		resp, err := az.InterfacesClient.CreateOrUpdate(ctx, az.ResourceGroup, *nic.Name, nic)
		glog.V(10).Infof("InterfacesClient.CreateOrUpdate(%s): end", *nic.Name)
		return processHTTPRetryResponse(resp, err)
	})
}

// DeletePublicIPWithRetry invokes az.PublicIPAddressesClient.Delete with exponential backoff retry
func (az *Cloud) DeletePublicIPWithRetry(pipResourceGroup string, pipName string) error {
	return wait.ExponentialBackoff(az.requestBackoff(), func() (bool, error) {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		resp, err := az.PublicIPAddressesClient.Delete(ctx, pipResourceGroup, pipName)
		return processHTTPRetryResponse(resp, err)
	})
}

// DeleteLBWithRetry invokes az.LoadBalancerClient.Delete with exponential backoff retry
func (az *Cloud) DeleteLBWithRetry(lbName string) error {
	return wait.ExponentialBackoff(az.requestBackoff(), func() (bool, error) {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		resp, err := az.LoadBalancerClient.Delete(ctx, az.ResourceGroup, lbName)
		done, err := processHTTPRetryResponse(resp, err)
		if done && err == nil {
			// Invalidate the cache right after deleting
			az.lbCache.Delete(lbName)
		}
		return done, err
	})
}

// CreateOrUpdateRouteTableWithRetry invokes az.RouteTablesClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) CreateOrUpdateRouteTableWithRetry(routeTable network.RouteTable) error {
	return wait.ExponentialBackoff(az.requestBackoff(), func() (bool, error) {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		resp, err := az.RouteTablesClient.CreateOrUpdate(ctx, az.ResourceGroup, az.RouteTableName, routeTable)
		return processHTTPRetryResponse(resp, err)
	})
}

// CreateOrUpdateRouteWithRetry invokes az.RoutesClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) CreateOrUpdateRouteWithRetry(route network.Route) error {
	return wait.ExponentialBackoff(az.requestBackoff(), func() (bool, error) {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		resp, err := az.RoutesClient.CreateOrUpdate(ctx, az.ResourceGroup, az.RouteTableName, *route.Name, route)
		glog.V(10).Infof("RoutesClient.CreateOrUpdate(%s): end", *route.Name)
		return processHTTPRetryResponse(resp, err)
	})
}

// DeleteRouteWithRetry invokes az.RoutesClient.Delete with exponential backoff retry
func (az *Cloud) DeleteRouteWithRetry(routeName string) error {
	return wait.ExponentialBackoff(az.requestBackoff(), func() (bool, error) {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		resp, err := az.RoutesClient.Delete(ctx, az.ResourceGroup, az.RouteTableName, routeName)
		glog.V(10).Infof("RoutesClient.Delete(%s): end", az.RouteTableName)
		return processHTTPRetryResponse(resp, err)
	})
}

// CreateOrUpdateVMWithRetry invokes az.VirtualMachinesClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) CreateOrUpdateVMWithRetry(vmName string, newVM compute.VirtualMachine) error {
	return wait.ExponentialBackoff(az.requestBackoff(), func() (bool, error) {
		ctx, cancel := getContextWithCancel()
		defer cancel()

		resp, err := az.VirtualMachinesClient.CreateOrUpdate(ctx, az.ResourceGroup, vmName, newVM)
		glog.V(10).Infof("VirtualMachinesClient.CreateOrUpdate(%s): end", vmName)
		return processHTTPRetryResponse(resp, err)
	})
}

// UpdateVmssVMWithRetry invokes az.VirtualMachineScaleSetVMsClient.Update with exponential backoff retry
func (az *Cloud) UpdateVmssVMWithRetry(ctx context.Context, resourceGroupName string, VMScaleSetName string, instanceID string, parameters compute.VirtualMachineScaleSetVM) error {
	return wait.ExponentialBackoff(az.requestBackoff(), func() (bool, error) {
		resp, err := az.VirtualMachineScaleSetVMsClient.Update(ctx, resourceGroupName, VMScaleSetName, instanceID, parameters)
		glog.V(10).Infof("VirtualMachinesClient.CreateOrUpdate(%s,%s): end", VMScaleSetName, instanceID)
		return processHTTPRetryResponse(resp, err)
	})
}

// A wait.ConditionFunc function to deal with common HTTP backoff response conditions
func processRetryResponse(resp autorest.Response, err error) (bool, error) {
	if isSuccessHTTPResponse(resp) {
		glog.V(2).Infof("processRetryResponse: backoff success, HTTP response=%d", resp.StatusCode)
		return true, nil
	}
	if shouldRetryAPIRequest(resp, err) {
		glog.Errorf("processRetryResponse: backoff failure, will retry, HTTP response=%d, err=%v", resp.StatusCode, err)
		// suppress the error object so that backoff process continues
		return false, nil
	}
	// Fall-through: stop periodic backoff
	return true, nil
}

// shouldRetryAPIRequest determines if the response from an HTTP request suggests periodic retry behavior
func shouldRetryAPIRequest(resp autorest.Response, err error) bool {
	if err != nil {
		return true
	}
	// HTTP 4xx or 5xx suggests we should retry
	if 399 < resp.StatusCode && resp.StatusCode < 600 {
		return true
	}
	return false
}

// isSuccessHTTPResponse determines if the response from an HTTP request suggests success
func isSuccessHTTPResponse(resp autorest.Response) bool {
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

func processHTTPRetryResponse(resp *http.Response, err error) (bool, error) {
	if resp != nil {
		// HTTP 2xx suggests a successful response
		if 199 < resp.StatusCode && resp.StatusCode < 300 {
			return true, nil
		}
	}

	if shouldRetryHTTPRequest(resp, err) {
		glog.Errorf("processHTTPRetryResponse: backoff failure, will retry, HTTP response=%d, err=%v", resp.StatusCode, err)
		// suppress the error object so that backoff process continues
		return false, nil
	}

	// Fall-through: stop periodic backoff
	return true, nil
}

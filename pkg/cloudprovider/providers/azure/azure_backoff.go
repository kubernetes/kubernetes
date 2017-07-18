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
	"k8s.io/apimachinery/pkg/util/wait"

	"github.com/Azure/azure-sdk-for-go/arm/compute"
	"github.com/Azure/azure-sdk-for-go/arm/network"
	"github.com/Azure/go-autorest/autorest"
	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
)

// GetVirtualMachineWithRetry invokes az.getVirtualMachine with exponential backoff retry
func (az *Cloud) GetVirtualMachineWithRetry(name types.NodeName) (compute.VirtualMachine, bool, error) {
	var machine compute.VirtualMachine
	var exists bool
	err := wait.ExponentialBackoff(az.resourceRequestBackoff, func() (bool, error) {
		var retryErr error
		machine, exists, retryErr = az.getVirtualMachine(name)
		if retryErr != nil {
			glog.Errorf("backoff: failure, will retry,err=%v", retryErr)
			return false, nil
		}
		glog.V(2).Infof("backoff: success")
		return true, nil
	})
	return machine, exists, err
}

// CreateOrUpdateSGWithRetry invokes az.SecurityGroupsClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) CreateOrUpdateSGWithRetry(sg network.SecurityGroup) error {
	return wait.ExponentialBackoff(az.resourceRequestBackoff, func() (bool, error) {
		az.operationPollRateLimiter.Accept()
		respChan, errChan := az.SecurityGroupsClient.CreateOrUpdate(az.ResourceGroup, *sg.Name, sg, nil)
		resp := <-respChan
		err := <-errChan
		return processRetryResponse(resp.Response, err)
	})
}

// CreateOrUpdateLBWithRetry invokes az.LoadBalancerClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) CreateOrUpdateLBWithRetry(lb network.LoadBalancer) error {
	return wait.ExponentialBackoff(az.resourceRequestBackoff, func() (bool, error) {
		az.operationPollRateLimiter.Accept()
		respChan, errChan := az.LoadBalancerClient.CreateOrUpdate(az.ResourceGroup, *lb.Name, lb, nil)
		resp := <-respChan
		err := <-errChan
		return processRetryResponse(resp.Response, err)
	})
}

// CreateOrUpdatePIPWithRetry invokes az.PublicIPAddressesClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) CreateOrUpdatePIPWithRetry(pip network.PublicIPAddress) error {
	return wait.ExponentialBackoff(az.resourceRequestBackoff, func() (bool, error) {
		az.operationPollRateLimiter.Accept()
		respChan, errChan := az.PublicIPAddressesClient.CreateOrUpdate(az.ResourceGroup, *pip.Name, pip, nil)
		resp := <-respChan
		err := <-errChan
		return processRetryResponse(resp.Response, err)
	})
}

// CreateOrUpdateInterfaceWithRetry invokes az.PublicIPAddressesClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) CreateOrUpdateInterfaceWithRetry(nic network.Interface) error {
	return wait.ExponentialBackoff(az.resourceRequestBackoff, func() (bool, error) {
		az.operationPollRateLimiter.Accept()
		respChan, errChan := az.InterfacesClient.CreateOrUpdate(az.ResourceGroup, *nic.Name, nic, nil)
		resp := <-respChan
		err := <-errChan
		return processRetryResponse(resp.Response, err)
	})
}

// DeletePublicIPWithRetry invokes az.PublicIPAddressesClient.Delete with exponential backoff retry
func (az *Cloud) DeletePublicIPWithRetry(pipName string) error {
	return wait.ExponentialBackoff(az.resourceRequestBackoff, func() (bool, error) {
		az.operationPollRateLimiter.Accept()
		respChan, errChan := az.PublicIPAddressesClient.Delete(az.ResourceGroup, pipName, nil)
		resp := <-respChan
		err := <-errChan
		return processRetryResponse(resp, err)
	})
}

// DeleteLBWithRetry invokes az.LoadBalancerClient.Delete with exponential backoff retry
func (az *Cloud) DeleteLBWithRetry(lbName string) error {
	return wait.ExponentialBackoff(az.resourceRequestBackoff, func() (bool, error) {
		az.operationPollRateLimiter.Accept()
		respChan, errChan := az.LoadBalancerClient.Delete(az.ResourceGroup, lbName, nil)
		resp := <-respChan
		err := <-errChan
		return processRetryResponse(resp, err)
	})
}

// CreateOrUpdateRouteTableWithRetry invokes az.RouteTablesClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) CreateOrUpdateRouteTableWithRetry(routeTable network.RouteTable) error {
	return wait.ExponentialBackoff(az.resourceRequestBackoff, func() (bool, error) {
		az.operationPollRateLimiter.Accept()
		respChan, errChan := az.RouteTablesClient.CreateOrUpdate(az.ResourceGroup, az.RouteTableName, routeTable, nil)
		resp := <-respChan
		err := <-errChan
		return processRetryResponse(resp.Response, err)
	})
}

// CreateOrUpdateRouteWithRetry invokes az.RoutesClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) CreateOrUpdateRouteWithRetry(route network.Route) error {
	return wait.ExponentialBackoff(az.resourceRequestBackoff, func() (bool, error) {
		az.operationPollRateLimiter.Accept()
		respChan, errChan := az.RoutesClient.CreateOrUpdate(az.ResourceGroup, az.RouteTableName, *route.Name, route, nil)
		resp := <-respChan
		err := <-errChan
		return processRetryResponse(resp.Response, err)
	})
}

// DeleteRouteWithRetry invokes az.RoutesClient.Delete with exponential backoff retry
func (az *Cloud) DeleteRouteWithRetry(routeName string) error {
	return wait.ExponentialBackoff(az.resourceRequestBackoff, func() (bool, error) {
		az.operationPollRateLimiter.Accept()
		respChan, errChan := az.RoutesClient.Delete(az.ResourceGroup, az.RouteTableName, routeName, nil)
		resp := <-respChan
		err := <-errChan
		return processRetryResponse(resp, err)
	})
}

// CreateOrUpdateVMWithRetry invokes az.VirtualMachinesClient.CreateOrUpdate with exponential backoff retry
func (az *Cloud) CreateOrUpdateVMWithRetry(vmName string, newVM compute.VirtualMachine) error {
	return wait.ExponentialBackoff(az.resourceRequestBackoff, func() (bool, error) {
		az.operationPollRateLimiter.Accept()
		respChan, errChan := az.VirtualMachinesClient.CreateOrUpdate(az.ResourceGroup, vmName, newVM, nil)
		resp := <-respChan
		err := <-errChan
		return processRetryResponse(resp.Response, err)
	})
}

// A wait.ConditionFunc function to deal with common HTTP backoff response conditions
func processRetryResponse(resp autorest.Response, err error) (bool, error) {
	if isSuccessHTTPResponse(resp) {
		glog.V(2).Infof("backoff: success, HTTP response=%d", resp.StatusCode)
		return true, nil
	}
	if shouldRetryAPIRequest(resp, err) {
		glog.Errorf("backoff: failure, will retry, HTTP response=%d, err=%v", resp.StatusCode, err)
		// suppress the error object so that backoff process continues
		return false, nil
	}
	// Fall-through: stop periodic backoff, return error object from most recent request
	return true, err
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

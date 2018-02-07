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
	"net/http"
	"time"

	"github.com/Azure/azure-sdk-for-go/arm/compute"
	"github.com/Azure/azure-sdk-for-go/arm/network"
	"github.com/Azure/go-autorest/autorest"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

var (
	vmCacheTTL = time.Minute
)

// checkExistsFromError inspects an error and returns a true if err is nil,
// false if error is an autorest.Error with StatusCode=404 and will return the
// error back if error is another status code or another type of error.
func checkResourceExistsFromError(err error) (bool, error) {
	if err == nil {
		return true, nil
	}
	v, ok := err.(autorest.DetailedError)
	if !ok {
		return false, err
	}
	if v.StatusCode == http.StatusNotFound {
		return false, nil
	}
	return false, v
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
	var realErr error

	routeTable, err = az.RouteTablesClient.Get(az.ResourceGroup, az.RouteTableName, "")
	exists, realErr = checkResourceExistsFromError(err)
	if realErr != nil {
		return routeTable, false, realErr
	}

	if !exists {
		return routeTable, false, nil
	}

	return routeTable, exists, err
}

func (az *Cloud) getPublicIPAddress(pipResourceGroup string, pipName string) (pip network.PublicIPAddress, exists bool, err error) {
	resourceGroup := az.ResourceGroup
	if pipResourceGroup != "" {
		resourceGroup = pipResourceGroup
	}

	var realErr error
	pip, err = az.PublicIPAddressesClient.Get(resourceGroup, pipName, "")
	exists, realErr = checkResourceExistsFromError(err)
	if realErr != nil {
		return pip, false, realErr
	}

	if !exists {
		return pip, false, nil
	}

	return pip, exists, err
}

func (az *Cloud) getSubnet(virtualNetworkName string, subnetName string) (subnet network.Subnet, exists bool, err error) {
	var realErr error
	var rg string

	if len(az.VnetResourceGroup) > 0 {
		rg = az.VnetResourceGroup
	} else {
		rg = az.ResourceGroup
	}

	subnet, err = az.SubnetsClient.Get(rg, virtualNetworkName, subnetName, "")
	exists, realErr = checkResourceExistsFromError(err)
	if realErr != nil {
		return subnet, false, realErr
	}

	if !exists {
		return subnet, false, nil
	}

	return subnet, exists, err
}

func (az *Cloud) getAzureLoadBalancer(name string) (lb network.LoadBalancer, exists bool, err error) {
	var realErr error

	lb, err = az.LoadBalancerClient.Get(az.ResourceGroup, name, "")
	exists, realErr = checkResourceExistsFromError(err)
	if realErr != nil {
		return lb, false, realErr
	}

	if !exists {
		return lb, false, nil
	}

	return lb, exists, err
}

func (az *Cloud) newVMCache() (*timedCache, error) {
	getter := func(key string) (interface{}, error) {
		vm, err := az.VirtualMachinesClient.Get(az.ResourceGroup, key, compute.InstanceView)
		exists, realErr := checkResourceExistsFromError(err)
		if realErr != nil {
			return nil, realErr
		}

		if !exists {
			return nil, nil
		}

		return &vm, nil
	}

	return newTimedcache(vmCacheTTL, getter)
}

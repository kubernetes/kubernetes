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

	"github.com/Azure/azure-sdk-for-go/arm/compute"
	"github.com/Azure/azure-sdk-for-go/arm/network"
	"github.com/Azure/go-autorest/autorest"
	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
)

// checkExistsFromError inspects an error and returns a true if err is nil,
// false if error is an autorest.Error with StatusCode=404 and will return the
// error back if error is another status code or another type of error.
func checkResourceExistsFromError(err error) (bool, error) {
	if err == nil {
		return true, nil
	}
	v, ok := err.(autorest.DetailedError)
	if ok && v.StatusCode == http.StatusNotFound {
		return false, nil
	}
	return false, v
}

func (az *Cloud) getVirtualMachine(nodeName types.NodeName) (vm compute.VirtualMachine, exists bool, err error) {
	var realErr error

	vmName := string(nodeName)
	az.operationPollRateLimiter.Accept()
	glog.V(10).Infof("VirtualMachinesClient.Get(%s): start", vmName)
	vm, err = az.VirtualMachinesClient.Get(az.ResourceGroup, vmName, "")
	glog.V(10).Infof("VirtualMachinesClient.Get(%s): end", vmName)

	exists, realErr = checkResourceExistsFromError(err)
	if realErr != nil {
		return vm, false, realErr
	}

	if !exists {
		return vm, false, nil
	}

	return vm, exists, err
}

func (az *Cloud) getRouteTable() (routeTable network.RouteTable, exists bool, err error) {
	var realErr error

	az.operationPollRateLimiter.Accept()
	glog.V(10).Infof("RouteTablesClient.Get(%s): start", az.RouteTableName)
	routeTable, err = az.RouteTablesClient.Get(az.ResourceGroup, az.RouteTableName, "")
	glog.V(10).Infof("RouteTablesClient.Get(%s): end", az.RouteTableName)

	exists, realErr = checkResourceExistsFromError(err)
	if realErr != nil {
		return routeTable, false, realErr
	}

	if !exists {
		return routeTable, false, nil
	}

	return routeTable, exists, err
}

func (az *Cloud) getSecurityGroup() (sg network.SecurityGroup, exists bool, err error) {
	var realErr error

	az.operationPollRateLimiter.Accept()
	glog.V(10).Infof("SecurityGroupsClient.Get(%s): start", az.SecurityGroupName)
	sg, err = az.SecurityGroupsClient.Get(az.ResourceGroup, az.SecurityGroupName, "")
	glog.V(10).Infof("SecurityGroupsClient.Get(%s): end", az.SecurityGroupName)

	exists, realErr = checkResourceExistsFromError(err)
	if realErr != nil {
		return sg, false, realErr
	}

	if !exists {
		return sg, false, nil
	}

	return sg, exists, err
}

func (az *Cloud) getAzureLoadBalancer(name string) (lb network.LoadBalancer, exists bool, err error) {
	var realErr error

	az.operationPollRateLimiter.Accept()
	glog.V(10).Infof("LoadBalancerClient.Get(%s): start", name)
	lb, err = az.LoadBalancerClient.Get(az.ResourceGroup, name, "")
	glog.V(10).Infof("LoadBalancerClient.Get(%s): end", name)

	exists, realErr = checkResourceExistsFromError(err)
	if realErr != nil {
		return lb, false, realErr
	}

	if !exists {
		return lb, false, nil
	}

	return lb, exists, err
}

func (az *Cloud) getPublicIPAddress(name string) (pip network.PublicIPAddress, exists bool, err error) {
	var realErr error

	az.operationPollRateLimiter.Accept()
	glog.V(10).Infof("PublicIPAddressesClient.Get(%s): start", name)
	pip, err = az.PublicIPAddressesClient.Get(az.ResourceGroup, name, "")
	glog.V(10).Infof("PublicIPAddressesClient.Get(%s): end", name)

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

	az.operationPollRateLimiter.Accept()
	glog.V(10).Infof("SubnetsClient.Get(%s): start", subnetName)
	subnet, err = az.SubnetsClient.Get(rg, virtualNetworkName, subnetName, "")
	glog.V(10).Infof("SubnetsClient.Get(%s): end", subnetName)

	exists, realErr = checkResourceExistsFromError(err)
	if realErr != nil {
		return subnet, false, realErr
	}

	if !exists {
		return subnet, false, nil
	}

	return subnet, exists, err
}

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
	"math/rand"
	"net/http"
	"strings"
	"sync"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/cloudprovider"

	"github.com/Azure/azure-sdk-for-go/arm/compute"
	"github.com/Azure/azure-sdk-for-go/arm/disk"
	"github.com/Azure/azure-sdk-for-go/arm/network"
	"github.com/Azure/azure-sdk-for-go/arm/storage"
	computepreview "github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2017-12-01/compute"
	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/to"
)

type fakeAzureLBClient struct {
	mutex     *sync.Mutex
	FakeStore map[string]map[string]network.LoadBalancer
}

func newFakeAzureLBClient() *fakeAzureLBClient {
	fLBC := &fakeAzureLBClient{}
	fLBC.FakeStore = make(map[string]map[string]network.LoadBalancer)
	fLBC.mutex = &sync.Mutex{}
	return fLBC
}

func (fLBC *fakeAzureLBClient) CreateOrUpdate(resourceGroupName string, loadBalancerName string, parameters network.LoadBalancer, cancel <-chan struct{}) (<-chan network.LoadBalancer, <-chan error) {
	fLBC.mutex.Lock()
	defer fLBC.mutex.Unlock()
	resultChan := make(chan network.LoadBalancer, 1)
	errChan := make(chan error, 1)
	var result network.LoadBalancer
	var err error
	defer func() {
		resultChan <- result
		errChan <- err
		close(resultChan)
		close(errChan)
	}()
	if _, ok := fLBC.FakeStore[resourceGroupName]; !ok {
		fLBC.FakeStore[resourceGroupName] = make(map[string]network.LoadBalancer)
	}

	// For dynamic ip allocation, just fill in the PrivateIPAddress
	if parameters.FrontendIPConfigurations != nil {
		for idx, config := range *parameters.FrontendIPConfigurations {
			if config.PrivateIPAllocationMethod == network.Dynamic {
				// Here we randomly assign an ip as private ip
				// It doesn't smart enough to know whether it is in the subnet's range
				(*parameters.FrontendIPConfigurations)[idx].PrivateIPAddress = getRandomIPPtr()
			}
		}
	}
	fLBC.FakeStore[resourceGroupName][loadBalancerName] = parameters
	result = fLBC.FakeStore[resourceGroupName][loadBalancerName]
	result.Response.Response = &http.Response{
		StatusCode: http.StatusOK,
	}
	err = nil
	return resultChan, errChan
}

func (fLBC *fakeAzureLBClient) Delete(resourceGroupName string, loadBalancerName string, cancel <-chan struct{}) (<-chan autorest.Response, <-chan error) {
	fLBC.mutex.Lock()
	defer fLBC.mutex.Unlock()
	respChan := make(chan autorest.Response, 1)
	errChan := make(chan error, 1)
	var resp autorest.Response
	var err error
	defer func() {
		respChan <- resp
		errChan <- err
		close(respChan)
		close(errChan)
	}()
	if rgLBs, ok := fLBC.FakeStore[resourceGroupName]; ok {
		if _, ok := rgLBs[loadBalancerName]; ok {
			delete(rgLBs, loadBalancerName)
			resp.Response = &http.Response{
				StatusCode: http.StatusAccepted,
			}
			err = nil
			return respChan, errChan
		}
	}
	resp.Response = &http.Response{
		StatusCode: http.StatusNotFound,
	}
	err = autorest.DetailedError{
		StatusCode: http.StatusNotFound,
		Message:    "Not such LB",
	}
	return respChan, errChan
}

func (fLBC *fakeAzureLBClient) Get(resourceGroupName string, loadBalancerName string, expand string) (result network.LoadBalancer, err error) {
	fLBC.mutex.Lock()
	defer fLBC.mutex.Unlock()
	if _, ok := fLBC.FakeStore[resourceGroupName]; ok {
		if entity, ok := fLBC.FakeStore[resourceGroupName][loadBalancerName]; ok {
			return entity, nil
		}
	}
	return result, autorest.DetailedError{
		StatusCode: http.StatusNotFound,
		Message:    "Not such LB",
	}
}

func (fLBC *fakeAzureLBClient) List(resourceGroupName string) (result network.LoadBalancerListResult, err error) {
	fLBC.mutex.Lock()
	defer fLBC.mutex.Unlock()
	var value []network.LoadBalancer
	if _, ok := fLBC.FakeStore[resourceGroupName]; ok {
		for _, v := range fLBC.FakeStore[resourceGroupName] {
			value = append(value, v)
		}
	}
	result.Response.Response = &http.Response{
		StatusCode: http.StatusOK,
	}
	result.NextLink = nil
	result.Value = &value
	return result, nil
}

func (fLBC *fakeAzureLBClient) ListNextResults(resourceGroupName string, lastResult network.LoadBalancerListResult) (result network.LoadBalancerListResult, err error) {
	fLBC.mutex.Lock()
	defer fLBC.mutex.Unlock()
	result.Response.Response = &http.Response{
		StatusCode: http.StatusOK,
	}
	result.NextLink = nil
	result.Value = nil
	return result, nil
}

type fakeAzurePIPClient struct {
	mutex          *sync.Mutex
	FakeStore      map[string]map[string]network.PublicIPAddress
	SubscriptionID string
}

const publicIPAddressIDTemplate = "/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Network/publicIPAddresses/%s"

// returns the full identifier of a publicIPAddress.
func getpublicIPAddressID(subscriptionID string, resourceGroupName, pipName string) string {
	return fmt.Sprintf(
		publicIPAddressIDTemplate,
		subscriptionID,
		resourceGroupName,
		pipName)
}

func newFakeAzurePIPClient(subscriptionID string) *fakeAzurePIPClient {
	fAPC := &fakeAzurePIPClient{}
	fAPC.FakeStore = make(map[string]map[string]network.PublicIPAddress)
	fAPC.SubscriptionID = subscriptionID
	fAPC.mutex = &sync.Mutex{}
	return fAPC
}

func (fAPC *fakeAzurePIPClient) CreateOrUpdate(resourceGroupName string, publicIPAddressName string, parameters network.PublicIPAddress, cancel <-chan struct{}) (<-chan network.PublicIPAddress, <-chan error) {
	fAPC.mutex.Lock()
	defer fAPC.mutex.Unlock()
	resultChan := make(chan network.PublicIPAddress, 1)
	errChan := make(chan error, 1)
	var result network.PublicIPAddress
	var err error
	defer func() {
		resultChan <- result
		errChan <- err
		close(resultChan)
		close(errChan)
	}()
	if _, ok := fAPC.FakeStore[resourceGroupName]; !ok {
		fAPC.FakeStore[resourceGroupName] = make(map[string]network.PublicIPAddress)
	}

	// assign id
	pipID := getpublicIPAddressID(fAPC.SubscriptionID, resourceGroupName, publicIPAddressName)
	parameters.ID = &pipID

	// only create in the case user has not provided
	if parameters.PublicIPAddressPropertiesFormat != nil &&
		parameters.PublicIPAddressPropertiesFormat.PublicIPAllocationMethod == network.Static {
		// assign ip
		parameters.IPAddress = getRandomIPPtr()
	}

	fAPC.FakeStore[resourceGroupName][publicIPAddressName] = parameters
	result = fAPC.FakeStore[resourceGroupName][publicIPAddressName]
	result.Response.Response = &http.Response{
		StatusCode: http.StatusOK,
	}
	err = nil
	return resultChan, errChan
}

func (fAPC *fakeAzurePIPClient) Delete(resourceGroupName string, publicIPAddressName string, cancel <-chan struct{}) (<-chan autorest.Response, <-chan error) {
	fAPC.mutex.Lock()
	defer fAPC.mutex.Unlock()
	respChan := make(chan autorest.Response, 1)
	errChan := make(chan error, 1)
	var resp autorest.Response
	var err error
	defer func() {
		respChan <- resp
		errChan <- err
		close(respChan)
		close(errChan)
	}()
	if rgPIPs, ok := fAPC.FakeStore[resourceGroupName]; ok {
		if _, ok := rgPIPs[publicIPAddressName]; ok {
			delete(rgPIPs, publicIPAddressName)
			resp.Response = &http.Response{
				StatusCode: http.StatusAccepted,
			}
			err = nil
			return respChan, errChan
		}
	}
	resp.Response = &http.Response{
		StatusCode: http.StatusNotFound,
	}
	err = autorest.DetailedError{
		StatusCode: http.StatusNotFound,
		Message:    "Not such PIP",
	}
	return respChan, errChan
}

func (fAPC *fakeAzurePIPClient) Get(resourceGroupName string, publicIPAddressName string, expand string) (result network.PublicIPAddress, err error) {
	fAPC.mutex.Lock()
	defer fAPC.mutex.Unlock()
	if _, ok := fAPC.FakeStore[resourceGroupName]; ok {
		if entity, ok := fAPC.FakeStore[resourceGroupName][publicIPAddressName]; ok {
			return entity, nil
		}
	}
	return result, autorest.DetailedError{
		StatusCode: http.StatusNotFound,
		Message:    "Not such PIP",
	}
}

func (fAPC *fakeAzurePIPClient) ListNextResults(resourceGroupName string, lastResults network.PublicIPAddressListResult) (result network.PublicIPAddressListResult, err error) {
	fAPC.mutex.Lock()
	defer fAPC.mutex.Unlock()
	return network.PublicIPAddressListResult{}, nil
}

func (fAPC *fakeAzurePIPClient) List(resourceGroupName string) (result network.PublicIPAddressListResult, err error) {
	fAPC.mutex.Lock()
	defer fAPC.mutex.Unlock()
	var value []network.PublicIPAddress
	if _, ok := fAPC.FakeStore[resourceGroupName]; ok {
		for _, v := range fAPC.FakeStore[resourceGroupName] {
			value = append(value, v)
		}
	}
	result.Response.Response = &http.Response{
		StatusCode: http.StatusOK,
	}
	result.NextLink = nil
	result.Value = &value
	return result, nil
}

type fakeAzureInterfacesClient struct {
	mutex     *sync.Mutex
	FakeStore map[string]map[string]network.Interface
}

func newFakeAzureInterfacesClient() *fakeAzureInterfacesClient {
	fIC := &fakeAzureInterfacesClient{}
	fIC.FakeStore = make(map[string]map[string]network.Interface)
	fIC.mutex = &sync.Mutex{}

	return fIC
}

func (fIC *fakeAzureInterfacesClient) CreateOrUpdate(resourceGroupName string, networkInterfaceName string, parameters network.Interface, cancel <-chan struct{}) (<-chan network.Interface, <-chan error) {
	fIC.mutex.Lock()
	defer fIC.mutex.Unlock()
	resultChan := make(chan network.Interface, 1)
	errChan := make(chan error, 1)
	var result network.Interface
	var err error
	defer func() {
		resultChan <- result
		errChan <- err
		close(resultChan)
		close(errChan)
	}()
	if _, ok := fIC.FakeStore[resourceGroupName]; !ok {
		fIC.FakeStore[resourceGroupName] = make(map[string]network.Interface)
	}
	fIC.FakeStore[resourceGroupName][networkInterfaceName] = parameters
	result = fIC.FakeStore[resourceGroupName][networkInterfaceName]
	result.Response.Response = &http.Response{
		StatusCode: http.StatusOK,
	}
	err = nil

	return resultChan, errChan
}

func (fIC *fakeAzureInterfacesClient) Get(resourceGroupName string, networkInterfaceName string, expand string) (result network.Interface, err error) {
	fIC.mutex.Lock()
	defer fIC.mutex.Unlock()
	if _, ok := fIC.FakeStore[resourceGroupName]; ok {
		if entity, ok := fIC.FakeStore[resourceGroupName][networkInterfaceName]; ok {
			return entity, nil
		}
	}
	return result, autorest.DetailedError{
		StatusCode: http.StatusNotFound,
		Message:    "Not such Interface",
	}
}

func (fIC *fakeAzureInterfacesClient) GetVirtualMachineScaleSetNetworkInterface(resourceGroupName string, virtualMachineScaleSetName string, virtualmachineIndex string, networkInterfaceName string, expand string) (result network.Interface, err error) {
	return result, nil
}

type fakeAzureVirtualMachinesClient struct {
	mutex     *sync.Mutex
	FakeStore map[string]map[string]compute.VirtualMachine
}

func newFakeAzureVirtualMachinesClient() *fakeAzureVirtualMachinesClient {
	fVMC := &fakeAzureVirtualMachinesClient{}
	fVMC.FakeStore = make(map[string]map[string]compute.VirtualMachine)
	fVMC.mutex = &sync.Mutex{}
	return fVMC
}

func (fVMC *fakeAzureVirtualMachinesClient) CreateOrUpdate(resourceGroupName string, VMName string, parameters compute.VirtualMachine, cancel <-chan struct{}) (<-chan compute.VirtualMachine, <-chan error) {
	fVMC.mutex.Lock()
	defer fVMC.mutex.Unlock()
	resultChan := make(chan compute.VirtualMachine, 1)
	errChan := make(chan error, 1)
	var result compute.VirtualMachine
	var err error
	defer func() {
		resultChan <- result
		errChan <- err
		close(resultChan)
		close(errChan)
	}()
	if _, ok := fVMC.FakeStore[resourceGroupName]; !ok {
		fVMC.FakeStore[resourceGroupName] = make(map[string]compute.VirtualMachine)
	}
	fVMC.FakeStore[resourceGroupName][VMName] = parameters
	result = fVMC.FakeStore[resourceGroupName][VMName]
	result.Response.Response = &http.Response{
		StatusCode: http.StatusOK,
	}
	err = nil
	return resultChan, errChan
}

func (fVMC *fakeAzureVirtualMachinesClient) Get(resourceGroupName string, VMName string, expand compute.InstanceViewTypes) (result compute.VirtualMachine, err error) {
	fVMC.mutex.Lock()
	defer fVMC.mutex.Unlock()
	if _, ok := fVMC.FakeStore[resourceGroupName]; ok {
		if entity, ok := fVMC.FakeStore[resourceGroupName][VMName]; ok {
			return entity, nil
		}
	}
	return result, autorest.DetailedError{
		StatusCode: http.StatusNotFound,
		Message:    "Not such VM",
	}
}

func (fVMC *fakeAzureVirtualMachinesClient) List(resourceGroupName string) (result compute.VirtualMachineListResult, err error) {
	fVMC.mutex.Lock()
	defer fVMC.mutex.Unlock()
	var value []compute.VirtualMachine
	if _, ok := fVMC.FakeStore[resourceGroupName]; ok {
		for _, v := range fVMC.FakeStore[resourceGroupName] {
			value = append(value, v)
		}
	}
	result.Response.Response = &http.Response{
		StatusCode: http.StatusOK,
	}
	result.NextLink = nil
	result.Value = &value
	return result, nil
}
func (fVMC *fakeAzureVirtualMachinesClient) ListNextResults(resourceGroupName string, lastResults compute.VirtualMachineListResult) (result compute.VirtualMachineListResult, err error) {
	fVMC.mutex.Lock()
	defer fVMC.mutex.Unlock()
	return compute.VirtualMachineListResult{}, nil
}

type fakeAzureSubnetsClient struct {
	mutex     *sync.Mutex
	FakeStore map[string]map[string]network.Subnet
}

func newFakeAzureSubnetsClient() *fakeAzureSubnetsClient {
	fASC := &fakeAzureSubnetsClient{}
	fASC.FakeStore = make(map[string]map[string]network.Subnet)
	fASC.mutex = &sync.Mutex{}
	return fASC
}

func (fASC *fakeAzureSubnetsClient) CreateOrUpdate(resourceGroupName string, virtualNetworkName string, subnetName string, subnetParameters network.Subnet, cancel <-chan struct{}) (<-chan network.Subnet, <-chan error) {
	fASC.mutex.Lock()
	defer fASC.mutex.Unlock()
	resultChan := make(chan network.Subnet, 1)
	errChan := make(chan error, 1)
	var result network.Subnet
	var err error
	defer func() {
		resultChan <- result
		errChan <- err
		close(resultChan)
		close(errChan)
	}()
	rgVnet := strings.Join([]string{resourceGroupName, virtualNetworkName}, "AND")
	if _, ok := fASC.FakeStore[rgVnet]; !ok {
		fASC.FakeStore[rgVnet] = make(map[string]network.Subnet)
	}
	fASC.FakeStore[rgVnet][subnetName] = subnetParameters
	result = fASC.FakeStore[rgVnet][subnetName]
	result.Response.Response = &http.Response{
		StatusCode: http.StatusOK,
	}
	err = nil
	return resultChan, errChan
}

func (fASC *fakeAzureSubnetsClient) Delete(resourceGroupName string, virtualNetworkName string, subnetName string, cancel <-chan struct{}) (<-chan autorest.Response, <-chan error) {
	fASC.mutex.Lock()
	defer fASC.mutex.Unlock()
	respChan := make(chan autorest.Response, 1)
	errChan := make(chan error, 1)
	var resp autorest.Response
	var err error
	defer func() {
		respChan <- resp
		errChan <- err
		close(respChan)
		close(errChan)
	}()

	rgVnet := strings.Join([]string{resourceGroupName, virtualNetworkName}, "AND")
	if rgSubnets, ok := fASC.FakeStore[rgVnet]; ok {
		if _, ok := rgSubnets[subnetName]; ok {
			delete(rgSubnets, subnetName)
			resp.Response = &http.Response{
				StatusCode: http.StatusAccepted,
			}
			err = nil
			return respChan, errChan
		}
	}
	resp.Response = &http.Response{
		StatusCode: http.StatusNotFound,
	}
	err = autorest.DetailedError{
		StatusCode: http.StatusNotFound,
		Message:    "Not such Subnet",
	}
	return respChan, errChan
}
func (fASC *fakeAzureSubnetsClient) Get(resourceGroupName string, virtualNetworkName string, subnetName string, expand string) (result network.Subnet, err error) {
	fASC.mutex.Lock()
	defer fASC.mutex.Unlock()
	rgVnet := strings.Join([]string{resourceGroupName, virtualNetworkName}, "AND")
	if _, ok := fASC.FakeStore[rgVnet]; ok {
		if entity, ok := fASC.FakeStore[rgVnet][subnetName]; ok {
			return entity, nil
		}
	}
	return result, autorest.DetailedError{
		StatusCode: http.StatusNotFound,
		Message:    "Not such Subnet",
	}
}
func (fASC *fakeAzureSubnetsClient) List(resourceGroupName string, virtualNetworkName string) (result network.SubnetListResult, err error) {
	fASC.mutex.Lock()
	defer fASC.mutex.Unlock()
	rgVnet := strings.Join([]string{resourceGroupName, virtualNetworkName}, "AND")
	var value []network.Subnet
	if _, ok := fASC.FakeStore[rgVnet]; ok {
		for _, v := range fASC.FakeStore[rgVnet] {
			value = append(value, v)
		}
	}
	result.Response.Response = &http.Response{
		StatusCode: http.StatusOK,
	}
	result.NextLink = nil
	result.Value = &value
	return result, nil
}

type fakeAzureNSGClient struct {
	mutex     *sync.Mutex
	FakeStore map[string]map[string]network.SecurityGroup
}

func newFakeAzureNSGClient() *fakeAzureNSGClient {
	fNSG := &fakeAzureNSGClient{}
	fNSG.FakeStore = make(map[string]map[string]network.SecurityGroup)
	fNSG.mutex = &sync.Mutex{}
	return fNSG
}

func (fNSG *fakeAzureNSGClient) CreateOrUpdate(resourceGroupName string, networkSecurityGroupName string, parameters network.SecurityGroup, cancel <-chan struct{}) (<-chan network.SecurityGroup, <-chan error) {
	fNSG.mutex.Lock()
	defer fNSG.mutex.Unlock()
	resultChan := make(chan network.SecurityGroup, 1)
	errChan := make(chan error, 1)
	var result network.SecurityGroup
	var err error
	defer func() {
		resultChan <- result
		errChan <- err
		close(resultChan)
		close(errChan)
	}()
	if _, ok := fNSG.FakeStore[resourceGroupName]; !ok {
		fNSG.FakeStore[resourceGroupName] = make(map[string]network.SecurityGroup)
	}
	fNSG.FakeStore[resourceGroupName][networkSecurityGroupName] = parameters
	result = fNSG.FakeStore[resourceGroupName][networkSecurityGroupName]
	result.Response.Response = &http.Response{
		StatusCode: http.StatusOK,
	}
	err = nil
	return resultChan, errChan
}

func (fNSG *fakeAzureNSGClient) Delete(resourceGroupName string, networkSecurityGroupName string, cancel <-chan struct{}) (<-chan autorest.Response, <-chan error) {
	fNSG.mutex.Lock()
	defer fNSG.mutex.Unlock()
	respChan := make(chan autorest.Response, 1)
	errChan := make(chan error, 1)
	var resp autorest.Response
	var err error
	defer func() {
		respChan <- resp
		errChan <- err
		close(respChan)
		close(errChan)
	}()
	if rgSGs, ok := fNSG.FakeStore[resourceGroupName]; ok {
		if _, ok := rgSGs[networkSecurityGroupName]; ok {
			delete(rgSGs, networkSecurityGroupName)
			resp.Response = &http.Response{
				StatusCode: http.StatusAccepted,
			}
			err = nil
			return respChan, errChan
		}
	}
	resp.Response = &http.Response{
		StatusCode: http.StatusNotFound,
	}
	err = autorest.DetailedError{
		StatusCode: http.StatusNotFound,
		Message:    "Not such NSG",
	}
	return respChan, errChan
}

func (fNSG *fakeAzureNSGClient) Get(resourceGroupName string, networkSecurityGroupName string, expand string) (result network.SecurityGroup, err error) {
	fNSG.mutex.Lock()
	defer fNSG.mutex.Unlock()
	if _, ok := fNSG.FakeStore[resourceGroupName]; ok {
		if entity, ok := fNSG.FakeStore[resourceGroupName][networkSecurityGroupName]; ok {
			return entity, nil
		}
	}
	return result, autorest.DetailedError{
		StatusCode: http.StatusNotFound,
		Message:    "Not such NSG",
	}
}

func (fNSG *fakeAzureNSGClient) List(resourceGroupName string) (result network.SecurityGroupListResult, err error) {
	fNSG.mutex.Lock()
	defer fNSG.mutex.Unlock()
	var value []network.SecurityGroup
	if _, ok := fNSG.FakeStore[resourceGroupName]; ok {
		for _, v := range fNSG.FakeStore[resourceGroupName] {
			value = append(value, v)
		}
	}
	result.Response.Response = &http.Response{
		StatusCode: http.StatusOK,
	}
	result.NextLink = nil
	result.Value = &value
	return result, nil
}

func getRandomIPPtr() *string {
	rand.Seed(time.Now().UnixNano())
	return to.StringPtr(fmt.Sprintf("%d.%d.%d.%d", rand.Intn(256), rand.Intn(256), rand.Intn(256), rand.Intn(256)))
}

type fakeVirtualMachineScaleSetVMsClient struct {
	mutex     *sync.Mutex
	FakeStore map[string]map[string]computepreview.VirtualMachineScaleSetVM
}

func newFakeVirtualMachineScaleSetVMsClient() *fakeVirtualMachineScaleSetVMsClient {
	fVMC := &fakeVirtualMachineScaleSetVMsClient{}
	fVMC.FakeStore = make(map[string]map[string]computepreview.VirtualMachineScaleSetVM)
	fVMC.mutex = &sync.Mutex{}

	return fVMC
}

func (fVMC *fakeVirtualMachineScaleSetVMsClient) setFakeStore(store map[string]map[string]computepreview.VirtualMachineScaleSetVM) {
	fVMC.mutex.Lock()
	defer fVMC.mutex.Unlock()

	fVMC.FakeStore = store
}

func (fVMC *fakeVirtualMachineScaleSetVMsClient) List(ctx context.Context, resourceGroupName string, virtualMachineScaleSetName string, filter string, selectParameter string, expand string) (result []computepreview.VirtualMachineScaleSetVM, err error) {
	fVMC.mutex.Lock()
	defer fVMC.mutex.Unlock()

	result = []computepreview.VirtualMachineScaleSetVM{}
	if _, ok := fVMC.FakeStore[resourceGroupName]; ok {
		for _, v := range fVMC.FakeStore[resourceGroupName] {
			result = append(result, v)
		}
	}

	return result, nil
}

func (fVMC *fakeVirtualMachineScaleSetVMsClient) Get(ctx context.Context, resourceGroupName string, VMScaleSetName string, instanceID string) (result computepreview.VirtualMachineScaleSetVM, err error) {
	fVMC.mutex.Lock()
	defer fVMC.mutex.Unlock()

	vmKey := fmt.Sprintf("%s_%s", VMScaleSetName, instanceID)
	if scaleSetMap, ok := fVMC.FakeStore[resourceGroupName]; ok {
		if entity, ok := scaleSetMap[vmKey]; ok {
			return entity, nil
		}
	}

	return result, autorest.DetailedError{
		StatusCode: http.StatusNotFound,
		Message:    "No such VirtualMachineScaleSetVM",
	}
}

func (fVMC *fakeVirtualMachineScaleSetVMsClient) GetInstanceView(ctx context.Context, resourceGroupName string, VMScaleSetName string, instanceID string) (result computepreview.VirtualMachineScaleSetVMInstanceView, err error) {
	_, err = fVMC.Get(ctx, resourceGroupName, VMScaleSetName, instanceID)
	if err != nil {
		return result, err
	}

	return result, nil
}

func (fVMC *fakeVirtualMachineScaleSetVMsClient) Update(ctx context.Context, resourceGroupName string, VMScaleSetName string, instanceID string, parameters computepreview.VirtualMachineScaleSetVM) (resp *http.Response, err error) {
	fVMC.mutex.Lock()
	defer fVMC.mutex.Unlock()

	vmKey := fmt.Sprintf("%s_%s", VMScaleSetName, instanceID)
	if scaleSetMap, ok := fVMC.FakeStore[resourceGroupName]; ok {
		if _, ok := scaleSetMap[vmKey]; ok {
			scaleSetMap[vmKey] = parameters
		}
	}

	return nil, nil
}

type fakeVirtualMachineScaleSetsClient struct {
	mutex     *sync.Mutex
	FakeStore map[string]map[string]computepreview.VirtualMachineScaleSet
}

func newFakeVirtualMachineScaleSetsClient() *fakeVirtualMachineScaleSetsClient {
	fVMSSC := &fakeVirtualMachineScaleSetsClient{}
	fVMSSC.FakeStore = make(map[string]map[string]computepreview.VirtualMachineScaleSet)
	fVMSSC.mutex = &sync.Mutex{}

	return fVMSSC
}

func (fVMSSC *fakeVirtualMachineScaleSetsClient) setFakeStore(store map[string]map[string]computepreview.VirtualMachineScaleSet) {
	fVMSSC.mutex.Lock()
	defer fVMSSC.mutex.Unlock()

	fVMSSC.FakeStore = store
}

func (fVMSSC *fakeVirtualMachineScaleSetsClient) CreateOrUpdate(ctx context.Context, resourceGroupName string, VMScaleSetName string, parameters computepreview.VirtualMachineScaleSet) (resp *http.Response, err error) {
	fVMSSC.mutex.Lock()
	defer fVMSSC.mutex.Unlock()

	if _, ok := fVMSSC.FakeStore[resourceGroupName]; !ok {
		fVMSSC.FakeStore[resourceGroupName] = make(map[string]computepreview.VirtualMachineScaleSet)
	}
	fVMSSC.FakeStore[resourceGroupName][VMScaleSetName] = parameters

	return nil, nil
}

func (fVMSSC *fakeVirtualMachineScaleSetsClient) Get(ctx context.Context, resourceGroupName string, VMScaleSetName string) (result computepreview.VirtualMachineScaleSet, err error) {
	fVMSSC.mutex.Lock()
	defer fVMSSC.mutex.Unlock()

	if scaleSetMap, ok := fVMSSC.FakeStore[resourceGroupName]; ok {
		if entity, ok := scaleSetMap[VMScaleSetName]; ok {
			return entity, nil
		}
	}

	return result, autorest.DetailedError{
		StatusCode: http.StatusNotFound,
		Message:    "No such ScaleSet",
	}
}

func (fVMSSC *fakeVirtualMachineScaleSetsClient) List(ctx context.Context, resourceGroupName string) (result []computepreview.VirtualMachineScaleSet, err error) {
	fVMSSC.mutex.Lock()
	defer fVMSSC.mutex.Unlock()

	result = []computepreview.VirtualMachineScaleSet{}
	if _, ok := fVMSSC.FakeStore[resourceGroupName]; ok {
		for _, v := range fVMSSC.FakeStore[resourceGroupName] {
			result = append(result, v)
		}
	}

	return result, nil
}

func (fVMSSC *fakeVirtualMachineScaleSetsClient) UpdateInstances(ctx context.Context, resourceGroupName string, VMScaleSetName string, VMInstanceIDs computepreview.VirtualMachineScaleSetVMInstanceRequiredIDs) (resp *http.Response, err error) {
	return nil, nil
}

type fakeRoutesClient struct {
	mutex     *sync.Mutex
	FakeStore map[string]map[string]network.Route
}

func newFakeRoutesClient() *fakeRoutesClient {
	fRC := &fakeRoutesClient{}
	fRC.FakeStore = make(map[string]map[string]network.Route)
	fRC.mutex = &sync.Mutex{}
	return fRC
}

func (fRC *fakeRoutesClient) CreateOrUpdate(resourceGroupName string, routeTableName string, routeName string, routeParameters network.Route, cancel <-chan struct{}) (<-chan network.Route, <-chan error) {
	fRC.mutex.Lock()
	defer fRC.mutex.Unlock()

	resultChan := make(chan network.Route, 1)
	errChan := make(chan error, 1)
	var result network.Route
	var err error
	defer func() {
		resultChan <- result
		errChan <- err
		close(resultChan)
		close(errChan)
	}()

	if _, ok := fRC.FakeStore[routeTableName]; !ok {
		fRC.FakeStore[routeTableName] = make(map[string]network.Route)
	}
	fRC.FakeStore[routeTableName][routeName] = routeParameters
	result = fRC.FakeStore[routeTableName][routeName]
	result.Response.Response = &http.Response{
		StatusCode: http.StatusOK,
	}
	err = nil
	return resultChan, errChan
}

func (fRC *fakeRoutesClient) Delete(resourceGroupName string, routeTableName string, routeName string, cancel <-chan struct{}) (<-chan autorest.Response, <-chan error) {
	fRC.mutex.Lock()
	defer fRC.mutex.Unlock()

	respChan := make(chan autorest.Response, 1)
	errChan := make(chan error, 1)
	var resp autorest.Response
	var err error
	defer func() {
		respChan <- resp
		errChan <- err
		close(respChan)
		close(errChan)
	}()
	if routes, ok := fRC.FakeStore[routeTableName]; ok {
		if _, ok := routes[routeName]; ok {
			delete(routes, routeName)
			resp.Response = &http.Response{
				StatusCode: http.StatusAccepted,
			}

			err = nil
			return respChan, errChan
		}
	}
	resp.Response = &http.Response{
		StatusCode: http.StatusNotFound,
	}
	err = autorest.DetailedError{
		StatusCode: http.StatusNotFound,
		Message:    "Not such Route",
	}
	return respChan, errChan
}

type fakeRouteTablesClient struct {
	mutex     *sync.Mutex
	FakeStore map[string]map[string]network.RouteTable
	Calls     []string
}

func newFakeRouteTablesClient() *fakeRouteTablesClient {
	fRTC := &fakeRouteTablesClient{}
	fRTC.FakeStore = make(map[string]map[string]network.RouteTable)
	fRTC.mutex = &sync.Mutex{}
	return fRTC
}

func (fRTC *fakeRouteTablesClient) CreateOrUpdate(resourceGroupName string, routeTableName string, parameters network.RouteTable, cancel <-chan struct{}) (<-chan network.RouteTable, <-chan error) {
	fRTC.mutex.Lock()
	defer fRTC.mutex.Unlock()

	fRTC.Calls = append(fRTC.Calls, "CreateOrUpdate")

	resultChan := make(chan network.RouteTable, 1)
	errChan := make(chan error, 1)
	var result network.RouteTable
	var err error
	defer func() {
		resultChan <- result
		errChan <- err
		close(resultChan)
		close(errChan)
	}()

	if _, ok := fRTC.FakeStore[resourceGroupName]; !ok {
		fRTC.FakeStore[resourceGroupName] = make(map[string]network.RouteTable)
	}
	fRTC.FakeStore[resourceGroupName][routeTableName] = parameters
	result = fRTC.FakeStore[resourceGroupName][routeTableName]
	result.Response.Response = &http.Response{
		StatusCode: http.StatusOK,
	}
	err = nil
	return resultChan, errChan
}

func (fRTC *fakeRouteTablesClient) Get(resourceGroupName string, routeTableName string, expand string) (result network.RouteTable, err error) {
	fRTC.mutex.Lock()
	defer fRTC.mutex.Unlock()

	fRTC.Calls = append(fRTC.Calls, "Get")

	if _, ok := fRTC.FakeStore[resourceGroupName]; ok {
		if entity, ok := fRTC.FakeStore[resourceGroupName][routeTableName]; ok {
			return entity, nil
		}
	}
	return result, autorest.DetailedError{
		StatusCode: http.StatusNotFound,
		Message:    "Not such RouteTable",
	}
}

type fakeFileClient struct {
}

func (fFC *fakeFileClient) createFileShare(accountName, accountKey, name string, sizeGiB int) error {
	return nil
}

func (fFC *fakeFileClient) deleteFileShare(accountName, accountKey, name string) error {
	return nil
}

func (fFC *fakeFileClient) resizeFileShare(accountName, accountKey, name string, sizeGiB int) error {
	return nil
}

type fakeStorageAccountClient struct {
	mutex     *sync.Mutex
	FakeStore map[string]map[string]storage.Account
	Keys      storage.AccountListKeysResult
	Accounts  storage.AccountListResult
	Err       error
}

func newFakeStorageAccountClient() *fakeStorageAccountClient {
	fSAC := &fakeStorageAccountClient{}
	fSAC.FakeStore = make(map[string]map[string]storage.Account)
	fSAC.mutex = &sync.Mutex{}
	return fSAC
}

func (fSAC *fakeStorageAccountClient) Create(resourceGroupName string, accountName string, parameters storage.AccountCreateParameters, cancel <-chan struct{}) (<-chan storage.Account, <-chan error) {
	fSAC.mutex.Lock()
	defer fSAC.mutex.Unlock()

	resultChan := make(chan storage.Account, 1)
	errChan := make(chan error, 1)
	var result storage.Account
	var err error
	defer func() {
		resultChan <- result
		errChan <- err
		close(resultChan)
		close(errChan)
	}()

	if _, ok := fSAC.FakeStore[resourceGroupName]; !ok {
		fSAC.FakeStore[resourceGroupName] = make(map[string]storage.Account)
	}
	fSAC.FakeStore[resourceGroupName][accountName] = storage.Account{
		Name:              &accountName,
		Sku:               parameters.Sku,
		Kind:              parameters.Kind,
		Location:          parameters.Location,
		Identity:          parameters.Identity,
		Tags:              parameters.Tags,
		AccountProperties: &storage.AccountProperties{},
	}
	result = fSAC.FakeStore[resourceGroupName][accountName]
	result.Response.Response = &http.Response{
		StatusCode: http.StatusOK,
	}
	err = nil
	return resultChan, errChan
}

func (fSAC *fakeStorageAccountClient) Delete(resourceGroupName string, accountName string) (result autorest.Response, err error) {
	fSAC.mutex.Lock()
	defer fSAC.mutex.Unlock()

	if rgAccounts, ok := fSAC.FakeStore[resourceGroupName]; ok {
		if _, ok := rgAccounts[accountName]; ok {
			delete(rgAccounts, accountName)
			result.Response = &http.Response{
				StatusCode: http.StatusAccepted,
			}
			return result, nil
		}
	}

	result.Response = &http.Response{
		StatusCode: http.StatusNotFound,
	}
	err = autorest.DetailedError{
		StatusCode: http.StatusNotFound,
		Message:    "Not such StorageAccount",
	}
	return result, err
}

func (fSAC *fakeStorageAccountClient) ListKeys(resourceGroupName string, accountName string) (result storage.AccountListKeysResult, err error) {
	return fSAC.Keys, fSAC.Err
}

func (fSAC *fakeStorageAccountClient) ListByResourceGroup(resourceGroupName string) (result storage.AccountListResult, err error) {
	return fSAC.Accounts, fSAC.Err
}

func (fSAC *fakeStorageAccountClient) GetProperties(resourceGroupName string, accountName string) (result storage.Account, err error) {
	fSAC.mutex.Lock()
	defer fSAC.mutex.Unlock()

	if _, ok := fSAC.FakeStore[resourceGroupName]; ok {
		if entity, ok := fSAC.FakeStore[resourceGroupName][accountName]; ok {
			return entity, nil
		}
	}

	return result, autorest.DetailedError{
		StatusCode: http.StatusNotFound,
		Message:    "Not such StorageAccount",
	}
}

type fakeDisksClient struct {
	mutex     *sync.Mutex
	FakeStore map[string]map[string]disk.Model
}

func newFakeDisksClient() *fakeDisksClient {
	fDC := &fakeDisksClient{}
	fDC.FakeStore = make(map[string]map[string]disk.Model)
	fDC.mutex = &sync.Mutex{}
	return fDC
}

func (fDC *fakeDisksClient) CreateOrUpdate(resourceGroupName string, diskName string, diskParameter disk.Model, cancel <-chan struct{}) (<-chan disk.Model, <-chan error) {
	fDC.mutex.Lock()
	defer fDC.mutex.Unlock()

	resultChan := make(chan disk.Model, 1)
	errChan := make(chan error, 1)
	var result disk.Model
	var err error
	defer func() {
		resultChan <- result
		errChan <- err
		close(resultChan)
		close(errChan)
	}()

	if _, ok := fDC.FakeStore[resourceGroupName]; !ok {
		fDC.FakeStore[resourceGroupName] = make(map[string]disk.Model)
	}
	fDC.FakeStore[resourceGroupName][diskName] = diskParameter
	result = fDC.FakeStore[resourceGroupName][diskName]
	result.Response.Response = &http.Response{
		StatusCode: http.StatusOK,
	}
	err = nil
	return resultChan, errChan
}

func (fDC *fakeDisksClient) Delete(resourceGroupName string, diskName string, cancel <-chan struct{}) (<-chan disk.OperationStatusResponse, <-chan error) {
	fDC.mutex.Lock()
	defer fDC.mutex.Unlock()

	respChan := make(chan disk.OperationStatusResponse, 1)
	errChan := make(chan error, 1)
	var resp disk.OperationStatusResponse
	var err error
	defer func() {
		respChan <- resp
		errChan <- err
		close(respChan)
		close(errChan)
	}()
	if rgDisks, ok := fDC.FakeStore[resourceGroupName]; ok {
		if _, ok := rgDisks[diskName]; ok {
			delete(rgDisks, diskName)
			resp.Response = autorest.Response{
				Response: &http.Response{
					StatusCode: http.StatusAccepted,
				},
			}

			err = nil
			return respChan, errChan
		}
	}
	resp.Response = autorest.Response{
		Response: &http.Response{
			StatusCode: http.StatusNotFound,
		},
	}
	err = autorest.DetailedError{
		StatusCode: http.StatusNotFound,
		Message:    "Not such Disk",
	}
	return respChan, errChan
}

func (fDC *fakeDisksClient) Get(resourceGroupName string, diskName string) (result disk.Model, err error) {
	fDC.mutex.Lock()
	defer fDC.mutex.Unlock()

	if _, ok := fDC.FakeStore[resourceGroupName]; ok {
		if entity, ok := fDC.FakeStore[resourceGroupName][diskName]; ok {
			return entity, nil
		}
	}

	return result, autorest.DetailedError{
		StatusCode: http.StatusNotFound,
		Message:    "Not such Disk",
	}
}

type fakeVMSet struct {
	NodeToIP map[string]map[string]string
	Err      error
}

func (f *fakeVMSet) GetInstanceIDByNodeName(name string) (string, error) {
	return "", fmt.Errorf("unimplemented")
}

func (f *fakeVMSet) GetInstanceTypeByNodeName(name string) (string, error) {
	return "", fmt.Errorf("unimplemented")
}

func (f *fakeVMSet) GetIPByNodeName(name, vmSetName string) (string, error) {
	nodes, found := f.NodeToIP[vmSetName]
	if !found {
		return "", fmt.Errorf("not found")
	}
	ip, found := nodes[name]
	if !found {
		return "", fmt.Errorf("not found")
	}
	return ip, nil
}

func (f *fakeVMSet) GetPrimaryInterface(nodeName, vmSetName string) (network.Interface, error) {
	return network.Interface{}, fmt.Errorf("unimplemented")
}

func (f *fakeVMSet) GetNodeNameByProviderID(providerID string) (types.NodeName, error) {
	return types.NodeName(""), fmt.Errorf("unimplemented")
}

func (f *fakeVMSet) GetZoneByNodeName(name string) (cloudprovider.Zone, error) {
	return cloudprovider.Zone{}, fmt.Errorf("unimplemented")
}

func (f *fakeVMSet) GetPrimaryVMSetName() string {
	return ""
}

func (f *fakeVMSet) GetVMSetNames(service *v1.Service, nodes []*v1.Node) (availabilitySetNames *[]string, err error) {
	return nil, fmt.Errorf("unimplemented")
}

func (f *fakeVMSet) EnsureHostsInPool(serviceName string, nodes []*v1.Node, backendPoolID string, vmSetName string) error {
	return fmt.Errorf("unimplemented")
}

func (f *fakeVMSet) EnsureBackendPoolDeleted(poolID, vmSetName string) error {
	return fmt.Errorf("unimplemented")
}

func (f *fakeVMSet) AttachDisk(isManagedDisk bool, diskName, diskURI string, nodeName types.NodeName, lun int32, cachingMode compute.CachingTypes) error {
	return fmt.Errorf("unimplemented")
}

func (f *fakeVMSet) DetachDiskByName(diskName, diskURI string, nodeName types.NodeName) error {
	return fmt.Errorf("unimplemented")
}

func (f *fakeVMSet) GetDiskLun(diskName, diskURI string, nodeName types.NodeName) (int32, error) {
	return -1, fmt.Errorf("unimplemented")
}

func (f *fakeVMSet) GetNextDiskLun(nodeName types.NodeName) (int32, error) {
	return -1, fmt.Errorf("unimplemented")
}

func (f *fakeVMSet) DisksAreAttached(diskNames []string, nodeName types.NodeName) (map[string]bool, error) {
	return nil, fmt.Errorf("unimplemented")
}

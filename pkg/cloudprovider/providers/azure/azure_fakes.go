package azure

import (
	"fmt"
	"math/rand"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/Azure/go-autorest/autorest/to"

	"github.com/Azure/azure-sdk-for-go/arm/compute"
	"github.com/Azure/azure-sdk-for-go/arm/network"
	"github.com/Azure/go-autorest/autorest"
)

type fakeAzureLBClient struct {
	mutex     *sync.Mutex
	FakeStore map[string]map[string]network.LoadBalancer
}

func NewFakeAzureLBClient() fakeAzureLBClient {
	fLBC := fakeAzureLBClient{}
	fLBC.FakeStore = make(map[string]map[string]network.LoadBalancer)
	fLBC.mutex = &sync.Mutex{}
	return fLBC
}

func (fLBC fakeAzureLBClient) CreateOrUpdate(resourceGroupName string, loadBalancerName string, parameters network.LoadBalancer, cancel <-chan struct{}) (<-chan network.LoadBalancer, <-chan error) {
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
				(*parameters.FrontendIPConfigurations)[idx].PrivateIPAddress = to.StringPtr("10.0.0.19")
			}
		}
	}
	fLBC.FakeStore[resourceGroupName][loadBalancerName] = parameters
	result = fLBC.FakeStore[resourceGroupName][loadBalancerName]
	err = nil
	return resultChan, errChan
}

func (fLBC fakeAzureLBClient) Delete(resourceGroupName string, loadBalancerName string, cancel <-chan struct{}) (<-chan autorest.Response, <-chan error) {
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
	if _, ok := fLBC.FakeStore[resourceGroupName]; ok {
		if _, ok := fLBC.FakeStore[resourceGroupName][loadBalancerName]; ok {
			delete(fLBC.FakeStore[resourceGroupName], loadBalancerName)
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

func (fLBC fakeAzureLBClient) Get(resourceGroupName string, loadBalancerName string, expand string) (result network.LoadBalancer, err error) {
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

func (fLBC fakeAzureLBClient) List(resourceGroupName string) (result network.LoadBalancerListResult, err error) {
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

func (fLBC fakeAzureLBClient) ListNextResults(lastResult network.LoadBalancerListResult) (result network.LoadBalancerListResult, err error) {
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

func NewFakeAzurePIPClient(subscriptionID string) fakeAzurePIPClient {
	fAPC := fakeAzurePIPClient{}
	fAPC.FakeStore = make(map[string]map[string]network.PublicIPAddress)
	fAPC.SubscriptionID = subscriptionID
	fAPC.mutex = &sync.Mutex{}
	return fAPC
}

func (fAPC fakeAzurePIPClient) CreateOrUpdate(resourceGroupName string, publicIPAddressName string, parameters network.PublicIPAddress, cancel <-chan struct{}) (<-chan network.PublicIPAddress, <-chan error) {
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
		rand.Seed(time.Now().UnixNano())
		randomIP := fmt.Sprintf("%d.%d.%d.%d", rand.Intn(256), rand.Intn(256), rand.Intn(256), rand.Intn(256))
		parameters.IPAddress = &randomIP
	}

	fAPC.FakeStore[resourceGroupName][publicIPAddressName] = parameters
	result = fAPC.FakeStore[resourceGroupName][publicIPAddressName]
	err = nil
	return resultChan, errChan
}

func (fAPC fakeAzurePIPClient) Delete(resourceGroupName string, publicIPAddressName string, cancel <-chan struct{}) (<-chan autorest.Response, <-chan error) {
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
	if _, ok := fAPC.FakeStore[resourceGroupName]; ok {
		if _, ok := fAPC.FakeStore[resourceGroupName][publicIPAddressName]; ok {
			delete(fAPC.FakeStore[resourceGroupName], publicIPAddressName)
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

func (fAPC fakeAzurePIPClient) Get(resourceGroupName string, publicIPAddressName string, expand string) (result network.PublicIPAddress, err error) {
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

func (fAPC fakeAzurePIPClient) ListNextResults(lastResults network.PublicIPAddressListResult) (result network.PublicIPAddressListResult, err error) {
	fAPC.mutex.Lock()
	defer fAPC.mutex.Unlock()
	return network.PublicIPAddressListResult{}, nil
}

func (fAPC fakeAzurePIPClient) List(resourceGroupName string) (result network.PublicIPAddressListResult, err error) {
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

type fakeInterfacesClient struct {
	mutex     *sync.Mutex
	FakeStore map[string]map[string]network.Interface
}

func NewFakeInterfacesClient() fakeInterfacesClient {
	fIC := fakeInterfacesClient{}
	fIC.FakeStore = make(map[string]map[string]network.Interface)
	fIC.mutex = &sync.Mutex{}

	return fIC
}

func (fIC fakeInterfacesClient) CreateOrUpdate(resourceGroupName string, networkInterfaceName string, parameters network.Interface, cancel <-chan struct{}) (<-chan network.Interface, <-chan error) {
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
	err = nil

	return resultChan, errChan
}

func (fIC fakeInterfacesClient) Get(resourceGroupName string, networkInterfaceName string, expand string) (result network.Interface, err error) {
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

type fakeVirtualMachinesClient struct {
	mutex     *sync.Mutex
	FakeStore map[string]map[string]compute.VirtualMachine
}

func NewFakeVirtualMachinesClient() fakeVirtualMachinesClient {
	fVMC := fakeVirtualMachinesClient{}
	fVMC.FakeStore = make(map[string]map[string]compute.VirtualMachine)
	fVMC.mutex = &sync.Mutex{}
	return fVMC
}

func (fVMC fakeVirtualMachinesClient) CreateOrUpdate(resourceGroupName string, VMName string, parameters compute.VirtualMachine, cancel <-chan struct{}) (<-chan compute.VirtualMachine, <-chan error) {
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
	err = nil
	return resultChan, errChan
}

func (fVMC fakeVirtualMachinesClient) Get(resourceGroupName string, VMName string, expand compute.InstanceViewTypes) (result compute.VirtualMachine, err error) {
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

func (fVMC fakeVirtualMachinesClient) List(resourceGroupName string) (result compute.VirtualMachineListResult, err error) {
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
func (fVMC fakeVirtualMachinesClient) ListNextResults(lastResults compute.VirtualMachineListResult) (result compute.VirtualMachineListResult, err error) {
	fVMC.mutex.Lock()
	defer fVMC.mutex.Unlock()
	return compute.VirtualMachineListResult{}, nil
}

type fakeAzureSubnetsClient struct {
	mutex     *sync.Mutex
	FakeStore map[string]map[string]network.Subnet
}

func NewFakeAzureSubnetsClient() fakeAzureSubnetsClient {
	fASC := fakeAzureSubnetsClient{}
	fASC.FakeStore = make(map[string]map[string]network.Subnet)
	fASC.mutex = &sync.Mutex{}
	return fASC
}

func (fASC fakeAzureSubnetsClient) CreateOrUpdate(resourceGroupName string, virtualNetworkName string, subnetName string, subnetParameters network.Subnet, cancel <-chan struct{}) (<-chan network.Subnet, <-chan error) {
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
	err = nil
	return resultChan, errChan
}

func (fASC fakeAzureSubnetsClient) Delete(resourceGroupName string, virtualNetworkName string, subnetName string, cancel <-chan struct{}) (<-chan autorest.Response, <-chan error) {
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
	if _, ok := fASC.FakeStore[rgVnet]; ok {
		if _, ok := fASC.FakeStore[rgVnet][subnetName]; ok {
			delete(fASC.FakeStore[rgVnet], subnetName)
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
func (fASC fakeAzureSubnetsClient) Get(resourceGroupName string, virtualNetworkName string, subnetName string, expand string) (result network.Subnet, err error) {
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
func (fASC fakeAzureSubnetsClient) List(resourceGroupName string, virtualNetworkName string) (result network.SubnetListResult, err error) {
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

func NewFakeAzureNSGClient() fakeAzureNSGClient {
	fNSG := fakeAzureNSGClient{}
	fNSG.FakeStore = make(map[string]map[string]network.SecurityGroup)
	fNSG.mutex = &sync.Mutex{}
	return fNSG
}

func (fNSG fakeAzureNSGClient) CreateOrUpdate(resourceGroupName string, networkSecurityGroupName string, parameters network.SecurityGroup, cancel <-chan struct{}) (<-chan network.SecurityGroup, <-chan error) {
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
	err = nil
	return resultChan, errChan
}

func (fNSG fakeAzureNSGClient) Delete(resourceGroupName string, networkSecurityGroupName string, cancel <-chan struct{}) (<-chan autorest.Response, <-chan error) {
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
	if _, ok := fNSG.FakeStore[resourceGroupName]; ok {
		if _, ok := fNSG.FakeStore[resourceGroupName][networkSecurityGroupName]; ok {
			delete(fNSG.FakeStore[resourceGroupName], networkSecurityGroupName)
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

func (fNSG fakeAzureNSGClient) Get(resourceGroupName string, networkSecurityGroupName string, expand string) (result network.SecurityGroup, err error) {
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

func (fNSG fakeAzureNSGClient) List(resourceGroupName string) (result network.SecurityGroupListResult, err error) {
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

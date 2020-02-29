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
	"errors"
	"fmt"
	"math/rand"
	"net/http"
	"sync"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-07-01/compute"
	"github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-06-01/network"
	"github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2019-06-01/storage"
	"github.com/Azure/go-autorest/autorest/to"
	"github.com/golang/mock/gomock"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/record"
	"k8s.io/legacy-cloud-providers/azure/auth"
	"k8s.io/legacy-cloud-providers/azure/clients/routeclient/mockrouteclient"
	"k8s.io/legacy-cloud-providers/azure/clients/routetableclient/mockroutetableclient"
	"k8s.io/legacy-cloud-providers/azure/clients/subnetclient/mocksubnetclient"
	"k8s.io/legacy-cloud-providers/azure/retry"
)

var (
	errPreconditionFailedEtagMismatch = fmt.Errorf("PreconditionFailedEtagMismatch")
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

func (fLBC *fakeAzureLBClient) CreateOrUpdate(ctx context.Context, resourceGroupName string, loadBalancerName string, parameters network.LoadBalancer, etag string) *retry.Error {
	fLBC.mutex.Lock()
	defer fLBC.mutex.Unlock()

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

	return nil
}

func (fLBC *fakeAzureLBClient) Delete(ctx context.Context, resourceGroupName string, loadBalancerName string) *retry.Error {
	fLBC.mutex.Lock()
	defer fLBC.mutex.Unlock()

	if rgLBs, ok := fLBC.FakeStore[resourceGroupName]; ok {
		if _, ok := rgLBs[loadBalancerName]; ok {
			delete(rgLBs, loadBalancerName)
			return nil
		}
	}

	return nil
}

func (fLBC *fakeAzureLBClient) Get(ctx context.Context, resourceGroupName string, loadBalancerName string, expand string) (result network.LoadBalancer, err *retry.Error) {
	fLBC.mutex.Lock()
	defer fLBC.mutex.Unlock()
	if _, ok := fLBC.FakeStore[resourceGroupName]; ok {
		if entity, ok := fLBC.FakeStore[resourceGroupName][loadBalancerName]; ok {
			return entity, nil
		}
	}
	return result, retry.GetError(
		&http.Response{
			StatusCode: http.StatusNotFound,
		},
		errors.New("Not such LB"))
}

func (fLBC *fakeAzureLBClient) List(ctx context.Context, resourceGroupName string) (result []network.LoadBalancer, err *retry.Error) {
	fLBC.mutex.Lock()
	defer fLBC.mutex.Unlock()

	var value []network.LoadBalancer
	if _, ok := fLBC.FakeStore[resourceGroupName]; ok {
		for _, v := range fLBC.FakeStore[resourceGroupName] {
			value = append(value, v)
		}
	}
	return value, nil
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

func (fAPC *fakeAzurePIPClient) CreateOrUpdate(ctx context.Context, resourceGroupName string, publicIPAddressName string, parameters network.PublicIPAddress) *retry.Error {
	fAPC.mutex.Lock()
	defer fAPC.mutex.Unlock()

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

	return nil
}

func (fAPC *fakeAzurePIPClient) Delete(ctx context.Context, resourceGroupName string, publicIPAddressName string) *retry.Error {
	fAPC.mutex.Lock()
	defer fAPC.mutex.Unlock()

	if rgPIPs, ok := fAPC.FakeStore[resourceGroupName]; ok {
		if _, ok := rgPIPs[publicIPAddressName]; ok {
			delete(rgPIPs, publicIPAddressName)
			return nil
		}
	}

	return retry.GetError(
		&http.Response{
			StatusCode: http.StatusNotFound,
		},
		errors.New("Not such PIP"))
}

func (fAPC *fakeAzurePIPClient) Get(ctx context.Context, resourceGroupName string, publicIPAddressName string, expand string) (result network.PublicIPAddress, err *retry.Error) {
	fAPC.mutex.Lock()
	defer fAPC.mutex.Unlock()
	if _, ok := fAPC.FakeStore[resourceGroupName]; ok {
		if entity, ok := fAPC.FakeStore[resourceGroupName][publicIPAddressName]; ok {
			return entity, nil
		}
	}
	return result, retry.GetError(
		&http.Response{
			StatusCode: http.StatusNotFound,
		},
		errors.New("Not such PIP"))
}

func (fAPC *fakeAzurePIPClient) GetVirtualMachineScaleSetPublicIPAddress(ctx context.Context, resourceGroupName string, virtualMachineScaleSetName string, virtualmachineIndex string, networkInterfaceName string, IPConfigurationName string, publicIPAddressName string, expand string) (result network.PublicIPAddress, err *retry.Error) {
	fAPC.mutex.Lock()
	defer fAPC.mutex.Unlock()
	if _, ok := fAPC.FakeStore[resourceGroupName]; ok {
		if entity, ok := fAPC.FakeStore[resourceGroupName][publicIPAddressName]; ok {
			return entity, nil
		}
	}
	return result, retry.GetError(
		&http.Response{
			StatusCode: http.StatusNotFound,
		},
		errors.New("Not such PIP"))
}

func (fAPC *fakeAzurePIPClient) List(ctx context.Context, resourceGroupName string) (result []network.PublicIPAddress, err *retry.Error) {
	fAPC.mutex.Lock()
	defer fAPC.mutex.Unlock()

	var value []network.PublicIPAddress
	if _, ok := fAPC.FakeStore[resourceGroupName]; ok {
		for _, v := range fAPC.FakeStore[resourceGroupName] {
			value = append(value, v)
		}
	}

	return value, nil
}

func (fAPC *fakeAzurePIPClient) setFakeStore(store map[string]map[string]network.PublicIPAddress) {
	fAPC.mutex.Lock()
	defer fAPC.mutex.Unlock()

	fAPC.FakeStore = store
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

func (fIC *fakeAzureInterfacesClient) CreateOrUpdate(ctx context.Context, resourceGroupName string, networkInterfaceName string, parameters network.Interface) *retry.Error {
	fIC.mutex.Lock()
	defer fIC.mutex.Unlock()

	if _, ok := fIC.FakeStore[resourceGroupName]; !ok {
		fIC.FakeStore[resourceGroupName] = make(map[string]network.Interface)
	}
	fIC.FakeStore[resourceGroupName][networkInterfaceName] = parameters

	return nil
}

func (fIC *fakeAzureInterfacesClient) Get(ctx context.Context, resourceGroupName string, networkInterfaceName string, expand string) (result network.Interface, err *retry.Error) {
	fIC.mutex.Lock()
	defer fIC.mutex.Unlock()
	if _, ok := fIC.FakeStore[resourceGroupName]; ok {
		if entity, ok := fIC.FakeStore[resourceGroupName][networkInterfaceName]; ok {
			return entity, nil
		}
	}
	return result, retry.GetError(
		&http.Response{
			StatusCode: http.StatusNotFound,
		},
		errors.New("Not such Interface"))
}

func (fIC *fakeAzureInterfacesClient) GetVirtualMachineScaleSetNetworkInterface(ctx context.Context, resourceGroupName string, virtualMachineScaleSetName string, virtualmachineIndex string, networkInterfaceName string, expand string) (result network.Interface, err *retry.Error) {
	fIC.mutex.Lock()
	defer fIC.mutex.Unlock()
	if _, ok := fIC.FakeStore[resourceGroupName]; ok {
		if entity, ok := fIC.FakeStore[resourceGroupName][networkInterfaceName]; ok {
			return entity, nil
		}
	}
	return result, retry.GetError(
		&http.Response{
			StatusCode: http.StatusNotFound,
		},
		errors.New("Not such Interface"))
}

func (fIC *fakeAzureInterfacesClient) Delete(ctx context.Context, resourceGroupName string, networkInterfaceName string) *retry.Error {
	return nil
}

func (fIC *fakeAzureInterfacesClient) setFakeStore(store map[string]map[string]network.Interface) {
	fIC.mutex.Lock()
	defer fIC.mutex.Unlock()

	fIC.FakeStore = store
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

func (fVMC *fakeAzureVirtualMachinesClient) CreateOrUpdate(ctx context.Context, resourceGroupName string, VMName string, parameters compute.VirtualMachine, source string) *retry.Error {
	fVMC.mutex.Lock()
	defer fVMC.mutex.Unlock()

	if _, ok := fVMC.FakeStore[resourceGroupName]; !ok {
		fVMC.FakeStore[resourceGroupName] = make(map[string]compute.VirtualMachine)
	}
	fVMC.FakeStore[resourceGroupName][VMName] = parameters

	return nil
}

func (fVMC *fakeAzureVirtualMachinesClient) Update(ctx context.Context, resourceGroupName string, VMName string, parameters compute.VirtualMachineUpdate, source string) *retry.Error {
	fVMC.mutex.Lock()
	defer fVMC.mutex.Unlock()

	if _, ok := fVMC.FakeStore[resourceGroupName]; !ok {
		fVMC.FakeStore[resourceGroupName] = make(map[string]compute.VirtualMachine)
	}

	return nil
}

func (fVMC *fakeAzureVirtualMachinesClient) Get(ctx context.Context, resourceGroupName string, VMName string, expand compute.InstanceViewTypes) (result compute.VirtualMachine, err *retry.Error) {
	fVMC.mutex.Lock()
	defer fVMC.mutex.Unlock()
	if _, ok := fVMC.FakeStore[resourceGroupName]; ok {
		if entity, ok := fVMC.FakeStore[resourceGroupName][VMName]; ok {
			return entity, nil
		}
	}
	return result, retry.GetError(
		&http.Response{
			StatusCode: http.StatusNotFound,
		},
		errors.New("Not such VM"))
}

func (fVMC *fakeAzureVirtualMachinesClient) List(ctx context.Context, resourceGroupName string) (result []compute.VirtualMachine, err *retry.Error) {
	fVMC.mutex.Lock()
	defer fVMC.mutex.Unlock()

	result = []compute.VirtualMachine{}
	if _, ok := fVMC.FakeStore[resourceGroupName]; ok {
		for _, v := range fVMC.FakeStore[resourceGroupName] {
			result = append(result, v)
		}
	}

	return result, nil
}

func (fVMC *fakeAzureVirtualMachinesClient) Delete(ctx context.Context, resourceGroupName string, VMName string) *retry.Error {
	return nil
}

func (fVMC *fakeAzureVirtualMachinesClient) setFakeStore(store map[string]map[string]compute.VirtualMachine) {
	fVMC.mutex.Lock()
	defer fVMC.mutex.Unlock()

	fVMC.FakeStore = store
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

func (fNSG *fakeAzureNSGClient) CreateOrUpdate(ctx context.Context, resourceGroupName string, networkSecurityGroupName string, parameters network.SecurityGroup, etag string) *retry.Error {
	fNSG.mutex.Lock()
	defer fNSG.mutex.Unlock()

	if _, ok := fNSG.FakeStore[resourceGroupName]; !ok {
		fNSG.FakeStore[resourceGroupName] = make(map[string]network.SecurityGroup)
	}

	if nsg, ok := fNSG.FakeStore[resourceGroupName][networkSecurityGroupName]; ok {
		if etag != "" && to.String(nsg.Etag) != "" && etag != to.String(nsg.Etag) {
			return retry.GetError(&http.Response{
				StatusCode: http.StatusPreconditionFailed,
			}, errPreconditionFailedEtagMismatch)
		}
	}
	fNSG.FakeStore[resourceGroupName][networkSecurityGroupName] = parameters

	return nil
}

func (fNSG *fakeAzureNSGClient) Delete(ctx context.Context, resourceGroupName string, networkSecurityGroupName string) *retry.Error {
	fNSG.mutex.Lock()
	defer fNSG.mutex.Unlock()

	if rgSGs, ok := fNSG.FakeStore[resourceGroupName]; ok {
		if _, ok := rgSGs[networkSecurityGroupName]; ok {
			delete(rgSGs, networkSecurityGroupName)
			return nil
		}
	}

	return retry.GetError(
		&http.Response{
			StatusCode: http.StatusNotFound,
		},
		errors.New("Not such NSG"))
}

func (fNSG *fakeAzureNSGClient) Get(ctx context.Context, resourceGroupName string, networkSecurityGroupName string, expand string) (result network.SecurityGroup, err *retry.Error) {
	fNSG.mutex.Lock()
	defer fNSG.mutex.Unlock()
	if _, ok := fNSG.FakeStore[resourceGroupName]; ok {
		if entity, ok := fNSG.FakeStore[resourceGroupName][networkSecurityGroupName]; ok {
			return entity, nil
		}
	}
	return result, retry.GetError(
		&http.Response{
			StatusCode: http.StatusNotFound,
		},
		errors.New("Not such NSG"))
}

func (fNSG *fakeAzureNSGClient) List(ctx context.Context, resourceGroupName string) (result []network.SecurityGroup, err *retry.Error) {
	fNSG.mutex.Lock()
	defer fNSG.mutex.Unlock()

	var value []network.SecurityGroup
	if _, ok := fNSG.FakeStore[resourceGroupName]; ok {
		for _, v := range fNSG.FakeStore[resourceGroupName] {
			value = append(value, v)
		}
	}

	return value, nil
}

func getRandomIPPtr() *string {
	return to.StringPtr(fmt.Sprintf("%d.%d.%d.%d", rand.Intn(256), rand.Intn(256), rand.Intn(256), rand.Intn(256)))
}

type fakeVirtualMachineScaleSetVMsClient struct {
	mutex     *sync.Mutex
	FakeStore map[string]map[string]compute.VirtualMachineScaleSetVM
}

func newFakeVirtualMachineScaleSetVMsClient() *fakeVirtualMachineScaleSetVMsClient {
	fVMC := &fakeVirtualMachineScaleSetVMsClient{}
	fVMC.FakeStore = make(map[string]map[string]compute.VirtualMachineScaleSetVM)
	fVMC.mutex = &sync.Mutex{}

	return fVMC
}

func (fVMC *fakeVirtualMachineScaleSetVMsClient) setFakeStore(store map[string]map[string]compute.VirtualMachineScaleSetVM) {
	fVMC.mutex.Lock()
	defer fVMC.mutex.Unlock()

	fVMC.FakeStore = store
}

func (fVMC *fakeVirtualMachineScaleSetVMsClient) List(ctx context.Context, resourceGroupName string, virtualMachineScaleSetName string, expand string) (result []compute.VirtualMachineScaleSetVM, err *retry.Error) {
	fVMC.mutex.Lock()
	defer fVMC.mutex.Unlock()

	result = []compute.VirtualMachineScaleSetVM{}
	if _, ok := fVMC.FakeStore[resourceGroupName]; ok {
		for _, v := range fVMC.FakeStore[resourceGroupName] {
			result = append(result, v)
		}
	}

	return result, nil
}

func (fVMC *fakeVirtualMachineScaleSetVMsClient) Get(ctx context.Context, resourceGroupName string, VMScaleSetName string, instanceID string, expand compute.InstanceViewTypes) (result compute.VirtualMachineScaleSetVM, err *retry.Error) {
	fVMC.mutex.Lock()
	defer fVMC.mutex.Unlock()

	vmKey := fmt.Sprintf("%s_%s", VMScaleSetName, instanceID)
	if scaleSetMap, ok := fVMC.FakeStore[resourceGroupName]; ok {
		if entity, ok := scaleSetMap[vmKey]; ok {
			return entity, nil
		}
	}

	return result, retry.GetError(
		&http.Response{
			StatusCode: http.StatusNotFound,
		},
		errors.New("Not such VirtualMachineScaleSetVM"))
}

func (fVMC *fakeVirtualMachineScaleSetVMsClient) Update(ctx context.Context, resourceGroupName string, VMScaleSetName string, instanceID string, parameters compute.VirtualMachineScaleSetVM, source string) *retry.Error {
	fVMC.mutex.Lock()
	defer fVMC.mutex.Unlock()

	vmKey := fmt.Sprintf("%s_%s", VMScaleSetName, instanceID)
	if scaleSetMap, ok := fVMC.FakeStore[resourceGroupName]; ok {
		if _, ok := scaleSetMap[vmKey]; ok {
			scaleSetMap[vmKey] = parameters
		}
	}

	return nil
}

func (fVMC *fakeVirtualMachineScaleSetVMsClient) UpdateVMs(ctx context.Context, resourceGroupName string, VMScaleSetName string, instances map[string]compute.VirtualMachineScaleSetVM, source string) *retry.Error {
	return nil
}

type fakeVirtualMachineScaleSetsClient struct {
	mutex     *sync.Mutex
	FakeStore map[string]map[string]compute.VirtualMachineScaleSet
}

func newFakeVirtualMachineScaleSetsClient() *fakeVirtualMachineScaleSetsClient {
	fVMSSC := &fakeVirtualMachineScaleSetsClient{}
	fVMSSC.FakeStore = make(map[string]map[string]compute.VirtualMachineScaleSet)
	fVMSSC.mutex = &sync.Mutex{}

	return fVMSSC
}

func (fVMSSC *fakeVirtualMachineScaleSetsClient) setFakeStore(store map[string]map[string]compute.VirtualMachineScaleSet) {
	fVMSSC.mutex.Lock()
	defer fVMSSC.mutex.Unlock()

	fVMSSC.FakeStore = store
}

func (fVMSSC *fakeVirtualMachineScaleSetsClient) CreateOrUpdate(ctx context.Context, resourceGroupName string, VMScaleSetName string, parameters compute.VirtualMachineScaleSet) *retry.Error {
	fVMSSC.mutex.Lock()
	defer fVMSSC.mutex.Unlock()

	if _, ok := fVMSSC.FakeStore[resourceGroupName]; !ok {
		fVMSSC.FakeStore[resourceGroupName] = make(map[string]compute.VirtualMachineScaleSet)
	}
	fVMSSC.FakeStore[resourceGroupName][VMScaleSetName] = parameters

	return nil
}

func (fVMSSC *fakeVirtualMachineScaleSetsClient) Get(ctx context.Context, resourceGroupName string, VMScaleSetName string) (result compute.VirtualMachineScaleSet, err *retry.Error) {
	fVMSSC.mutex.Lock()
	defer fVMSSC.mutex.Unlock()

	if scaleSetMap, ok := fVMSSC.FakeStore[resourceGroupName]; ok {
		if entity, ok := scaleSetMap[VMScaleSetName]; ok {
			return entity, nil
		}
	}

	return result, retry.GetError(
		&http.Response{
			StatusCode: http.StatusNotFound,
		},
		errors.New("Not such ScaleSet"))
}

func (fVMSSC *fakeVirtualMachineScaleSetsClient) List(ctx context.Context, resourceGroupName string) (result []compute.VirtualMachineScaleSet, err *retry.Error) {
	fVMSSC.mutex.Lock()
	defer fVMSSC.mutex.Unlock()

	result = []compute.VirtualMachineScaleSet{}
	if _, ok := fVMSSC.FakeStore[resourceGroupName]; ok {
		for _, v := range fVMSSC.FakeStore[resourceGroupName] {
			result = append(result, v)
		}
	}

	return result, nil
}

func (fVMSSC *fakeVirtualMachineScaleSetsClient) UpdateInstances(ctx context.Context, resourceGroupName string, VMScaleSetName string, VMInstanceIDs compute.VirtualMachineScaleSetVMInstanceRequiredIDs) *retry.Error {
	return nil
}

func (fVMSSC *fakeVirtualMachineScaleSetsClient) DeleteInstances(ctx context.Context, resourceGroupName string, vmScaleSetName string, vmInstanceIDs compute.VirtualMachineScaleSetVMInstanceRequiredIDs) *retry.Error {
	return nil
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
	Accounts  []storage.Account
	Err       error
}

func newFakeStorageAccountClient() *fakeStorageAccountClient {
	fSAC := &fakeStorageAccountClient{}
	fSAC.FakeStore = make(map[string]map[string]storage.Account)
	fSAC.mutex = &sync.Mutex{}
	return fSAC
}

func (fSAC *fakeStorageAccountClient) Create(ctx context.Context, resourceGroupName string, accountName string, parameters storage.AccountCreateParameters) *retry.Error {
	fSAC.mutex.Lock()
	defer fSAC.mutex.Unlock()

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

	return nil
}

func (fSAC *fakeStorageAccountClient) Delete(ctx context.Context, resourceGroupName string, accountName string) *retry.Error {
	fSAC.mutex.Lock()
	defer fSAC.mutex.Unlock()

	if rgAccounts, ok := fSAC.FakeStore[resourceGroupName]; ok {
		if _, ok := rgAccounts[accountName]; ok {
			delete(rgAccounts, accountName)
			return nil
		}
	}

	return retry.GetError(
		&http.Response{
			StatusCode: http.StatusNotFound,
		},
		errors.New("Not such StorageAccount"))
}

func (fSAC *fakeStorageAccountClient) ListKeys(ctx context.Context, resourceGroupName string, accountName string) (result storage.AccountListKeysResult, err *retry.Error) {
	return fSAC.Keys, nil
}

func (fSAC *fakeStorageAccountClient) ListByResourceGroup(ctx context.Context, resourceGroupName string) (result []storage.Account, err *retry.Error) {
	return fSAC.Accounts, nil
}

func (fSAC *fakeStorageAccountClient) GetProperties(ctx context.Context, resourceGroupName string, accountName string) (result storage.Account, err *retry.Error) {
	fSAC.mutex.Lock()
	defer fSAC.mutex.Unlock()

	if _, ok := fSAC.FakeStore[resourceGroupName]; ok {
		if entity, ok := fSAC.FakeStore[resourceGroupName][accountName]; ok {
			return entity, nil
		}
	}

	return result, retry.GetError(
		&http.Response{
			StatusCode: http.StatusNotFound,
		},
		errors.New("Not such StorageAccount"))
}

type fakeDisksClient struct {
	mutex     *sync.Mutex
	FakeStore map[string]map[string]compute.Disk
}

func newFakeDisksClient() *fakeDisksClient {
	fDC := &fakeDisksClient{}
	fDC.FakeStore = make(map[string]map[string]compute.Disk)
	fDC.mutex = &sync.Mutex{}
	return fDC
}

func (fDC *fakeDisksClient) CreateOrUpdate(ctx context.Context, resourceGroupName string, diskName string, diskParameter compute.Disk) *retry.Error {
	fDC.mutex.Lock()
	defer fDC.mutex.Unlock()

	provisioningStateSucceeded := string(compute.ProvisioningStateSucceeded)
	diskParameter.DiskProperties = &compute.DiskProperties{ProvisioningState: &provisioningStateSucceeded}
	diskParameter.ID = &diskName

	if _, ok := fDC.FakeStore[resourceGroupName]; !ok {
		fDC.FakeStore[resourceGroupName] = make(map[string]compute.Disk)
	}
	fDC.FakeStore[resourceGroupName][diskName] = diskParameter

	return nil
}

func (fDC *fakeDisksClient) Delete(ctx context.Context, resourceGroupName string, diskName string) *retry.Error {
	fDC.mutex.Lock()
	defer fDC.mutex.Unlock()

	if rgDisks, ok := fDC.FakeStore[resourceGroupName]; ok {
		if _, ok := rgDisks[diskName]; ok {
			delete(rgDisks, diskName)
			return nil
		}
	}

	return retry.GetError(
		&http.Response{
			StatusCode: http.StatusNotFound,
		},
		errors.New("Not such Disk"))
}

func (fDC *fakeDisksClient) Get(ctx context.Context, resourceGroupName string, diskName string) (result compute.Disk, err *retry.Error) {
	fDC.mutex.Lock()
	defer fDC.mutex.Unlock()

	if _, ok := fDC.FakeStore[resourceGroupName]; ok {
		if entity, ok := fDC.FakeStore[resourceGroupName][diskName]; ok {
			return entity, nil
		}
	}

	return result, retry.GetError(
		&http.Response{
			StatusCode: http.StatusNotFound,
		},
		errors.New("Not such Disk"))
}

// GetTestCloud returns a fake azure cloud for unit tests in Azure related CSI drivers
func GetTestCloud(ctrl *gomock.Controller) (az *Cloud) {
	az = &Cloud{
		Config: Config{
			AzureAuthConfig: auth.AzureAuthConfig{
				TenantID:       "tenant",
				SubscriptionID: "subscription",
			},
			ResourceGroup:                "rg",
			VnetResourceGroup:            "rg",
			RouteTableResourceGroup:      "rg",
			SecurityGroupResourceGroup:   "rg",
			Location:                     "westus",
			VnetName:                     "vnet",
			SubnetName:                   "subnet",
			SecurityGroupName:            "nsg",
			RouteTableName:               "rt",
			PrimaryAvailabilitySetName:   "as",
			MaximumLoadBalancerRuleCount: 250,
			VMType:                       vmTypeStandard,
		},
		nodeZones:          map[string]sets.String{},
		nodeInformerSynced: func() bool { return true },
		nodeResourceGroups: map[string]string{},
		unmanagedNodes:     sets.NewString(),
		routeCIDRs:         map[string]string{},
		eventRecorder:      &record.FakeRecorder{},
	}
	az.DisksClient = newFakeDisksClient()
	az.InterfacesClient = newFakeAzureInterfacesClient()
	az.LoadBalancerClient = newFakeAzureLBClient()
	az.PublicIPAddressesClient = newFakeAzurePIPClient(az.Config.SubscriptionID)
	az.RoutesClient = mockrouteclient.NewMockInterface(ctrl)
	az.RouteTablesClient = mockroutetableclient.NewMockInterface(ctrl)
	az.SecurityGroupsClient = newFakeAzureNSGClient()
	az.SubnetsClient = mocksubnetclient.NewMockInterface(ctrl)
	az.VirtualMachineScaleSetsClient = newFakeVirtualMachineScaleSetsClient()
	az.VirtualMachineScaleSetVMsClient = newFakeVirtualMachineScaleSetVMsClient()
	az.VirtualMachinesClient = newFakeAzureVirtualMachinesClient()
	az.vmSet = newAvailabilitySet(az)
	az.vmCache, _ = az.newVMCache()
	az.lbCache, _ = az.newLBCache()
	az.nsgCache, _ = az.newNSGCache()
	az.rtCache, _ = az.newRouteTableCache()

	common := &controllerCommon{cloud: az}
	az.controllerCommon = common
	az.ManagedDiskController = &ManagedDiskController{common: common}

	return az
}

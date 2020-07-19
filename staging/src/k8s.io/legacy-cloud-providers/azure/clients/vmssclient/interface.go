// +build !providerless

/*
Copyright 2019 The Kubernetes Authors.

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

package vmssclient

import (
	"context"
	"net/http"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-12-01/compute"
	"github.com/Azure/go-autorest/autorest/azure"

	"k8s.io/legacy-cloud-providers/azure/retry"
)

const (
	// APIVersion is the API version for VMSS.
	APIVersion = "2019-07-01"
)

// Interface is the client interface for VirtualMachineScaleSet.
// Don't forget to run the following command to generate the mock client:
// mockgen -source=$GOPATH/src/k8s.io/kubernetes/staging/src/k8s.io/legacy-cloud-providers/azure/clients/vmssclient/interface.go -package=mockvmssclient Interface > $GOPATH/src/k8s.io/kubernetes/staging/src/k8s.io/legacy-cloud-providers/azure/clients/vmssclient/mockvmssclient/interface.go
type Interface interface {
	// Get gets a VirtualMachineScaleSet.
	Get(ctx context.Context, resourceGroupName string, VMScaleSetName string) (result compute.VirtualMachineScaleSet, rerr *retry.Error)

	// List gets a list of VirtualMachineScaleSets in the resource group.
	List(ctx context.Context, resourceGroupName string) (result []compute.VirtualMachineScaleSet, rerr *retry.Error)

	// CreateOrUpdate creates or updates a VirtualMachineScaleSet.
	CreateOrUpdate(ctx context.Context, resourceGroupName string, VMScaleSetName string, parameters compute.VirtualMachineScaleSet) *retry.Error

	// CreateOrUpdateSync sends the request to arm client and DO NOT wait for the response
	CreateOrUpdateAsync(ctx context.Context, resourceGroupName string, VMScaleSetName string, parameters compute.VirtualMachineScaleSet) (*azure.Future, *retry.Error)

	// WaitForAsyncOperationResult waits for the response of the request
	WaitForAsyncOperationResult(ctx context.Context, future *azure.Future) (*http.Response, error)

	// DeleteInstances deletes the instances for a VirtualMachineScaleSet.
	DeleteInstances(ctx context.Context, resourceGroupName string, vmScaleSetName string, vmInstanceIDs compute.VirtualMachineScaleSetVMInstanceRequiredIDs) *retry.Error

	// DeleteInstancesAsync sends the delete request to the ARM client and DOEST NOT wait on the future
	DeleteInstancesAsync(ctx context.Context, resourceGroupName string, vmScaleSetName string, vmInstanceIDs compute.VirtualMachineScaleSetVMInstanceRequiredIDs) (*azure.Future, *retry.Error)
}

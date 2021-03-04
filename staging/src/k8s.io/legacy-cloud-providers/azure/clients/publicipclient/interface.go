// +build !providerless

/*
Copyright 2020 The Kubernetes Authors.

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

package publicipclient

import (
	"context"

	"github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-06-01/network"
	"k8s.io/legacy-cloud-providers/azure/retry"
)

const (
	// APIVersion is the API version for network.
	APIVersion = "2019-06-01"

	// ComputeAPIVersion is the API version for compute. It is required to get VMSS public IP.
	ComputeAPIVersion = "2017-03-30"
)

// Interface is the client interface for PublicIPAddress.
// Don't forget to run the following command to generate the mock client:
// mockgen -source=$GOPATH/src/k8s.io/kubernetes/staging/src/k8s.io/legacy-cloud-providers/azure/clients/publicipclient/interface.go -package=mockpublicipclient Interface > $GOPATH/src/k8s.io/kubernetes/staging/src/k8s.io/legacy-cloud-providers/azure/clients/publicipclient/mockpublicipclient/interface.go
type Interface interface {
	// Get gets a PublicIPAddress.
	Get(ctx context.Context, resourceGroupName string, publicIPAddressName string, expand string) (result network.PublicIPAddress, rerr *retry.Error)

	// GetVirtualMachineScaleSetPublicIPAddress gets a PublicIPAddress for VMSS VM.
	GetVirtualMachineScaleSetPublicIPAddress(ctx context.Context, resourceGroupName string, virtualMachineScaleSetName string, virtualmachineIndex string, networkInterfaceName string, IPConfigurationName string, publicIPAddressName string, expand string) (result network.PublicIPAddress, rerr *retry.Error)

	// List gets a list of PublicIPAddress in the resource group.
	List(ctx context.Context, resourceGroupName string) (result []network.PublicIPAddress, rerr *retry.Error)

	// CreateOrUpdate creates or updates a PublicIPAddress.
	CreateOrUpdate(ctx context.Context, resourceGroupName string, publicIPAddressName string, parameters network.PublicIPAddress) *retry.Error

	// Delete deletes a PublicIPAddress by name.
	Delete(ctx context.Context, resourceGroupName string, publicIPAddressName string) *retry.Error
}

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

package containerserviceclient

import (
	"context"

	"k8s.io/legacy-cloud-providers/azure/retry"

	"github.com/Azure/azure-sdk-for-go/services/containerservice/mgmt/2020-04-01/containerservice"
)

const (
	// APIVersion is the API version for containerservice.
	APIVersion = "2020-04-01"
)

// Interface is the client interface for ContainerService.
// Don't forget to run the following command to generate the mock client:
// mockgen -source=$GOPATH/src/k8s.io/kubernetes/staging/src/k8s.io/legacy-cloud-providers/azure/clients/containerserviceclient/interface.go -package=mockcontainerserviceclient Interface > $GOPATH/src/k8s.io/kubernetes/staging/src/k8s.io/legacy-cloud-providers/azure/clients/containerserviceclient/mockcontainerserviceclient/interface.go
type Interface interface {
	CreateOrUpdate(ctx context.Context, resourceGroupName string, managedClusterName string, parameters containerservice.ManagedCluster, etag string) *retry.Error
	Delete(ctx context.Context, resourceGroupName string, managedClusterName string) *retry.Error
	Get(ctx context.Context, resourceGroupName string, managedClusterName string) (containerservice.ManagedCluster, *retry.Error)
	List(ctx context.Context, resourceGroupName string) ([]containerservice.ManagedCluster, *retry.Error)
}

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

package snapshotclient

import (
	"context"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-07-01/compute"
	"k8s.io/legacy-cloud-providers/azure/retry"
)

const (
	// APIVersion is the API version for compute.
	APIVersion = "2019-07-01"
)

// Interface is the client interface for Snapshots.
// Don't forget to run the following command to generate the mock client:
// mockgen -source=$GOPATH/src/k8s.io/kubernetes/staging/src/k8s.io/legacy-cloud-providers/azure/clients/snapshotclient/interface.go -package=mocksnapshotclient Interface > $GOPATH/src/k8s.io/kubernetes/staging/src/k8s.io/legacy-cloud-providers/azure/clients/snapshotclient/mocksnapshotclient/interface.go
type Interface interface {
	// Get gets a Snapshot.
	Get(ctx context.Context, resourceGroupName string, snapshotName string) (compute.Snapshot, *retry.Error)

	// Delete deletes a Snapshot by name.
	Delete(ctx context.Context, resourceGroupName string, snapshotName string) *retry.Error

	// ListByResourceGroup get a list snapshots by resourceGroup.
	ListByResourceGroup(ctx context.Context, resourceGroupName string) ([]compute.Snapshot, *retry.Error)

	// CreateOrUpdate creates or updates a Snapshot.
	CreateOrUpdate(ctx context.Context, resourceGroupName string, snapshotName string, snapshot compute.Snapshot) *retry.Error
}

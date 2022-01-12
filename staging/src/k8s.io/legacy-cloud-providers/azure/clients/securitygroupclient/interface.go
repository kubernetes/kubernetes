//go:build !providerless
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

//go:generate mockgen -copyright_file=$BUILD_TAG_FILE -source=interface.go  -destination=mocksecuritygroupclient/interface.go -package=mocksecuritygroupclient Interface
package securitygroupclient

import (
	"context"

	"github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-06-01/network"
	"k8s.io/legacy-cloud-providers/azure/retry"
)

const (
	// APIVersion is the API version for network.
	APIVersion = "2019-06-01"
)

// Interface is the client interface for SecurityGroups.
type Interface interface {
	// Get gets a SecurityGroup.
	Get(ctx context.Context, resourceGroupName string, networkSecurityGroupName string, expand string) (result network.SecurityGroup, rerr *retry.Error)

	// List gets a list of SecurityGroup in the resource group.
	List(ctx context.Context, resourceGroupName string) (result []network.SecurityGroup, rerr *retry.Error)

	// CreateOrUpdate creates or updates a SecurityGroup.
	CreateOrUpdate(ctx context.Context, resourceGroupName string, networkSecurityGroupName string, parameters network.SecurityGroup, etag string) *retry.Error

	// Delete deletes a SecurityGroup by name.
	Delete(ctx context.Context, resourceGroupName string, networkSecurityGroupName string) *retry.Error
}

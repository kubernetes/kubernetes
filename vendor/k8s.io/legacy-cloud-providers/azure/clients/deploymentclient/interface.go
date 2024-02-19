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

//go:generate mockgen -copyright_file=$BUILD_TAG_FILE -source=interface.go  -destination=mockdeploymentclient/interface.go -package=mockdeploymentclient Interface
package deploymentclient

import (
	"context"

	"github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2017-05-10/resources"
	"k8s.io/legacy-cloud-providers/azure/retry"
)

const (
	// APIVersion is the API version for resources.
	APIVersion = "2017-05-10"
)

// Interface is the client interface for Deployments.
type Interface interface {
	Get(ctx context.Context, resourceGroupName string, deploymentName string) (resources.DeploymentExtended, *retry.Error)
	List(ctx context.Context, resourceGroupName string) ([]resources.DeploymentExtended, *retry.Error)
	ExportTemplate(ctx context.Context, resourceGroupName string, deploymentName string) (result resources.DeploymentExportResult, rerr *retry.Error)
	CreateOrUpdate(ctx context.Context, resourceGroupName string, managedClusterName string, parameters resources.Deployment, etag string) *retry.Error
	Delete(ctx context.Context, resourceGroupName string, deploymentName string) *retry.Error
}

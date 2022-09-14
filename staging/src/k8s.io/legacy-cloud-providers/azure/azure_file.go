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

	"github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2019-06-01/storage"
	"k8s.io/legacy-cloud-providers/azure/clients/fileclient"
)

// create file share
func (az *Cloud) createFileShare(ctx context.Context, resourceGroupName, accountName string, shareOptions *fileclient.ShareOptions) error {
	return az.FileClient.CreateFileShare(ctx, resourceGroupName, accountName, shareOptions)
}

func (az *Cloud) deleteFileShare(ctx context.Context, resourceGroupName, accountName, name string) error {
	return az.FileClient.DeleteFileShare(ctx, resourceGroupName, accountName, name)
}

func (az *Cloud) resizeFileShare(ctx context.Context, resourceGroupName, accountName, name string, sizeGiB int) error {
	return az.FileClient.ResizeFileShare(ctx, resourceGroupName, accountName, name, sizeGiB)
}

func (az *Cloud) getFileShare(ctx context.Context, resourceGroupName, accountName, name string) (storage.FileShare, error) {
	return az.FileClient.GetFileShare(ctx, resourceGroupName, accountName, name)
}

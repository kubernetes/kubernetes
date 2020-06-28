// +build !providerless

/*
Copyright 2016 The Kubernetes Authors.

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
	"fmt"

	"github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2019-06-01/storage"

	"k8s.io/klog/v2"
)

const (
	defaultStorageAccountType      = string(storage.StandardLRS)
	defaultStorageAccountKind      = storage.StorageV2
	fileShareAccountNamePrefix     = "f"
	sharedDiskAccountNamePrefix    = "ds"
	dedicatedDiskAccountNamePrefix = "dd"
)

// CreateFileShare creates a file share, using a matching storage account type, account kind, etc.
// storage account will be created if specified account is not found
func (az *Cloud) CreateFileShare(shareName, accountName, accountType, accountKind, resourceGroup, location string, protocol storage.EnabledProtocols, requestGiB int) (string, string, error) {
	if resourceGroup == "" {
		resourceGroup = az.resourceGroup
	}

	enableHTTPSTrafficOnly := true
	if protocol == storage.NFS {
		enableHTTPSTrafficOnly = false
	}
	account, key, err := az.EnsureStorageAccount(accountName, accountType, accountKind, resourceGroup, location, fileShareAccountNamePrefix, enableHTTPSTrafficOnly)
	if err != nil {
		return "", "", fmt.Errorf("could not get storage key for storage account %s: %v", accountName, err)
	}

	if err := az.createFileShare(resourceGroup, account, shareName, protocol, requestGiB); err != nil {
		return "", "", fmt.Errorf("failed to create share %s in account %s: %v", shareName, account, err)
	}
	klog.V(4).Infof("created share %s in account %s", shareName, account)
	return account, key, nil
}

// DeleteFileShare deletes a file share using storage account name and key
func (az *Cloud) DeleteFileShare(resourceGroup, accountName, shareName string) error {
	if err := az.deleteFileShare(resourceGroup, accountName, shareName); err != nil {
		return err
	}
	klog.V(4).Infof("share %s deleted", shareName)
	return nil
}

// ResizeFileShare resizes a file share
func (az *Cloud) ResizeFileShare(resourceGroup, accountName, name string, sizeGiB int) error {
	return az.resizeFileShare(resourceGroup, accountName, name, sizeGiB)
}

// GetFileShare gets a file share
func (az *Cloud) GetFileShare(resourceGroupName, accountName, name string) (storage.FileShare, error) {
	return az.getFileShare(resourceGroupName, accountName, name)
}

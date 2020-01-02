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
	"fmt"

	azs "github.com/Azure/azure-sdk-for-go/storage"
	"github.com/Azure/go-autorest/autorest/azure"

	"k8s.io/klog"
)

const (
	useHTTPS = true
)

// FileClient is the interface for creating file shares, interface for test
// injection.
type FileClient interface {
	createFileShare(accountName, accountKey, name string, sizeGiB int) error
	deleteFileShare(accountName, accountKey, name string) error
	resizeFileShare(accountName, accountKey, name string, sizeGiB int) error
}

// create file share
func (az *Cloud) createFileShare(accountName, accountKey, name string, sizeGiB int) error {
	return az.FileClient.createFileShare(accountName, accountKey, name, sizeGiB)
}

func (az *Cloud) deleteFileShare(accountName, accountKey, name string) error {
	return az.FileClient.deleteFileShare(accountName, accountKey, name)
}

func (az *Cloud) resizeFileShare(accountName, accountKey, name string, sizeGiB int) error {
	return az.FileClient.resizeFileShare(accountName, accountKey, name, sizeGiB)
}

type azureFileClient struct {
	env azure.Environment
}

func (f *azureFileClient) createFileShare(accountName, accountKey, name string, sizeGiB int) error {
	fileClient, err := f.getFileSvcClient(accountName, accountKey)
	if err != nil {
		return err
	}
	share := fileClient.GetShareReference(name)
	share.Properties.Quota = sizeGiB
	newlyCreated, err := share.CreateIfNotExists(nil)
	if err != nil {
		return fmt.Errorf("failed to create file share, err: %v", err)
	}
	if !newlyCreated {
		klog.V(2).Infof("file share(%s) under account(%s) already exists", name, accountName)
	}
	return nil
}

// delete a file share
func (f *azureFileClient) deleteFileShare(accountName, accountKey, name string) error {
	fileClient, err := f.getFileSvcClient(accountName, accountKey)
	if err != nil {
		return err
	}
	return fileClient.GetShareReference(name).Delete(nil)
}

func (f *azureFileClient) resizeFileShare(accountName, accountKey, name string, sizeGiB int) error {
	fileClient, err := f.getFileSvcClient(accountName, accountKey)
	if err != nil {
		return err
	}
	share := fileClient.GetShareReference(name)
	if share.Properties.Quota >= sizeGiB {
		klog.Warningf("file share size(%dGi) is already greater or equal than requested size(%dGi), accountName: %s, shareName: %s",
			share.Properties.Quota, sizeGiB, accountName, name)
		return nil
	}
	share.Properties.Quota = sizeGiB
	if err = share.SetProperties(nil); err != nil {
		return fmt.Errorf("failed to set quota on file share %s, err: %v", name, err)
	}
	klog.V(4).Infof("resize file share completed, accountName: %s, shareName: %s, sizeGiB: %d", accountName, name, sizeGiB)
	return nil
}

func (f *azureFileClient) getFileSvcClient(accountName, accountKey string) (*azs.FileServiceClient, error) {
	fileClient, err := azs.NewClient(accountName, accountKey, f.env.StorageEndpointSuffix, azs.DefaultAPIVersion, useHTTPS)
	if err != nil {
		return nil, fmt.Errorf("error creating azure client: %v", err)
	}
	fc := fileClient.GetFileService()
	return &fc, nil
}

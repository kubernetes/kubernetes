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

package fileclient

import (
	"context"
	"fmt"
	"net/http"
	"strings"

	"github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2019-06-01/storage"

	"k8s.io/klog/v2"
	azclients "k8s.io/legacy-cloud-providers/azure/clients"
)

// Client implements the azure file client interface
type Client struct {
	fileSharesClient storage.FileSharesClient
}

// ShareOptions contains the fields which are used to create file share.
type ShareOptions struct {
	Name       string
	Protocol   storage.EnabledProtocols
	RequestGiB int
}

// New creates a azure file client
func New(config *azclients.ClientConfig) *Client {
	client := storage.NewFileSharesClientWithBaseURI(config.ResourceManagerEndpoint, config.SubscriptionID)
	client.Authorizer = config.Authorizer

	return &Client{
		fileSharesClient: client,
	}
}

// CreateFileShare creates a file share
func (c *Client) CreateFileShare(resourceGroupName, accountName string, shareOptions *ShareOptions) error {
	if shareOptions == nil {
		return fmt.Errorf("share options is nil")
	}
	result, err := c.GetFileShare(resourceGroupName, accountName, shareOptions.Name)
	if err == nil {
		klog.V(2).Infof("file share(%s) under account(%s) rg(%s) already exists", shareOptions.Name, accountName, resourceGroupName)
		return nil
	} else if result.Response.Response == nil || (err != nil && result.Response.Response.StatusCode != http.StatusNotFound && !strings.Contains(err.Error(), "ShareNotFound")) {
		return fmt.Errorf("failed to get file share(%s), err: %v", shareOptions.Name, err)
	}

	quota := int32(shareOptions.RequestGiB)
	fileShareProperties := &storage.FileShareProperties{
		ShareQuota: &quota,
	}
	if shareOptions.Protocol == storage.NFS {
		fileShareProperties.EnabledProtocols = shareOptions.Protocol
	}
	fileShare := storage.FileShare{
		Name:                &shareOptions.Name,
		FileShareProperties: fileShareProperties,
	}
	_, err = c.fileSharesClient.Create(context.Background(), resourceGroupName, accountName, shareOptions.Name, fileShare)

	return err
}

// DeleteFileShare deletes a file share
func (c *Client) DeleteFileShare(resourceGroupName, accountName, name string) error {
	_, err := c.fileSharesClient.Delete(context.Background(), resourceGroupName, accountName, name)

	return err
}

// ResizeFileShare resizes a file share
func (c *Client) ResizeFileShare(resourceGroupName, accountName, name string, sizeGiB int) error {
	quota := int32(sizeGiB)

	share, err := c.fileSharesClient.Get(context.Background(), resourceGroupName, accountName, name, storage.Stats)
	if err != nil {
		return fmt.Errorf("failed to get file share(%s), : %v", name, err)
	}
	if *share.FileShareProperties.ShareQuota >= quota {
		klog.Warningf("file share size(%dGi) is already greater or equal than requested size(%dGi), accountName: %s, shareName: %s",
			share.FileShareProperties.ShareQuota, sizeGiB, accountName, name)
		return nil

	}

	share.FileShareProperties.ShareQuota = &quota
	_, err = c.fileSharesClient.Update(context.Background(), resourceGroupName, accountName, name, share)
	if err != nil {
		return fmt.Errorf("failed to update quota on file share(%s), err: %v", name, err)
	}

	klog.V(4).Infof("resize file share completed, resourceGroupName(%s), accountName: %s, shareName: %s, sizeGiB: %d", resourceGroupName, accountName, name, sizeGiB)

	return nil
}

// GetFileShare gets a file share
func (c *Client) GetFileShare(resourceGroupName, accountName, name string) (storage.FileShare, error) {
	return c.fileSharesClient.Get(context.Background(), resourceGroupName, accountName, name, storage.Stats)
}

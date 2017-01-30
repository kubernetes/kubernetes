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
	"regexp"
	"strings"

	azs "github.com/Azure/azure-sdk-for-go/storage"
)

const (
	vhdContainerName = "vhds"
	useHTTPS         = true
	blobServiceName  = "blob"
)

// create page blob
func (az *Cloud) createVhdBlob(accountName, accountKey, name string, sizeGB int64, tags map[string]string) (string, string, error) {
	blobClient, err := az.getBlobClient(accountName, accountKey)
	if err != nil {
		return "", "", err
	}
	size := 1024 * 1024 * 1024 * sizeGB
	vhdSize := size + vhdHeaderSize /* header size */
	// Blob name in URL must end with '.vhd' extension.
	name = name + ".vhd"
	err = blobClient.PutPageBlob(vhdContainerName, name, vhdSize, tags)
	if err != nil {
		// if container doesn't exist, create one and retry PutPageBlob
		detail := err.Error()
		if strings.Contains(detail, errContainerNotFound) {
			err = blobClient.CreateContainer(vhdContainerName, azs.ContainerAccessTypeContainer)
			if err == nil {
				err = blobClient.PutPageBlob(vhdContainerName, name, vhdSize, tags)
			}
		}
	}
	if err != nil {
		return "", "", fmt.Errorf("failed to put page blob: %v", err)
	}

	// add VHD signature to the blob
	h, err := createVHDHeader(uint64(size))
	if err != nil {
		az.deleteVhdBlob(accountName, accountKey, name)
		return "", "", fmt.Errorf("failed to create vhd header, err: %v", err)
	}
	if err = blobClient.PutPage(vhdContainerName, name, size, vhdSize-1, azs.PageWriteTypeUpdate, h[:vhdHeaderSize], nil); err != nil {
		az.deleteVhdBlob(accountName, accountKey, name)
		return "", "", fmt.Errorf("failed to update vhd header, err: %v", err)
	}

	scheme := "http"
	if useHTTPS {
		scheme = "https"
	}
	host := fmt.Sprintf("%s://%s.%s.%s", scheme, accountName, blobServiceName, az.Environment.StorageEndpointSuffix)
	uri := fmt.Sprintf("%s/%s/%s", host, vhdContainerName, name)
	return name, uri, nil

}

// delete a vhd blob
func (az *Cloud) deleteVhdBlob(accountName, accountKey, blobName string) error {
	blobClient, err := az.getBlobClient(accountName, accountKey)
	if err == nil {
		return blobClient.DeleteBlob(vhdContainerName, blobName, nil)
	}
	return err
}

func (az *Cloud) getBlobClient(accountName, accountKey string) (*azs.BlobStorageClient, error) {
	client, err := azs.NewClient(accountName, accountKey, az.Environment.StorageEndpointSuffix, azs.DefaultAPIVersion, useHTTPS)
	if err != nil {
		return nil, fmt.Errorf("error creating azure client: %v", err)
	}
	b := client.GetBlobService()
	return &b, nil
}

// get uri https://foo.blob.core.windows.net/vhds/bar.vhd and return foo (account) and bar.vhd (blob name)
func (az *Cloud) getBlobNameAndAccountFromURI(uri string) (string, string, error) {
	scheme := "http"
	if useHTTPS {
		scheme = "https"
	}
	host := fmt.Sprintf("%s://(.*).%s.%s", scheme, blobServiceName, az.Environment.StorageEndpointSuffix)
	reStr := fmt.Sprintf("%s/%s/(.*)", host, vhdContainerName)
	re := regexp.MustCompile(reStr)
	res := re.FindSubmatch([]byte(uri))
	if len(res) < 3 {
		return "", "", fmt.Errorf("invalid vhd URI for regex %s: %s", reStr, uri)
	}
	return string(res[1]), string(res[2]), nil
}

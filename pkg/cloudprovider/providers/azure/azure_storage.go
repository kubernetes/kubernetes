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

	"github.com/Azure/azure-sdk-for-go/arm/storage"
	"github.com/Azure/go-autorest/autorest/to"
	"github.com/golang/glog"
)

const (
	defaultStorageAccountType  = string(storage.StandardLRS)
	fileShareAccountNamePrefix = "f"
)

// CreateFileShare creates a file share, using a matching storage account
func (az *Cloud) CreateFileShare(shareName, accountName, accountType, location string, requestGiB int) (string, string, error) {
	if len(accountName) == 0 {
		// find a storage account that matches accountType
		accounts, err := az.getStorageAccounts(accountType, location)
		if err != nil {
			return "", "", fmt.Errorf("could not list storage accounts for account type %s: %v", accountType, err)
		}

		if len(accounts) > 0 {
			accountName = accounts[0].Name
			glog.V(4).Infof("found a matching account %s type %s location %s", accounts[0].Name, accounts[0].StorageType, accounts[0].Location)
		}

		if len(accountName) == 0 {
			// not found a matching account, now create a new account in current resource group
			accountName = generateStorageAccountName(fileShareAccountNamePrefix)
			if location == "" {
				location = az.Location
			}
			if accountType == "" {
				accountType = defaultStorageAccountType
			}

			glog.V(2).Infof("azureFile - no matching account found, begin to create a new account %s in resource group %s, location: %s, accountType: %s",
				accountName, az.ResourceGroup, location, accountType)
			cp := storage.AccountCreateParameters{
				Sku:      &storage.Sku{Name: storage.SkuName(accountType)},
				Tags:     &map[string]*string{"created-by": to.StringPtr("azure-file")},
				Location: &location}
			cancel := make(chan struct{})

			_, errchan := az.StorageAccountClient.Create(az.ResourceGroup, accountName, cp, cancel)
			err := <-errchan
			if err != nil {
				return "", "", fmt.Errorf(fmt.Sprintf("Failed to create storage account %s, error: %s", accountName, err))
			}
		}
	}

	// find the access key with this account
	accountKey, err := az.getStorageAccesskey(accountName)
	if err != nil {
		return "", "", fmt.Errorf("could not get storage key for storage account %s: %v", accountName, err)
	}

	if err := az.createFileShare(accountName, accountKey, shareName, requestGiB); err != nil {
		return "", "", fmt.Errorf("failed to create share %s in account %s: %v", shareName, accountName, err)
	}
	glog.V(4).Infof("created share %s in account %s", shareName, accountName)
	return accountName, accountKey, nil
}

// DeleteFileShare deletes a file share using storage account name and key
func (az *Cloud) DeleteFileShare(accountName, accountKey, shareName string) error {
	if err := az.deleteFileShare(accountName, accountKey, shareName); err != nil {
		return err
	}
	glog.V(4).Infof("share %s deleted", shareName)
	return nil
}

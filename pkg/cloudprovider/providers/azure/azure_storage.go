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

	"github.com/golang/glog"
)

// CreateFileShare creates a file share, using a matching storage account
func (az *Cloud) CreateFileShare(name, storageAccount, storageType, location string, requestGB int) (string, string, error) {
	var errResult error
	accounts := []accountWithLocation{}
	if len(storageAccount) > 0 {
		accounts = append(accounts, accountWithLocation{Name: storageAccount})
	} else {
		// find a storage account
		accounts, errResult = az.getStorageAccounts()
		if errResult != nil {
			// TODO: create a storage account and container
			return "", "", errResult
		}
	}
	for _, account := range accounts {
		glog.V(4).Infof("account %s type %s location %s", account.Name, account.StorageType, account.Location)
		if ((storageType == "" || account.StorageType == storageType) && (location == "" || account.Location == location)) || len(storageAccount) > 0 {
			// find the access key with this account
			key, innerErr := az.getStorageAccesskey(account.Name)
			if innerErr != nil {
				errResult = fmt.Errorf("could not get storage key for storage account %s: %v", account.Name, innerErr)
				continue
			}

			if innerErr = az.createFileShare(account.Name, key, name, requestGB); innerErr != nil {
				errResult = fmt.Errorf("failed to create share %s in account %s: %v", name, account.Name, innerErr)
				continue
			}
			glog.V(4).Infof("created share %s in account %s", name, account.Name)
			return account.Name, key, nil
		}
	}

	if errResult == nil {
		errResult = fmt.Errorf("failed to find a matching storage account")
	}
	return "", "", errResult
}

// DeleteFileShare deletes a file share using storage account name and key
func (az *Cloud) DeleteFileShare(accountName, key, name string) error {
	if err := az.deleteFileShare(accountName, key, name); err != nil {
		return err
	}
	glog.V(4).Infof("share %s deleted", name)
	return nil
}

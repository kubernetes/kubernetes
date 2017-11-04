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
	var err error
	accounts := []accountWithLocation{}
	if len(storageAccount) > 0 {
		accounts = append(accounts, accountWithLocation{Name: storageAccount})
	} else {
		// find a storage account
		accounts, err = az.getStorageAccounts()
		if err != nil {
			// TODO: create a storage account and container
			return "", "", err
		}
	}
	for _, account := range accounts {
		glog.V(4).Infof("account %s type %s location %s", account.Name, account.StorageType, account.Location)
		if ((storageType == "" || account.StorageType == storageType) && (location == "" || account.Location == location)) || len(storageAccount) > 0 {
			// find the access key with this account
			key, err := az.getStorageAccesskey(account.Name)
			if err != nil {
				glog.V(2).Infof("no key found for storage account %s", account.Name)
				continue
			}

			err = az.createFileShare(account.Name, key, name, requestGB)
			if err != nil {
				glog.V(2).Infof("failed to create share %s in account %s: %v", name, account.Name, err)
				continue
			}
			glog.V(4).Infof("created share %s in account %s", name, account.Name)
			return account.Name, key, err
		}
	}
	return "", "", fmt.Errorf("failed to find a matching storage account")
}

// DeleteFileShare deletes a file share using storage account name and key
func (az *Cloud) DeleteFileShare(accountName, key, name string) error {
	err := az.deleteFileShare(accountName, key, name)
	if err != nil {
		return err
	}
	glog.V(4).Infof("share %s deleted", name)
	return nil

}

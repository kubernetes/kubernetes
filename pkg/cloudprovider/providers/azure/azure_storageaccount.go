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
	"strings"
)

type accountWithLocation struct {
	Name, StorageType, Location string
}

// getStorageAccounts gets the storage accounts' name, type, location in a resource group
func (az *Cloud) getStorageAccounts() ([]accountWithLocation, error) {
	az.operationPollRateLimiter.Accept()
	result, err := az.StorageAccountClient.ListByResourceGroup(az.ResourceGroup)
	if err != nil {
		return nil, err
	}
	if result.Value == nil {
		return nil, fmt.Errorf("no storage accounts from resource group %s", az.ResourceGroup)
	}

	accounts := []accountWithLocation{}
	for _, acct := range *result.Value {
		if acct.Name != nil {
			name := *acct.Name
			loc := ""
			if acct.Location != nil {
				loc = *acct.Location
			}
			storageType := ""
			if acct.Sku != nil {
				storageType = string((*acct.Sku).Name)
			}
			accounts = append(accounts, accountWithLocation{Name: name, StorageType: storageType, Location: loc})
		}
	}

	return accounts, nil
}

// getStorageAccesskey gets the storage account access key
func (az *Cloud) getStorageAccesskey(account string) (string, error) {
	az.operationPollRateLimiter.Accept()
	result, err := az.StorageAccountClient.ListKeys(az.ResourceGroup, account)
	if err != nil {
		return "", err
	}
	if result.Keys == nil {
		return "", fmt.Errorf("empty keys")
	}

	for _, k := range *result.Keys {
		if k.Value != nil && *k.Value != "" {
			v := *k.Value
			if ind := strings.LastIndex(v, " "); ind >= 0 {
				v = v[(ind + 1):]
			}
			return v, nil
		}
	}
	return "", fmt.Errorf("no valid keys")
}

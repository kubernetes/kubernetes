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
	"strings"

	"github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2019-06-01/storage"
	"github.com/Azure/go-autorest/autorest/to"

	"k8s.io/klog/v2"
)

// AccountOptions contains the fields which are used to create storage account.
type AccountOptions struct {
	Name, Type, Kind, ResourceGroup, Location string
	EnableHTTPSTrafficOnly                    bool
	Tags                                      map[string]string
	VirtualNetworkResourceIDs                 []string
}

type accountWithLocation struct {
	Name, StorageType, Location string
}

// getStorageAccounts get matching storage accounts
func (az *Cloud) getStorageAccounts(accountOptions *AccountOptions) ([]accountWithLocation, error) {
	ctx, cancel := getContextWithCancel()
	defer cancel()
	result, rerr := az.StorageAccountClient.ListByResourceGroup(ctx, accountOptions.ResourceGroup)
	if rerr != nil {
		return nil, rerr.Error()
	}

	accounts := []accountWithLocation{}
	for _, acct := range result {
		if acct.Name != nil && acct.Location != nil && acct.Sku != nil {
			storageType := string((*acct.Sku).Name)
			if accountOptions.Type != "" && !strings.EqualFold(accountOptions.Type, storageType) {
				continue
			}

			if accountOptions.Kind != "" && !strings.EqualFold(accountOptions.Kind, string(acct.Kind)) {
				continue
			}

			location := *acct.Location
			if accountOptions.Location != "" && !strings.EqualFold(accountOptions.Location, location) {
				continue
			}

			if len(accountOptions.VirtualNetworkResourceIDs) > 0 {
				if acct.AccountProperties == nil || acct.AccountProperties.NetworkRuleSet == nil ||
					acct.AccountProperties.NetworkRuleSet.VirtualNetworkRules == nil {
					continue
				}

				found := false
				for _, subnetID := range accountOptions.VirtualNetworkResourceIDs {
					for _, rule := range *acct.AccountProperties.NetworkRuleSet.VirtualNetworkRules {
						if strings.EqualFold(to.String(rule.VirtualNetworkResourceID), subnetID) && rule.Action == storage.Allow {
							found = true
							break
						}
					}
				}
				if !found {
					continue
				}
			}

			accounts = append(accounts, accountWithLocation{Name: *acct.Name, StorageType: storageType, Location: location})
		}
	}

	return accounts, nil
}

// GetStorageAccesskey gets the storage account access key
func (az *Cloud) GetStorageAccesskey(account, resourceGroup string) (string, error) {
	ctx, cancel := getContextWithCancel()
	defer cancel()

	result, rerr := az.StorageAccountClient.ListKeys(ctx, resourceGroup, account)
	if rerr != nil {
		return "", rerr.Error()
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

// EnsureStorageAccount search storage account, create one storage account(with genAccountNamePrefix) if not found, return accountName, accountKey
func (az *Cloud) EnsureStorageAccount(accountOptions *AccountOptions, genAccountNamePrefix string) (string, string, error) {
	if accountOptions == nil {
		return "", "", fmt.Errorf("account options is nil")
	}
	accountName := accountOptions.Name
	accountType := accountOptions.Type
	accountKind := accountOptions.Kind
	resourceGroup := accountOptions.ResourceGroup
	location := accountOptions.Location
	enableHTTPSTrafficOnly := accountOptions.EnableHTTPSTrafficOnly

	if len(accountName) == 0 {
		// find a storage account that matches accountType
		accounts, err := az.getStorageAccounts(accountOptions)
		if err != nil {
			return "", "", fmt.Errorf("could not list storage accounts for account type %s: %v", accountType, err)
		}

		if len(accounts) > 0 {
			accountName = accounts[0].Name
			klog.V(4).Infof("found a matching account %s type %s location %s", accounts[0].Name, accounts[0].StorageType, accounts[0].Location)
		}

		if len(accountName) == 0 {
			// set network rules for storage account
			var networkRuleSet *storage.NetworkRuleSet
			virtualNetworkRules := []storage.VirtualNetworkRule{}
			for _, subnetID := range accountOptions.VirtualNetworkResourceIDs {
				vnetRule := storage.VirtualNetworkRule{
					VirtualNetworkResourceID: &subnetID,
					Action:                   storage.Allow,
				}
				virtualNetworkRules = append(virtualNetworkRules, vnetRule)
				klog.V(4).Infof("subnetID(%s) has been set", subnetID)
			}
			if len(virtualNetworkRules) > 0 {
				networkRuleSet = &storage.NetworkRuleSet{
					VirtualNetworkRules: &virtualNetworkRules,
					DefaultAction:       storage.DefaultActionDeny,
				}
			}

			// not found a matching account, now create a new account in current resource group
			accountName = generateStorageAccountName(genAccountNamePrefix)
			if location == "" {
				location = az.Location
			}
			if accountType == "" {
				accountType = defaultStorageAccountType
			}

			// use StorageV2 by default per https://docs.microsoft.com/en-us/azure/storage/common/storage-account-options
			kind := defaultStorageAccountKind
			if accountKind != "" {
				kind = storage.Kind(accountKind)
			}
			if len(accountOptions.Tags) == 0 {
				accountOptions.Tags = make(map[string]string)
			}
			accountOptions.Tags["created-by"] = "azure"
			tags := convertMapToMapPointer(accountOptions.Tags)

			klog.V(2).Infof("azure - no matching account found, begin to create a new account %s in resource group %s, location: %s, accountType: %s, accountKind: %s, tags: %+v",
				accountName, resourceGroup, location, accountType, kind, accountOptions.Tags)

			cp := storage.AccountCreateParameters{
				Sku:  &storage.Sku{Name: storage.SkuName(accountType)},
				Kind: kind,
				AccountPropertiesCreateParameters: &storage.AccountPropertiesCreateParameters{
					EnableHTTPSTrafficOnly: &enableHTTPSTrafficOnly,
					NetworkRuleSet:         networkRuleSet,
				},
				Tags:     tags,
				Location: &location}

			ctx, cancel := getContextWithCancel()
			defer cancel()
			rerr := az.StorageAccountClient.Create(ctx, resourceGroup, accountName, cp)
			if rerr != nil {
				return "", "", fmt.Errorf(fmt.Sprintf("Failed to create storage account %s, error: %v", accountName, rerr))
			}
		}
	}

	// find the access key with this account
	accountKey, err := az.GetStorageAccesskey(accountName, resourceGroup)
	if err != nil {
		return "", "", fmt.Errorf("could not get storage key for storage account %s: %v", accountName, err)
	}

	return accountName, accountKey, nil
}

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
	"testing"

	"k8s.io/legacy-cloud-providers/azure/clients/storageaccountclient/mockstorageaccountclient"

	"github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2019-06-01/storage"
	"github.com/golang/mock/gomock"
)

func TestGetStorageAccessKeys(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	cloud := &Cloud{}
	value := "foo bar"

	tests := []struct {
		results     storage.AccountListKeysResult
		expectedKey string
		expectErr   bool
		err         error
	}{
		{storage.AccountListKeysResult{}, "", true, nil},
		{
			storage.AccountListKeysResult{
				Keys: &[]storage.AccountKey{
					{Value: &value},
				},
			},
			"bar",
			false,
			nil,
		},
		{
			storage.AccountListKeysResult{
				Keys: &[]storage.AccountKey{
					{},
					{Value: &value},
				},
			},
			"bar",
			false,
			nil,
		},
		{storage.AccountListKeysResult{}, "", true, fmt.Errorf("test error")},
	}

	for _, test := range tests {
		mockStorageAccountsClient := mockstorageaccountclient.NewMockInterface(ctrl)
		cloud.StorageAccountClient = mockStorageAccountsClient
		mockStorageAccountsClient.EXPECT().ListKeys(gomock.Any(), "rg", gomock.Any()).Return(test.results, nil).AnyTimes()
		key, err := cloud.GetStorageAccesskey("acct", "rg")
		if test.expectErr && err == nil {
			t.Errorf("Unexpected non-error")
			continue
		}
		if !test.expectErr && err != nil {
			t.Errorf("Unexpected error: %v", err)
			continue
		}
		if key != test.expectedKey {
			t.Errorf("expected: %s, saw %s", test.expectedKey, key)
		}
	}
}

func TestGetStorageAccount(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	cloud := &Cloud{}

	name := "testAccount"
	location := "testLocation"
	networkID := "networkID"
	accountProperties := storage.AccountProperties{
		NetworkRuleSet: &storage.NetworkRuleSet{
			VirtualNetworkRules: &[]storage.VirtualNetworkRule{
				{
					VirtualNetworkResourceID: &networkID,
					Action:                   storage.Allow,
					State:                    "state",
				},
			},
		}}

	account := storage.Account{
		Sku: &storage.Sku{
			Name: "testSku",
			Tier: "testSkuTier",
		},
		Kind:              "testKind",
		Location:          &location,
		Name:              &name,
		AccountProperties: &accountProperties,
	}

	testResourceGroups := []storage.Account{account}

	accountOptions := &AccountOptions{
		ResourceGroup:             "rg",
		VirtualNetworkResourceIDs: []string{networkID},
	}

	mockStorageAccountsClient := mockstorageaccountclient.NewMockInterface(ctrl)
	cloud.StorageAccountClient = mockStorageAccountsClient

	mockStorageAccountsClient.EXPECT().ListByResourceGroup(gomock.Any(), "rg").Return(testResourceGroups, nil).Times(1)

	accountsWithLocations, err := cloud.getStorageAccounts(accountOptions)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if accountsWithLocations == nil {
		t.Error("unexpected error as returned accounts are nil")
	}

	if len(accountsWithLocations) == 0 {
		t.Error("unexpected error as returned accounts slice is empty")
	}

	expectedAccountWithLocation := accountWithLocation{
		Name:        "testAccount",
		StorageType: "testSku",
		Location:    "testLocation",
	}

	accountWithLocation := accountsWithLocations[0]
	if accountWithLocation.Name != expectedAccountWithLocation.Name {
		t.Errorf("expected %s, but was %s", accountWithLocation.Name, expectedAccountWithLocation.Name)
	}

	if accountWithLocation.StorageType != expectedAccountWithLocation.StorageType {
		t.Errorf("expected %s, but was %s", accountWithLocation.StorageType, expectedAccountWithLocation.StorageType)
	}

	if accountWithLocation.Location != expectedAccountWithLocation.Location {
		t.Errorf("expected %s, but was %s", accountWithLocation.Location, expectedAccountWithLocation.Location)
	}
}

func TestGetStorageAccountEdgeCases(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	cloud := &Cloud{}

	// default account with name, location, sku, kind
	name := "testAccount"
	location := "testLocation"
	sku := &storage.Sku{
		Name: "testSku",
		Tier: "testSkuTier",
	}
	account := storage.Account{
		Sku:      sku,
		Kind:     "testKind",
		Location: &location,
		Name:     &name,
	}

	accountPropertiesWithoutNetworkRuleSet := storage.AccountProperties{NetworkRuleSet: nil}
	accountPropertiesWithoutVirtualNetworkRules := storage.AccountProperties{
		NetworkRuleSet: &storage.NetworkRuleSet{
			VirtualNetworkRules: nil,
		}}

	tests := []struct {
		testCase           string
		testAccountOptions *AccountOptions
		testResourceGroups []storage.Account
		expectedResult     []accountWithLocation
		expectedError      error
	}{
		{
			testCase: "account name is nil",
			testAccountOptions: &AccountOptions{
				ResourceGroup: "rg",
			},
			testResourceGroups: []storage.Account{},
			expectedResult:     []accountWithLocation{},
			expectedError:      nil,
		},
		{
			testCase: "account location is nil",
			testAccountOptions: &AccountOptions{
				ResourceGroup: "rg",
			},
			testResourceGroups: []storage.Account{{Name: &name}},
			expectedResult:     []accountWithLocation{},
			expectedError:      nil,
		},
		{
			testCase: "account sku is nil",
			testAccountOptions: &AccountOptions{
				ResourceGroup: "rg",
			},
			testResourceGroups: []storage.Account{{Name: &name, Location: &location}},
			expectedResult:     []accountWithLocation{},
			expectedError:      nil,
		},
		{
			testCase: "account options type is not empty and not equal account storage type",
			testAccountOptions: &AccountOptions{
				ResourceGroup: "rg",
				Type:          "testAccountOptionsType",
			},
			testResourceGroups: []storage.Account{account},
			expectedResult:     []accountWithLocation{},
			expectedError:      nil,
		},
		{
			testCase: "account options kind is not empty and not equal account type",
			testAccountOptions: &AccountOptions{
				ResourceGroup: "rg",
				Kind:          "testAccountOptionsKind",
			},
			testResourceGroups: []storage.Account{account},
			expectedResult:     []accountWithLocation{},
			expectedError:      nil,
		},
		{
			testCase: "account options location is not empty and not equal account location",
			testAccountOptions: &AccountOptions{
				ResourceGroup: "rg",
				Location:      "testAccountOptionsLocation",
			},
			testResourceGroups: []storage.Account{account},
			expectedResult:     []accountWithLocation{},
			expectedError:      nil,
		},
		{
			testCase: "account options account properties are nil",
			testAccountOptions: &AccountOptions{
				ResourceGroup:             "rg",
				VirtualNetworkResourceIDs: []string{"id"},
			},
			testResourceGroups: []storage.Account{},
			expectedResult:     []accountWithLocation{},
			expectedError:      nil,
		},
		{
			testCase: "account options account properties network rule set is nil",
			testAccountOptions: &AccountOptions{
				ResourceGroup:             "rg",
				VirtualNetworkResourceIDs: []string{"id"},
			},
			testResourceGroups: []storage.Account{{Name: &name, Kind: "kind", Location: &location, Sku: sku, AccountProperties: &accountPropertiesWithoutNetworkRuleSet}},
			expectedResult:     []accountWithLocation{},
			expectedError:      nil,
		},
		{
			testCase: "account options account properties virtual network rule is nil",
			testAccountOptions: &AccountOptions{
				ResourceGroup:             "rg",
				VirtualNetworkResourceIDs: []string{"id"},
			},
			testResourceGroups: []storage.Account{{Name: &name, Kind: "kind", Location: &location, Sku: sku, AccountProperties: &accountPropertiesWithoutVirtualNetworkRules}},
			expectedResult:     []accountWithLocation{},
			expectedError:      nil,
		},
	}

	for _, test := range tests {
		t.Logf("running test case: %s", test.testCase)
		mockStorageAccountsClient := mockstorageaccountclient.NewMockInterface(ctrl)
		cloud.StorageAccountClient = mockStorageAccountsClient

		mockStorageAccountsClient.EXPECT().ListByResourceGroup(gomock.Any(), "rg").Return(test.testResourceGroups, nil).AnyTimes()

		accountsWithLocations, err := cloud.getStorageAccounts(test.testAccountOptions)
		if err != test.expectedError {
			t.Errorf("unexpected error: %v", err)
		}

		if len(accountsWithLocations) != len(test.expectedResult) {
			t.Error("unexpected error as returned accounts slice is not empty")
		}
	}
}

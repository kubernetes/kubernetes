//go:build !providerless
// +build !providerless

/*
Copyright 2018 The Kubernetes Authors.

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

	"github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2019-06-01/storage"
	"github.com/golang/mock/gomock"

	"k8s.io/legacy-cloud-providers/azure/clients/fileclient"
	"k8s.io/legacy-cloud-providers/azure/clients/fileclient/mockfileclient"
	"k8s.io/legacy-cloud-providers/azure/clients/storageaccountclient/mockstorageaccountclient"
)

func TestCreateFileShare(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	cloud := &Cloud{controllerCommon: &controllerCommon{resourceGroup: "rg"}}
	name := "baz"
	sku := "sku"
	kind := "StorageV2"
	location := "centralus"
	value := "foo key"
	bogus := "bogus"

	tests := []struct {
		rg       string
		name     string
		acct     string
		acctType string
		acctKind string
		loc      string
		gb       int
		accounts []storage.Account
		keys     storage.AccountListKeysResult
		err      error

		expectErr  bool
		expectAcct string
		expectKey  string
	}{
		{
			name:      "foo",
			acct:      "bar",
			acctType:  "type",
			acctKind:  "StorageV2",
			loc:       "eastus",
			gb:        10,
			expectErr: true,
		},
		{
			name:      "foo",
			acct:      "",
			acctType:  "type",
			acctKind:  "StorageV2",
			loc:       "eastus",
			gb:        10,
			expectErr: true,
		},
		{
			name:     "foo",
			acct:     "",
			acctType: sku,
			acctKind: kind,
			loc:      location,
			gb:       10,
			accounts: []storage.Account{
				{Name: &name, Sku: &storage.Sku{Name: storage.SkuName(sku)}, Kind: storage.Kind(kind), Location: &location},
			},
			keys: storage.AccountListKeysResult{
				Keys: &[]storage.AccountKey{
					{Value: &value},
				},
			},
			expectAcct: "baz",
			expectKey:  "key",
		},
		{
			rg:       "rg",
			name:     "foo",
			acct:     "",
			acctType: sku,
			acctKind: kind,
			loc:      location,
			gb:       10,
			accounts: []storage.Account{
				{Name: &name, Sku: &storage.Sku{Name: storage.SkuName(sku)}, Kind: storage.Kind(kind), Location: &location},
			},
			keys: storage.AccountListKeysResult{
				Keys: &[]storage.AccountKey{
					{Value: &value},
				},
			},
			err:       fmt.Errorf("create fileshare error"),
			expectErr: true,
		},
		{
			name:     "foo",
			acct:     "",
			acctType: sku,
			acctKind: kind,
			loc:      location,
			gb:       10,
			accounts: []storage.Account{
				{Name: &bogus, Sku: &storage.Sku{Name: storage.SkuName(sku)}, Location: &location},
			},
			expectErr: true,
		},
		{
			name:     "foo",
			acct:     "",
			acctType: sku,
			acctKind: kind,
			loc:      location,
			gb:       10,
			accounts: []storage.Account{
				{Name: &name, Sku: &storage.Sku{Name: storage.SkuName(sku)}, Location: &bogus},
			},
			expectErr: true,
		},
	}

	for _, test := range tests {
		mockFileClient := mockfileclient.NewMockInterface(ctrl)
		cloud.FileClient = mockFileClient
		mockFileClient.EXPECT().CreateFileShare(gomock.Any(), gomock.Any(), gomock.Any()).Return(test.err).AnyTimes()

		mockStorageAccountsClient := mockstorageaccountclient.NewMockInterface(ctrl)
		cloud.StorageAccountClient = mockStorageAccountsClient
		mockStorageAccountsClient.EXPECT().ListKeys(gomock.Any(), "rg", gomock.Any()).Return(test.keys, nil).AnyTimes()
		mockStorageAccountsClient.EXPECT().ListByResourceGroup(gomock.Any(), "rg").Return(test.accounts, nil).AnyTimes()
		mockStorageAccountsClient.EXPECT().Create(gomock.Any(), "rg", gomock.Any(), gomock.Any()).Return(nil).AnyTimes()

		mockAccount := &AccountOptions{
			Name:          test.acct,
			Type:          test.acctType,
			Kind:          test.acctKind,
			ResourceGroup: test.rg,
			Location:      test.loc,
		}

		mockFileShare := &fileclient.ShareOptions{
			Name:       test.name,
			Protocol:   storage.SMB,
			RequestGiB: test.gb,
		}

		account, key, err := cloud.CreateFileShare(mockAccount, mockFileShare)
		if test.expectErr && err == nil {
			t.Errorf("unexpected non-error")
			continue
		}
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}
		if test.expectAcct != account {
			t.Errorf("Expected: %s, got %s", test.expectAcct, account)
		}
		if test.expectKey != key {
			t.Errorf("Expected: %s, got %s", test.expectKey, key)
		}
	}
}

func TestDeleteFileShare(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	cloud := &Cloud{}
	tests := []struct {
		rg   string
		acct string
		name string

		err       error
		expectErr bool
	}{
		{
			rg:   "rg",
			acct: "bar",
			name: "foo",

			expectErr: false,
		},
		{
			rg:   "rg",
			acct: "bar",
			name: "",

			err:       fmt.Errorf("delete fileshare error"),
			expectErr: true,
		},
	}

	for _, test := range tests {
		mockFileClient := mockfileclient.NewMockInterface(ctrl)
		cloud.FileClient = mockFileClient
		mockFileClient.EXPECT().DeleteFileShare(gomock.Any(), gomock.Any(), gomock.Any()).Return(test.err).Times(1)

		err := cloud.DeleteFileShare(test.rg, test.acct, test.name)
		if test.expectErr && err == nil {
			t.Errorf("unexpected non-error")
			continue
		}
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}
	}
}

func TestResizeFileShare(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	cloud := &Cloud{}
	mockFileClient := mockfileclient.NewMockInterface(ctrl)
	mockFileClient.EXPECT().ResizeFileShare(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Return(nil).AnyTimes()
	cloud.FileClient = mockFileClient

	tests := []struct {
		rg   string
		acct string
		name string
		gb   int

		expectErr bool
	}{
		{
			rg:   "rg",
			acct: "bar",
			name: "foo",
			gb:   10,

			expectErr: false,
		},
	}

	for _, test := range tests {
		mockStorageAccountsClient := mockstorageaccountclient.NewMockInterface(ctrl)
		cloud.StorageAccountClient = mockStorageAccountsClient

		err := cloud.ResizeFileShare(test.rg, test.acct, test.name, test.gb)
		if test.expectErr && err == nil {
			t.Errorf("unexpected non-error")
			continue
		}
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}
	}
}

func TestGetFileShare(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	cloud := &Cloud{}
	mockFileClient := mockfileclient.NewMockInterface(ctrl)
	mockFileClient.EXPECT().GetFileShare(gomock.Any(), gomock.Any(), gomock.Any()).Return(storage.FileShare{}, nil).AnyTimes()
	cloud.FileClient = mockFileClient

	tests := []struct {
		rg   string
		acct string
		name string

		expectErr bool
	}{
		{
			rg:   "rg",
			acct: "bar",
			name: "foo",

			expectErr: false,
		},
	}

	for _, test := range tests {
		mockStorageAccountsClient := mockstorageaccountclient.NewMockInterface(ctrl)
		cloud.StorageAccountClient = mockStorageAccountsClient

		_, err := cloud.GetFileShare(test.rg, test.acct, test.name)
		if test.expectErr && err == nil {
			t.Errorf("unexpected non-error")
			continue
		}
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}
	}
}

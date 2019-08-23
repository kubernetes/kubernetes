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
	"testing"

	"github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2019-04-01/storage"
)

func TestCreateFileShare(t *testing.T) {
	cloud := &Cloud{}
	fake := newFakeStorageAccountClient()
	cloud.StorageAccountClient = fake
	cloud.FileClient = &fakeFileClient{}

	name := "baz"
	sku := "sku"
	kind := "StorageV2"
	location := "centralus"
	value := "foo key"
	bogus := "bogus"

	tests := []struct {
		name     string
		acct     string
		acctType string
		acctKind string
		loc      string
		gb       int
		accounts storage.AccountListResult
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
			accounts: storage.AccountListResult{
				Value: &[]storage.Account{
					{Name: &name, Sku: &storage.Sku{Name: storage.SkuName(sku)}, Kind: storage.Kind(kind), Location: &location},
				},
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
			name:     "foo",
			acct:     "",
			acctType: sku,
			acctKind: kind,
			loc:      location,
			gb:       10,
			accounts: storage.AccountListResult{
				Value: &[]storage.Account{
					{Name: &bogus, Sku: &storage.Sku{Name: storage.SkuName(sku)}, Location: &location},
				},
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
			accounts: storage.AccountListResult{
				Value: &[]storage.Account{
					{Name: &name, Sku: &storage.Sku{Name: storage.SkuName(sku)}, Location: &bogus},
				},
			},
			expectErr: true,
		},
	}

	for _, test := range tests {
		fake.Accounts = test.accounts
		fake.Keys = test.keys
		fake.Err = test.err

		account, key, err := cloud.CreateFileShare(test.name, test.acct, test.acctType, test.acctKind, "rg", test.loc, test.gb)
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

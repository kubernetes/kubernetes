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

	"github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2018-07-01/storage"
)

func TestGetStorageAccessKeys(t *testing.T) {
	cloud := &Cloud{}
	fake := newFakeStorageAccountClient()
	cloud.StorageAccountClient = fake
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
		expectedKey := test.expectedKey
		fake.Keys = test.results
		fake.Err = test.err
		key, err := cloud.GetStorageAccesskey("acct", "rg")
		if test.expectErr && err == nil {
			t.Errorf("Unexpected non-error")
			continue
		}
		if !test.expectErr && err != nil {
			t.Errorf("Unexpected error: %v", err)
			continue
		}
		if key != expectedKey {
			t.Errorf("expected: %s, saw %s", expectedKey, key)
		}
	}
}

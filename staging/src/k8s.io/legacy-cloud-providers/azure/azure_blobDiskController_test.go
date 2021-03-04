// +build !providerless

/*
Copyright 2020 The Kubernetes Authors.

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
	"net/http"
	"strings"
	"testing"

	"k8s.io/legacy-cloud-providers/azure/clients/storageaccountclient/mockstorageaccountclient"
	"k8s.io/legacy-cloud-providers/azure/retry"

	"github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2019-06-01/storage"
	azstorage "github.com/Azure/azure-sdk-for-go/storage"
	autorestazure "github.com/Azure/go-autorest/autorest/azure"
	"github.com/Azure/go-autorest/autorest/to"
	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"
)

var retryError500 = retry.Error{HTTPStatusCode: http.StatusInternalServerError}

func GetTestBlobDiskController(t *testing.T) BlobDiskController {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	az := GetTestCloud(ctrl)
	az.Environment = autorestazure.PublicCloud
	common := &controllerCommon{cloud: az, resourceGroup: "rg", location: "westus"}

	return BlobDiskController{
		common:   common,
		accounts: make(map[string]*storageAccountState),
	}
}

func TestInitStorageAccounts(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	b := GetTestBlobDiskController(t)
	b.accounts = nil

	mockSAClient := mockstorageaccountclient.NewMockInterface(ctrl)
	mockSAClient.EXPECT().ListByResourceGroup(gomock.Any(), b.common.resourceGroup).Return([]storage.Account{}, &retryError500)
	b.common.cloud.StorageAccountClient = mockSAClient

	b.initStorageAccounts()
	assert.Empty(t, b.accounts)

	mockSAClient.EXPECT().ListByResourceGroup(gomock.Any(), b.common.resourceGroup).Return([]storage.Account{
		{
			Name: to.StringPtr("ds-0"),
			Sku:  &storage.Sku{Name: "sku"},
		},
	}, nil)
	b.common.cloud.StorageAccountClient = mockSAClient

	b.initStorageAccounts()
	assert.Equal(t, 1, len(b.accounts))
}

func TestCreateVolume(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	b := GetTestBlobDiskController(t)

	mockSAClient := mockstorageaccountclient.NewMockInterface(ctrl)
	mockSAClient.EXPECT().ListKeys(gomock.Any(), b.common.resourceGroup, "testsa").Return(storage.AccountListKeysResult{}, &retryError500)
	b.common.cloud.StorageAccountClient = mockSAClient

	diskName, diskURI, requestGB, err := b.CreateVolume("testBlob", "testsa", "type", b.common.location, 10)
	expectedErr := fmt.Errorf("could not get storage key for storage account testsa: could not get storage key for " +
		"storage account testsa: Retriable: false, RetryAfter: 0s, HTTPStatusCode: 500, RawError: <nil>")
	assert.Equal(t, expectedErr, err)
	assert.Empty(t, diskName)
	assert.Empty(t, diskURI)
	assert.Zero(t, requestGB)

	mockSAClient.EXPECT().ListKeys(gomock.Any(), b.common.resourceGroup, "testsa").Return(storage.AccountListKeysResult{
		Keys: &[]storage.AccountKey{
			{
				KeyName: to.StringPtr("key1"),
				Value:   to.StringPtr("dmFsdWUK"),
			},
		},
	}, nil)
	diskName, diskURI, requestGB, err = b.CreateVolume("testBlob", "testsa", "type", b.common.location, 10)
	expectedErrStr := "failed to put page blob testBlob.vhd in container vhds: storage: service returned error: StatusCode=403, ErrorCode=AccountIsDisabled, ErrorMessage=The specified account is disabled."
	assert.Error(t, err)
	assert.True(t, strings.Contains(err.Error(), expectedErrStr))
	assert.Empty(t, diskName)
	assert.Empty(t, diskURI)
	assert.Zero(t, requestGB)
}

func TestDeleteVolume(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	b := GetTestBlobDiskController(t)
	b.common.cloud.BlobDiskController = &b

	mockSAClient := mockstorageaccountclient.NewMockInterface(ctrl)
	mockSAClient.EXPECT().ListKeys(gomock.Any(), b.common.resourceGroup, "foo").Return(storage.AccountListKeysResult{}, &retryError500).Times(2)
	b.common.cloud.StorageAccountClient = mockSAClient

	fakeDiskURL := "fake"
	diskURL := "https://foo.blob./vhds/bar.vhd"
	err := b.DeleteVolume(diskURL)
	expectedErr := fmt.Errorf("no key for storage account foo, err Retriable: false, RetryAfter: 0s, HTTPStatusCode: 500, RawError: <nil>")
	assert.Equal(t, expectedErr, err)

	err = b.DeleteVolume(diskURL)
	assert.Equal(t, expectedErr, err)

	mockSAClient.EXPECT().ListKeys(gomock.Any(), b.common.resourceGroup, "foo").Return(storage.AccountListKeysResult{
		Keys: &[]storage.AccountKey{
			{
				KeyName: to.StringPtr("key1"),
				Value:   to.StringPtr("dmFsdWUK"),
			},
		},
	}, nil)

	err = b.DeleteVolume(fakeDiskURL)
	expectedErr = fmt.Errorf("failed to parse vhd URI invalid vhd URI for regex https://(.*).blob./vhds/(.*): fake")
	assert.Equal(t, expectedErr, err)

	err = b.DeleteVolume(diskURL)
	expectedErrStr := "failed to delete vhd https://foo.blob./vhds/bar.vhd, account foo, blob bar.vhd, err: storage: service returned error: " +
		"StatusCode=403, ErrorCode=AccountIsDisabled, ErrorMessage=The specified account is disabled."
	assert.Error(t, err)
	assert.True(t, strings.Contains(err.Error(), expectedErrStr))
}

func TestCreateVHDBlobDisk(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	b := GetTestBlobDiskController(t)

	b.common.cloud.Environment = autorestazure.PublicCloud
	client, err := azstorage.NewBasicClientOnSovereignCloud("testsa", "a2V5Cg==", b.common.cloud.Environment)
	assert.NoError(t, err)
	blobClient := client.GetBlobService()

	_, _, err = b.createVHDBlobDisk(blobClient, "testsa", "blob", vhdContainerName, int64(10))
	expectedErr := "failed to put page blob blob.vhd in container vhds: storage: service returned error: StatusCode=403, ErrorCode=AccountIsDisabled, ErrorMessage=The specified account is disabled."
	assert.Error(t, err)
	assert.True(t, strings.Contains(err.Error(), expectedErr))
}

func TestGetAllStorageAccounts(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	b := GetTestBlobDiskController(t)

	expectedStorageAccounts := []storage.Account{
		{
			Name: to.StringPtr("this-should-be-skipped"),
		},
		{
			Name: to.StringPtr("this-should-be-skipped"),
			Sku:  &storage.Sku{Name: "sku"},
		},
		{
			Name: to.StringPtr("ds-0"),
			Sku:  &storage.Sku{Name: "sku"},
		},
	}

	mockSAClient := mockstorageaccountclient.NewMockInterface(ctrl)
	mockSAClient.EXPECT().ListByResourceGroup(gomock.Any(), b.common.resourceGroup).Return(expectedStorageAccounts, nil)
	b.common.cloud.StorageAccountClient = mockSAClient

	accounts, err := b.getAllStorageAccounts()
	assert.NoError(t, err)
	assert.Equal(t, 1, len(accounts))
}

func TestEnsureDefaultContainer(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	b := GetTestBlobDiskController(t)

	mockSAClient := mockstorageaccountclient.NewMockInterface(ctrl)
	b.common.cloud.StorageAccountClient = mockSAClient

	mockSAClient.EXPECT().GetProperties(gomock.Any(), b.common.resourceGroup, "testsa").Return(storage.Account{}, &retryError500)
	err := b.ensureDefaultContainer("testsa")
	expectedErr := fmt.Errorf("azureDisk - account testsa does not exist while trying to create/ensure default container")
	assert.Equal(t, expectedErr, err)

	b.accounts["testsa"] = &storageAccountState{defaultContainerCreated: true}
	err = b.ensureDefaultContainer("testsa")
	assert.NoError(t, err)

	b.accounts["testsa"] = &storageAccountState{isValidating: 0}
	mockSAClient.EXPECT().GetProperties(gomock.Any(), b.common.resourceGroup, "testsa").Return(storage.Account{
		AccountProperties: &storage.AccountProperties{ProvisioningState: storage.Creating},
	}, nil)
	mockSAClient.EXPECT().GetProperties(gomock.Any(), b.common.resourceGroup, "testsa").Return(storage.Account{}, &retryError500)
	mockSAClient.EXPECT().GetProperties(gomock.Any(), b.common.resourceGroup, "testsa").Return(storage.Account{
		AccountProperties: &storage.AccountProperties{ProvisioningState: storage.Succeeded},
	}, nil)
	mockSAClient.EXPECT().ListKeys(gomock.Any(), b.common.resourceGroup, "testsa").Return(storage.AccountListKeysResult{
		Keys: &[]storage.AccountKey{
			{
				KeyName: to.StringPtr("key1"),
				Value:   to.StringPtr("key1"),
			},
		},
	}, nil)
	err = b.ensureDefaultContainer("testsa")
	expectedErrStr := "storage: service returned error: StatusCode=403, ErrorCode=AccountIsDisabled, ErrorMessage=The specified account is disabled."
	assert.Error(t, err)
	assert.True(t, strings.Contains(err.Error(), expectedErrStr))
}

func TestGetDiskCount(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	b := GetTestBlobDiskController(t)

	mockSAClient := mockstorageaccountclient.NewMockInterface(ctrl)
	b.common.cloud.StorageAccountClient = mockSAClient

	b.accounts["testsa"] = &storageAccountState{diskCount: 1}
	count, err := b.getDiskCount("testsa")
	assert.Equal(t, 1, count)
	assert.NoError(t, err)

	b.accounts["testsa"] = &storageAccountState{diskCount: -1}
	mockSAClient.EXPECT().GetProperties(gomock.Any(), b.common.resourceGroup, "testsa").Return(storage.Account{}, &retryError500)
	count, err = b.getDiskCount("testsa")
	assert.Zero(t, count)
	expectedErr := fmt.Errorf("azureDisk - account testsa does not exist while trying to create/ensure default container")
	assert.Equal(t, expectedErr, err)

	b.accounts["testsa"].defaultContainerCreated = true
	mockSAClient.EXPECT().ListKeys(gomock.Any(), b.common.resourceGroup, "testsa").Return(storage.AccountListKeysResult{
		Keys: &[]storage.AccountKey{
			{
				KeyName: to.StringPtr("key1"),
				Value:   to.StringPtr("key1"),
			},
		},
	}, nil)
	count, err = b.getDiskCount("testsa")
	expectedErrStr := "storage: service returned error: StatusCode=403, ErrorCode=AccountIsDisabled, ErrorMessage=The specified account is disabled."
	assert.Error(t, err)
	assert.True(t, strings.Contains(err.Error(), expectedErrStr))
	assert.Zero(t, count)
}

func TestFindSANameForDisk(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	b := GetTestBlobDiskController(t)

	mockSAClient := mockstorageaccountclient.NewMockInterface(ctrl)
	b.common.cloud.StorageAccountClient = mockSAClient

	b.accounts = map[string]*storageAccountState{
		"this-shall-be-skipped": {name: "fake"},
		"ds0": {
			name:      "ds0",
			saType:    storage.StandardGRS,
			diskCount: 50,
		},
	}
	mockSAClient.EXPECT().GetProperties(gomock.Any(), b.common.resourceGroup, gomock.Any()).Return(storage.Account{}, &retryError500).Times(2)
	mockSAClient.EXPECT().GetProperties(gomock.Any(), b.common.resourceGroup, gomock.Any()).Return(storage.Account{
		AccountProperties: &storage.AccountProperties{ProvisioningState: storage.Succeeded},
	}, nil).Times(2)
	mockSAClient.EXPECT().ListKeys(gomock.Any(), b.common.resourceGroup, gomock.Any()).Return(storage.AccountListKeysResult{
		Keys: &[]storage.AccountKey{
			{
				KeyName: to.StringPtr("key1"),
				Value:   to.StringPtr("key1"),
			},
		},
	}, nil)
	mockSAClient.EXPECT().Create(gomock.Any(), b.common.resourceGroup, gomock.Any(), gomock.Any()).Return(nil)
	name, err := b.findSANameForDisk(storage.StandardGRS)
	expectedErr := "does not exist while trying to create/ensure default container"
	assert.True(t, strings.Contains(err.Error(), expectedErr))
	assert.Error(t, err)
	assert.Empty(t, name)

	b.accounts = make(map[string]*storageAccountState)
	name, err = b.findSANameForDisk(storage.StandardGRS)
	assert.Error(t, err)
	assert.Empty(t, name)

	b.accounts = map[string]*storageAccountState{
		"ds0": {
			name:      "ds0",
			saType:    storage.StandardGRS,
			diskCount: 0,
		},
	}
	name, err = b.findSANameForDisk(storage.StandardGRS)
	assert.Equal(t, "ds0", name)
	assert.NoError(t, err)

	for i := 0; i < maxStorageAccounts; i++ {
		b.accounts[fmt.Sprintf("ds%d", i)] = &storageAccountState{
			name:      fmt.Sprintf("ds%d", i),
			saType:    storage.StandardGRS,
			diskCount: 59,
		}
	}
	name, err = b.findSANameForDisk(storage.StandardGRS)
	assert.NotEmpty(t, name)
	assert.NoError(t, err)
}

func TestCreateBlobDisk(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	b := GetTestBlobDiskController(t)
	b.accounts = map[string]*storageAccountState{
		"ds0": {
			name:      "ds0",
			saType:    storage.StandardGRS,
			diskCount: 0,
		},
	}

	mockSAClient := mockstorageaccountclient.NewMockInterface(ctrl)
	b.common.cloud.StorageAccountClient = mockSAClient
	mockSAClient.EXPECT().ListKeys(gomock.Any(), b.common.resourceGroup, gomock.Any()).Return(storage.AccountListKeysResult{
		Keys: &[]storage.AccountKey{
			{
				KeyName: to.StringPtr("key1"),
				Value:   to.StringPtr("key1"),
			},
		},
	}, nil)
	diskURI, err := b.CreateBlobDisk("datadisk", storage.StandardGRS, 10)
	expectedErr := "failed to put page blob datadisk.vhd in container vhds: storage: service returned error: StatusCode=403"
	assert.Error(t, err)
	assert.True(t, strings.Contains(err.Error(), expectedErr))
	assert.Empty(t, diskURI)
}

/*
Copyright 2019 The Kubernetes Authors.

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

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-03-01/compute"
	"github.com/stretchr/testify/assert"
)

func TestAttachDisk(t *testing.T) {
	c := getTestCloud()

	common := &controllerCommon{
		location:              c.Location,
		storageEndpointSuffix: c.Environment.StorageEndpointSuffix,
		resourceGroup:         c.ResourceGroup,
		subscriptionID:        c.SubscriptionID,
		cloud:                 c,
		vmLockMap:             newLockMap(),
	}

	diskURI := fmt.Sprintf("/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Compute/disks/disk-name", c.SubscriptionID, c.ResourceGroup)

	_, err := common.AttachDisk(true, "", diskURI, "node1", compute.CachingTypesReadOnly)
	if err != nil {
		fmt.Printf("TestAttachDisk return expected error: %v", err)
	} else {
		t.Errorf("TestAttachDisk unexpected nil err")
	}
}

func TestDetachDisk(t *testing.T) {
	c := getTestCloud()

	common := &controllerCommon{
		location:              c.Location,
		storageEndpointSuffix: c.Environment.StorageEndpointSuffix,
		resourceGroup:         c.ResourceGroup,
		subscriptionID:        c.SubscriptionID,
		cloud:                 c,
		vmLockMap:             newLockMap(),
	}

	diskURI := fmt.Sprintf("/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Compute/disks/disk-name", c.SubscriptionID, c.ResourceGroup)

	err := common.DetachDisk("", diskURI, "node1")
	if err != nil {
		t.Errorf("TestAttachDisk got unexpected error: %v", err)
	}
}

func TestCheckDiskExists(t *testing.T) {
	ctx, cancel := getContextWithCancel()
	defer cancel()

	testCloud := getTestCloud()
	common := &controllerCommon{
		location:              testCloud.Location,
		storageEndpointSuffix: testCloud.Environment.StorageEndpointSuffix,
		resourceGroup:         testCloud.ResourceGroup,
		subscriptionID:        testCloud.SubscriptionID,
		cloud:                 testCloud,
		vmLockMap:             newLockMap(),
	}
	// create a new disk before running test
	newDiskName := "newdisk"
	newDiskURI := fmt.Sprintf("/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Compute/disks/%s",
		testCloud.SubscriptionID, testCloud.ResourceGroup, newDiskName)
	fDC := newFakeDisksClient()
	_, rerr := fDC.CreateOrUpdate(ctx, testCloud.ResourceGroup, newDiskName, compute.Disk{})
	assert.Equal(t, rerr == nil, true, "return error: %v", rerr)
	testCloud.DisksClient = fDC

	testCases := []struct {
		diskURI        string
		expectedResult bool
		expectedErr    bool
	}{
		{
			diskURI:        "incorrect disk URI format",
			expectedResult: false,
			expectedErr:    true,
		},
		{
			diskURI:        "/subscriptions/xxx/resourceGroups/xxx/providers/Microsoft.Compute/disks/non-existing-disk",
			expectedResult: false,
			expectedErr:    false,
		},
		{
			diskURI:        newDiskURI,
			expectedResult: true,
			expectedErr:    false,
		},
	}

	for i, test := range testCases {
		exist, err := common.checkDiskExists(ctx, test.diskURI)
		assert.Equal(t, test.expectedResult, exist, "TestCase[%d]", i, exist)
		assert.Equal(t, test.expectedErr, err != nil, "TestCase[%d], return error: %v", i, err)
	}
}

// +build !providerless

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

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-07-01/compute"
	"github.com/stretchr/testify/assert"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/pointer"
)

func TestCommonAttachDisk(t *testing.T) {
	testCases := []struct {
		desc            string
		vmList          map[string]string
		nodeName        types.NodeName
		isDataDisksFull bool
		expectedLun     int32
		expectedErr     bool
	}{
		{
			desc:        "LUN -1 and error shall be returned if there's no such instance corresponding to given nodeName",
			nodeName:    "vm1",
			expectedLun: -1,
			expectedErr: true,
		},
		{
			desc:            "LUN -1 and error shall be returned if there's no available LUN for instance",
			vmList:          map[string]string{"vm1": "PowerState/Running"},
			nodeName:        "vm1",
			isDataDisksFull: true,
			expectedLun:     -1,
			expectedErr:     true,
		},
		{
			desc:        "correct LUN and no error shall be returned if everything is good",
			vmList:      map[string]string{"vm1": "PowerState/Running"},
			nodeName:    "vm1",
			expectedLun: -1,
			expectedErr: true,
		},
	}

	for i, test := range testCases {
		testCloud := getTestCloud()
		common := &controllerCommon{
			location:              testCloud.Location,
			storageEndpointSuffix: testCloud.Environment.StorageEndpointSuffix,
			resourceGroup:         testCloud.ResourceGroup,
			subscriptionID:        testCloud.SubscriptionID,
			cloud:                 testCloud,
			vmLockMap:             newLockMap(),
		}
		diskURI := fmt.Sprintf("/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Compute/disks/disk-name",
			testCloud.SubscriptionID, testCloud.ResourceGroup)
		setTestVirtualMachines(testCloud, test.vmList, test.isDataDisksFull)

		lun, err := common.AttachDisk(true, "", diskURI, test.nodeName, compute.CachingTypesReadOnly)
		assert.Equal(t, test.expectedLun, lun, "TestCase[%d]: %s", i, test.desc)
		assert.Equal(t, test.expectedErr, err != nil, "TestCase[%d]: %s, return error: %v", i, test.desc, err)
	}
}

func TestCommonDetachDisk(t *testing.T) {
	testCases := []struct {
		desc        string
		vmList      map[string]string
		nodeName    types.NodeName
		diskName    string
		expectedErr bool
	}{
		{
			desc:        "error should not be returned if there's no such instance corresponding to given nodeName",
			nodeName:    "vm1",
			expectedErr: false,
		},
		{
			desc:        "no error shall be returned if there's no matching disk according to given diskName",
			vmList:      map[string]string{"vm1": "PowerState/Running"},
			nodeName:    "vm1",
			diskName:    "disk2",
			expectedErr: false,
		},
		{
			desc:        "no error shall be returned if the disk exsists",
			vmList:      map[string]string{"vm1": "PowerState/Running"},
			nodeName:    "vm1",
			diskName:    "disk1",
			expectedErr: false,
		},
	}

	for i, test := range testCases {
		testCloud := getTestCloud()
		common := &controllerCommon{
			location:              testCloud.Location,
			storageEndpointSuffix: testCloud.Environment.StorageEndpointSuffix,
			resourceGroup:         testCloud.ResourceGroup,
			subscriptionID:        testCloud.SubscriptionID,
			cloud:                 testCloud,
			vmLockMap:             newLockMap(),
		}
		diskURI := fmt.Sprintf("/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Compute/disks/disk-name",
			testCloud.SubscriptionID, testCloud.ResourceGroup)
		setTestVirtualMachines(testCloud, test.vmList, false)

		err := common.DetachDisk(test.diskName, diskURI, test.nodeName)
		assert.Equal(t, test.expectedErr, err != nil, "TestCase[%d]: %s", i, test.desc)
	}
}

func TestGetDiskLun(t *testing.T) {
	testCases := []struct {
		desc        string
		diskName    string
		diskURI     string
		expectedLun int32
		expectedErr bool
	}{
		{
			desc:        "LUN -1 and error shall be returned if diskName != disk.Name or diskURI != disk.Vhd.URI",
			diskName:    "disk2",
			expectedLun: -1,
			expectedErr: true,
		},
		{
			desc:        "correct LUN and no error shall be returned if diskName = disk.Name",
			diskName:    "disk1",
			expectedLun: 0,
			expectedErr: false,
		},
	}

	for i, test := range testCases {
		testCloud := getTestCloud()
		common := &controllerCommon{
			location:              testCloud.Location,
			storageEndpointSuffix: testCloud.Environment.StorageEndpointSuffix,
			resourceGroup:         testCloud.ResourceGroup,
			subscriptionID:        testCloud.SubscriptionID,
			cloud:                 testCloud,
			vmLockMap:             newLockMap(),
		}
		setTestVirtualMachines(testCloud, map[string]string{"vm1": "PowerState/Running"}, false)

		lun, err := common.GetDiskLun(test.diskName, test.diskURI, "vm1")
		assert.Equal(t, test.expectedLun, lun, "TestCase[%d]: %s", i, test.desc)
		assert.Equal(t, test.expectedErr, err != nil, "TestCase[%d]: %s", i, test.desc)
	}
}

func TestGetNextDiskLun(t *testing.T) {
	testCases := []struct {
		desc            string
		isDataDisksFull bool
		expectedLun     int32
		expectedErr     bool
	}{
		{
			desc:            "the minimal LUN shall be returned if there's enough room for extra disks",
			isDataDisksFull: false,
			expectedLun:     1,
			expectedErr:     false,
		},
		{
			desc:            "LUN -1 and  error shall be returned if there's no available LUN",
			isDataDisksFull: true,
			expectedLun:     -1,
			expectedErr:     true,
		},
	}

	for i, test := range testCases {
		testCloud := getTestCloud()
		common := &controllerCommon{
			location:              testCloud.Location,
			storageEndpointSuffix: testCloud.Environment.StorageEndpointSuffix,
			resourceGroup:         testCloud.ResourceGroup,
			subscriptionID:        testCloud.SubscriptionID,
			cloud:                 testCloud,
			vmLockMap:             newLockMap(),
		}
		setTestVirtualMachines(testCloud, map[string]string{"vm1": "PowerState/Running"}, test.isDataDisksFull)

		lun, err := common.GetNextDiskLun("vm1")
		assert.Equal(t, test.expectedLun, lun, "TestCase[%d]: %s", i, test.desc)
		assert.Equal(t, test.expectedErr, err != nil, "TestCase[%d]: %s", i, test.desc)
	}
}

func TestDisksAreAttached(t *testing.T) {
	testCases := []struct {
		desc             string
		diskNames        []string
		nodeName         types.NodeName
		expectedAttached map[string]bool
		expectedErr      bool
	}{
		{
			desc:             "an error shall be returned if there's no such instance corresponding to given nodeName",
			diskNames:        []string{"disk1"},
			nodeName:         "vm2",
			expectedAttached: map[string]bool{"disk1": false},
			expectedErr:      false,
		},
		{
			desc:             "proper attach map shall be returned if everything is good",
			diskNames:        []string{"disk1", "disk2"},
			nodeName:         "vm1",
			expectedAttached: map[string]bool{"disk1": true, "disk2": false},
			expectedErr:      false,
		},
	}

	for i, test := range testCases {
		testCloud := getTestCloud()
		common := &controllerCommon{
			location:              testCloud.Location,
			storageEndpointSuffix: testCloud.Environment.StorageEndpointSuffix,
			resourceGroup:         testCloud.ResourceGroup,
			subscriptionID:        testCloud.SubscriptionID,
			cloud:                 testCloud,
			vmLockMap:             newLockMap(),
		}
		setTestVirtualMachines(testCloud, map[string]string{"vm1": "PowerState/Running"}, false)

		attached, err := common.DisksAreAttached(test.diskNames, test.nodeName)
		assert.Equal(t, test.expectedAttached, attached, "TestCase[%d]: %s", i, test.desc)
		assert.Equal(t, test.expectedErr, err != nil, "TestCase[%d]: %s", i, test.desc)
	}
}

func TestFilteredDetatchingDisks(t *testing.T) {

	disks := []compute.DataDisk{
		{
			Name:         pointer.StringPtr("DiskName1"),
			ToBeDetached: pointer.BoolPtr(false),
			ManagedDisk: &compute.ManagedDiskParameters{
				ID: pointer.StringPtr("ManagedID"),
			},
		},
		{
			Name:         pointer.StringPtr("DiskName2"),
			ToBeDetached: pointer.BoolPtr(true),
		},
		{
			Name:         pointer.StringPtr("DiskName3"),
			ToBeDetached: nil,
		},
		{
			Name:         pointer.StringPtr("DiskName4"),
			ToBeDetached: nil,
		},
	}

	filteredDisks := filterDetachingDisks(disks)
	assert.Equal(t, 3, len(filteredDisks))
	assert.Equal(t, "DiskName1", *filteredDisks[0].Name)
	assert.Equal(t, "ManagedID", *filteredDisks[0].ManagedDisk.ID)
	assert.Equal(t, "DiskName3", *filteredDisks[1].Name)

	disks = []compute.DataDisk{}
	filteredDisks = filterDetachingDisks(disks)
	assert.Equal(t, 0, len(filteredDisks))
}

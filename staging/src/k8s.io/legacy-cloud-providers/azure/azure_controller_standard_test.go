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
	"net/http"
	"testing"
	"time"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-12-01/compute"
	"github.com/Azure/go-autorest/autorest/to"
	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"

	"k8s.io/apimachinery/pkg/types"
	cloudprovider "k8s.io/cloud-provider"
	azcache "k8s.io/legacy-cloud-providers/azure/cache"
	"k8s.io/legacy-cloud-providers/azure/clients/vmclient/mockvmclient"
	"k8s.io/legacy-cloud-providers/azure/retry"
)

var (
	fakeCacheTTL = 2 * time.Second
)

func TestStandardAttachDisk(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	testCases := []struct {
		desc          string
		nodeName      types.NodeName
		isManagedDisk bool
		isAttachFail  bool
		expectedErr   bool
	}{
		{
			desc:          "an error shall be returned if there's no corresponding vms",
			nodeName:      "vm2",
			isManagedDisk: true,
			expectedErr:   true,
		},
		{
			desc:          "no error shall be returned if everything's good",
			nodeName:      "vm1",
			isManagedDisk: true,
			expectedErr:   false,
		},
		{
			desc:          "no error shall be returned if everything's good with non managed disk",
			nodeName:      "vm1",
			isManagedDisk: false,
			expectedErr:   false,
		},
		{
			desc:          "an error shall be returned if update attach disk failed",
			nodeName:      "vm1",
			isManagedDisk: true,
			isAttachFail:  true,
			expectedErr:   true,
		},
	}

	for i, test := range testCases {
		testCloud := GetTestCloud(ctrl)
		vmSet := testCloud.VMSet
		expectedVMs := setTestVirtualMachines(testCloud, map[string]string{"vm1": "PowerState/Running"}, false)
		mockVMsClient := testCloud.VirtualMachinesClient.(*mockvmclient.MockInterface)
		for _, vm := range expectedVMs {
			vm.StorageProfile = &compute.StorageProfile{
				OsDisk: &compute.OSDisk{
					Name: to.StringPtr("osdisk1"),
					ManagedDisk: &compute.ManagedDiskParameters{
						ID: to.StringPtr("ManagedID"),
						DiskEncryptionSet: &compute.DiskEncryptionSetParameters{
							ID: to.StringPtr("DiskEncryptionSetID"),
						},
					},
				},
				DataDisks: &[]compute.DataDisk{},
			}
			mockVMsClient.EXPECT().Get(gomock.Any(), testCloud.ResourceGroup, *vm.Name, gomock.Any()).Return(vm, nil).AnyTimes()
		}
		mockVMsClient.EXPECT().Get(gomock.Any(), testCloud.ResourceGroup, "vm2", gomock.Any()).Return(compute.VirtualMachine{}, &retry.Error{HTTPStatusCode: http.StatusNotFound, RawError: cloudprovider.InstanceNotFound}).AnyTimes()
		if test.isAttachFail {
			mockVMsClient.EXPECT().Update(gomock.Any(), testCloud.ResourceGroup, gomock.Any(), gomock.Any(), gomock.Any()).Return(&retry.Error{HTTPStatusCode: http.StatusNotFound, RawError: cloudprovider.InstanceNotFound}).AnyTimes()
		} else {
			mockVMsClient.EXPECT().Update(gomock.Any(), testCloud.ResourceGroup, gomock.Any(), gomock.Any(), gomock.Any()).Return(nil).AnyTimes()
		}

		err := vmSet.AttachDisk(test.isManagedDisk, "",
			"uri", test.nodeName, 0, compute.CachingTypesReadOnly, "", false)
		assert.Equal(t, test.expectedErr, err != nil, "TestCase[%d]: %s, err: %v", i, test.desc, err)
	}
}

func TestStandardDetachDisk(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	testCases := []struct {
		desc          string
		nodeName      types.NodeName
		diskName      string
		isDetachFail  bool
		expectedError bool
	}{
		{
			desc:          "no error shall be returned if there's no corresponding vm",
			nodeName:      "vm2",
			expectedError: false,
		},
		{
			desc:          "no error shall be returned if there's no corresponding disk",
			nodeName:      "vm1",
			diskName:      "disk2",
			expectedError: false,
		},
		{
			desc:          "no error shall be returned if there's a corresponding disk",
			nodeName:      "vm1",
			diskName:      "disk1",
			expectedError: false,
		},
		{
			desc:          "an error shall be returned if detach disk failed",
			nodeName:      "vm1",
			isDetachFail:  true,
			expectedError: true,
		},
	}

	for i, test := range testCases {
		testCloud := GetTestCloud(ctrl)
		vmSet := testCloud.VMSet
		expectedVMs := setTestVirtualMachines(testCloud, map[string]string{"vm1": "PowerState/Running"}, false)
		mockVMsClient := testCloud.VirtualMachinesClient.(*mockvmclient.MockInterface)
		for _, vm := range expectedVMs {
			mockVMsClient.EXPECT().Get(gomock.Any(), testCloud.ResourceGroup, *vm.Name, gomock.Any()).Return(vm, nil).AnyTimes()
		}
		mockVMsClient.EXPECT().Get(gomock.Any(), testCloud.ResourceGroup, "vm2", gomock.Any()).Return(compute.VirtualMachine{}, &retry.Error{HTTPStatusCode: http.StatusNotFound, RawError: cloudprovider.InstanceNotFound}).AnyTimes()
		if test.isDetachFail {
			mockVMsClient.EXPECT().Update(gomock.Any(), testCloud.ResourceGroup, gomock.Any(), gomock.Any(), gomock.Any()).Return(&retry.Error{HTTPStatusCode: http.StatusNotFound, RawError: cloudprovider.InstanceNotFound}).AnyTimes()
		} else {
			mockVMsClient.EXPECT().Update(gomock.Any(), testCloud.ResourceGroup, gomock.Any(), gomock.Any(), gomock.Any()).Return(nil).AnyTimes()
		}

		err := vmSet.DetachDisk(test.diskName, "", test.nodeName)
		assert.Equal(t, test.expectedError, err != nil, "TestCase[%d]: %s", i, test.desc)
		if !test.expectedError && test.diskName != "" {
			dataDisks, err := vmSet.GetDataDisks(test.nodeName, azcache.CacheReadTypeDefault)
			assert.Equal(t, true, len(dataDisks) == 1, "TestCase[%d]: %s, err: %v", i, test.desc, err)
		}
	}
}

func TestGetDataDisks(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	var testCases = []struct {
		desc              string
		nodeName          types.NodeName
		isDataDiskNull    bool
		expectedDataDisks []compute.DataDisk
		expectedError     bool
		crt               azcache.AzureCacheReadType
	}{
		{
			desc:              "an error shall be returned if there's no corresponding vm",
			nodeName:          "vm2",
			expectedDataDisks: nil,
			expectedError:     true,
			crt:               azcache.CacheReadTypeDefault,
		},
		{
			desc:     "correct list of data disks shall be returned if everything is good",
			nodeName: "vm1",
			expectedDataDisks: []compute.DataDisk{
				{
					Lun:  to.Int32Ptr(0),
					Name: to.StringPtr("disk1"),
				},
			},
			expectedError: false,
			crt:           azcache.CacheReadTypeDefault,
		},
		{
			desc:     "correct list of data disks shall be returned if everything is good",
			nodeName: "vm1",
			expectedDataDisks: []compute.DataDisk{
				{
					Lun:  to.Int32Ptr(0),
					Name: to.StringPtr("disk1"),
				},
			},
			expectedError: false,
			crt:           azcache.CacheReadTypeUnsafe,
		},
		{
			desc:              "nil shall be returned if DataDisk is null",
			nodeName:          "vm1",
			isDataDiskNull:    true,
			expectedDataDisks: nil,
			expectedError:     false,
			crt:               azcache.CacheReadTypeDefault,
		},
	}
	for i, test := range testCases {
		testCloud := GetTestCloud(ctrl)
		vmSet := testCloud.VMSet
		expectedVMs := setTestVirtualMachines(testCloud, map[string]string{"vm1": "PowerState/Running"}, false)
		mockVMsClient := testCloud.VirtualMachinesClient.(*mockvmclient.MockInterface)
		for _, vm := range expectedVMs {
			if test.isDataDiskNull {
				vm.StorageProfile = &compute.StorageProfile{}
			}
			mockVMsClient.EXPECT().Get(gomock.Any(), testCloud.ResourceGroup, *vm.Name, gomock.Any()).Return(vm, nil).AnyTimes()
		}
		mockVMsClient.EXPECT().Get(gomock.Any(), testCloud.ResourceGroup, gomock.Not("vm1"), gomock.Any()).Return(compute.VirtualMachine{}, &retry.Error{HTTPStatusCode: http.StatusNotFound, RawError: cloudprovider.InstanceNotFound}).AnyTimes()
		mockVMsClient.EXPECT().Update(gomock.Any(), testCloud.ResourceGroup, gomock.Any(), gomock.Any(), gomock.Any()).Return(nil).AnyTimes()

		dataDisks, err := vmSet.GetDataDisks(test.nodeName, test.crt)
		assert.Equal(t, test.expectedDataDisks, dataDisks, "TestCase[%d]: %s", i, test.desc)
		assert.Equal(t, test.expectedError, err != nil, "TestCase[%d]: %s", i, test.desc)

		if test.crt == azcache.CacheReadTypeUnsafe {
			time.Sleep(fakeCacheTTL)
			dataDisks, err := vmSet.GetDataDisks(test.nodeName, test.crt)
			assert.Equal(t, test.expectedDataDisks, dataDisks, "TestCase[%d]: %s", i, test.desc)
			assert.Equal(t, test.expectedError, err != nil, "TestCase[%d]: %s", i, test.desc)
		}
	}
}

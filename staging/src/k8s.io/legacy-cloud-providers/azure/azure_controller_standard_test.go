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
	"testing"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-03-01/compute"
	"github.com/Azure/go-autorest/autorest/to"
	"github.com/stretchr/testify/assert"

	"k8s.io/apimachinery/pkg/types"
)

func TestStandardAttachDisk(t *testing.T) {
	testCases := []struct {
		desc        string
		nodeName    types.NodeName
		expectedErr bool
	}{
		{
			desc:        "an error shall be returned if there's no corresponding vms",
			nodeName:    "vm2",
			expectedErr: true,
		},
		{
			desc:        "no error shall be returned if everything's good",
			nodeName:    "vm1",
			expectedErr: false,
		},
	}

	for i, test := range testCases {
		testCloud := getTestCloud()
		vmSet := testCloud.vmSet
		setTestVirtualMachines(testCloud, map[string]string{"vm1": "PowerState/Running"}, false)

		err := vmSet.AttachDisk(true, "",
			"uri", test.nodeName, 0, compute.CachingTypesReadOnly)
		assert.Equal(t, test.expectedErr, err != nil, "TestCase[%d]: %s", i, test.desc)
	}
}

func TestStandardDetachDisk(t *testing.T) {
	testCases := []struct {
		desc          string
		nodeName      types.NodeName
		diskName      string
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
	}

	for i, test := range testCases {
		testCloud := getTestCloud()
		vmSet := testCloud.vmSet
		setTestVirtualMachines(testCloud, map[string]string{"vm1": "PowerState/Running"}, false)

		_, err := vmSet.DetachDisk(test.diskName, "", test.nodeName)
		assert.Equal(t, test.expectedError, err != nil, "TestCase[%d]: %s", i, test.desc)
	}
}

func TestGetDataDisks(t *testing.T) {
	var testCases = []struct {
		desc              string
		nodeName          types.NodeName
		expectedDataDisks []compute.DataDisk
		expectedError     bool
	}{
		{
			desc:              "an error shall be returned if there's no corresponding vm",
			nodeName:          "vm2",
			expectedDataDisks: nil,
			expectedError:     true,
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
		},
	}
	for i, test := range testCases {
		testCloud := getTestCloud()
		vmSet := testCloud.vmSet
		setTestVirtualMachines(testCloud, map[string]string{"vm1": "PowerState/Running"}, false)

		dataDisks, err := vmSet.GetDataDisks(test.nodeName)
		assert.Equal(t, test.expectedDataDisks, dataDisks, "TestCase[%d]: %s", i, test.desc)
		assert.Equal(t, test.expectedError, err != nil, "TestCase[%d]: %s", i, test.desc)
	}
}

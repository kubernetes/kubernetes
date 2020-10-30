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

package vsphere

import (
	"testing"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/types"
	k8stypes "k8s.io/apimachinery/pkg/types"
)

func TestVsphereVolumeMap(t *testing.T) {
	tests := []struct {
		name            string
		deviceToAdd     object.VirtualDeviceList
		nodeToAdd       k8stypes.NodeName
		volumeToCheck   string
		runVerification bool
		expectInMap     bool
	}{
		{
			name:          "adding new volume",
			deviceToAdd:   getVirtualDeviceList("[foobar] kubevols/foo.vmdk"),
			nodeToAdd:     convertToK8sType("node1.lan"),
			volumeToCheck: "[foobar] kubevols/foo.vmdk",
			expectInMap:   true,
		},
		{
			name:          "mismatching volume",
			deviceToAdd:   getVirtualDeviceList("[foobar] kubevols/foo.vmdk"),
			nodeToAdd:     convertToK8sType("node1.lan"),
			volumeToCheck: "[foobar] kubevols/bar.vmdk",
			expectInMap:   false,
		},
		{
			name:            "should remove unverified devices",
			deviceToAdd:     getVirtualDeviceList("[foobar] kubevols/foo.vmdk"),
			nodeToAdd:       convertToK8sType("node1.lan"),
			volumeToCheck:   "[foobar] kubevols/foo.vmdk",
			runVerification: true,
			expectInMap:     false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			vMap := NewVsphereVolumeMap()
			vMap.Add(tc.nodeToAdd, tc.deviceToAdd)

			if tc.runVerification {
				vMap.StartDiskVerification()
				vMap.RemoveUnverified()
			}
			_, ok := vMap.CheckForVolume(tc.volumeToCheck)
			if ok != tc.expectInMap {
				t.Errorf("error checking volume %s, expected %v got %v", tc.volumeToCheck, tc.expectInMap, ok)
			}
		})
	}
}

func getVirtualDeviceList(vPath string) object.VirtualDeviceList {
	return object.VirtualDeviceList{
		&types.VirtualDisk{
			VirtualDevice: types.VirtualDevice{
				Key: 1000,
				Backing: &types.VirtualDiskFlatVer2BackingInfo{
					VirtualDeviceFileBackingInfo: types.VirtualDeviceFileBackingInfo{
						FileName: vPath,
					},
				},
			},
		},
	}
}

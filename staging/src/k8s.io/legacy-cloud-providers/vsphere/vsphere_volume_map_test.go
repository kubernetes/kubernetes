//go:build !providerless
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
		name        string
		deviceToAdd object.VirtualDeviceList
		nodeToAdd   k8stypes.NodeName
		checkRunner func(volumeMap *VsphereVolumeMap)
	}{
		{
			name:        "adding new volume",
			deviceToAdd: getVirtualDeviceList("[foobar] kubevols/foo.vmdk"),
			nodeToAdd:   convertToK8sType("node1.lan"),
			checkRunner: func(volumeMap *VsphereVolumeMap) {
				volumeToCheck := "[foobar] kubevols/foo.vmdk"
				_, ok := volumeMap.CheckForVolume(volumeToCheck)
				if !ok {
					t.Errorf("error checking volume %s, expected true got %v", volumeToCheck, ok)
				}
			},
		},
		{
			name:        "mismatching volume",
			deviceToAdd: getVirtualDeviceList("[foobar] kubevols/foo.vmdk"),
			nodeToAdd:   convertToK8sType("node1.lan"),
			checkRunner: func(volumeMap *VsphereVolumeMap) {
				volumeToCheck := "[foobar] kubevols/bar.vmdk"
				_, ok := volumeMap.CheckForVolume(volumeToCheck)
				if ok {
					t.Errorf("error checking volume %s, expected false got %v", volumeToCheck, ok)
				}
			},
		},
		{
			name:        "should remove unverified devices",
			deviceToAdd: getVirtualDeviceList("[foobar] kubevols/foo.vmdk"),
			nodeToAdd:   convertToK8sType("node1.lan"),
			checkRunner: func(volumeMap *VsphereVolumeMap) {
				volumeMap.StartDiskVerification()
				volumeMap.RemoveUnverified()
				volumeToCheck := "[foobar] kubevols/foo.vmdk"
				_, ok := volumeMap.CheckForVolume(volumeToCheck)
				if ok {
					t.Errorf("error checking volume %s, expected false got %v", volumeToCheck, ok)
				}
				node := k8stypes.NodeName("node1.lan")
				ok = volumeMap.CheckForNode(node)
				if ok {
					t.Errorf("unexpected node %s in node map", node)
				}
			},
		},
		{
			name:        "node check should return false for previously added node",
			deviceToAdd: getVirtualDeviceList("[foobar] kubevols/foo.vmdk"),
			nodeToAdd:   convertToK8sType("node1.lan"),
			checkRunner: func(volumeMap *VsphereVolumeMap) {
				volumeMap.StartDiskVerification()
				node := k8stypes.NodeName("node1.lan")
				ok := volumeMap.CheckForNode(node)
				if ok {
					t.Errorf("unexpected node %s in node map", node)
				}
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			vMap := NewVsphereVolumeMap()
			vMap.Add(tc.nodeToAdd, tc.deviceToAdd)
			tc.checkRunner(vMap)
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

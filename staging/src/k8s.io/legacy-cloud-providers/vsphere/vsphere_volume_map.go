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
	"sync"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/types"
	k8stypes "k8s.io/apimachinery/pkg/types"
)

type volumePath string

type nodeVolumeStatus struct {
	nodeName k8stypes.NodeName
	verified bool
}

// VsphereVolumeMap stores last known state of node and volume mapping
type VsphereVolumeMap struct {
	volumeNodeMap map[volumePath]nodeVolumeStatus
	lock          sync.RWMutex
}

func NewVsphereVolumeMap() *VsphereVolumeMap {
	return &VsphereVolumeMap{
		volumeNodeMap: map[volumePath]nodeVolumeStatus{},
	}
}

// StartDiskVerification marks all known volumes as unverified so as
// disks which aren't verified can be removed at the end of verification process
func (vsphereVolume *VsphereVolumeMap) StartDiskVerification() {
	vsphereVolume.lock.Lock()
	defer vsphereVolume.lock.Unlock()
	for k, v := range vsphereVolume.volumeNodeMap {
		v.verified = false
		vsphereVolume.volumeNodeMap[k] = v
	}
}

// CheckForVolume verifies if disk is attached to some node in the cluster.
// This check is not definitive and should be followed up by separate verification.
func (vsphereVolume *VsphereVolumeMap) CheckForVolume(path string) (k8stypes.NodeName, bool) {
	vsphereVolume.lock.RLock()
	defer vsphereVolume.lock.RUnlock()
	vPath := volumePath(path)
	ns, ok := vsphereVolume.volumeNodeMap[vPath]
	if ok {
		return ns.nodeName, true
	}
	return "", false
}

// Add all devices found on a node to the device map
func (vsphereVolume *VsphereVolumeMap) Add(node k8stypes.NodeName, vmDevices object.VirtualDeviceList) {
	vsphereVolume.lock.Lock()
	defer vsphereVolume.lock.Unlock()
	for _, device := range vmDevices {
		if vmDevices.TypeName(device) == "VirtualDisk" {
			virtualDevice := device.GetVirtualDevice()
			if backing, ok := virtualDevice.Backing.(*types.VirtualDiskFlatVer2BackingInfo); ok {
				filename := volumePath(backing.FileName)
				vsphereVolume.volumeNodeMap[filename] = nodeVolumeStatus{node, true}
			}
		}
	}
}

// RemoveUnverified will remove any device which we could not verify to be attached to a node.
func (vsphereVolume *VsphereVolumeMap) RemoveUnverified() {
	vsphereVolume.lock.Lock()
	defer vsphereVolume.lock.Unlock()
	for k, v := range vsphereVolume.volumeNodeMap {
		if !v.verified {
			delete(vsphereVolume.volumeNodeMap, k)
		}
	}
}

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
	"net/http"
	"strings"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-03-01/compute"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog"
)

// AttachDisk attaches a vhd to vm
// the vhd must exist, can be identified by diskName, diskURI, and lun.
func (as *availabilitySet) AttachDisk(isManagedDisk bool, diskName, diskURI string, nodeName types.NodeName, lun int32, cachingMode compute.CachingTypes) error {
	vm, err := as.getVirtualMachine(nodeName)
	if err != nil {
		return err
	}

	vmName := mapNodeNameToVMName(nodeName)
	nodeResourceGroup, err := as.GetNodeResourceGroup(vmName)
	if err != nil {
		return err
	}

	disks := make([]compute.DataDisk, len(*vm.StorageProfile.DataDisks))
	copy(disks, *vm.StorageProfile.DataDisks)

	if isManagedDisk {
		disks = append(disks,
			compute.DataDisk{
				Name:         &diskName,
				Lun:          &lun,
				Caching:      cachingMode,
				CreateOption: "attach",
				ManagedDisk: &compute.ManagedDiskParameters{
					ID: &diskURI,
				},
			})
	} else {
		disks = append(disks,
			compute.DataDisk{
				Name: &diskName,
				Vhd: &compute.VirtualHardDisk{
					URI: &diskURI,
				},
				Lun:          &lun,
				Caching:      cachingMode,
				CreateOption: "attach",
			})
	}

	newVM := compute.VirtualMachineUpdate{
		VirtualMachineProperties: &compute.VirtualMachineProperties{
			HardwareProfile: vm.HardwareProfile,
			StorageProfile: &compute.StorageProfile{
				DataDisks: &disks,
			},
		},
	}
	klog.V(2).Infof("azureDisk - update(%s): vm(%s) - attach disk(%s, %s)", nodeResourceGroup, vmName, diskName, diskURI)
	ctx, cancel := getContextWithCancel()
	defer cancel()

	// Invalidate the cache right after updating
	defer as.cloud.vmCache.Delete(vmName)

	_, err = as.VirtualMachinesClient.Update(ctx, nodeResourceGroup, vmName, newVM, "attach_disk")
	if err != nil {
		klog.Errorf("azureDisk - attach disk(%s, %s) failed, err: %v", diskName, diskURI, err)
		detail := err.Error()
		if strings.Contains(detail, errLeaseFailed) || strings.Contains(detail, errDiskBlobNotFound) {
			// if lease cannot be acquired or disk not found, immediately detach the disk and return the original error
			klog.V(2).Infof("azureDisk - err %v, try detach disk(%s, %s)", err, diskName, diskURI)
			as.DetachDisk(diskName, diskURI, nodeName)
		}
	} else {
		klog.V(2).Infof("azureDisk - attach disk(%s, %s) succeeded", diskName, diskURI)
	}
	return err
}

// DetachDisk detaches a disk from host
// the vhd can be identified by diskName or diskURI
func (as *availabilitySet) DetachDisk(diskName, diskURI string, nodeName types.NodeName) (*http.Response, error) {
	vm, err := as.getVirtualMachine(nodeName)
	if err != nil {
		// if host doesn't exist, no need to detach
		klog.Warningf("azureDisk - cannot find node %s, skip detaching disk(%s, %s)", nodeName, diskName, diskURI)
		return nil, nil
	}

	vmName := mapNodeNameToVMName(nodeName)
	nodeResourceGroup, err := as.GetNodeResourceGroup(vmName)
	if err != nil {
		return nil, err
	}

	disks := make([]compute.DataDisk, len(*vm.StorageProfile.DataDisks))
	copy(disks, *vm.StorageProfile.DataDisks)

	bFoundDisk := false
	for i, disk := range disks {
		if disk.Lun != nil && (disk.Name != nil && diskName != "" && strings.EqualFold(*disk.Name, diskName)) ||
			(disk.Vhd != nil && disk.Vhd.URI != nil && diskURI != "" && strings.EqualFold(*disk.Vhd.URI, diskURI)) ||
			(disk.ManagedDisk != nil && diskURI != "" && strings.EqualFold(*disk.ManagedDisk.ID, diskURI)) {
			// found the disk
			klog.V(2).Infof("azureDisk - detach disk: name %q uri %q", diskName, diskURI)
			disks = append(disks[:i], disks[i+1:]...)
			bFoundDisk = true
			break
		}
	}

	if !bFoundDisk {
		// only log here, next action is to update VM status with original meta data
		klog.Errorf("detach azure disk: disk %s not found, diskURI: %s", diskName, diskURI)
	}

	newVM := compute.VirtualMachineUpdate{
		VirtualMachineProperties: &compute.VirtualMachineProperties{
			HardwareProfile: vm.HardwareProfile,
			StorageProfile: &compute.StorageProfile{
				DataDisks: &disks,
			},
		},
	}
	klog.V(2).Infof("azureDisk - update(%s): vm(%s) - detach disk(%s, %s)", nodeResourceGroup, vmName, diskName, diskURI)
	ctx, cancel := getContextWithCancel()
	defer cancel()

	// Invalidate the cache right after updating
	defer as.cloud.vmCache.Delete(vmName)

	return as.VirtualMachinesClient.Update(ctx, nodeResourceGroup, vmName, newVM, "detach_disk")
}

// GetDataDisks gets a list of data disks attached to the node.
func (as *availabilitySet) GetDataDisks(nodeName types.NodeName) ([]compute.DataDisk, error) {
	vm, err := as.getVirtualMachine(nodeName)
	if err != nil {
		return nil, err
	}

	if vm.StorageProfile.DataDisks == nil {
		return nil, nil
	}

	return *vm.StorageProfile.DataDisks, nil
}

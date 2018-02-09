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
	"fmt"
	"strings"

	"github.com/Azure/azure-sdk-for-go/arm/compute"
	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

// AttachDisk attaches a vhd to vm
// the vhd must exist, can be identified by diskName, diskURI, and lun.
func (as *availabilitySet) AttachDisk(isManagedDisk bool, diskName, diskURI string, nodeName types.NodeName, lun int32, cachingMode compute.CachingTypes) error {
	vm, err := as.getVirtualMachine(nodeName)
	if err != nil {
		return err
	}

	disks := *vm.StorageProfile.DataDisks
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

	newVM := compute.VirtualMachine{
		Location: vm.Location,
		VirtualMachineProperties: &compute.VirtualMachineProperties{
			StorageProfile: &compute.StorageProfile{
				DataDisks: &disks,
			},
		},
	}
	vmName := mapNodeNameToVMName(nodeName)
	glog.V(2).Infof("azureDisk - update(%s): vm(%s) - attach disk", as.resourceGroup, vmName)
	respChan, errChan := as.VirtualMachinesClient.CreateOrUpdate(as.resourceGroup, vmName, newVM, nil)
	resp := <-respChan
	err = <-errChan
	if as.CloudProviderBackoff && shouldRetryAPIRequest(resp.Response, err) {
		glog.V(2).Infof("azureDisk - update(%s) backing off: vm(%s)", as.resourceGroup, vmName)
		retryErr := as.CreateOrUpdateVMWithRetry(vmName, newVM)
		if retryErr != nil {
			err = retryErr
			glog.V(2).Infof("azureDisk - update(%s) abort backoff: vm(%s)", as.resourceGroup, vmName)
		}
	}
	if err != nil {
		glog.Errorf("azureDisk - azure attach failed, err: %v", err)
		detail := err.Error()
		if strings.Contains(detail, errLeaseFailed) || strings.Contains(detail, errDiskBlobNotFound) {
			// if lease cannot be acquired or disk not found, immediately detach the disk and return the original error
			glog.Infof("azureDisk - err %s, try detach", detail)
			as.DetachDiskByName(diskName, diskURI, nodeName)
		}
	} else {
		glog.V(4).Info("azureDisk - azure attach succeeded")
		// Invalidate the cache right after updating
		as.cloud.vmCache.Delete(vmName)
	}
	return err
}

// DetachDiskByName detaches a vhd from host
// the vhd can be identified by diskName or diskURI
func (as *availabilitySet) DetachDiskByName(diskName, diskURI string, nodeName types.NodeName) error {
	vm, err := as.getVirtualMachine(nodeName)
	if err != nil {
		// if host doesn't exist, no need to detach
		glog.Warningf("azureDisk - cannot find node %s, skip detaching disk %s", nodeName, diskName)
		return nil
	}

	disks := *vm.StorageProfile.DataDisks
	bFoundDisk := false
	for i, disk := range disks {
		if disk.Lun != nil && (disk.Name != nil && diskName != "" && *disk.Name == diskName) ||
			(disk.Vhd != nil && disk.Vhd.URI != nil && diskURI != "" && *disk.Vhd.URI == diskURI) ||
			(disk.ManagedDisk != nil && diskURI != "" && *disk.ManagedDisk.ID == diskURI) {
			// found the disk
			glog.V(4).Infof("azureDisk - detach disk: name %q uri %q", diskName, diskURI)
			disks = append(disks[:i], disks[i+1:]...)
			bFoundDisk = true
			break
		}
	}

	if !bFoundDisk {
		return fmt.Errorf("detach azure disk failure, disk %s not found, diskURI: %s", diskName, diskURI)
	}

	newVM := compute.VirtualMachine{
		Location: vm.Location,
		VirtualMachineProperties: &compute.VirtualMachineProperties{
			StorageProfile: &compute.StorageProfile{
				DataDisks: &disks,
			},
		},
	}
	vmName := mapNodeNameToVMName(nodeName)
	glog.V(2).Infof("azureDisk - update(%s): vm(%s) - detach disk", as.resourceGroup, vmName)
	respChan, errChan := as.VirtualMachinesClient.CreateOrUpdate(as.resourceGroup, vmName, newVM, nil)
	resp := <-respChan
	err = <-errChan
	if as.CloudProviderBackoff && shouldRetryAPIRequest(resp.Response, err) {
		glog.V(2).Infof("azureDisk - update(%s) backing off: vm(%s)", as.resourceGroup, vmName)
		retryErr := as.CreateOrUpdateVMWithRetry(vmName, newVM)
		if retryErr != nil {
			err = retryErr
			glog.V(2).Infof("azureDisk - update(%s) abort backoff: vm(%s)", as.ResourceGroup, vmName)
		}
	}
	if err != nil {
		glog.Errorf("azureDisk - azure disk detach failed, err: %v", err)
	} else {
		glog.V(4).Info("azureDisk - azure disk detach succeeded")
		// Invalidate the cache right after updating
		as.cloud.vmCache.Delete(vmName)
	}
	return err
}

// GetDiskLun finds the lun on the host that the vhd is attached to, given a vhd's diskName and diskURI
func (as *availabilitySet) GetDiskLun(diskName, diskURI string, nodeName types.NodeName) (int32, error) {
	vm, err := as.getVirtualMachine(nodeName)
	if err != nil {
		return -1, err
	}
	disks := *vm.StorageProfile.DataDisks
	for _, disk := range disks {
		if disk.Lun != nil && (disk.Name != nil && diskName != "" && *disk.Name == diskName) ||
			(disk.Vhd != nil && disk.Vhd.URI != nil && diskURI != "" && *disk.Vhd.URI == diskURI) ||
			(disk.ManagedDisk != nil && *disk.ManagedDisk.ID == diskURI) {
			// found the disk
			glog.V(4).Infof("azureDisk - find disk: lun %d name %q uri %q", *disk.Lun, diskName, diskURI)
			return *disk.Lun, nil
		}
	}
	return -1, fmt.Errorf("Cannot find Lun for disk %s", diskName)
}

// GetNextDiskLun searches all vhd attachment on the host and find unused lun
// return -1 if all luns are used
func (as *availabilitySet) GetNextDiskLun(nodeName types.NodeName) (int32, error) {
	vm, err := as.getVirtualMachine(nodeName)
	if err != nil {
		return -1, err
	}
	used := make([]bool, maxLUN)
	disks := *vm.StorageProfile.DataDisks
	for _, disk := range disks {
		if disk.Lun != nil {
			used[*disk.Lun] = true
		}
	}
	for k, v := range used {
		if !v {
			return int32(k), nil
		}
	}
	return -1, fmt.Errorf("All Luns are used")
}

// DisksAreAttached checks if a list of volumes are attached to the node with the specified NodeName
func (as *availabilitySet) DisksAreAttached(diskNames []string, nodeName types.NodeName) (map[string]bool, error) {
	attached := make(map[string]bool)
	for _, diskName := range diskNames {
		attached[diskName] = false
	}
	vm, err := as.getVirtualMachine(nodeName)
	if err == cloudprovider.InstanceNotFound {
		// if host doesn't exist, no need to detach
		glog.Warningf("azureDisk - Cannot find node %q, DisksAreAttached will assume disks %v are not attached to it.",
			nodeName, diskNames)
		return attached, nil
	} else if err != nil {
		return attached, err
	}

	disks := *vm.StorageProfile.DataDisks
	for _, disk := range disks {
		for _, diskName := range diskNames {
			if disk.Name != nil && diskName != "" && *disk.Name == diskName {
				attached[diskName] = true
			}
		}
	}

	return attached, nil
}

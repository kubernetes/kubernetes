// +build !providerless

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

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-07-01/compute"
	"github.com/Azure/go-autorest/autorest/to"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog"
)

// AttachDisk attaches a vhd to vm
// the vhd must exist, can be identified by diskName, diskURI, and lun.
func (ss *scaleSet) AttachDisk(isManagedDisk bool, diskName, diskURI string, nodeName types.NodeName, lun int32, cachingMode compute.CachingTypes) error {
	vmName := mapNodeNameToVMName(nodeName)
	ssName, instanceID, vm, err := ss.getVmssVM(vmName, cacheReadTypeDefault)
	if err != nil {
		return err
	}

	nodeResourceGroup, err := ss.GetNodeResourceGroup(vmName)
	if err != nil {
		return err
	}

	disks := []compute.DataDisk{}
	if vm.StorageProfile != nil && vm.StorageProfile.DataDisks != nil {
		disks = make([]compute.DataDisk, len(*vm.StorageProfile.DataDisks))
		copy(disks, *vm.StorageProfile.DataDisks)
	}
	if isManagedDisk {
		disks = append(disks,
			compute.DataDisk{
				Name:         &diskName,
				Lun:          &lun,
				Caching:      compute.CachingTypes(cachingMode),
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
				Caching:      compute.CachingTypes(cachingMode),
				CreateOption: "attach",
			})
	}
	newVM := compute.VirtualMachineScaleSetVM{
		Sku:      vm.Sku,
		Location: vm.Location,
		VirtualMachineScaleSetVMProperties: &compute.VirtualMachineScaleSetVMProperties{
			HardwareProfile: vm.HardwareProfile,
			StorageProfile: &compute.StorageProfile{
				OsDisk:    vm.StorageProfile.OsDisk,
				DataDisks: &disks,
			},
		},
	}

	ctx, cancel := getContextWithCancel()
	defer cancel()

	// Invalidate the cache right after updating
	defer ss.deleteCacheForNode(vmName)

	klog.V(2).Infof("azureDisk - update(%s): vm(%s) - attach disk(%s, %s)", nodeResourceGroup, nodeName, diskName, diskURI)
	_, err = ss.VirtualMachineScaleSetVMsClient.Update(ctx, nodeResourceGroup, ssName, instanceID, newVM, "attach_disk")
	if err != nil {
		klog.Errorf("azureDisk - attach disk(%s, %s) on rg(%s) vm(%s) failed, err: %v", diskName, diskURI, nodeResourceGroup, nodeName, err)
		if strings.Contains(err.Error(), errDiskNotFound) {
			klog.Errorf("azureDisk - begin to filterNonExistingDisks(%s, %s) on rg(%s) vm(%s)", diskName, diskURI, nodeResourceGroup, nodeName)
			disks := ss.filterNonExistingDisks(ctx, *newVM.VirtualMachineScaleSetVMProperties.StorageProfile.DataDisks)
			newVM.VirtualMachineScaleSetVMProperties.StorageProfile.DataDisks = &disks
			_, err = ss.VirtualMachineScaleSetVMsClient.Update(ctx, nodeResourceGroup, ssName, instanceID, newVM, "attach_disk")
		}
	}
	klog.V(2).Infof("azureDisk - update(%s): vm(%s) - attach disk(%s, %s) returned with %v", nodeResourceGroup, nodeName, diskName, diskURI, err)
	return err
}

// DetachDisk detaches a disk from host
// the vhd can be identified by diskName or diskURI
func (ss *scaleSet) DetachDisk(diskName, diskURI string, nodeName types.NodeName) (*http.Response, error) {
	vmName := mapNodeNameToVMName(nodeName)
	ssName, instanceID, vm, err := ss.getVmssVM(vmName, cacheReadTypeDefault)
	if err != nil {
		return nil, err
	}

	nodeResourceGroup, err := ss.GetNodeResourceGroup(vmName)
	if err != nil {
		return nil, err
	}

	disks := []compute.DataDisk{}
	if vm.StorageProfile != nil && vm.StorageProfile.DataDisks != nil {
		disks = make([]compute.DataDisk, len(*vm.StorageProfile.DataDisks))
		copy(disks, *vm.StorageProfile.DataDisks)
	}
	bFoundDisk := false
	for i, disk := range disks {
		if disk.Lun != nil && (disk.Name != nil && diskName != "" && strings.EqualFold(*disk.Name, diskName)) ||
			(disk.Vhd != nil && disk.Vhd.URI != nil && diskURI != "" && strings.EqualFold(*disk.Vhd.URI, diskURI)) ||
			(disk.ManagedDisk != nil && diskURI != "" && strings.EqualFold(*disk.ManagedDisk.ID, diskURI)) {
			// found the disk
			klog.V(2).Infof("azureDisk - detach disk: name %q uri %q", diskName, diskURI)
			disks[i].ToBeDetached = to.BoolPtr(true)
			bFoundDisk = true
			break
		}
	}

	if !bFoundDisk {
		// only log here, next action is to update VM status with original meta data
		klog.Errorf("detach azure disk: disk %s not found, diskURI: %s", diskName, diskURI)
	}

	newVM := compute.VirtualMachineScaleSetVM{
		Sku:      vm.Sku,
		Location: vm.Location,
		VirtualMachineScaleSetVMProperties: &compute.VirtualMachineScaleSetVMProperties{
			HardwareProfile: vm.HardwareProfile,
			StorageProfile: &compute.StorageProfile{
				OsDisk:    vm.StorageProfile.OsDisk,
				DataDisks: &disks,
			},
		},
	}

	ctx, cancel := getContextWithCancel()
	defer cancel()

	// Invalidate the cache right after updating
	defer ss.deleteCacheForNode(vmName)

	klog.V(2).Infof("azureDisk - update(%s): vm(%s) - detach disk(%s, %s)", nodeResourceGroup, nodeName, diskName, diskURI)
	httpResponse, err := ss.VirtualMachineScaleSetVMsClient.Update(ctx, nodeResourceGroup, ssName, instanceID, newVM, "detach_disk")
	if err != nil {
		klog.Errorf("azureDisk - detach disk(%s, %s) on rg(%s) vm(%s) failed, err: %v", diskName, diskURI, nodeResourceGroup, nodeName, err)
		if strings.Contains(err.Error(), errDiskNotFound) {
			klog.Errorf("azureDisk - begin to filterNonExistingDisks(%s, %s) on rg(%s) vm(%s)", diskName, diskURI, nodeResourceGroup, nodeName)
			ss.filterNonExistingDisks(ctx, *newVM.VirtualMachineScaleSetVMProperties.StorageProfile.DataDisks)
			newVM.VirtualMachineScaleSetVMProperties.StorageProfile.DataDisks = &disks
			httpResponse, err = ss.VirtualMachineScaleSetVMsClient.Update(ctx, nodeResourceGroup, ssName, instanceID, newVM, "detach_disk")
		}
	}

	klog.V(2).Infof("azureDisk - update(%s): vm(%s) - detach disk(%s, %s) returned with %v", nodeResourceGroup, nodeName, diskName, diskURI, err)
	return httpResponse, err
}

// GetDataDisks gets a list of data disks attached to the node.
func (ss *scaleSet) GetDataDisks(nodeName types.NodeName, crt cacheReadType) ([]compute.DataDisk, error) {
	_, _, vm, err := ss.getVmssVM(string(nodeName), crt)
	if err != nil {
		return nil, err
	}

	if vm.StorageProfile == nil || vm.StorageProfile.DataDisks == nil {
		return nil, nil
	}

	return *vm.StorageProfile.DataDisks, nil
}

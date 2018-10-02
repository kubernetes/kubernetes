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

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2018-04-01/compute"
	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/types"
)

// AttachDisk attaches a vhd to vm
// the vhd must exist, can be identified by diskName, diskURI, and lun.
func (ss *scaleSet) AttachDisk(isManagedDisk bool, diskName, diskURI string, nodeName types.NodeName, lun int32, cachingMode compute.CachingTypes) error {
	vmName := mapNodeNameToVMName(nodeName)
	ssName, instanceID, vm, err := ss.getVmssVM(vmName)
	if err != nil {
		return err
	}

	nodeResourceGroup, err := ss.GetNodeResourceGroup(vmName)
	if err != nil {
		return err
	}

	disks := []compute.DataDisk{}
	if vm.StorageProfile != nil && vm.StorageProfile.DataDisks != nil {
		disks = *vm.StorageProfile.DataDisks
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
	vm.StorageProfile.DataDisks = &disks

	ctx, cancel := getContextWithCancel()
	defer cancel()
	glog.V(2).Infof("azureDisk - update(%s): vm(%s) - attach disk", nodeResourceGroup, nodeName)
	resp, err := ss.VirtualMachineScaleSetVMsClient.Update(ctx, nodeResourceGroup, ssName, instanceID, vm)
	if ss.CloudProviderBackoff && shouldRetryHTTPRequest(resp, err) {
		glog.V(2).Infof("azureDisk - update(%s) backing off: vm(%s)", nodeResourceGroup, nodeName)
		retryErr := ss.UpdateVmssVMWithRetry(ctx, nodeResourceGroup, ssName, instanceID, vm)
		if retryErr != nil {
			err = retryErr
			glog.V(2).Infof("azureDisk - update(%s) abort backoff: vm(%s)", nodeResourceGroup, nodeName)
		}
	}
	if err != nil {
		detail := err.Error()
		if strings.Contains(detail, errLeaseFailed) || strings.Contains(detail, errDiskBlobNotFound) {
			// if lease cannot be acquired or disk not found, immediately detach the disk and return the original error
			glog.Infof("azureDisk - err %s, try detach", detail)
			ss.DetachDiskByName(diskName, diskURI, nodeName)
		}
	} else {
		glog.V(4).Info("azureDisk - azure attach succeeded")
		// Invalidate the cache right after updating
		key := buildVmssCacheKey(nodeResourceGroup, ss.makeVmssVMName(ssName, instanceID))
		ss.vmssVMCache.Delete(key)
	}
	return err
}

// DetachDiskByName detaches a vhd from host
// the vhd can be identified by diskName or diskURI
func (ss *scaleSet) DetachDiskByName(diskName, diskURI string, nodeName types.NodeName) error {
	vmName := mapNodeNameToVMName(nodeName)
	ssName, instanceID, vm, err := ss.getVmssVM(vmName)
	if err != nil {
		return err
	}

	nodeResourceGroup, err := ss.GetNodeResourceGroup(vmName)
	if err != nil {
		return err
	}

	disks := []compute.DataDisk{}
	if vm.StorageProfile != nil && vm.StorageProfile.DataDisks != nil {
		disks = *vm.StorageProfile.DataDisks
	}
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

	vm.StorageProfile.DataDisks = &disks
	ctx, cancel := getContextWithCancel()
	defer cancel()
	glog.V(2).Infof("azureDisk - update(%s): vm(%s) - detach disk", nodeResourceGroup, nodeName)
	resp, err := ss.VirtualMachineScaleSetVMsClient.Update(ctx, nodeResourceGroup, ssName, instanceID, vm)
	if ss.CloudProviderBackoff && shouldRetryHTTPRequest(resp, err) {
		glog.V(2).Infof("azureDisk - update(%s) backing off: vm(%s)", nodeResourceGroup, nodeName)
		retryErr := ss.UpdateVmssVMWithRetry(ctx, nodeResourceGroup, ssName, instanceID, vm)
		if retryErr != nil {
			err = retryErr
			glog.V(2).Infof("azureDisk - update(%s) abort backoff: vm(%s)", nodeResourceGroup, nodeName)
		}
	}
	if err != nil {
		glog.Errorf("azureDisk - azure disk detach %q from %s failed, err: %v", diskName, nodeName, err)
	} else {
		glog.V(4).Info("azureDisk - azure detach succeeded")
		// Invalidate the cache right after updating
		key := buildVmssCacheKey(nodeResourceGroup, ss.makeVmssVMName(ssName, instanceID))
		ss.vmssVMCache.Delete(key)
	}

	return err
}

// GetDataDisks gets a list of data disks attached to the node.
func (ss *scaleSet) GetDataDisks(nodeName types.NodeName) ([]compute.DataDisk, error) {
	_, _, vm, err := ss.getVmssVM(string(nodeName))
	if err != nil {
		return nil, err
	}

	if vm.StorageProfile == nil || vm.StorageProfile.DataDisks == nil {
		return nil, nil
	}

	return *vm.StorageProfile.DataDisks, nil
}

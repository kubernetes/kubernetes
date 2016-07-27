/*
Copyright 2016 The Kubernetes Authors.

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
	"github.com/Azure/azure-sdk-for-go/arm/compute"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

// attach a vhd to vm
// the vhd must exist, can be identified by diskName, diskUri, and lun.
func (az *Cloud) AttachDisk(diskName, diskUri, vmName string, lun int32, cachingMode compute.CachingTypes) error {
	vm, exists, err := az.getVirtualMachine(vmName)
	if err != nil {
		return err
	} else if !exists {
		return cloudprovider.InstanceNotFound
	}
	disks := *vm.Properties.StorageProfile.DataDisks
	disks = append(disks,
		compute.DataDisk{
			Name: &diskName,
			Vhd: &compute.VirtualHardDisk{
				URI: &diskUri,
			},
			Lun:          &lun,
			Caching:      cachingMode,
			CreateOption: "attach",
		})

	newVM := compute.VirtualMachine{
		Location: vm.Location,
		Properties: &compute.VirtualMachineProperties{
			StorageProfile: &compute.StorageProfile{
				DataDisks: &disks,
			},
		},
	}
	res, err := az.VirtualMachinesClient.CreateOrUpdate(az.ResourceGroup, vmName,
		newVM, nil)
	glog.V(4).Infof("azure attach result:%#v, err: %v", res, err)
	return err
}

// detach a vhd from host
// the vhd can be identified by diskName or diskUri
func (az *Cloud) DetachDiskByName(diskName, diskUri, vmName string) error {
	vm, exists, err := az.getVirtualMachine(vmName)
	if err != nil {
		return err
	} else if !exists {
		return cloudprovider.InstanceNotFound
	}
	disks := *vm.Properties.StorageProfile.DataDisks
	for i, disk := range disks {
		if (disk.Name != nil && diskName != "" && *disk.Name == diskName) ||
			(disk.Vhd.URI != nil && diskUri != "" && *disk.Vhd.URI == diskUri) {
			// found the disk
			glog.V(4).Infof("detach disk: lun %d name %q uri %q", *disk.Lun, diskName, diskUri)
			disks = append(disks[:i], disks[i+1:]...)
			break
		}
	}
	newVM := compute.VirtualMachine{
		Location: vm.Location,
		Properties: &compute.VirtualMachineProperties{
			StorageProfile: &compute.StorageProfile{
				DataDisks: &disks,
			},
		},
	}
	res, err := az.VirtualMachinesClient.CreateOrUpdate(az.ResourceGroup, vmName,
		newVM, nil)
	glog.V(4).Infof("azure detach result:%#v, err: %v", res, err)
	return err
}

// given a vhd's diskName and diskUri, find the lun on the host that the vhd is attached to
func (az *Cloud) GetDiskLun(diskName, diskUri, vmName string) (int32, error) {
	vm, exists, err := az.getVirtualMachine(vmName)
	if err != nil {
		return -1, err
	} else if !exists {
		return -1, cloudprovider.InstanceNotFound
	}
	disks := *vm.Properties.StorageProfile.DataDisks
	for _, disk := range disks {
		if disk.Lun != nil {
			if (disk.Name != nil && diskName != "" && *disk.Name == diskName) ||
				(disk.Vhd.URI != nil && diskUri != "" && *disk.Vhd.URI == diskUri) {
				// found the disk
				glog.V(4).Infof("find disk: lun %d name %q uri %q", *disk.Lun, diskName, diskUri)
				return *disk.Lun, nil
			}
		}
	}
	return -1, cloudprovider.VolumeNotFound
}

// search all vhd attachment on the host and find unused lun
// return -1 if all luns are used
func (az *Cloud) GetNextDiskLun(vmName string) (int32, error) {
	vm, exists, err := az.getVirtualMachine(vmName)
	if err != nil {
		return -1, err
	} else if !exists {
		return -1, cloudprovider.InstanceNotFound
	}
	used := make([]bool, 64)
	disks := *vm.Properties.StorageProfile.DataDisks
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
	return -1, cloudprovider.VolumeNotFound
}

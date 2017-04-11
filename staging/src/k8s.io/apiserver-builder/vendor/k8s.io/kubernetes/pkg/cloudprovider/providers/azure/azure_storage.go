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
	"fmt"
	"strings"

	"github.com/Azure/azure-sdk-for-go/arm/compute"
	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/volume"
)

const (
	maxLUN               = 64 // max number of LUNs per VM
	errLeaseFailed       = "AcquireDiskLeaseFailed"
	errLeaseIDMissing    = "LeaseIdMissing"
	errContainerNotFound = "ContainerNotFound"
)

// AttachDisk attaches a vhd to vm
// the vhd must exist, can be identified by diskName, diskURI, and lun.
func (az *Cloud) AttachDisk(diskName, diskURI string, nodeName types.NodeName, lun int32, cachingMode compute.CachingTypes) error {
	vm, exists, err := az.getVirtualMachine(nodeName)
	if err != nil {
		return err
	} else if !exists {
		return cloudprovider.InstanceNotFound
	}
	disks := *vm.StorageProfile.DataDisks
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

	newVM := compute.VirtualMachine{
		Location: vm.Location,
		VirtualMachineProperties: &compute.VirtualMachineProperties{
			StorageProfile: &compute.StorageProfile{
				DataDisks: &disks,
			},
		},
	}
	vmName := mapNodeNameToVMName(nodeName)
	_, err = az.VirtualMachinesClient.CreateOrUpdate(az.ResourceGroup, vmName, newVM, nil)
	if err != nil {
		glog.Errorf("azure attach failed, err: %v", err)
		detail := err.Error()
		if strings.Contains(detail, errLeaseFailed) {
			// if lease cannot be acquired, immediately detach the disk and return the original error
			glog.Infof("failed to acquire disk lease, try detach")
			az.DetachDiskByName(diskName, diskURI, nodeName)
		}
	} else {
		glog.V(4).Infof("azure attach succeeded")
	}
	return err
}

// DisksAreAttached checks if a list of volumes are attached to the node with the specified NodeName
func (az *Cloud) DisksAreAttached(diskNames []string, nodeName types.NodeName) (map[string]bool, error) {
	attached := make(map[string]bool)
	for _, diskName := range diskNames {
		attached[diskName] = false
	}
	vm, exists, err := az.getVirtualMachine(nodeName)
	if !exists {
		// if host doesn't exist, no need to detach
		glog.Warningf("Cannot find node %q, DisksAreAttached will assume disks %v are not attached to it.",
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

// DetachDiskByName detaches a vhd from host
// the vhd can be identified by diskName or diskURI
func (az *Cloud) DetachDiskByName(diskName, diskURI string, nodeName types.NodeName) error {
	vm, exists, err := az.getVirtualMachine(nodeName)
	if err != nil || !exists {
		// if host doesn't exist, no need to detach
		glog.Warningf("cannot find node %s, skip detaching disk %s", nodeName, diskName)
		return nil
	}

	disks := *vm.StorageProfile.DataDisks
	for i, disk := range disks {
		if (disk.Name != nil && diskName != "" && *disk.Name == diskName) || (disk.Vhd.URI != nil && diskURI != "" && *disk.Vhd.URI == diskURI) {
			// found the disk
			glog.V(4).Infof("detach disk: name %q uri %q", diskName, diskURI)
			disks = append(disks[:i], disks[i+1:]...)
			break
		}
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
	_, err = az.VirtualMachinesClient.CreateOrUpdate(az.ResourceGroup, vmName, newVM, nil)
	if err != nil {
		glog.Errorf("azure disk detach failed, err: %v", err)
	} else {
		glog.V(4).Infof("azure disk detach succeeded")
	}
	return err
}

// GetDiskLun finds the lun on the host that the vhd is attached to, given a vhd's diskName and diskURI
func (az *Cloud) GetDiskLun(diskName, diskURI string, nodeName types.NodeName) (int32, error) {
	vm, exists, err := az.getVirtualMachine(nodeName)
	if err != nil {
		return -1, err
	} else if !exists {
		return -1, cloudprovider.InstanceNotFound
	}
	disks := *vm.StorageProfile.DataDisks
	for _, disk := range disks {
		if disk.Lun != nil && (disk.Name != nil && diskName != "" && *disk.Name == diskName) || (disk.Vhd.URI != nil && diskURI != "" && *disk.Vhd.URI == diskURI) {
			// found the disk
			glog.V(4).Infof("find disk: lun %d name %q uri %q", *disk.Lun, diskName, diskURI)
			return *disk.Lun, nil
		}
	}
	return -1, fmt.Errorf("Cannot find Lun for disk %s", diskName)
}

// GetNextDiskLun searches all vhd attachment on the host and find unused lun
// return -1 if all luns are used
func (az *Cloud) GetNextDiskLun(nodeName types.NodeName) (int32, error) {
	vm, exists, err := az.getVirtualMachine(nodeName)
	if err != nil {
		return -1, err
	} else if !exists {
		return -1, cloudprovider.InstanceNotFound
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

// CreateVolume creates a VHD blob in a storage account that has storageType and location using the given storage account.
// If no storage account is given, search all the storage accounts associated with the resource group and pick one that
// fits storage type and location.
func (az *Cloud) CreateVolume(name, storageAccount, storageType, location string, requestGB int) (string, string, int, error) {
	var err error
	accounts := []accountWithLocation{}
	if len(storageAccount) > 0 {
		accounts = append(accounts, accountWithLocation{Name: storageAccount})
	} else {
		// find a storage account
		accounts, err = az.getStorageAccounts()
		if err != nil {
			// TODO: create a storage account and container
			return "", "", 0, err
		}
	}
	for _, account := range accounts {
		glog.V(4).Infof("account %s type %s location %s", account.Name, account.StorageType, account.Location)
		if ((storageType == "" || account.StorageType == storageType) && (location == "" || account.Location == location)) || len(storageAccount) > 0 {
			// find the access key with this account
			key, err := az.getStorageAccesskey(account.Name)
			if err != nil {
				glog.V(2).Infof("no key found for storage account %s", account.Name)
				continue
			}

			// create a page blob in this account's vhd container
			name, uri, err := az.createVhdBlob(account.Name, key, name, int64(requestGB), nil)
			if err != nil {
				glog.V(2).Infof("failed to create vhd in account %s: %v", account.Name, err)
				continue
			}
			glog.V(4).Infof("created vhd blob uri: %s", uri)
			return name, uri, requestGB, err
		}
	}
	return "", "", 0, fmt.Errorf("failed to find a matching storage account")
}

// DeleteVolume deletes a VHD blob
func (az *Cloud) DeleteVolume(name, uri string) error {
	accountName, blob, err := az.getBlobNameAndAccountFromURI(uri)
	if err != nil {
		return fmt.Errorf("failed to parse vhd URI %v", err)
	}
	key, err := az.getStorageAccesskey(accountName)
	if err != nil {
		return fmt.Errorf("no key for storage account %s, err %v", accountName, err)
	}
	err = az.deleteVhdBlob(accountName, key, blob)
	if err != nil {
		glog.Warningf("failed to delete blob %s err: %v", uri, err)
		detail := err.Error()
		if strings.Contains(detail, errLeaseIDMissing) {
			// disk is still being used
			// see https://msdn.microsoft.com/en-us/library/microsoft.windowsazure.storage.blob.protocol.bloberrorcodestrings.leaseidmissing.aspx
			return volume.NewDeletedVolumeInUseError(fmt.Sprintf("disk %q is still in use while being deleted", name))
		}
		return fmt.Errorf("failed to delete vhd %v, account %s, blob %s, err: %v", uri, accountName, blob, err)
	}
	glog.V(4).Infof("blob %s deleted", uri)
	return nil

}

// CreateFileShare creates a file share, using a matching storage account
func (az *Cloud) CreateFileShare(name, storageAccount, storageType, location string, requestGB int) (string, string, error) {
	var err error
	accounts := []accountWithLocation{}
	if len(storageAccount) > 0 {
		accounts = append(accounts, accountWithLocation{Name: storageAccount})
	} else {
		// find a storage account
		accounts, err = az.getStorageAccounts()
		if err != nil {
			// TODO: create a storage account and container
			return "", "", err
		}
	}
	for _, account := range accounts {
		glog.V(4).Infof("account %s type %s location %s", account.Name, account.StorageType, account.Location)
		if ((storageType == "" || account.StorageType == storageType) && (location == "" || account.Location == location)) || len(storageAccount) > 0 {
			// find the access key with this account
			key, err := az.getStorageAccesskey(account.Name)
			if err != nil {
				glog.V(2).Infof("no key found for storage account %s", account.Name)
				continue
			}

			err = az.createFileShare(account.Name, key, name, requestGB)
			if err != nil {
				glog.V(2).Infof("failed to create share in account %s: %v", account.Name, err)
				continue
			}
			glog.V(4).Infof("created share %s in account %s", name, account.Name)
			return account.Name, key, err
		}
	}
	return "", "", fmt.Errorf("failed to find a matching storage account")
}

// DeleteFileShare deletes a file share using storage account name and key
func (az *Cloud) DeleteFileShare(accountName, key, name string) error {
	err := az.deleteFileShare(accountName, key, name)
	if err != nil {
		return err
	}
	glog.V(4).Infof("share %s deleted", name)
	return nil

}

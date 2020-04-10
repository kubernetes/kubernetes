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

package diskmanagers

import (
	"context"
	"fmt"
	"hash/fnv"
	"strings"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/types"

	"k8s.io/klog/v2"
	"k8s.io/legacy-cloud-providers/vsphere/vclib"
)

// vmDiskManager implements VirtualDiskProvider interface for creating volume using Virtual Machine Reconfigure approach
type vmDiskManager struct {
	diskPath      string
	volumeOptions *vclib.VolumeOptions
	vmOptions     *vclib.VMOptions
}

// Create implements Disk's Create interface
// Contains implementation of VM based Provisioning to provision disk with SPBM Policy or VSANStorageProfileData
func (vmdisk vmDiskManager) Create(ctx context.Context, datastore *vclib.Datastore) (canonicalDiskPath string, err error) {
	if vmdisk.volumeOptions.SCSIControllerType == "" {
		vmdisk.volumeOptions.SCSIControllerType = vclib.PVSCSIControllerType
	}
	pbmClient, err := vclib.NewPbmClient(ctx, datastore.Client())
	if err != nil {
		klog.Errorf("error occurred while creating new pbmClient, err: %+v", err)
		return "", err
	}

	if vmdisk.volumeOptions.StoragePolicyID == "" && vmdisk.volumeOptions.StoragePolicyName != "" {
		vmdisk.volumeOptions.StoragePolicyID, err = pbmClient.ProfileIDByName(ctx, vmdisk.volumeOptions.StoragePolicyName)
		if err != nil {
			klog.Errorf("error occurred while getting Profile Id from Profile Name: %s, err: %+v", vmdisk.volumeOptions.StoragePolicyName, err)
			return "", err
		}
	}
	if vmdisk.volumeOptions.StoragePolicyID != "" {
		compatible, faultMessage, err := datastore.IsCompatibleWithStoragePolicy(ctx, vmdisk.volumeOptions.StoragePolicyID)
		if err != nil {
			klog.Errorf("error occurred while checking datastore compatibility with storage policy id: %s, err: %+v", vmdisk.volumeOptions.StoragePolicyID, err)
			return "", err
		}

		if !compatible {
			klog.Errorf("Datastore: %s is not compatible with Policy: %s", datastore.Name(), vmdisk.volumeOptions.StoragePolicyName)
			return "", fmt.Errorf("user specified datastore is not compatible with the storagePolicy: %q. Failed with faults: %+q", vmdisk.volumeOptions.StoragePolicyName, faultMessage)
		}
	}

	storageProfileSpec := &types.VirtualMachineDefinedProfileSpec{}
	// Is PBM storage policy ID is present, set the storage spec profile ID,
	// else, set raw the VSAN policy string.
	if vmdisk.volumeOptions.StoragePolicyID != "" {
		storageProfileSpec.ProfileId = vmdisk.volumeOptions.StoragePolicyID
	} else if vmdisk.volumeOptions.VSANStorageProfileData != "" {
		// Check Datastore type - VSANStorageProfileData is only applicable to vSAN Datastore
		dsType, err := datastore.GetType(ctx)
		if err != nil {
			return "", err
		}
		if dsType != vclib.VSANDatastoreType {
			klog.Errorf("The specified datastore: %q is not a VSAN datastore", datastore.Name())
			return "", fmt.Errorf("the specified datastore: %q is not a VSAN datastore."+
				" the policy parameters will work only with VSAN Datastore."+
				" so, please specify a valid VSAN datastore in Storage class definition", datastore.Name())
		}
		storageProfileSpec.ProfileId = ""
		storageProfileSpec.ProfileData = &types.VirtualMachineProfileRawData{
			ExtensionKey: "com.vmware.vim.sps",
			ObjectData:   vmdisk.volumeOptions.VSANStorageProfileData,
		}
	} else {
		klog.Errorf("Both volumeOptions.StoragePolicyID and volumeOptions.VSANStorageProfileData are not set. One of them should be set")
		return "", fmt.Errorf("both volumeOptions.StoragePolicyID and volumeOptions.VSANStorageProfileData are not set. One of them should be set")
	}
	var dummyVM *vclib.VirtualMachine
	// Check if VM already exist in the folder.
	// If VM is already present, use it, else create a new dummy VM.
	fnvHash := fnv.New32a()
	fnvHash.Write([]byte(vmdisk.volumeOptions.Name))
	dummyVMFullName := vclib.DummyVMPrefixName + "-" + fmt.Sprint(fnvHash.Sum32())
	dummyVM, err = datastore.Datacenter.GetVMByPath(ctx, vmdisk.vmOptions.VMFolder.InventoryPath+"/"+dummyVMFullName)
	if err != nil {
		// Create a dummy VM
		klog.V(1).Infof("Creating Dummy VM: %q", dummyVMFullName)
		dummyVM, err = vmdisk.createDummyVM(ctx, datastore.Datacenter, dummyVMFullName)
		if err != nil {
			klog.Errorf("failed to create Dummy VM. err: %v", err)
			return "", err
		}
	}

	// Reconfigure the VM to attach the disk with the VSAN policy configured
	virtualMachineConfigSpec := types.VirtualMachineConfigSpec{}
	disk, _, err := dummyVM.CreateDiskSpec(ctx, vmdisk.diskPath, datastore, vmdisk.volumeOptions)
	if err != nil {
		klog.Errorf("failed to create Disk Spec. err: %v", err)
		return "", err
	}
	deviceConfigSpec := &types.VirtualDeviceConfigSpec{
		Device:        disk,
		Operation:     types.VirtualDeviceConfigSpecOperationAdd,
		FileOperation: types.VirtualDeviceConfigSpecFileOperationCreate,
	}

	deviceConfigSpec.Profile = append(deviceConfigSpec.Profile, storageProfileSpec)
	virtualMachineConfigSpec.DeviceChange = append(virtualMachineConfigSpec.DeviceChange, deviceConfigSpec)
	fileAlreadyExist := false
	task, err := dummyVM.Reconfigure(ctx, virtualMachineConfigSpec)
	if err != nil {
		klog.Errorf("Failed to reconfig. err: %v", err)
		return "", err
	}
	err = task.Wait(ctx)
	if err != nil {
		fileAlreadyExist = isAlreadyExists(vmdisk.diskPath, err)
		if fileAlreadyExist {
			//Skip error and continue to detach the disk as the disk was already created on the datastore.
			klog.V(vclib.LogLevel).Infof("File: %v already exists", vmdisk.diskPath)
		} else {
			klog.Errorf("Failed to attach the disk to VM: %q with err: %+v", dummyVMFullName, err)
			return "", err
		}
	}
	// Detach the disk from the dummy VM.
	err = dummyVM.DetachDisk(ctx, vmdisk.diskPath)
	if err != nil {
		if vclib.DiskNotFoundErrMsg == err.Error() && fileAlreadyExist {
			// Skip error if disk was already detached from the dummy VM but still present on the datastore.
			klog.V(vclib.LogLevel).Infof("File: %v is already detached", vmdisk.diskPath)
		} else {
			klog.Errorf("Failed to detach the disk: %q from VM: %q with err: %+v", vmdisk.diskPath, dummyVMFullName, err)
			return "", err
		}
	}
	//  Delete the dummy VM
	err = dummyVM.DeleteVM(ctx)
	if err != nil {
		klog.Errorf("Failed to destroy the vm: %q with err: %+v", dummyVMFullName, err)
	}
	return vmdisk.diskPath, nil
}

func (vmdisk vmDiskManager) Delete(ctx context.Context, datacenter *vclib.Datacenter) error {
	return fmt.Errorf("vmDiskManager.Delete is not supported")
}

// CreateDummyVM create a Dummy VM at specified location with given name.
func (vmdisk vmDiskManager) createDummyVM(ctx context.Context, datacenter *vclib.Datacenter, vmName string) (*vclib.VirtualMachine, error) {
	// Create a virtual machine config spec with 1 SCSI adapter.
	virtualMachineConfigSpec := types.VirtualMachineConfigSpec{
		Name: vmName,
		Files: &types.VirtualMachineFileInfo{
			VmPathName: "[" + vmdisk.volumeOptions.Datastore + "]",
		},
		NumCPUs:  1,
		MemoryMB: 4,
		DeviceChange: []types.BaseVirtualDeviceConfigSpec{
			&types.VirtualDeviceConfigSpec{
				Operation: types.VirtualDeviceConfigSpecOperationAdd,
				Device: &types.ParaVirtualSCSIController{
					VirtualSCSIController: types.VirtualSCSIController{
						SharedBus: types.VirtualSCSISharingNoSharing,
						VirtualController: types.VirtualController{
							BusNumber: 0,
							VirtualDevice: types.VirtualDevice{
								Key: 1000,
							},
						},
					},
				},
			},
		},
	}

	task, err := vmdisk.vmOptions.VMFolder.CreateVM(ctx, virtualMachineConfigSpec, vmdisk.vmOptions.VMResourcePool, nil)
	if err != nil {
		klog.Errorf("Failed to create VM. err: %+v", err)
		return nil, err
	}

	dummyVMTaskInfo, err := task.WaitForResult(ctx, nil)
	if err != nil {
		klog.Errorf("Error occurred while waiting for create VM task result. err: %+v", err)
		return nil, err
	}

	vmRef := dummyVMTaskInfo.Result.(object.Reference)
	dummyVM := object.NewVirtualMachine(datacenter.Client(), vmRef.Reference())
	return &vclib.VirtualMachine{VirtualMachine: dummyVM, Datacenter: datacenter}, nil
}

// CleanUpDummyVMs deletes stale dummyVM's
func CleanUpDummyVMs(ctx context.Context, folder *vclib.Folder) error {
	vmList, err := folder.GetVirtualMachines(ctx)
	if err != nil {
		klog.V(4).Infof("Failed to get virtual machines in the kubernetes cluster: %s, err: %+v", folder.InventoryPath, err)
		return err
	}
	if vmList == nil || len(vmList) == 0 {
		klog.Errorf("No virtual machines found in the kubernetes cluster: %s", folder.InventoryPath)
		return fmt.Errorf("no virtual machines found in the kubernetes cluster: %s", folder.InventoryPath)
	}
	var dummyVMList []*vclib.VirtualMachine
	// Loop through VM's in the Kubernetes cluster to find dummy VM's
	for _, vm := range vmList {
		vmName, err := vm.ObjectName(ctx)
		if err != nil {
			klog.V(4).Infof("Unable to get name from VM with err: %+v", err)
			continue
		}
		if strings.HasPrefix(vmName, vclib.DummyVMPrefixName) {
			vmObj := vclib.VirtualMachine{VirtualMachine: object.NewVirtualMachine(folder.Client(), vm.Reference())}
			dummyVMList = append(dummyVMList, &vmObj)
		}
	}
	for _, vm := range dummyVMList {
		err = vm.DeleteVM(ctx)
		if err != nil {
			klog.V(4).Infof("Unable to delete dummy VM with err: %+v", err)
			continue
		}
	}
	return nil
}

func isAlreadyExists(path string, err error) bool {
	errorMessage := fmt.Sprintf("Cannot complete the operation because the file or folder %s already exists", path)
	if errorMessage == err.Error() {
		return true
	}
	return false
}

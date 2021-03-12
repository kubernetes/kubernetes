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

package vclib

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
	"k8s.io/klog/v2"
)

// VirtualMachine extends the govmomi VirtualMachine object
type VirtualMachine struct {
	*object.VirtualMachine
	Datacenter *Datacenter
}

// IsDiskAttached checks if disk is attached to the VM.
func (vm *VirtualMachine) IsDiskAttached(ctx context.Context, diskPath string) (bool, error) {
	device, err := vm.getVirtualDeviceByPath(ctx, diskPath)
	if err != nil {
		return false, err
	}
	if device != nil {
		return true, nil
	}
	return false, nil
}

// DeleteVM deletes the VM.
func (vm *VirtualMachine) DeleteVM(ctx context.Context) error {
	destroyTask, err := vm.Destroy(ctx)
	if err != nil {
		klog.Errorf("Failed to delete the VM: %q. err: %+v", vm.InventoryPath, err)
		return err
	}
	return destroyTask.Wait(ctx)
}

// AttachDisk attaches the disk at location - vmDiskPath from Datastore - dsObj to the Virtual Machine
// Additionally the disk can be configured with SPBM policy if volumeOptions.StoragePolicyID is non-empty.
func (vm *VirtualMachine) AttachDisk(ctx context.Context, vmDiskPath string, volumeOptions *VolumeOptions) (string, error) {
	// Check if the diskControllerType is valid
	if !CheckControllerSupported(volumeOptions.SCSIControllerType) {
		return "", fmt.Errorf("Not a valid SCSI Controller Type. Valid options are %q", SCSIControllerTypeValidOptions())
	}
	vmDiskPathCopy := vmDiskPath
	vmDiskPath = RemoveStorageClusterORFolderNameFromVDiskPath(vmDiskPath)
	attached, err := vm.IsDiskAttached(ctx, vmDiskPath)
	if err != nil {
		klog.Errorf("Error occurred while checking if disk is attached on VM: %q. vmDiskPath: %q, err: %+v", vm.InventoryPath, vmDiskPath, err)
		return "", err
	}
	// If disk is already attached, return the disk UUID
	if attached {
		diskUUID, _ := vm.Datacenter.GetVirtualDiskPage83Data(ctx, vmDiskPath)
		return diskUUID, nil
	}

	if volumeOptions.StoragePolicyName != "" {
		pbmClient, err := NewPbmClient(ctx, vm.Client())
		if err != nil {
			klog.Errorf("Error occurred while creating new pbmClient. err: %+v", err)
			return "", err
		}

		volumeOptions.StoragePolicyID, err = pbmClient.ProfileIDByName(ctx, volumeOptions.StoragePolicyName)
		if err != nil {
			klog.Errorf("Failed to get Profile ID by name: %s. err: %+v", volumeOptions.StoragePolicyName, err)
			return "", err
		}
	}

	dsObj, err := vm.Datacenter.GetDatastoreByPath(ctx, vmDiskPathCopy)
	if err != nil {
		klog.Errorf("Failed to get datastore from vmDiskPath: %q. err: %+v", vmDiskPath, err)
		return "", err
	}
	// If disk is not attached, create a disk spec for disk to be attached to the VM.
	disk, newSCSIController, err := vm.CreateDiskSpec(ctx, vmDiskPath, dsObj, volumeOptions)
	if err != nil {
		klog.Errorf("Error occurred while creating disk spec. err: %+v", err)
		return "", err
	}
	vmDevices, err := vm.Device(ctx)
	if err != nil {
		klog.Errorf("Failed to retrieve VM devices for VM: %q. err: %+v", vm.InventoryPath, err)
		return "", err
	}
	virtualMachineConfigSpec := types.VirtualMachineConfigSpec{}
	deviceConfigSpec := &types.VirtualDeviceConfigSpec{
		Device:    disk,
		Operation: types.VirtualDeviceConfigSpecOperationAdd,
	}
	// Configure the disk with the SPBM profile only if ProfileID is not empty.
	if volumeOptions.StoragePolicyID != "" {
		profileSpec := &types.VirtualMachineDefinedProfileSpec{
			ProfileId: volumeOptions.StoragePolicyID,
		}
		deviceConfigSpec.Profile = append(deviceConfigSpec.Profile, profileSpec)
	}
	virtualMachineConfigSpec.DeviceChange = append(virtualMachineConfigSpec.DeviceChange, deviceConfigSpec)
	requestTime := time.Now()
	task, err := vm.Reconfigure(ctx, virtualMachineConfigSpec)
	if err != nil {
		RecordvSphereMetric(APIAttachVolume, requestTime, err)
		klog.Errorf("Failed to attach the disk with storagePolicy: %q on VM: %q. err - %+v", volumeOptions.StoragePolicyID, vm.InventoryPath, err)
		if newSCSIController != nil {
			nestedErr := vm.deleteController(ctx, newSCSIController, vmDevices)
			if nestedErr != nil {
				return "", fmt.Errorf("failed to delete SCSI Controller after reconfiguration failed with err=%v: %v", err, nestedErr)
			}
		}
		return "", err
	}
	err = task.Wait(ctx)
	RecordvSphereMetric(APIAttachVolume, requestTime, err)
	if err != nil {
		klog.Errorf("Failed to attach the disk with storagePolicy: %+q on VM: %q. err - %+v", volumeOptions.StoragePolicyID, vm.InventoryPath, err)
		if newSCSIController != nil {
			nestedErr := vm.deleteController(ctx, newSCSIController, vmDevices)
			if nestedErr != nil {
				return "", fmt.Errorf("failed to delete SCSI Controller after waiting for reconfiguration failed with err='%v': %v", err, nestedErr)
			}
		}
		return "", err
	}

	// Once disk is attached, get the disk UUID.
	diskUUID, err := vm.Datacenter.GetVirtualDiskPage83Data(ctx, vmDiskPath)
	if err != nil {
		klog.Errorf("Error occurred while getting Disk Info from VM: %q. err: %v", vm.InventoryPath, err)
		nestedErr := vm.DetachDisk(ctx, vmDiskPath)
		if nestedErr != nil {
			return "", fmt.Errorf("failed to detach disk after getting VM UUID failed with err='%v': %v", err, nestedErr)
		}
		if newSCSIController != nil {
			nestedErr = vm.deleteController(ctx, newSCSIController, vmDevices)
			if nestedErr != nil {
				return "", fmt.Errorf("failed to delete SCSI Controller after getting VM UUID failed with err='%v': %v", err, nestedErr)
			}
		}
		return "", err
	}
	return diskUUID, nil
}

// DetachDisk detaches the disk specified by vmDiskPath
func (vm *VirtualMachine) DetachDisk(ctx context.Context, vmDiskPath string) error {
	device, err := vm.getVirtualDeviceByPath(ctx, vmDiskPath)
	if err != nil {
		klog.Errorf("Disk ID not found for VM: %q with diskPath: %q", vm.InventoryPath, vmDiskPath)
		return err
	}
	if device == nil {
		klog.Errorf("No virtual device found with diskPath: %q on VM: %q", vmDiskPath, vm.InventoryPath)
		return fmt.Errorf("No virtual device found with diskPath: %q on VM: %q", vmDiskPath, vm.InventoryPath)
	}
	// Detach disk from VM
	requestTime := time.Now()
	err = vm.RemoveDevice(ctx, true, device)
	RecordvSphereMetric(APIDetachVolume, requestTime, err)
	if err != nil {
		klog.Errorf("Error occurred while removing disk device for VM: %q. err: %v", vm.InventoryPath, err)
		return err
	}
	return nil
}

// GetResourcePool gets the resource pool for VM.
func (vm *VirtualMachine) GetResourcePool(ctx context.Context) (*object.ResourcePool, error) {
	vmMoList, err := vm.Datacenter.GetVMMoList(ctx, []*VirtualMachine{vm}, []string{"resourcePool"})
	if err != nil {
		klog.Errorf("Failed to get resource pool from VM: %q. err: %+v", vm.InventoryPath, err)
		return nil, err
	}
	return object.NewResourcePool(vm.Client(), vmMoList[0].ResourcePool.Reference()), nil
}

// IsActive checks if the VM is active.
// Returns true if VM is in poweredOn state.
func (vm *VirtualMachine) IsActive(ctx context.Context) (bool, error) {
	vmMoList, err := vm.Datacenter.GetVMMoList(ctx, []*VirtualMachine{vm}, []string{"summary"})
	if err != nil {
		klog.Errorf("Failed to get VM Managed object with property summary. err: +%v", err)
		return false, err
	}
	if vmMoList[0].Summary.Runtime.PowerState == ActivePowerState {
		return true, nil
	}

	return false, nil
}

// Exists checks if VM exists and is not terminated
func (vm *VirtualMachine) Exists(ctx context.Context) (bool, error) {
	vmMoList, err := vm.Datacenter.GetVMMoList(ctx, []*VirtualMachine{vm}, []string{"summary.runtime.powerState"})
	if err != nil {
		klog.Errorf("Failed to get VM Managed object with property summary. err: +%v", err)
		return false, err
	}
	// We check for VMs which are still available in vcenter and has not been terminated/removed from
	// disk and hence we consider PoweredOn,PoweredOff and Suspended as alive states.
	aliveStates := []types.VirtualMachinePowerState{
		types.VirtualMachinePowerStatePoweredOff,
		types.VirtualMachinePowerStatePoweredOn,
		types.VirtualMachinePowerStateSuspended,
	}
	currentState := vmMoList[0].Summary.Runtime.PowerState
	for _, state := range aliveStates {
		if state == currentState {
			return true, nil
		}
	}
	return false, nil
}

// GetAllAccessibleDatastores gets the list of accessible Datastores for the given Virtual Machine
func (vm *VirtualMachine) GetAllAccessibleDatastores(ctx context.Context) ([]*DatastoreInfo, error) {
	host, err := vm.HostSystem(ctx)
	if err != nil {
		klog.Errorf("Failed to get host system for VM: %q. err: %+v", vm.InventoryPath, err)
		return nil, err
	}
	var hostSystemMo mo.HostSystem
	s := object.NewSearchIndex(vm.Client())
	err = s.Properties(ctx, host.Reference(), []string{DatastoreProperty}, &hostSystemMo)
	if err != nil {
		klog.Errorf("Failed to retrieve datastores for host: %+v. err: %+v", host, err)
		return nil, err
	}
	var dsRefList []types.ManagedObjectReference
	dsRefList = append(dsRefList, hostSystemMo.Datastore...)

	var dsMoList []mo.Datastore
	pc := property.DefaultCollector(vm.Client())
	properties := []string{DatastoreInfoProperty, NameProperty}
	err = pc.Retrieve(ctx, dsRefList, properties, &dsMoList)
	if err != nil {
		klog.Errorf("Failed to get Datastore managed objects from datastore objects."+
			" dsObjList: %+v, properties: %+v, err: %v", dsRefList, properties, err)
		return nil, err
	}
	klog.V(9).Infof("Result dsMoList: %+v", dsMoList)
	finder := getFinder(vm.Datacenter)
	var dsObjList []*DatastoreInfo
	for _, dsMo := range dsMoList {
		// use the finder so that InventoryPath is set correctly in ds
		ds, err := finder.Datastore(ctx, dsMo.Name)
		if err != nil {
			klog.Errorf("Failed finding datastore: %s. err: %+v", dsMo.Name, err)
			return nil, err
		}
		datastore := Datastore{ds, vm.Datacenter}
		dsObjList = append(dsObjList,
			&DatastoreInfo{
				&datastore,
				dsMo.Info.GetDatastoreInfo()})
	}
	return dsObjList, nil
}

// CreateDiskSpec creates a disk spec for disk
func (vm *VirtualMachine) CreateDiskSpec(ctx context.Context, diskPath string, dsObj *Datastore, volumeOptions *VolumeOptions) (*types.VirtualDisk, types.BaseVirtualDevice, error) {
	var newSCSIController types.BaseVirtualDevice
	vmDevices, err := vm.Device(ctx)
	if err != nil {
		klog.Errorf("Failed to retrieve VM devices. err: %+v", err)
		return nil, nil, err
	}
	// find SCSI controller of particular type from VM devices
	scsiControllersOfRequiredType := getSCSIControllersOfType(vmDevices, volumeOptions.SCSIControllerType)
	scsiController := getAvailableSCSIController(scsiControllersOfRequiredType)
	if scsiController == nil {
		newSCSIController, err = vm.createAndAttachSCSIController(ctx, volumeOptions.SCSIControllerType)
		if err != nil {
			klog.Errorf("Failed to create SCSI controller for VM :%q with err: %+v", vm.InventoryPath, err)
			return nil, nil, err
		}
		// Get VM device list
		vmDevices, err := vm.Device(ctx)
		if err != nil {
			klog.Errorf("Failed to retrieve VM devices. err: %v", err)
			return nil, nil, err
		}
		// verify scsi controller in virtual machine
		scsiControllersOfRequiredType := getSCSIControllersOfType(vmDevices, volumeOptions.SCSIControllerType)
		scsiController = getAvailableSCSIController(scsiControllersOfRequiredType)
		if scsiController == nil {
			klog.Errorf("Cannot find SCSI controller of type: %q in VM", volumeOptions.SCSIControllerType)
			// attempt clean up of scsi controller
			if err := vm.deleteController(ctx, newSCSIController, vmDevices); err != nil {
				return nil, nil, fmt.Errorf("failed to delete SCSI controller after failing to find it on VM: %v", err)
			}
			return nil, nil, fmt.Errorf("cannot find SCSI controller of type: %q in VM", volumeOptions.SCSIControllerType)
		}
	}
	disk := vmDevices.CreateDisk(scsiController, dsObj.Reference(), diskPath)
	unitNumber, err := getNextUnitNumber(vmDevices, scsiController)
	if err != nil {
		klog.Errorf("Cannot attach disk to VM, unitNumber limit reached - %+v.", err)
		return nil, nil, err
	}
	*disk.UnitNumber = unitNumber
	backing := disk.Backing.(*types.VirtualDiskFlatVer2BackingInfo)
	backing.DiskMode = string(types.VirtualDiskModeIndependent_persistent)

	if volumeOptions.CapacityKB != 0 {
		disk.CapacityInKB = int64(volumeOptions.CapacityKB)
	}
	if volumeOptions.DiskFormat != "" {
		var diskFormat string
		diskFormat = DiskFormatValidType[volumeOptions.DiskFormat]
		switch diskFormat {
		case ThinDiskType:
			backing.ThinProvisioned = types.NewBool(true)
		case EagerZeroedThickDiskType:
			backing.EagerlyScrub = types.NewBool(true)
		default:
			backing.ThinProvisioned = types.NewBool(false)
		}
	}
	return disk, newSCSIController, nil
}

// GetVirtualDiskPath gets the first available virtual disk devicePath from the VM
func (vm *VirtualMachine) GetVirtualDiskPath(ctx context.Context) (string, error) {
	vmDevices, err := vm.Device(ctx)
	if err != nil {
		klog.Errorf("Failed to get the devices for VM: %q. err: %+v", vm.InventoryPath, err)
		return "", err
	}
	// filter vm devices to retrieve device for the given vmdk file identified by disk path
	for _, device := range vmDevices {
		if vmDevices.TypeName(device) == "VirtualDisk" {
			virtualDevice := device.GetVirtualDevice()
			if backing, ok := virtualDevice.Backing.(*types.VirtualDiskFlatVer2BackingInfo); ok {
				return backing.FileName, nil
			}
		}
	}
	return "", nil
}

// createAndAttachSCSIController creates and attachs the SCSI controller to the VM.
func (vm *VirtualMachine) createAndAttachSCSIController(ctx context.Context, diskControllerType string) (types.BaseVirtualDevice, error) {
	// Get VM device list
	vmDevices, err := vm.Device(ctx)
	if err != nil {
		klog.Errorf("Failed to retrieve VM devices for VM: %q. err: %+v", vm.InventoryPath, err)
		return nil, err
	}
	allSCSIControllers := getSCSIControllers(vmDevices)
	if len(allSCSIControllers) >= SCSIControllerLimit {
		// we reached the maximum number of controllers we can attach
		klog.Errorf("SCSI Controller Limit of %d has been reached, cannot create another SCSI controller", SCSIControllerLimit)
		return nil, fmt.Errorf("SCSI Controller Limit of %d has been reached, cannot create another SCSI controller", SCSIControllerLimit)
	}
	newSCSIController, err := vmDevices.CreateSCSIController(diskControllerType)
	if err != nil {
		klog.Errorf("Failed to create new SCSI controller on VM: %q. err: %+v", vm.InventoryPath, err)
		return nil, err
	}
	configNewSCSIController := newSCSIController.(types.BaseVirtualSCSIController).GetVirtualSCSIController()
	hotAndRemove := true
	configNewSCSIController.HotAddRemove = &hotAndRemove
	configNewSCSIController.SharedBus = types.VirtualSCSISharing(types.VirtualSCSISharingNoSharing)

	// add the scsi controller to virtual machine
	err = vm.AddDevice(context.TODO(), newSCSIController)
	if err != nil {
		klog.V(LogLevel).Infof("Cannot add SCSI controller to VM: %q. err: %+v", vm.InventoryPath, err)
		// attempt clean up of scsi controller
		nestedErr := vm.deleteController(ctx, newSCSIController, vmDevices)
		if nestedErr != nil {
			return nil, fmt.Errorf("failed to delete SCSI controller after failing to add it to vm with err='%v': %v", err, nestedErr)
		}
		return nil, err
	}
	return newSCSIController, nil
}

// getVirtualDeviceByPath gets the virtual device by path
func (vm *VirtualMachine) getVirtualDeviceByPath(ctx context.Context, diskPath string) (types.BaseVirtualDevice, error) {
	vmDevices, err := vm.Device(ctx)
	if err != nil {
		klog.Errorf("Failed to get the devices for VM: %q. err: %+v", vm.InventoryPath, err)
		return nil, err
	}

	// filter vm devices to retrieve device for the given vmdk file identified by disk path
	for _, device := range vmDevices {
		if vmDevices.TypeName(device) == "VirtualDisk" {
			virtualDevice := device.GetVirtualDevice()
			if backing, ok := virtualDevice.Backing.(*types.VirtualDiskFlatVer2BackingInfo); ok {
				if matchVirtualDiskAndVolPath(backing.FileName, diskPath) {
					klog.V(LogLevel).Infof("Found VirtualDisk backing with filename %q for diskPath %q", backing.FileName, diskPath)
					return device, nil
				}
			}
		}
	}
	return nil, nil
}

func matchVirtualDiskAndVolPath(diskPath, volPath string) bool {
	fileExt := ".vmdk"
	diskPath = strings.TrimSuffix(diskPath, fileExt)
	volPath = strings.TrimSuffix(volPath, fileExt)
	return diskPath == volPath
}

// deleteController removes latest added SCSI controller from VM.
func (vm *VirtualMachine) deleteController(ctx context.Context, controllerDevice types.BaseVirtualDevice, vmDevices object.VirtualDeviceList) error {
	controllerDeviceList := vmDevices.SelectByType(controllerDevice)
	if len(controllerDeviceList) < 1 {
		return ErrNoDevicesFound
	}
	device := controllerDeviceList[len(controllerDeviceList)-1]
	err := vm.RemoveDevice(ctx, true, device)
	if err != nil {
		klog.Errorf("Error occurred while removing device on VM: %q. err: %+v", vm.InventoryPath, err)
		return err
	}
	return nil
}

// RenewVM renews this virtual machine with new client connection.
func (vm *VirtualMachine) RenewVM(client *vim25.Client) VirtualMachine {
	dc := Datacenter{Datacenter: object.NewDatacenter(client, vm.Datacenter.Reference())}
	newVM := object.NewVirtualMachine(client, vm.VirtualMachine.Reference())
	return VirtualMachine{VirtualMachine: newVM, Datacenter: &dc}
}

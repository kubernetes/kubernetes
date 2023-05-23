/*
Copyright (c) 2017-2018 VMware, Inc. All Rights Reserved.

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

package simulator

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"log"
	"net"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/google/uuid"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/simulator/esx"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type VirtualMachine struct {
	mo.VirtualMachine

	log string
	sid int32
	run container
	uid uuid.UUID
	imc *types.CustomizationSpec
}

func asVirtualMachineMO(obj mo.Reference) (*mo.VirtualMachine, bool) {
	vm, ok := getManagedObject(obj).Addr().Interface().(*mo.VirtualMachine)
	return vm, ok
}

func NewVirtualMachine(ctx *Context, parent types.ManagedObjectReference, spec *types.VirtualMachineConfigSpec) (*VirtualMachine, types.BaseMethodFault) {
	vm := &VirtualMachine{}
	vm.Parent = &parent
	ctx.Map.reference(vm)

	folder := ctx.Map.Get(parent)

	if spec.Name == "" {
		return vm, &types.InvalidVmConfig{Property: "configSpec.name"}
	}

	if spec.Files == nil || spec.Files.VmPathName == "" {
		return vm, &types.InvalidVmConfig{Property: "configSpec.files.vmPathName"}
	}

	rspec := types.DefaultResourceConfigSpec()
	vm.Guest = &types.GuestInfo{}
	vm.Config = &types.VirtualMachineConfigInfo{
		ExtraConfig:        []types.BaseOptionValue{&types.OptionValue{Key: "govcsim", Value: "TRUE"}},
		Tools:              &types.ToolsConfigInfo{},
		MemoryAllocation:   &rspec.MemoryAllocation,
		CpuAllocation:      &rspec.CpuAllocation,
		LatencySensitivity: &types.LatencySensitivity{Level: types.LatencySensitivitySensitivityLevelNormal},
		BootOptions:        &types.VirtualMachineBootOptions{},
		CreateDate:         types.NewTime(time.Now()),
	}
	vm.Layout = &types.VirtualMachineFileLayout{}
	vm.LayoutEx = &types.VirtualMachineFileLayoutEx{
		Timestamp: time.Now(),
	}
	vm.Snapshot = nil // intentionally set to nil until a snapshot is created
	vm.Storage = &types.VirtualMachineStorageInfo{
		Timestamp: time.Now(),
	}
	vm.Summary.Guest = &types.VirtualMachineGuestSummary{}
	vm.Summary.Vm = &vm.Self
	vm.Summary.Storage = &types.VirtualMachineStorageSummary{
		Timestamp: time.Now(),
	}

	vmx := vm.vmx(spec)
	if vmx.Path == "" {
		// Append VM Name as the directory name if not specified
		vmx.Path = spec.Name
	}

	dc := ctx.Map.getEntityDatacenter(folder.(mo.Entity))
	ds := ctx.Map.FindByName(vmx.Datastore, dc.Datastore).(*Datastore)
	dir := path.Join(ds.Info.GetDatastoreInfo().Url, vmx.Path)

	if path.Ext(vmx.Path) == ".vmx" {
		dir = path.Dir(dir)
		// Ignore error here, deferring to createFile
		_ = os.Mkdir(dir, 0700)
	} else {
		// Create VM directory, renaming if already exists
		name := dir

		for i := 0; i < 1024; /* just in case */ i++ {
			err := os.Mkdir(name, 0700)
			if err != nil {
				if os.IsExist(err) {
					name = fmt.Sprintf("%s (%d)", dir, i)
					continue
				}
				return nil, &types.FileFault{File: name}
			}
			break
		}
		vmx.Path = path.Join(path.Base(name), spec.Name+".vmx")
	}

	spec.Files.VmPathName = vmx.String()

	dsPath := path.Dir(spec.Files.VmPathName)
	vm.uid = sha1UUID(spec.Files.VmPathName)

	defaults := types.VirtualMachineConfigSpec{
		NumCPUs:           1,
		NumCoresPerSocket: 1,
		MemoryMB:          32,
		Uuid:              vm.uid.String(),
		InstanceUuid:      newUUID(strings.ToUpper(spec.Files.VmPathName)),
		Version:           esx.HardwareVersion,
		Firmware:          string(types.GuestOsDescriptorFirmwareTypeBios),
		Files: &types.VirtualMachineFileInfo{
			SnapshotDirectory: dsPath,
			SuspendDirectory:  dsPath,
			LogDirectory:      dsPath,
		},
	}

	// Add the default devices
	defaults.DeviceChange, _ = object.VirtualDeviceList(esx.VirtualDevice).ConfigSpec(types.VirtualDeviceConfigSpecOperationAdd)

	err := vm.configure(ctx, &defaults)
	if err != nil {
		return vm, err
	}

	vm.Runtime.PowerState = types.VirtualMachinePowerStatePoweredOff
	vm.Runtime.ConnectionState = types.VirtualMachineConnectionStateConnected
	vm.Summary.Runtime = vm.Runtime

	vm.Capability.ChangeTrackingSupported = types.NewBool(changeTrackingSupported(spec))

	vm.Summary.QuickStats.GuestHeartbeatStatus = types.ManagedEntityStatusGray
	vm.Summary.OverallStatus = types.ManagedEntityStatusGreen
	vm.ConfigStatus = types.ManagedEntityStatusGreen

	// put vm in the folder only if no errors occurred
	f, _ := asFolderMO(folder)
	folderPutChild(ctx, f, vm)

	return vm, nil
}

func (o *VirtualMachine) RenameTask(ctx *Context, r *types.Rename_Task) soap.HasFault {
	return RenameTask(ctx, o, r)
}

func (*VirtualMachine) Reload(*types.Reload) soap.HasFault {
	return &methods.ReloadBody{Res: new(types.ReloadResponse)}
}

func (vm *VirtualMachine) event() types.VmEvent {
	host := Map.Get(*vm.Runtime.Host).(*HostSystem)

	return types.VmEvent{
		Event: types.Event{
			Datacenter:      datacenterEventArgument(host),
			ComputeResource: host.eventArgumentParent(),
			Host:            host.eventArgument(),
			Ds:              Map.Get(vm.Datastore[0]).(*Datastore).eventArgument(),
			Vm: &types.VmEventArgument{
				EntityEventArgument: types.EntityEventArgument{Name: vm.Name},
				Vm:                  vm.Self,
			},
		},
	}
}

func (vm *VirtualMachine) hostInMM(ctx *Context) bool {
	return ctx.Map.Get(*vm.Runtime.Host).(*HostSystem).Runtime.InMaintenanceMode
}

func (vm *VirtualMachine) apply(spec *types.VirtualMachineConfigSpec) {
	if spec.Files == nil {
		spec.Files = new(types.VirtualMachineFileInfo)
	}

	apply := []struct {
		src string
		dst *string
	}{
		{spec.AlternateGuestName, &vm.Config.AlternateGuestName},
		{spec.Annotation, &vm.Config.Annotation},
		{spec.Firmware, &vm.Config.Firmware},
		{spec.InstanceUuid, &vm.Config.InstanceUuid},
		{spec.LocationId, &vm.Config.LocationId},
		{spec.NpivWorldWideNameType, &vm.Config.NpivWorldWideNameType},
		{spec.Name, &vm.Name},
		{spec.Name, &vm.Config.Name},
		{spec.Name, &vm.Summary.Config.Name},
		{spec.GuestId, &vm.Config.GuestId},
		{spec.GuestId, &vm.Config.GuestFullName},
		{spec.GuestId, &vm.Summary.Guest.GuestId},
		{spec.GuestId, &vm.Summary.Config.GuestId},
		{spec.GuestId, &vm.Summary.Config.GuestFullName},
		{spec.Uuid, &vm.Config.Uuid},
		{spec.Uuid, &vm.Summary.Config.Uuid},
		{spec.InstanceUuid, &vm.Config.InstanceUuid},
		{spec.InstanceUuid, &vm.Summary.Config.InstanceUuid},
		{spec.Version, &vm.Config.Version},
		{spec.Files.VmPathName, &vm.Config.Files.VmPathName},
		{spec.Files.VmPathName, &vm.Summary.Config.VmPathName},
		{spec.Files.SnapshotDirectory, &vm.Config.Files.SnapshotDirectory},
		{spec.Files.SuspendDirectory, &vm.Config.Files.SuspendDirectory},
		{spec.Files.LogDirectory, &vm.Config.Files.LogDirectory},
	}

	for _, f := range apply {
		if f.src != "" {
			*f.dst = f.src
		}
	}

	applyb := []struct {
		src *bool
		dst **bool
	}{
		{spec.NestedHVEnabled, &vm.Config.NestedHVEnabled},
		{spec.CpuHotAddEnabled, &vm.Config.CpuHotAddEnabled},
		{spec.CpuHotRemoveEnabled, &vm.Config.CpuHotRemoveEnabled},
		{spec.GuestAutoLockEnabled, &vm.Config.GuestAutoLockEnabled},
		{spec.MemoryHotAddEnabled, &vm.Config.MemoryHotAddEnabled},
		{spec.MemoryReservationLockedToMax, &vm.Config.MemoryReservationLockedToMax},
		{spec.MessageBusTunnelEnabled, &vm.Config.MessageBusTunnelEnabled},
		{spec.NpivTemporaryDisabled, &vm.Config.NpivTemporaryDisabled},
		{spec.NpivOnNonRdmDisks, &vm.Config.NpivOnNonRdmDisks},
		{spec.ChangeTrackingEnabled, &vm.Config.ChangeTrackingEnabled},
	}

	for _, f := range applyb {
		if f.src != nil {
			*f.dst = f.src
		}
	}

	if spec.Flags != nil {
		vm.Config.Flags = *spec.Flags
	}

	if spec.LatencySensitivity != nil {
		vm.Config.LatencySensitivity = spec.LatencySensitivity
	}

	if spec.ManagedBy != nil {
		vm.Config.ManagedBy = spec.ManagedBy
	}

	if spec.BootOptions != nil {
		vm.Config.BootOptions = spec.BootOptions
	}

	if spec.RepConfig != nil {
		vm.Config.RepConfig = spec.RepConfig
	}

	if spec.Tools != nil {
		vm.Config.Tools = spec.Tools
	}

	if spec.ConsolePreferences != nil {
		vm.Config.ConsolePreferences = spec.ConsolePreferences
	}

	if spec.CpuAffinity != nil {
		vm.Config.CpuAffinity = spec.CpuAffinity
	}

	if spec.CpuAllocation != nil {
		vm.Config.CpuAllocation = spec.CpuAllocation
	}

	if spec.MemoryAffinity != nil {
		vm.Config.MemoryAffinity = spec.MemoryAffinity
	}

	if spec.MemoryAllocation != nil {
		vm.Config.MemoryAllocation = spec.MemoryAllocation
	}

	if spec.LatencySensitivity != nil {
		vm.Config.LatencySensitivity = spec.LatencySensitivity
	}

	if spec.MemoryMB != 0 {
		vm.Config.Hardware.MemoryMB = int32(spec.MemoryMB)
		vm.Summary.Config.MemorySizeMB = vm.Config.Hardware.MemoryMB
	}

	if spec.NumCPUs != 0 {
		vm.Config.Hardware.NumCPU = spec.NumCPUs
		vm.Summary.Config.NumCpu = vm.Config.Hardware.NumCPU
	}

	if spec.NumCoresPerSocket != 0 {
		vm.Config.Hardware.NumCoresPerSocket = spec.NumCoresPerSocket
	}

	if spec.GuestId != "" {
		vm.Guest.GuestFamily = guestFamily(spec.GuestId)
	}

	vm.Config.Modified = time.Now()
}

// updateVAppProperty updates the simulator VM with the specified VApp properties.
func (vm *VirtualMachine) updateVAppProperty(spec *types.VmConfigSpec) types.BaseMethodFault {
	ps := make([]types.VAppPropertyInfo, 0)

	if vm.Config.VAppConfig != nil && vm.Config.VAppConfig.GetVmConfigInfo() != nil {
		ps = vm.Config.VAppConfig.GetVmConfigInfo().Property
	}

	for _, prop := range spec.Property {
		var foundIndex int
		exists := false
		// Check if the specified property exists or not. This helps rejecting invalid
		// operations (e.g., Adding a VApp property that already exists)
		for i, p := range ps {
			if p.Key == prop.Info.Key {
				exists = true
				foundIndex = i
				break
			}
		}

		switch prop.Operation {
		case types.ArrayUpdateOperationAdd:
			if exists {
				return new(types.InvalidArgument)
			}
			ps = append(ps, *prop.Info)
		case types.ArrayUpdateOperationEdit:
			if !exists {
				return new(types.InvalidArgument)
			}
			ps[foundIndex] = *prop.Info
		case types.ArrayUpdateOperationRemove:
			if !exists {
				return new(types.InvalidArgument)
			}
			ps = append(ps[:foundIndex], ps[foundIndex+1:]...)
		}
	}

	if vm.Config.VAppConfig == nil {
		vm.Config.VAppConfig = &types.VmConfigInfo{}
	}

	vm.Config.VAppConfig.GetVmConfigInfo().Property = ps

	return nil
}

var extraConfigAlias = map[string]string{
	"ip0": "SET.guest.ipAddress",
}

func extraConfigKey(key string) string {
	if k, ok := extraConfigAlias[key]; ok {
		return k
	}
	return key
}

func (vm *VirtualMachine) applyExtraConfig(spec *types.VirtualMachineConfigSpec) {
	var changes []types.PropertyChange
	for _, c := range spec.ExtraConfig {
		val := c.GetOptionValue()
		key := strings.TrimPrefix(extraConfigKey(val.Key), "SET.")
		if key == val.Key {
			vm.Config.ExtraConfig = append(vm.Config.ExtraConfig, c)
			continue
		}
		changes = append(changes, types.PropertyChange{Name: key, Val: val.Value})

		switch key {
		case "guest.ipAddress":
			if len(vm.Guest.Net) > 0 {
				ip := val.Value.(string)
				vm.Guest.Net[0].IpAddress = []string{ip}
				changes = append(changes,
					types.PropertyChange{Name: "summary." + key, Val: ip},
					types.PropertyChange{Name: "guest.net", Val: vm.Guest.Net},
				)
			}
		case "guest.hostName":
			changes = append(changes,
				types.PropertyChange{Name: "summary." + key, Val: val.Value},
			)
		}
	}
	if len(changes) != 0 {
		Map.Update(vm, changes)
	}
}

func validateGuestID(id string) types.BaseMethodFault {
	for _, x := range GuestID {
		if id == string(x) {
			return nil
		}
	}

	return &types.InvalidArgument{InvalidProperty: "configSpec.guestId"}
}

func (vm *VirtualMachine) configure(ctx *Context, spec *types.VirtualMachineConfigSpec) types.BaseMethodFault {
	vm.apply(spec)

	if spec.MemoryAllocation != nil {
		if err := updateResourceAllocation("memory", spec.MemoryAllocation, vm.Config.MemoryAllocation); err != nil {
			return err
		}
	}

	if spec.CpuAllocation != nil {
		if err := updateResourceAllocation("cpu", spec.CpuAllocation, vm.Config.CpuAllocation); err != nil {
			return err
		}
	}

	if spec.GuestId != "" {
		if err := validateGuestID(spec.GuestId); err != nil {
			return err
		}
	}

	if o := spec.BootOptions; o != nil {
		if isTrue(o.EfiSecureBootEnabled) && vm.Config.Firmware != string(types.GuestOsDescriptorFirmwareTypeEfi) {
			return &types.InvalidVmConfig{Property: "msg.hostd.configSpec.efi"}
		}
	}

	if spec.VAppConfig != nil {
		if err := vm.updateVAppProperty(spec.VAppConfig.GetVmConfigSpec()); err != nil {
			return err
		}
	}

	return vm.configureDevices(ctx, spec)
}

func getVMFileType(fileName string) types.VirtualMachineFileLayoutExFileType {
	var fileType types.VirtualMachineFileLayoutExFileType

	fileExt := path.Ext(fileName)
	fileNameNoExt := strings.TrimSuffix(fileName, fileExt)

	switch fileExt {
	case ".vmx":
		fileType = types.VirtualMachineFileLayoutExFileTypeConfig
	case ".core":
		fileType = types.VirtualMachineFileLayoutExFileTypeCore
	case ".vmdk":
		fileType = types.VirtualMachineFileLayoutExFileTypeDiskDescriptor
		if strings.HasSuffix(fileNameNoExt, "-digest") {
			fileType = types.VirtualMachineFileLayoutExFileTypeDigestDescriptor
		}

		extentSuffixes := []string{"-flat", "-delta", "-s", "-rdm", "-rdmp"}
		for _, suffix := range extentSuffixes {
			if strings.HasSuffix(fileNameNoExt, suffix) {
				fileType = types.VirtualMachineFileLayoutExFileTypeDiskExtent
			} else if strings.HasSuffix(fileNameNoExt, "-digest"+suffix) {
				fileType = types.VirtualMachineFileLayoutExFileTypeDigestExtent
			}
		}
	case ".psf":
		fileType = types.VirtualMachineFileLayoutExFileTypeDiskReplicationState
	case ".vmxf":
		fileType = types.VirtualMachineFileLayoutExFileTypeExtendedConfig
	case ".vmft":
		fileType = types.VirtualMachineFileLayoutExFileTypeFtMetadata
	case ".log":
		fileType = types.VirtualMachineFileLayoutExFileTypeLog
	case ".nvram":
		fileType = types.VirtualMachineFileLayoutExFileTypeNvram
	case ".png", ".bmp":
		fileType = types.VirtualMachineFileLayoutExFileTypeScreenshot
	case ".vmsn":
		fileType = types.VirtualMachineFileLayoutExFileTypeSnapshotData
	case ".vmsd":
		fileType = types.VirtualMachineFileLayoutExFileTypeSnapshotList
	case ".xml":
		if strings.HasSuffix(fileNameNoExt, "-aux") {
			fileType = types.VirtualMachineFileLayoutExFileTypeSnapshotManifestList
		}
	case ".stat":
		fileType = types.VirtualMachineFileLayoutExFileTypeStat
	case ".vmss":
		fileType = types.VirtualMachineFileLayoutExFileTypeSuspend
	case ".vmem":
		if strings.Contains(fileNameNoExt, "Snapshot") {
			fileType = types.VirtualMachineFileLayoutExFileTypeSnapshotMemory
		} else {
			fileType = types.VirtualMachineFileLayoutExFileTypeSuspendMemory
		}
	case ".vswp":
		if strings.HasPrefix(fileNameNoExt, "vmx-") {
			fileType = types.VirtualMachineFileLayoutExFileTypeUwswap
		} else {
			fileType = types.VirtualMachineFileLayoutExFileTypeSwap
		}
	case "":
		if strings.HasPrefix(fileNameNoExt, "imcf-") {
			fileType = types.VirtualMachineFileLayoutExFileTypeGuestCustomization
		}
	}

	return fileType
}

func (vm *VirtualMachine) addFileLayoutEx(datastorePath object.DatastorePath, fileSize int64) int32 {
	var newKey int32
	for _, layoutFile := range vm.LayoutEx.File {
		if layoutFile.Name == datastorePath.String() {
			return layoutFile.Key
		}

		if layoutFile.Key >= newKey {
			newKey = layoutFile.Key + 1
		}
	}

	fileType := getVMFileType(filepath.Base(datastorePath.Path))

	switch fileType {
	case types.VirtualMachineFileLayoutExFileTypeNvram, types.VirtualMachineFileLayoutExFileTypeSnapshotList:
		vm.addConfigLayout(datastorePath.Path)
	case types.VirtualMachineFileLayoutExFileTypeLog:
		vm.addLogLayout(datastorePath.Path)
	case types.VirtualMachineFileLayoutExFileTypeSwap:
		vm.addSwapLayout(datastorePath.String())
	}

	vm.LayoutEx.File = append(vm.LayoutEx.File, types.VirtualMachineFileLayoutExFileInfo{
		Accessible:      types.NewBool(true),
		BackingObjectId: "",
		Key:             newKey,
		Name:            datastorePath.String(),
		Size:            fileSize,
		Type:            string(fileType),
		UniqueSize:      fileSize,
	})

	vm.LayoutEx.Timestamp = time.Now()

	vm.updateStorage()

	return newKey
}

func (vm *VirtualMachine) addConfigLayout(name string) {
	for _, config := range vm.Layout.ConfigFile {
		if config == name {
			return
		}
	}

	vm.Layout.ConfigFile = append(vm.Layout.ConfigFile, name)

	vm.updateStorage()
}

func (vm *VirtualMachine) addLogLayout(name string) {
	for _, log := range vm.Layout.LogFile {
		if log == name {
			return
		}
	}

	vm.Layout.LogFile = append(vm.Layout.LogFile, name)

	vm.updateStorage()
}

func (vm *VirtualMachine) addSwapLayout(name string) {
	vm.Layout.SwapFile = name

	vm.updateStorage()
}

func (vm *VirtualMachine) addSnapshotLayout(snapshot types.ManagedObjectReference, dataKey int32) {
	for _, snapshotLayout := range vm.Layout.Snapshot {
		if snapshotLayout.Key == snapshot {
			return
		}
	}

	var snapshotFiles []string
	for _, file := range vm.LayoutEx.File {
		if file.Key == dataKey || file.Type == "diskDescriptor" {
			snapshotFiles = append(snapshotFiles, file.Name)
		}
	}

	vm.Layout.Snapshot = append(vm.Layout.Snapshot, types.VirtualMachineFileLayoutSnapshotLayout{
		Key:          snapshot,
		SnapshotFile: snapshotFiles,
	})

	vm.updateStorage()
}

func (vm *VirtualMachine) addSnapshotLayoutEx(snapshot types.ManagedObjectReference, dataKey int32, memoryKey int32) {
	for _, snapshotLayoutEx := range vm.LayoutEx.Snapshot {
		if snapshotLayoutEx.Key == snapshot {
			return
		}
	}

	vm.LayoutEx.Snapshot = append(vm.LayoutEx.Snapshot, types.VirtualMachineFileLayoutExSnapshotLayout{
		DataKey:   dataKey,
		Disk:      vm.LayoutEx.Disk,
		Key:       snapshot,
		MemoryKey: memoryKey,
	})

	vm.LayoutEx.Timestamp = time.Now()

	vm.updateStorage()
}

// Updates both vm.Layout.Disk and vm.LayoutEx.Disk
func (vm *VirtualMachine) updateDiskLayouts() types.BaseMethodFault {
	var disksLayout []types.VirtualMachineFileLayoutDiskLayout
	var disksLayoutEx []types.VirtualMachineFileLayoutExDiskLayout

	disks := object.VirtualDeviceList(vm.Config.Hardware.Device).SelectByType((*types.VirtualDisk)(nil))
	for _, disk := range disks {
		disk := disk.(*types.VirtualDisk)
		diskBacking := disk.Backing.(*types.VirtualDiskFlatVer2BackingInfo)

		diskLayout := &types.VirtualMachineFileLayoutDiskLayout{Key: disk.Key}
		diskLayoutEx := &types.VirtualMachineFileLayoutExDiskLayout{Key: disk.Key}

		// Iterate through disk and its parents
		for {
			dFileName := diskBacking.GetVirtualDeviceFileBackingInfo().FileName

			var fileKeys []int32

			// Add disk descriptor and extent files
			for _, diskName := range vdmNames(dFileName) {
				// get full path including datastore location
				p, fault := parseDatastorePath(diskName)
				if fault != nil {
					return fault
				}

				datastore := vm.useDatastore(p.Datastore)
				dFilePath := path.Join(datastore.Info.GetDatastoreInfo().Url, p.Path)

				var fileSize int64
				// If file can not be opened - fileSize will be 0
				if dFileInfo, err := os.Stat(dFilePath); err == nil {
					fileSize = dFileInfo.Size()
				}

				diskKey := vm.addFileLayoutEx(*p, fileSize)
				fileKeys = append(fileKeys, diskKey)
			}

			diskLayout.DiskFile = append(diskLayout.DiskFile, dFileName)
			diskLayoutEx.Chain = append(diskLayoutEx.Chain, types.VirtualMachineFileLayoutExDiskUnit{
				FileKey: fileKeys,
			})

			if parent := diskBacking.Parent; parent != nil {
				diskBacking = parent
			} else {
				break
			}
		}

		disksLayout = append(disksLayout, *diskLayout)
		disksLayoutEx = append(disksLayoutEx, *diskLayoutEx)
	}

	vm.Layout.Disk = disksLayout

	vm.LayoutEx.Disk = disksLayoutEx
	vm.LayoutEx.Timestamp = time.Now()

	vm.updateStorage()

	return nil
}

func (vm *VirtualMachine) updateStorage() types.BaseMethodFault {
	// Committed - sum of Size for each file in vm.LayoutEx.File
	// Unshared  - sum of Size for each disk (.vmdk) in vm.LayoutEx.File
	// Uncommitted - disk capacity minus disk usage (only currently used disk)
	var datastoresUsage []types.VirtualMachineUsageOnDatastore

	disks := object.VirtualDeviceList(vm.Config.Hardware.Device).SelectByType((*types.VirtualDisk)(nil))

	for _, file := range vm.LayoutEx.File {
		p, fault := parseDatastorePath(file.Name)
		if fault != nil {
			return fault
		}

		datastore := vm.useDatastore(p.Datastore)
		dsUsage := &types.VirtualMachineUsageOnDatastore{
			Datastore: datastore.Self,
		}

		for idx, usage := range datastoresUsage {
			if usage.Datastore == datastore.Self {
				datastoresUsage = append(datastoresUsage[:idx], datastoresUsage[idx+1:]...)
				dsUsage = &usage
				break
			}
		}

		dsUsage.Committed = file.Size

		if path.Ext(file.Name) == ".vmdk" {
			dsUsage.Unshared = file.Size
		}

		for _, disk := range disks {
			disk := disk.(*types.VirtualDisk)
			backing := disk.Backing.(types.BaseVirtualDeviceFileBackingInfo).GetVirtualDeviceFileBackingInfo()

			if backing.FileName == file.Name {
				dsUsage.Uncommitted = disk.CapacityInBytes
			}
		}

		datastoresUsage = append(datastoresUsage, *dsUsage)
	}

	vm.Storage.PerDatastoreUsage = datastoresUsage
	vm.Storage.Timestamp = time.Now()

	storageSummary := &types.VirtualMachineStorageSummary{
		Timestamp: time.Now(),
	}

	for _, usage := range datastoresUsage {
		storageSummary.Committed += usage.Committed
		storageSummary.Uncommitted += usage.Uncommitted
		storageSummary.Unshared += usage.Unshared
	}

	vm.Summary.Storage = storageSummary

	return nil
}

func (vm *VirtualMachine) RefreshStorageInfo(ctx *Context, req *types.RefreshStorageInfo) soap.HasFault {
	body := new(methods.RefreshStorageInfoBody)

	if vm.Runtime.Host == nil {
		// VM not fully created
		return body
	}

	// Validate that all files in vm.LayoutEx.File can still be found
	for idx := len(vm.LayoutEx.File) - 1; idx >= 0; idx-- {
		file := vm.LayoutEx.File[idx]

		p, fault := parseDatastorePath(file.Name)
		if fault != nil {
			body.Fault_ = Fault("", fault)
			return body
		}

		if _, err := os.Stat(p.String()); err != nil {
			vm.LayoutEx.File = append(vm.LayoutEx.File[:idx], vm.LayoutEx.File[idx+1:]...)
		}
	}

	// Directories will be used to locate VM files.
	// Does not include information about virtual disk file locations.
	locations := []string{
		vm.Config.Files.VmPathName,
		vm.Config.Files.SnapshotDirectory,
		vm.Config.Files.LogDirectory,
		vm.Config.Files.SuspendDirectory,
		vm.Config.Files.FtMetadataDirectory,
	}

	for _, directory := range locations {
		if directory == "" {
			continue
		}

		p, fault := parseDatastorePath(directory)
		if fault != nil {
			body.Fault_ = Fault("", fault)
			return body
		}

		datastore := vm.useDatastore(p.Datastore)
		directory := path.Join(datastore.Info.GetDatastoreInfo().Url, p.Path)

		if path.Ext(p.Path) == ".vmx" {
			directory = path.Dir(directory) // vm.Config.Files.VmPathName can be a directory or full path to .vmx
		}

		if _, err := os.Stat(directory); err != nil {
			// Can not access the directory
			continue
		}

		files, err := ioutil.ReadDir(directory)
		if err != nil {
			body.Fault_ = Fault("", ctx.Map.FileManager().fault(directory, err, new(types.CannotAccessFile)))
			return body
		}

		for _, file := range files {
			datastorePath := object.DatastorePath{
				Datastore: p.Datastore,
				Path:      strings.TrimPrefix(file.Name(), datastore.Info.GetDatastoreInfo().Url),
			}

			vm.addFileLayoutEx(datastorePath, file.Size())
		}
	}

	fault := vm.updateDiskLayouts()
	if fault != nil {
		body.Fault_ = Fault("", fault)
		return body
	}

	vm.LayoutEx.Timestamp = time.Now()

	body.Res = new(types.RefreshStorageInfoResponse)

	return body
}

func (vm *VirtualMachine) findDatastore(name string) *Datastore {
	host := Map.Get(*vm.Runtime.Host).(*HostSystem)

	return Map.FindByName(name, host.Datastore).(*Datastore)
}

func (vm *VirtualMachine) useDatastore(name string) *Datastore {
	ds := vm.findDatastore(name)
	if FindReference(vm.Datastore, ds.Self) == nil {
		vm.Datastore = append(vm.Datastore, ds.Self)
	}

	return ds
}

func (vm *VirtualMachine) vmx(spec *types.VirtualMachineConfigSpec) object.DatastorePath {
	var p object.DatastorePath
	vmx := vm.Config.Files.VmPathName
	if spec != nil {
		vmx = spec.Files.VmPathName
	}
	p.FromString(vmx)
	return p
}

func (vm *VirtualMachine) createFile(spec string, name string, register bool) (*os.File, types.BaseMethodFault) {
	p, fault := parseDatastorePath(spec)
	if fault != nil {
		return nil, fault
	}

	ds := vm.useDatastore(p.Datastore)

	nhost := len(ds.Host)
	if ds.Name == "vsanDatastore" && nhost < 3 {
		fault := new(types.CannotCreateFile)
		fault.FaultMessage = []types.LocalizableMessage{
			{
				Key:     "vob.vsanprovider.object.creation.failed",
				Message: "Failed to create object.",
			},
			{
				Key:     "vob.vsan.clomd.needMoreFaultDomains2",
				Message: fmt.Sprintf("There are currently %d usable fault domains. The operation requires %d more usable fault domains.", nhost, 3-nhost),
			},
		}
		fault.File = p.Path
		return nil, fault
	}

	file := path.Join(ds.Info.GetDatastoreInfo().Url, p.Path)

	if name != "" {
		if path.Ext(p.Path) == ".vmx" {
			file = path.Dir(file) // vm.Config.Files.VmPathName can be a directory or full path to .vmx
		}

		file = path.Join(file, name)
	}

	if register {
		f, err := os.Open(filepath.Clean(file))
		if err != nil {
			log.Printf("register %s: %s", vm.Reference(), err)
			if os.IsNotExist(err) {
				return nil, &types.NotFound{}
			}

			return nil, &types.InvalidArgument{}
		}

		return f, nil
	}

	_, err := os.Stat(file)
	if err == nil {
		fault := &types.FileAlreadyExists{FileFault: types.FileFault{File: file}}
		log.Printf("%T: %s", fault, file)
		return nil, fault
	}

	// Create parent directory if needed
	dir := path.Dir(file)
	_, err = os.Stat(dir)
	if err != nil {
		if os.IsNotExist(err) {
			_ = os.Mkdir(dir, 0700)
		}
	}

	f, err := os.Create(file)
	if err != nil {
		log.Printf("create(%s): %s", file, err)
		return nil, &types.FileFault{
			File: file,
		}
	}

	return f, nil
}

// Rather than keep an fd open for each VM, open/close the log for each messages.
// This is ok for now as we do not do any heavy VM logging.
func (vm *VirtualMachine) logPrintf(format string, v ...interface{}) {
	f, err := os.OpenFile(vm.log, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0)
	if err != nil {
		log.Println(err)
		return
	}
	log.New(f, "vmx ", log.Flags()).Printf(format, v...)
	_ = f.Close()
}

func (vm *VirtualMachine) create(ctx *Context, spec *types.VirtualMachineConfigSpec, register bool) types.BaseMethodFault {
	vm.apply(spec)

	if spec.Version != "" {
		v := strings.TrimPrefix(spec.Version, "vmx-")
		_, err := strconv.Atoi(v)
		if err != nil {
			log.Printf("unsupported hardware version: %s", spec.Version)
			return new(types.NotSupported)
		}
	}

	files := []struct {
		spec string
		name string
		use  *string
	}{
		{vm.Config.Files.VmPathName, "", nil},
		{vm.Config.Files.VmPathName, fmt.Sprintf("%s.nvram", vm.Name), nil},
		{vm.Config.Files.LogDirectory, "vmware.log", &vm.log},
	}

	for _, file := range files {
		f, err := vm.createFile(file.spec, file.name, register)
		if err != nil {
			return err
		}
		if file.use != nil {
			*file.use = f.Name()
		}
		_ = f.Close()
	}

	vm.logPrintf("created")

	return vm.configureDevices(ctx, spec)
}

var vmwOUI = net.HardwareAddr([]byte{0x0, 0xc, 0x29})

// From http://pubs.vmware.com/vsphere-60/index.jsp?topic=%2Fcom.vmware.vsphere.networking.doc%2FGUID-DC7478FF-DC44-4625-9AD7-38208C56A552.html
// "The host generates generateMAC addresses that consists of the VMware OUI 00:0C:29 and the last three octets in hexadecimal
//  format of the virtual machine UUID.  The virtual machine UUID is based on a hash calculated by using the UUID of the
//  ESXi physical machine and the path to the configuration file (.vmx) of the virtual machine."
func (vm *VirtualMachine) generateMAC(unit int32) string {
	id := []byte(vm.Config.Uuid)

	offset := len(id) - len(vmwOUI)
	key := id[offset] + byte(unit) // add device unit number, giving each VM NIC a unique MAC
	id = append([]byte{key}, id[offset+1:]...)

	mac := append(vmwOUI, id...)

	return mac.String()
}

func numberToString(n int64, sep rune) string {
	buf := &bytes.Buffer{}
	if n < 0 {
		n = -n
		buf.WriteRune('-')
	}
	s := strconv.FormatInt(n, 10)
	pos := 3 - (len(s) % 3)
	for i := 0; i < len(s); i++ {
		if pos == 3 {
			if i != 0 {
				buf.WriteRune(sep)
			}
			pos = 0
		}
		pos++
		buf.WriteByte(s[i])
	}

	return buf.String()
}

func getDiskSize(disk *types.VirtualDisk) int64 {
	if disk.CapacityInBytes == 0 {
		return disk.CapacityInKB * 1024
	}
	return disk.CapacityInBytes
}

func changedDiskSize(oldDisk *types.VirtualDisk, newDiskSpec *types.VirtualDisk) (int64, bool) {
	// capacity cannot be decreased
	if newDiskSpec.CapacityInBytes < oldDisk.CapacityInBytes || newDiskSpec.CapacityInKB < oldDisk.CapacityInKB {
		return 0, false
	}

	// NOTE: capacity is ignored if specified value is same as before
	if newDiskSpec.CapacityInBytes == oldDisk.CapacityInBytes {
		return newDiskSpec.CapacityInKB * 1024, true
	}
	if newDiskSpec.CapacityInKB == oldDisk.CapacityInKB {
		return newDiskSpec.CapacityInBytes, true
	}

	// CapacityInBytes and CapacityInKB indicate different values
	if newDiskSpec.CapacityInBytes != newDiskSpec.CapacityInKB*1024 {
		return 0, false
	}
	return newDiskSpec.CapacityInBytes, true
}

func (vm *VirtualMachine) validateSwitchMembers(id string) types.BaseMethodFault {
	var dswitch *DistributedVirtualSwitch

	var find func(types.ManagedObjectReference)
	find = func(child types.ManagedObjectReference) {
		s, ok := Map.Get(child).(*DistributedVirtualSwitch)
		if ok && s.Uuid == id {
			dswitch = s
			return
		}
		walk(Map.Get(child), find)
	}
	f := Map.getEntityDatacenter(vm).NetworkFolder
	walk(Map.Get(f), find) // search in NetworkFolder and any sub folders

	if dswitch == nil {
		log.Printf("DVS %s cannot be found", id)
		return new(types.NotFound)
	}

	h := Map.Get(*vm.Runtime.Host).(*HostSystem)
	c := hostParent(&h.HostSystem)
	isMember := func(val types.ManagedObjectReference) bool {
		for _, mem := range dswitch.Summary.HostMember {
			if mem == val {
				return true
			}
		}
		log.Printf("%s is not a member of VDS %s", h.Name, dswitch.Name)
		return false
	}

	for _, ref := range c.Host {
		if !isMember(ref) {
			return &types.InvalidArgument{InvalidProperty: "spec.deviceChange.device.port.switchUuid"}
		}
	}

	return nil
}

func (vm *VirtualMachine) configureDevice(
	ctx *Context,
	devices object.VirtualDeviceList,
	spec *types.VirtualDeviceConfigSpec,
	oldDevice types.BaseVirtualDevice) types.BaseMethodFault {

	device := spec.Device
	d := device.GetVirtualDevice()
	var controller types.BaseVirtualController

	if d.Key <= 0 {
		// Keys can't be negative; Key 0 is reserved
		d.Key = devices.NewKey()
		d.Key *= -1
	}

	// Choose a unique key
	for {
		if devices.FindByKey(d.Key) == nil {
			break
		}
		d.Key++
	}

	label := devices.Name(device)
	summary := label
	dc := ctx.Map.getEntityDatacenter(ctx.Map.Get(*vm.Parent).(mo.Entity))

	switch x := device.(type) {
	case types.BaseVirtualEthernetCard:
		controller = devices.PickController((*types.VirtualPCIController)(nil))
		var net types.ManagedObjectReference
		var name string

		switch b := d.Backing.(type) {
		case *types.VirtualEthernetCardNetworkBackingInfo:
			name = b.DeviceName
			summary = name
			net = ctx.Map.FindByName(b.DeviceName, dc.Network).Reference()
			b.Network = &net
		case *types.VirtualEthernetCardDistributedVirtualPortBackingInfo:
			summary = fmt.Sprintf("DVSwitch: %s", b.Port.SwitchUuid)
			net.Type = "DistributedVirtualPortgroup"
			net.Value = b.Port.PortgroupKey
			if err := vm.validateSwitchMembers(b.Port.SwitchUuid); err != nil {
				return err
			}
		}

		ctx.Map.Update(vm, []types.PropertyChange{
			{Name: "summary.config.numEthernetCards", Val: vm.Summary.Config.NumEthernetCards + 1},
			{Name: "network", Val: append(vm.Network, net)},
		})

		c := x.GetVirtualEthernetCard()
		if c.MacAddress == "" {
			if c.UnitNumber == nil {
				devices.AssignController(device, controller)
			}
			c.MacAddress = vm.generateMAC(*c.UnitNumber - 7) // Note 7 == PCI offset
		}

		vm.Guest.Net = append(vm.Guest.Net, types.GuestNicInfo{
			Network:        name,
			IpAddress:      nil,
			MacAddress:     c.MacAddress,
			Connected:      true,
			DeviceConfigId: c.Key,
		})

		if spec.Operation == types.VirtualDeviceConfigSpecOperationAdd {
			if c.ResourceAllocation == nil {
				c.ResourceAllocation = &types.VirtualEthernetCardResourceAllocation{
					Reservation: types.NewInt64(0),
					Share: types.SharesInfo{
						Shares: 50,
						Level:  "normal",
					},
					Limit: types.NewInt64(-1),
				}
			}
		}
	case *types.VirtualDisk:
		if oldDevice == nil {
			// NOTE: either of capacityInBytes and capacityInKB may not be specified
			x.CapacityInBytes = getDiskSize(x)
			x.CapacityInKB = getDiskSize(x) / 1024
		} else {
			if oldDisk, ok := oldDevice.(*types.VirtualDisk); ok {
				diskSize, ok := changedDiskSize(oldDisk, x)
				if !ok {
					return &types.InvalidDeviceOperation{}
				}
				x.CapacityInBytes = diskSize
				x.CapacityInKB = diskSize / 1024
			}
		}

		summary = fmt.Sprintf("%s KB", numberToString(x.CapacityInKB, ','))
		switch b := d.Backing.(type) {
		case types.BaseVirtualDeviceFileBackingInfo:
			info := b.GetVirtualDeviceFileBackingInfo()
			var path object.DatastorePath
			path.FromString(info.FileName)

			if path.Path == "" {
				filename, err := vm.genVmdkPath(path)
				if err != nil {
					return err
				}

				info.FileName = filename
			}

			err := vdmCreateVirtualDisk(spec.FileOperation, &types.CreateVirtualDisk_Task{
				Datacenter: &dc.Self,
				Name:       info.FileName,
			})
			if err != nil {
				return err
			}

			ctx.Map.Update(vm, []types.PropertyChange{
				{Name: "summary.config.numVirtualDisks", Val: vm.Summary.Config.NumVirtualDisks + 1},
			})

			p, _ := parseDatastorePath(info.FileName)
			ds := vm.findDatastore(p.Datastore)
			info.Datastore = &ds.Self

			if oldDevice != nil {
				if oldDisk, ok := oldDevice.(*types.VirtualDisk); ok {
					// add previous capacity to datastore freespace
					ctx.WithLock(ds, func() {
						ds.Summary.FreeSpace += getDiskSize(oldDisk)
						ds.Info.GetDatastoreInfo().FreeSpace = ds.Summary.FreeSpace
					})
				}
			}

			// then subtract new capacity from datastore freespace
			// XXX: compare disk size and free space until windows stat is supported
			ctx.WithLock(ds, func() {
				ds.Summary.FreeSpace -= getDiskSize(x)
				ds.Info.GetDatastoreInfo().FreeSpace = ds.Summary.FreeSpace
			})

			vm.updateDiskLayouts()

			if disk, ok := b.(*types.VirtualDiskFlatVer2BackingInfo); ok {
				// These properties default to false
				props := []**bool{
					&disk.EagerlyScrub,
					&disk.ThinProvisioned,
					&disk.WriteThrough,
					&disk.Split,
					&disk.DigestEnabled,
				}
				for _, prop := range props {
					if *prop == nil {
						*prop = types.NewBool(false)
					}
				}
				disk.Uuid = virtualDiskUUID(&dc.Self, info.FileName)
			}
		}
	case *types.VirtualCdrom:
		if b, ok := d.Backing.(types.BaseVirtualDeviceFileBackingInfo); ok {
			summary = "ISO " + b.GetVirtualDeviceFileBackingInfo().FileName
		}
	case *types.VirtualFloppy:
		if b, ok := d.Backing.(types.BaseVirtualDeviceFileBackingInfo); ok {
			summary = "Image " + b.GetVirtualDeviceFileBackingInfo().FileName
		}
	case *types.VirtualSerialPort:
		switch b := d.Backing.(type) {
		case types.BaseVirtualDeviceFileBackingInfo:
			summary = "File " + b.GetVirtualDeviceFileBackingInfo().FileName
		case *types.VirtualSerialPortURIBackingInfo:
			summary = "Remote " + b.ServiceURI
		}
	}

	if d.UnitNumber == nil && controller != nil {
		devices.AssignController(device, controller)
	}

	if d.DeviceInfo == nil {
		d.DeviceInfo = &types.Description{
			Label:   label,
			Summary: summary,
		}
	} else {
		info := d.DeviceInfo.GetDescription()
		if info.Label == "" {
			info.Label = label
		}
		if info.Summary == "" {
			info.Summary = summary
		}
	}

	switch device.(type) {
	case types.BaseVirtualEthernetCard, *types.VirtualCdrom, *types.VirtualFloppy, *types.VirtualUSB, *types.VirtualSerialPort:
		if d.Connectable == nil {
			d.Connectable = &types.VirtualDeviceConnectInfo{StartConnected: true, Connected: true}
		}
	}

	// device can be connected only if vm is powered on
	if vm.Runtime.PowerState != types.VirtualMachinePowerStatePoweredOn {
		if d.Connectable != nil {
			d.Connectable.Connected = false
		}
	}

	return nil
}

func (vm *VirtualMachine) removeDevice(ctx *Context, devices object.VirtualDeviceList, spec *types.VirtualDeviceConfigSpec) object.VirtualDeviceList {
	key := spec.Device.GetVirtualDevice().Key

	for i, d := range devices {
		if d.GetVirtualDevice().Key != key {
			continue
		}

		devices = append(devices[:i], devices[i+1:]...)

		switch device := spec.Device.(type) {
		case *types.VirtualDisk:
			if spec.FileOperation == types.VirtualDeviceConfigSpecFileOperationDestroy {
				var file string

				switch b := device.Backing.(type) {
				case types.BaseVirtualDeviceFileBackingInfo:
					file = b.GetVirtualDeviceFileBackingInfo().FileName

					p, _ := parseDatastorePath(file)
					ds := vm.findDatastore(p.Datastore)

					ctx.WithLock(ds, func() {
						ds.Summary.FreeSpace += getDiskSize(device)
						ds.Info.GetDatastoreInfo().FreeSpace = ds.Summary.FreeSpace
					})
				}

				if file != "" {
					dc := ctx.Map.getEntityDatacenter(vm)
					dm := ctx.Map.VirtualDiskManager()
					if dc == nil {
						continue // parent was destroyed
					}
					res := dm.DeleteVirtualDiskTask(ctx, &types.DeleteVirtualDisk_Task{
						Name:       file,
						Datacenter: &dc.Self,
					})
					ctask := ctx.Map.Get(res.(*methods.DeleteVirtualDisk_TaskBody).Res.Returnval).(*Task)
					ctask.Wait()
				}
			}
			ctx.Map.Update(vm, []types.PropertyChange{
				{Name: "summary.config.numVirtualDisks", Val: vm.Summary.Config.NumVirtualDisks - 1},
			})

			vm.updateDiskLayouts()
		case types.BaseVirtualEthernetCard:
			var net types.ManagedObjectReference

			switch b := device.GetVirtualEthernetCard().Backing.(type) {
			case *types.VirtualEthernetCardNetworkBackingInfo:
				net = *b.Network
			case *types.VirtualEthernetCardDistributedVirtualPortBackingInfo:
				net.Type = "DistributedVirtualPortgroup"
				net.Value = b.Port.PortgroupKey
			}

			for j, nicInfo := range vm.Guest.Net {
				if nicInfo.DeviceConfigId == key {
					vm.Guest.Net = append(vm.Guest.Net[:j], vm.Guest.Net[j+1:]...)
					break
				}
			}

			networks := vm.Network
			RemoveReference(&networks, net)
			ctx.Map.Update(vm, []types.PropertyChange{
				{Name: "summary.config.numEthernetCards", Val: vm.Summary.Config.NumEthernetCards - 1},
				{Name: "network", Val: networks},
			})
		}

		break
	}

	return devices
}

func (vm *VirtualMachine) genVmdkPath(p object.DatastorePath) (string, types.BaseMethodFault) {
	if p.Datastore == "" {
		p.FromString(vm.Config.Files.VmPathName)
	}
	if p.Path == "" {
		p.Path = vm.Config.Name
	} else {
		p.Path = path.Dir(p.Path)
	}
	vmdir := p.String()
	index := 0
	for {
		var filename string
		if index == 0 {
			filename = fmt.Sprintf("%s.vmdk", vm.Config.Name)
		} else {
			filename = fmt.Sprintf("%s_%d.vmdk", vm.Config.Name, index)
		}

		f, err := vm.createFile(vmdir, filename, false)
		if err != nil {
			switch err.(type) {
			case *types.FileAlreadyExists:
				index++
				continue
			default:
				return "", err
			}
		}

		_ = f.Close()
		_ = os.Remove(f.Name())

		return path.Join(vmdir, filename), nil
	}
}

func (vm *VirtualMachine) configureDevices(ctx *Context, spec *types.VirtualMachineConfigSpec) types.BaseMethodFault {
	devices := object.VirtualDeviceList(vm.Config.Hardware.Device)

	for i, change := range spec.DeviceChange {
		dspec := change.GetVirtualDeviceConfigSpec()
		device := dspec.Device.GetVirtualDevice()
		invalid := &types.InvalidDeviceSpec{DeviceIndex: int32(i)}

		switch dspec.FileOperation {
		case types.VirtualDeviceConfigSpecFileOperationCreate:
			switch dspec.Device.(type) {
			case *types.VirtualDisk:
				if device.UnitNumber == nil {
					return invalid
				}
			}
		}

		switch dspec.Operation {
		case types.VirtualDeviceConfigSpecOperationAdd:
			if devices.FindByKey(device.Key) != nil && device.ControllerKey == 0 {
				// Note: real ESX does not allow adding base controllers (ControllerKey = 0)
				// after VM is created (returns success but device is not added).
				continue
			} else if device.UnitNumber != nil && devices.SelectByType(dspec.Device).Select(func(d types.BaseVirtualDevice) bool {
				base := d.GetVirtualDevice()
				if base.UnitNumber != nil {
					if base.ControllerKey != device.ControllerKey {
						return false
					}
					return *base.UnitNumber == *device.UnitNumber
				}
				return false
			}) != nil {
				// UnitNumber for this device type is taken
				return invalid
			}

			key := device.Key
			err := vm.configureDevice(ctx, devices, dspec, nil)
			if err != nil {
				return err
			}

			devices = append(devices, dspec.Device)
			if key != device.Key {
				// Update ControllerKey refs
				for i := range spec.DeviceChange {
					ckey := &spec.DeviceChange[i].GetVirtualDeviceConfigSpec().Device.GetVirtualDevice().ControllerKey
					if *ckey == key {
						*ckey = device.Key
					}
				}
			}
		case types.VirtualDeviceConfigSpecOperationEdit:
			rspec := *dspec
			oldDevice := devices.FindByKey(device.Key)
			if oldDevice == nil {
				return invalid
			}
			rspec.Device = oldDevice
			devices = vm.removeDevice(ctx, devices, &rspec)
			if device.DeviceInfo != nil {
				device.DeviceInfo.GetDescription().Summary = "" // regenerate summary
			}

			err := vm.configureDevice(ctx, devices, dspec, oldDevice)
			if err != nil {
				return err
			}

			devices = append(devices, dspec.Device)
		case types.VirtualDeviceConfigSpecOperationRemove:
			devices = vm.removeDevice(ctx, devices, dspec)
		}
	}

	ctx.Map.Update(vm, []types.PropertyChange{
		{Name: "config.hardware.device", Val: []types.BaseVirtualDevice(devices)},
	})

	vm.updateDiskLayouts()

	vm.applyExtraConfig(spec) // Do this after device config, as some may apply to the devices themselves (e.g. ethernet -> guest.net)

	return nil
}

type powerVMTask struct {
	*VirtualMachine

	state types.VirtualMachinePowerState
	ctx   *Context
}

func (c *powerVMTask) Run(task *Task) (types.AnyType, types.BaseMethodFault) {
	c.logPrintf("running power task: requesting %s, existing %s",
		c.state, c.VirtualMachine.Runtime.PowerState)

	if c.VirtualMachine.Runtime.PowerState == c.state {
		return nil, &types.InvalidPowerState{
			RequestedState: c.state,
			ExistingState:  c.VirtualMachine.Runtime.PowerState,
		}
	}

	var boot types.AnyType
	if c.state == types.VirtualMachinePowerStatePoweredOn {
		boot = time.Now()
	}

	event := c.event()
	switch c.state {
	case types.VirtualMachinePowerStatePoweredOn:
		if c.VirtualMachine.hostInMM(c.ctx) {
			return nil, new(types.InvalidState)
		}

		c.run.start(c.ctx, c.VirtualMachine)
		c.ctx.postEvent(
			&types.VmStartingEvent{VmEvent: event},
			&types.VmPoweredOnEvent{VmEvent: event},
		)
		c.customize(c.ctx)
	case types.VirtualMachinePowerStatePoweredOff:
		c.run.stop(c.ctx, c.VirtualMachine)
		c.ctx.postEvent(
			&types.VmStoppingEvent{VmEvent: event},
			&types.VmPoweredOffEvent{VmEvent: event},
		)
	case types.VirtualMachinePowerStateSuspended:
		if c.VirtualMachine.Runtime.PowerState != types.VirtualMachinePowerStatePoweredOn {
			return nil, &types.InvalidPowerState{
				RequestedState: types.VirtualMachinePowerStatePoweredOn,
				ExistingState:  c.VirtualMachine.Runtime.PowerState,
			}
		}

		c.run.pause(c.ctx, c.VirtualMachine)
		c.ctx.postEvent(
			&types.VmSuspendingEvent{VmEvent: event},
			&types.VmSuspendedEvent{VmEvent: event},
		)
	}

	// copy devices to prevent data race
	devices := c.VirtualMachine.cloneDevice()
	for _, d := range devices {
		conn := d.GetVirtualDevice().Connectable
		if conn == nil {
			continue
		}

		if c.state == types.VirtualMachinePowerStatePoweredOn {
			// apply startConnected to current connection
			conn.Connected = conn.StartConnected
		} else {
			conn.Connected = false
		}
	}

	c.ctx.Map.Update(c.VirtualMachine, []types.PropertyChange{
		{Name: "runtime.powerState", Val: c.state},
		{Name: "summary.runtime.powerState", Val: c.state},
		{Name: "summary.runtime.bootTime", Val: boot},
		{Name: "config.hardware.device", Val: devices},
	})

	return nil, nil
}

func (vm *VirtualMachine) PowerOnVMTask(ctx *Context, c *types.PowerOnVM_Task) soap.HasFault {
	if vm.Config.Template {
		return &methods.PowerOnVM_TaskBody{
			Fault_: Fault("cannot powerOn a template", &types.InvalidState{}),
		}
	}

	runner := &powerVMTask{vm, types.VirtualMachinePowerStatePoweredOn, ctx}
	task := CreateTask(runner.Reference(), "powerOn", runner.Run)

	return &methods.PowerOnVM_TaskBody{
		Res: &types.PowerOnVM_TaskResponse{
			Returnval: task.Run(ctx),
		},
	}
}

func (vm *VirtualMachine) PowerOffVMTask(ctx *Context, c *types.PowerOffVM_Task) soap.HasFault {
	runner := &powerVMTask{vm, types.VirtualMachinePowerStatePoweredOff, ctx}
	task := CreateTask(runner.Reference(), "powerOff", runner.Run)

	return &methods.PowerOffVM_TaskBody{
		Res: &types.PowerOffVM_TaskResponse{
			Returnval: task.Run(ctx),
		},
	}
}

func (vm *VirtualMachine) SuspendVMTask(ctx *Context, req *types.SuspendVM_Task) soap.HasFault {
	runner := &powerVMTask{vm, types.VirtualMachinePowerStateSuspended, ctx}
	task := CreateTask(runner.Reference(), "suspend", runner.Run)

	return &methods.SuspendVM_TaskBody{
		Res: &types.SuspendVM_TaskResponse{
			Returnval: task.Run(ctx),
		},
	}
}

func (vm *VirtualMachine) ResetVMTask(ctx *Context, req *types.ResetVM_Task) soap.HasFault {
	task := CreateTask(vm, "reset", func(task *Task) (types.AnyType, types.BaseMethodFault) {
		res := vm.PowerOffVMTask(ctx, &types.PowerOffVM_Task{This: vm.Self})
		ctask := ctx.Map.Get(res.(*methods.PowerOffVM_TaskBody).Res.Returnval).(*Task)
		ctask.Wait()
		if ctask.Info.Error != nil {
			return nil, ctask.Info.Error.Fault
		}

		res = vm.PowerOnVMTask(ctx, &types.PowerOnVM_Task{This: vm.Self})
		ctask = ctx.Map.Get(res.(*methods.PowerOnVM_TaskBody).Res.Returnval).(*Task)
		ctask.Wait()

		return nil, nil
	})

	return &methods.ResetVM_TaskBody{
		Res: &types.ResetVM_TaskResponse{
			Returnval: task.Run(ctx),
		},
	}
}

func (vm *VirtualMachine) RebootGuest(ctx *Context, req *types.RebootGuest) soap.HasFault {
	body := new(methods.RebootGuestBody)

	if vm.Runtime.PowerState != types.VirtualMachinePowerStatePoweredOn {
		body.Fault_ = Fault("", &types.InvalidPowerState{
			RequestedState: types.VirtualMachinePowerStatePoweredOn,
			ExistingState:  vm.Runtime.PowerState,
		})
		return body
	}

	if vm.Guest.ToolsRunningStatus == string(types.VirtualMachineToolsRunningStatusGuestToolsRunning) {
		vm.run.restart(ctx, vm)
		body.Res = new(types.RebootGuestResponse)
	} else {
		body.Fault_ = Fault("", new(types.ToolsUnavailable))
	}

	return body
}

func (vm *VirtualMachine) ReconfigVMTask(ctx *Context, req *types.ReconfigVM_Task) soap.HasFault {
	task := CreateTask(vm, "reconfigVm", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		ctx.postEvent(&types.VmReconfiguredEvent{
			VmEvent:    vm.event(),
			ConfigSpec: req.Spec,
		})

		if vm.Config.Template {
			expect := types.VirtualMachineConfigSpec{
				Name:       req.Spec.Name,
				Annotation: req.Spec.Annotation,
			}
			if !reflect.DeepEqual(&req.Spec, &expect) {
				log.Printf("template reconfigure only allows name and annotation change")
				return nil, new(types.NotSupported)
			}
		}

		err := vm.configure(ctx, &req.Spec)

		return nil, err
	})

	return &methods.ReconfigVM_TaskBody{
		Res: &types.ReconfigVM_TaskResponse{
			Returnval: task.Run(ctx),
		},
	}
}

func (vm *VirtualMachine) UpgradeVMTask(ctx *Context, req *types.UpgradeVM_Task) soap.HasFault {
	body := &methods.UpgradeVM_TaskBody{}

	task := CreateTask(vm, "upgradeVm", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		if vm.Config.Version != esx.HardwareVersion {
			ctx.Map.Update(vm, []types.PropertyChange{{
				Name: "config.version", Val: esx.HardwareVersion,
			}})
		}
		return nil, nil
	})

	body.Res = &types.UpgradeVM_TaskResponse{
		Returnval: task.Run(ctx),
	}

	return body
}

func (vm *VirtualMachine) DestroyTask(ctx *Context, req *types.Destroy_Task) soap.HasFault {
	dc := ctx.Map.getEntityDatacenter(vm)

	task := CreateTask(vm, "destroy", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		if dc == nil {
			return nil, &types.ManagedObjectNotFound{Obj: vm.Self} // If our Parent was destroyed, so were we.
		}

		r := vm.UnregisterVM(ctx, &types.UnregisterVM{
			This: req.This,
		})

		if r.Fault() != nil {
			return nil, r.Fault().VimFault().(types.BaseMethodFault)
		}

		// Remove all devices
		devices := object.VirtualDeviceList(vm.Config.Hardware.Device)
		spec, _ := devices.ConfigSpec(types.VirtualDeviceConfigSpecOperationRemove)
		vm.configureDevices(ctx, &types.VirtualMachineConfigSpec{DeviceChange: spec})

		// Delete VM files from the datastore (ignoring result for now)
		m := ctx.Map.FileManager()

		_ = m.DeleteDatastoreFileTask(ctx, &types.DeleteDatastoreFile_Task{
			This:       m.Reference(),
			Name:       vm.Config.Files.LogDirectory,
			Datacenter: &dc.Self,
		})

		vm.run.remove(vm)

		return nil, nil
	})

	return &methods.Destroy_TaskBody{
		Res: &types.Destroy_TaskResponse{
			Returnval: task.Run(ctx),
		},
	}
}

func (vm *VirtualMachine) SetCustomValue(ctx *Context, req *types.SetCustomValue) soap.HasFault {
	return SetCustomValue(ctx, req)
}

func (vm *VirtualMachine) UnregisterVM(ctx *Context, c *types.UnregisterVM) soap.HasFault {
	r := &methods.UnregisterVMBody{}

	if vm.Runtime.PowerState == types.VirtualMachinePowerStatePoweredOn {
		r.Fault_ = Fault("", &types.InvalidPowerState{
			RequestedState: types.VirtualMachinePowerStatePoweredOff,
			ExistingState:  vm.Runtime.PowerState,
		})

		return r
	}

	host := ctx.Map.Get(*vm.Runtime.Host).(*HostSystem)
	ctx.Map.RemoveReference(ctx, host, &host.Vm, vm.Self)

	if vm.ResourcePool != nil {
		switch pool := ctx.Map.Get(*vm.ResourcePool).(type) {
		case *ResourcePool:
			ctx.Map.RemoveReference(ctx, pool, &pool.Vm, vm.Self)
		case *VirtualApp:
			ctx.Map.RemoveReference(ctx, pool, &pool.Vm, vm.Self)
		}
	}

	for i := range vm.Datastore {
		ds := ctx.Map.Get(vm.Datastore[i]).(*Datastore)
		ctx.Map.RemoveReference(ctx, ds, &ds.Vm, vm.Self)
	}

	ctx.postEvent(&types.VmRemovedEvent{VmEvent: vm.event()})
	if f, ok := asFolderMO(ctx.Map.getEntityParent(vm, "Folder")); ok {
		folderRemoveChild(ctx, f, c.This)
	}

	r.Res = new(types.UnregisterVMResponse)

	return r
}

type vmFolder interface {
	CreateVMTask(ctx *Context, c *types.CreateVM_Task) soap.HasFault
}

func (vm *VirtualMachine) cloneDevice() []types.BaseVirtualDevice {
	src := types.ArrayOfVirtualDevice{
		VirtualDevice: vm.Config.Hardware.Device,
	}
	dst := types.ArrayOfVirtualDevice{}
	deepCopy(src, &dst)
	return dst.VirtualDevice
}

func (vm *VirtualMachine) CloneVMTask(ctx *Context, req *types.CloneVM_Task) soap.HasFault {
	pool := req.Spec.Location.Pool
	if pool == nil {
		if !vm.Config.Template {
			pool = vm.ResourcePool
		}
	}

	destHost := vm.Runtime.Host

	if req.Spec.Location.Host != nil {
		destHost = req.Spec.Location.Host
	}

	folder, _ := asFolderMO(ctx.Map.Get(req.Folder))
	host := ctx.Map.Get(*destHost).(*HostSystem)
	event := vm.event()

	ctx.postEvent(&types.VmBeingClonedEvent{
		VmCloneEvent: types.VmCloneEvent{
			VmEvent: event,
		},
		DestFolder: folderEventArgument(folder),
		DestName:   req.Name,
		DestHost:   *host.eventArgument(),
	})

	vmx := vm.vmx(nil)
	vmx.Path = req.Name
	if ref := req.Spec.Location.Datastore; ref != nil {
		ds := ctx.Map.Get(*ref).(*Datastore).Name
		vmx.Datastore = ds
	}

	task := CreateTask(vm, "cloneVm", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		if pool == nil {
			return nil, &types.InvalidArgument{InvalidProperty: "spec.location.pool"}
		}
		if obj := ctx.Map.FindByName(req.Name, folder.ChildEntity); obj != nil {
			return nil, &types.DuplicateName{
				Name:   req.Name,
				Object: obj.Reference(),
			}
		}
		config := types.VirtualMachineConfigSpec{
			Name:    req.Name,
			GuestId: vm.Config.GuestId,
			Files: &types.VirtualMachineFileInfo{
				VmPathName: vmx.String(),
			},
		}
		if req.Spec.Config != nil {
			config.ExtraConfig = req.Spec.Config.ExtraConfig
			config.InstanceUuid = req.Spec.Config.InstanceUuid
		}

		// Copying hardware properties
		config.NumCPUs = vm.Config.Hardware.NumCPU
		config.MemoryMB = int64(vm.Config.Hardware.MemoryMB)
		config.NumCoresPerSocket = vm.Config.Hardware.NumCoresPerSocket
		config.VirtualICH7MPresent = vm.Config.Hardware.VirtualICH7MPresent
		config.VirtualSMCPresent = vm.Config.Hardware.VirtualSMCPresent

		defaultDevices := object.VirtualDeviceList(esx.VirtualDevice)
		devices := vm.cloneDevice()

		for _, device := range devices {
			var fop types.VirtualDeviceConfigSpecFileOperation

			if defaultDevices.Find(object.VirtualDeviceList(devices).Name(device)) != nil {
				// Default devices are added during CreateVMTask
				continue
			}

			switch disk := device.(type) {
			case *types.VirtualDisk:
				// TODO: consider VirtualMachineCloneSpec.DiskMoveType
				fop = types.VirtualDeviceConfigSpecFileOperationCreate

				// Leave FileName empty so CreateVM will just create a new one under VmPathName
				disk.Backing.(*types.VirtualDiskFlatVer2BackingInfo).FileName = ""
				disk.Backing.(*types.VirtualDiskFlatVer2BackingInfo).Parent = nil
			}

			config.DeviceChange = append(config.DeviceChange, &types.VirtualDeviceConfigSpec{
				Operation:     types.VirtualDeviceConfigSpecOperationAdd,
				Device:        device,
				FileOperation: fop,
			})
		}

		res := ctx.Map.Get(req.Folder).(vmFolder).CreateVMTask(ctx, &types.CreateVM_Task{
			This:   folder.Self,
			Config: config,
			Pool:   *pool,
			Host:   destHost,
		})

		ctask := ctx.Map.Get(res.(*methods.CreateVM_TaskBody).Res.Returnval).(*Task)
		ctask.Wait()
		if ctask.Info.Error != nil {
			return nil, ctask.Info.Error.Fault
		}

		ref := ctask.Info.Result.(types.ManagedObjectReference)
		clone := ctx.Map.Get(ref).(*VirtualMachine)
		clone.configureDevices(ctx, &types.VirtualMachineConfigSpec{DeviceChange: req.Spec.Location.DeviceChange})
		if req.Spec.Config != nil && req.Spec.Config.DeviceChange != nil {
			clone.configureDevices(ctx, &types.VirtualMachineConfigSpec{DeviceChange: req.Spec.Config.DeviceChange})
		}

		if req.Spec.Template {
			_ = clone.MarkAsTemplate(&types.MarkAsTemplate{This: clone.Self})
		}

		ctx.postEvent(&types.VmClonedEvent{
			VmCloneEvent: types.VmCloneEvent{VmEvent: clone.event()},
			SourceVm:     *event.Vm,
		})

		return ref, nil
	})

	return &methods.CloneVM_TaskBody{
		Res: &types.CloneVM_TaskResponse{
			Returnval: task.Run(ctx),
		},
	}
}

func (vm *VirtualMachine) RelocateVMTask(ctx *Context, req *types.RelocateVM_Task) soap.HasFault {
	task := CreateTask(vm, "relocateVm", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		var changes []types.PropertyChange

		if ref := req.Spec.Datastore; ref != nil {
			ds := ctx.Map.Get(*ref).(*Datastore)
			ctx.Map.RemoveReference(ctx, ds, &ds.Vm, *ref)

			// TODO: migrate vm.Config.Files, vm.Summary.Config.VmPathName, vm.Layout and vm.LayoutEx

			changes = append(changes, types.PropertyChange{Name: "datastore", Val: []types.ManagedObjectReference{*ref}})
		}

		if ref := req.Spec.Pool; ref != nil {
			pool := ctx.Map.Get(*ref).(*ResourcePool)
			ctx.Map.RemoveReference(ctx, pool, &pool.Vm, *ref)

			changes = append(changes, types.PropertyChange{Name: "resourcePool", Val: ref})
		}

		if ref := req.Spec.Host; ref != nil {
			host := ctx.Map.Get(*ref).(*HostSystem)
			ctx.Map.RemoveReference(ctx, host, &host.Vm, *ref)

			changes = append(changes,
				types.PropertyChange{Name: "runtime.host", Val: ref},
				types.PropertyChange{Name: "summary.runtime.host", Val: ref},
			)
		}

		if ref := req.Spec.Folder; ref != nil {
			folder := ctx.Map.Get(*ref).(*Folder)
			folder.MoveIntoFolderTask(ctx, &types.MoveIntoFolder_Task{
				List: []types.ManagedObjectReference{vm.Self},
			})
		}

		ctx.postEvent(&types.VmMigratedEvent{
			VmEvent:          vm.event(),
			SourceHost:       *ctx.Map.Get(*vm.Runtime.Host).(*HostSystem).eventArgument(),
			SourceDatacenter: datacenterEventArgument(vm),
			SourceDatastore:  ctx.Map.Get(vm.Datastore[0]).(*Datastore).eventArgument(),
		})

		ctx.Map.Update(vm, changes)

		return nil, nil
	})

	return &methods.RelocateVM_TaskBody{
		Res: &types.RelocateVM_TaskResponse{
			Returnval: task.Run(ctx),
		},
	}
}

func (vm *VirtualMachine) customize(ctx *Context) {
	if vm.imc == nil {
		return
	}

	event := types.CustomizationEvent{VmEvent: vm.event()}
	ctx.postEvent(&types.CustomizationStartedEvent{CustomizationEvent: event})

	changes := []types.PropertyChange{
		{Name: "config.tools.pendingCustomization", Val: ""},
	}

	hostname := ""
	address := ""

	switch c := vm.imc.Identity.(type) {
	case *types.CustomizationLinuxPrep:
		hostname = customizeName(vm, c.HostName)
	case *types.CustomizationSysprep:
		hostname = customizeName(vm, c.UserData.ComputerName)
	}

	cards := object.VirtualDeviceList(vm.Config.Hardware.Device).SelectByType((*types.VirtualEthernetCard)(nil))

	for i, s := range vm.imc.NicSettingMap {
		nic := &vm.Guest.Net[i]
		if s.MacAddress != "" {
			nic.MacAddress = strings.ToLower(s.MacAddress) // MacAddress in guest will always be lowercase
			card := cards[i].(types.BaseVirtualEthernetCard).GetVirtualEthernetCard()
			card.MacAddress = s.MacAddress // MacAddress in Virtual NIC can be any case
			card.AddressType = string(types.VirtualEthernetCardMacTypeManual)
		}
		if nic.DnsConfig == nil {
			nic.DnsConfig = new(types.NetDnsConfigInfo)
		}
		if s.Adapter.DnsDomain != "" {
			nic.DnsConfig.DomainName = s.Adapter.DnsDomain
		}
		if len(s.Adapter.DnsServerList) != 0 {
			nic.DnsConfig.IpAddress = s.Adapter.DnsServerList
		}
		if hostname != "" {
			nic.DnsConfig.HostName = hostname
		}
		if len(vm.imc.GlobalIPSettings.DnsSuffixList) != 0 {
			nic.DnsConfig.SearchDomain = vm.imc.GlobalIPSettings.DnsSuffixList
		}
		if nic.IpConfig == nil {
			nic.IpConfig = new(types.NetIpConfigInfo)
		}

		switch ip := s.Adapter.Ip.(type) {
		case *types.CustomizationCustomIpGenerator:
		case *types.CustomizationDhcpIpGenerator:
		case *types.CustomizationFixedIp:
			if address == "" {
				address = ip.IpAddress
			}
			nic.IpAddress = []string{ip.IpAddress}
			nic.IpConfig.IpAddress = []types.NetIpConfigInfoIpAddress{{
				IpAddress: ip.IpAddress,
			}}
		case *types.CustomizationUnknownIpGenerator:
		}
	}

	if len(vm.imc.NicSettingMap) != 0 {
		changes = append(changes, types.PropertyChange{Name: "guest.net", Val: vm.Guest.Net})
	}
	if hostname != "" {
		changes = append(changes, types.PropertyChange{Name: "guest.hostName", Val: hostname})
		changes = append(changes, types.PropertyChange{Name: "summary.guest.hostName", Val: hostname})
	}
	if address != "" {
		changes = append(changes, types.PropertyChange{Name: "guest.ipAddress", Val: address})
		changes = append(changes, types.PropertyChange{Name: "summary.guest.ipAddress", Val: address})
	}

	vm.imc = nil
	ctx.Map.Update(vm, changes)
	ctx.postEvent(&types.CustomizationSucceeded{CustomizationEvent: event})
}

func (vm *VirtualMachine) CustomizeVMTask(ctx *Context, req *types.CustomizeVM_Task) soap.HasFault {
	task := CreateTask(vm, "customizeVm", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		if vm.hostInMM(ctx) {
			return nil, new(types.InvalidState)
		}

		if vm.Runtime.PowerState == types.VirtualMachinePowerStatePoweredOn {
			return nil, &types.InvalidPowerState{
				RequestedState: types.VirtualMachinePowerStatePoweredOff,
				ExistingState:  vm.Runtime.PowerState,
			}
		}
		if vm.Config.Tools.PendingCustomization != "" {
			return nil, new(types.CustomizationPending)
		}
		if len(vm.Guest.Net) != len(req.Spec.NicSettingMap) {
			return nil, &types.NicSettingMismatch{
				NumberOfNicsInSpec: int32(len(req.Spec.NicSettingMap)),
				NumberOfNicsInVM:   int32(len(vm.Guest.Net)),
			}
		}

		vm.imc = &req.Spec
		vm.Config.Tools.PendingCustomization = uuid.New().String()

		return nil, nil
	})

	return &methods.CustomizeVM_TaskBody{
		Res: &types.CustomizeVM_TaskResponse{
			Returnval: task.Run(ctx),
		},
	}
}

func (vm *VirtualMachine) CreateSnapshotTask(ctx *Context, req *types.CreateSnapshot_Task) soap.HasFault {
	task := CreateTask(vm, "createSnapshot", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		var changes []types.PropertyChange

		if vm.Snapshot == nil {
			vm.Snapshot = &types.VirtualMachineSnapshotInfo{}
		}

		snapshot := &VirtualMachineSnapshot{}
		snapshot.Vm = vm.Reference()
		snapshot.Config = *vm.Config

		ctx.Map.Put(snapshot)

		treeItem := types.VirtualMachineSnapshotTree{
			Snapshot:        snapshot.Self,
			Vm:              snapshot.Vm,
			Name:            req.Name,
			Description:     req.Description,
			Id:              atomic.AddInt32(&vm.sid, 1),
			CreateTime:      time.Now(),
			State:           vm.Runtime.PowerState,
			Quiesced:        req.Quiesce,
			BackupManifest:  "",
			ReplaySupported: types.NewBool(false),
		}

		cur := vm.Snapshot.CurrentSnapshot
		if cur != nil {
			parent := ctx.Map.Get(*cur).(*VirtualMachineSnapshot)
			parent.ChildSnapshot = append(parent.ChildSnapshot, snapshot.Self)

			ss := findSnapshotInTree(vm.Snapshot.RootSnapshotList, *cur)
			ss.ChildSnapshotList = append(ss.ChildSnapshotList, treeItem)
		} else {
			changes = append(changes, types.PropertyChange{
				Name: "snapshot.rootSnapshotList",
				Val:  append(vm.Snapshot.RootSnapshotList, treeItem),
			})
			changes = append(changes, types.PropertyChange{
				Name: "rootSnapshot",
				Val:  append(vm.RootSnapshot, treeItem.Snapshot),
			})
		}

		snapshot.createSnapshotFiles()

		changes = append(changes, types.PropertyChange{Name: "snapshot.currentSnapshot", Val: snapshot.Self})
		ctx.Map.Update(vm, changes)

		return snapshot.Self, nil
	})

	return &methods.CreateSnapshot_TaskBody{
		Res: &types.CreateSnapshot_TaskResponse{
			Returnval: task.Run(ctx),
		},
	}
}

func (vm *VirtualMachine) RevertToCurrentSnapshotTask(ctx *Context, req *types.RevertToCurrentSnapshot_Task) soap.HasFault {
	body := &methods.RevertToCurrentSnapshot_TaskBody{}

	if vm.Snapshot == nil || vm.Snapshot.CurrentSnapshot == nil {
		body.Fault_ = Fault("snapshot not found", &types.NotFound{})

		return body
	}

	task := CreateTask(vm, "revertSnapshot", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		return nil, nil
	})

	body.Res = &types.RevertToCurrentSnapshot_TaskResponse{
		Returnval: task.Run(ctx),
	}

	return body
}

func (vm *VirtualMachine) RemoveAllSnapshotsTask(ctx *Context, req *types.RemoveAllSnapshots_Task) soap.HasFault {
	task := CreateTask(vm, "RemoveAllSnapshots", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		if vm.Snapshot == nil {
			return nil, nil
		}

		refs := allSnapshotsInTree(vm.Snapshot.RootSnapshotList)

		ctx.Map.Update(vm, []types.PropertyChange{
			{Name: "snapshot", Val: nil},
			{Name: "rootSnapshot", Val: nil},
		})

		for _, ref := range refs {
			ctx.Map.Get(ref).(*VirtualMachineSnapshot).removeSnapshotFiles(ctx)
			ctx.Map.Remove(ctx, ref)
		}

		return nil, nil
	})

	return &methods.RemoveAllSnapshots_TaskBody{
		Res: &types.RemoveAllSnapshots_TaskResponse{
			Returnval: task.Run(ctx),
		},
	}
}

func (vm *VirtualMachine) ShutdownGuest(ctx *Context, c *types.ShutdownGuest) soap.HasFault {
	r := &methods.ShutdownGuestBody{}
	// should be poweron
	if vm.Runtime.PowerState == types.VirtualMachinePowerStatePoweredOff {
		r.Fault_ = Fault("", &types.InvalidPowerState{
			RequestedState: types.VirtualMachinePowerStatePoweredOn,
			ExistingState:  vm.Runtime.PowerState,
		})

		return r
	}
	// change state
	vm.Runtime.PowerState = types.VirtualMachinePowerStatePoweredOff
	vm.Summary.Runtime.PowerState = types.VirtualMachinePowerStatePoweredOff

	event := vm.event()
	ctx.postEvent(
		&types.VmGuestShutdownEvent{VmEvent: event},
		&types.VmPoweredOffEvent{VmEvent: event},
	)
	vm.run.stop(ctx, vm)

	ctx.Map.Update(vm, []types.PropertyChange{
		{Name: "runtime.powerState", Val: types.VirtualMachinePowerStatePoweredOff},
		{Name: "summary.runtime.powerState", Val: types.VirtualMachinePowerStatePoweredOff},
	})

	r.Res = new(types.ShutdownGuestResponse)

	return r
}

func (vm *VirtualMachine) MarkAsTemplate(req *types.MarkAsTemplate) soap.HasFault {
	r := &methods.MarkAsTemplateBody{}

	if vm.Config.Template {
		r.Fault_ = Fault("", new(types.NotSupported))
		return r
	}

	if vm.Runtime.PowerState != types.VirtualMachinePowerStatePoweredOff {
		r.Fault_ = Fault("", &types.InvalidPowerState{
			RequestedState: types.VirtualMachinePowerStatePoweredOff,
			ExistingState:  vm.Runtime.PowerState,
		})
		return r
	}

	vm.Config.Template = true
	vm.Summary.Config.Template = true
	vm.ResourcePool = nil

	r.Res = new(types.MarkAsTemplateResponse)

	return r
}

func (vm *VirtualMachine) MarkAsVirtualMachine(req *types.MarkAsVirtualMachine) soap.HasFault {
	r := &methods.MarkAsVirtualMachineBody{}

	if !vm.Config.Template {
		r.Fault_ = Fault("", new(types.NotSupported))
		return r
	}

	if vm.Runtime.PowerState != types.VirtualMachinePowerStatePoweredOff {
		r.Fault_ = Fault("", &types.InvalidPowerState{
			RequestedState: types.VirtualMachinePowerStatePoweredOff,
			ExistingState:  vm.Runtime.PowerState,
		})
		return r
	}

	vm.Config.Template = false
	vm.Summary.Config.Template = false
	vm.ResourcePool = &req.Pool
	if req.Host != nil {
		vm.Runtime.Host = req.Host
	}

	r.Res = new(types.MarkAsVirtualMachineResponse)

	return r
}

func findSnapshotInTree(tree []types.VirtualMachineSnapshotTree, ref types.ManagedObjectReference) *types.VirtualMachineSnapshotTree {
	if tree == nil {
		return nil
	}

	for i, ss := range tree {
		if ss.Snapshot == ref {
			return &tree[i]
		}

		target := findSnapshotInTree(ss.ChildSnapshotList, ref)
		if target != nil {
			return target
		}
	}

	return nil
}

func findParentSnapshot(tree types.VirtualMachineSnapshotTree, ref types.ManagedObjectReference) *types.ManagedObjectReference {
	for _, ss := range tree.ChildSnapshotList {
		if ss.Snapshot == ref {
			return &tree.Snapshot
		}

		res := findParentSnapshot(ss, ref)
		if res != nil {
			return res
		}
	}

	return nil
}

func findParentSnapshotInTree(tree []types.VirtualMachineSnapshotTree, ref types.ManagedObjectReference) *types.ManagedObjectReference {
	if tree == nil {
		return nil
	}

	for _, ss := range tree {
		res := findParentSnapshot(ss, ref)
		if res != nil {
			return res
		}
	}

	return nil
}

func removeSnapshotInTree(tree []types.VirtualMachineSnapshotTree, ref types.ManagedObjectReference, removeChildren bool) []types.VirtualMachineSnapshotTree {
	if tree == nil {
		return tree
	}

	var result []types.VirtualMachineSnapshotTree

	for _, ss := range tree {
		if ss.Snapshot == ref {
			if !removeChildren {
				result = append(result, ss.ChildSnapshotList...)
			}
		} else {
			ss.ChildSnapshotList = removeSnapshotInTree(ss.ChildSnapshotList, ref, removeChildren)
			result = append(result, ss)
		}
	}

	return result
}

func allSnapshotsInTree(tree []types.VirtualMachineSnapshotTree) []types.ManagedObjectReference {
	var result []types.ManagedObjectReference

	if tree == nil {
		return result
	}

	for _, ss := range tree {
		result = append(result, ss.Snapshot)
		result = append(result, allSnapshotsInTree(ss.ChildSnapshotList)...)
	}

	return result
}

func changeTrackingSupported(spec *types.VirtualMachineConfigSpec) bool {
	for _, device := range spec.DeviceChange {
		if dev, ok := device.GetVirtualDeviceConfigSpec().Device.(*types.VirtualDisk); ok {
			switch dev.Backing.(type) {
			case *types.VirtualDiskFlatVer2BackingInfo:
				return true
			case *types.VirtualDiskSparseVer2BackingInfo:
				return true
			case *types.VirtualDiskRawDiskMappingVer1BackingInfo:
				return true
			case *types.VirtualDiskRawDiskVer2BackingInfo:
				return true
			default:
				return false
			}
		}
	}
	return false
}

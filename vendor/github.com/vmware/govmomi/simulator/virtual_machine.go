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
}

func NewVirtualMachine(parent types.ManagedObjectReference, spec *types.VirtualMachineConfigSpec) (*VirtualMachine, types.BaseMethodFault) {
	vm := &VirtualMachine{}
	vm.Parent = &parent

	Map.Get(parent).(*Folder).putChild(vm)

	if spec.Name == "" {
		return vm, &types.InvalidVmConfig{Property: "configSpec.name"}
	}

	if spec.Files == nil || spec.Files.VmPathName == "" {
		return vm, &types.InvalidVmConfig{Property: "configSpec.files.vmPathName"}
	}

	rspec := types.DefaultResourceConfigSpec()
	vm.Guest = &types.GuestInfo{}
	vm.Config = &types.VirtualMachineConfigInfo{
		ExtraConfig:      []types.BaseOptionValue{&types.OptionValue{Key: "govcsim", Value: "TRUE"}},
		Tools:            &types.ToolsConfigInfo{},
		MemoryAllocation: &rspec.MemoryAllocation,
		CpuAllocation:    &rspec.CpuAllocation,
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

	// Append VM Name as the directory name if not specified
	if strings.HasSuffix(spec.Files.VmPathName, "]") { // e.g. "[datastore1]"
		spec.Files.VmPathName += " " + spec.Name
	}

	if !strings.HasSuffix(spec.Files.VmPathName, ".vmx") {
		spec.Files.VmPathName = path.Join(spec.Files.VmPathName, spec.Name+".vmx")
	}

	dsPath := path.Dir(spec.Files.VmPathName)

	defaults := types.VirtualMachineConfigSpec{
		NumCPUs:           1,
		NumCoresPerSocket: 1,
		MemoryMB:          32,
		Uuid:              uuid.New().String(),
		InstanceUuid:      uuid.New().String(),
		Version:           esx.HardwareVersion,
		Files: &types.VirtualMachineFileInfo{
			SnapshotDirectory: dsPath,
			SuspendDirectory:  dsPath,
			LogDirectory:      dsPath,
		},
	}

	// Add the default devices
	defaults.DeviceChange, _ = object.VirtualDeviceList(esx.VirtualDevice).ConfigSpec(types.VirtualDeviceConfigSpecOperationAdd)

	err := vm.configure(&defaults)
	if err != nil {
		return vm, err
	}

	vm.Runtime.PowerState = types.VirtualMachinePowerStatePoweredOff
	vm.Runtime.ConnectionState = types.VirtualMachineConnectionStateConnected
	vm.Summary.Runtime = vm.Runtime

	vm.Summary.QuickStats.GuestHeartbeatStatus = types.ManagedEntityStatusGray
	vm.Summary.OverallStatus = types.ManagedEntityStatusGreen
	vm.ConfigStatus = types.ManagedEntityStatusGreen

	return vm, nil
}

func (vm *VirtualMachine) event() types.VmEvent {
	host := Map.Get(*vm.Runtime.Host).(*HostSystem)

	return types.VmEvent{
		Event: types.Event{
			Datacenter:      datacenterEventArgument(host),
			ComputeResource: host.eventArgumentParent(),
			Host:            host.eventArgument(),
			Vm: &types.VmEventArgument{
				EntityEventArgument: types.EntityEventArgument{Name: vm.Name},
				Vm:                  vm.Self,
			},
		},
	}
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

	var changes []types.PropertyChange
	for _, c := range spec.ExtraConfig {
		val := c.GetOptionValue()
		key := strings.TrimPrefix(val.Key, "SET.")
		if key == val.Key {
			vm.Config.ExtraConfig = append(vm.Config.ExtraConfig, c)
			continue
		}
		changes = append(changes, types.PropertyChange{Name: key, Val: val.Value})

		switch key {
		case "guest.ipAddress":
			ip := val.Value.(string)
			vm.Guest.Net[0].IpAddress = []string{ip}
			changes = append(changes,
				types.PropertyChange{Name: "summary." + key, Val: ip},
				types.PropertyChange{Name: "guest.net", Val: vm.Guest.Net},
			)
		case "guest.hostName":
			changes = append(changes,
				types.PropertyChange{Name: "summary." + key, Val: val.Value},
			)
		}
	}
	if len(changes) != 0 {
		Map.Update(vm, changes)
	}

	vm.Config.Modified = time.Now()
}

func validateGuestID(id string) types.BaseMethodFault {
	for _, x := range GuestID {
		if id == string(x) {
			return nil
		}
	}

	return &types.InvalidArgument{InvalidProperty: "configSpec.guestId"}
}

func (vm *VirtualMachine) configure(spec *types.VirtualMachineConfigSpec) types.BaseMethodFault {
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

	return vm.configureDevices(spec)
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

			dm := Map.VirtualDiskManager()
			// Add disk descriptor and extent files
			for _, diskName := range dm.names(dFileName) {
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
			body.Fault_ = soap.ToSoapFault(err)
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

	return body
}

func (vm *VirtualMachine) useDatastore(name string) *Datastore {
	host := Map.Get(*vm.Runtime.Host).(*HostSystem)

	ds := Map.FindByName(name, host.Datastore).(*Datastore)

	if FindReference(vm.Datastore, ds.Self) == nil {
		vm.Datastore = append(vm.Datastore, ds.Self)
	}

	return ds
}

func (vm *VirtualMachine) createFile(spec string, name string, register bool) (*os.File, types.BaseMethodFault) {
	p, fault := parseDatastorePath(spec)
	if fault != nil {
		return nil, fault
	}

	ds := vm.useDatastore(p.Datastore)

	file := path.Join(ds.Info.GetDatastoreInfo().Url, p.Path)

	if name != "" {
		if path.Ext(p.Path) == ".vmx" {
			file = path.Dir(file) // vm.Config.Files.VmPathName can be a directory or full path to .vmx
		}

		file = path.Join(file, name)
	}

	if register {
		f, err := os.Open(file)
		if err != nil {
			log.Printf("register %s: %s", vm.Reference(), err)
			if os.IsNotExist(err) {
				return nil, &types.NotFound{}
			}

			return nil, &types.InvalidArgument{}
		}

		return f, nil
	}

	dir := path.Dir(file)

	_ = os.MkdirAll(dir, 0700)

	_, err := os.Stat(file)
	if err == nil {
		return nil, &types.FileAlreadyExists{
			FileFault: types.FileFault{
				File: file,
			},
		}
	}

	f, err := os.Create(file)
	if err != nil {
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

func (vm *VirtualMachine) create(spec *types.VirtualMachineConfigSpec, register bool) types.BaseMethodFault {
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

	return vm.configureDevices(spec)
}

var vmwOUI = net.HardwareAddr([]byte{0x0, 0xc, 0x29})

// From http://pubs.vmware.com/vsphere-60/index.jsp?topic=%2Fcom.vmware.vsphere.networking.doc%2FGUID-DC7478FF-DC44-4625-9AD7-38208C56A552.html
// "The host generates generateMAC addresses that consists of the VMware OUI 00:0C:29 and the last three octets in hexadecimal
//  format of the virtual machine UUID.  The virtual machine UUID is based on a hash calculated by using the UUID of the
//  ESXi physical machine and the path to the configuration file (.vmx) of the virtual machine."
func (vm *VirtualMachine) generateMAC() string {
	id := uuid.New() // Random is fine for now.

	offset := len(id) - len(vmwOUI)

	mac := append(vmwOUI, id[offset:]...)

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

func (vm *VirtualMachine) configureDevice(devices object.VirtualDeviceList, spec *types.VirtualDeviceConfigSpec) types.BaseMethodFault {
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
	dc := Map.getEntityDatacenter(Map.Get(*vm.Parent).(mo.Entity))
	dm := Map.VirtualDiskManager()

	switch x := device.(type) {
	case types.BaseVirtualEthernetCard:
		controller = devices.PickController((*types.VirtualPCIController)(nil))
		var net types.ManagedObjectReference
		var name string

		switch b := d.Backing.(type) {
		case *types.VirtualEthernetCardNetworkBackingInfo:
			name = b.DeviceName
			summary = name
			net = Map.FindByName(b.DeviceName, dc.Network).Reference()
			b.Network = &net
		case *types.VirtualEthernetCardDistributedVirtualPortBackingInfo:
			summary = fmt.Sprintf("DVSwitch: %s", b.Port.SwitchUuid)
			net.Type = "DistributedVirtualPortgroup"
			net.Value = b.Port.PortgroupKey
		}

		Map.Update(vm, []types.PropertyChange{
			{Name: "summary.config.numEthernetCards", Val: vm.Summary.Config.NumEthernetCards + 1},
			{Name: "network", Val: append(vm.Network, net)},
		})

		c := x.GetVirtualEthernetCard()
		if c.MacAddress == "" {
			c.MacAddress = vm.generateMAC()
		}

		if spec.Operation == types.VirtualDeviceConfigSpecOperationAdd {
			vm.Guest.Net = append(vm.Guest.Net, types.GuestNicInfo{
				Network:        name,
				IpAddress:      nil,
				MacAddress:     c.MacAddress,
				Connected:      true,
				DeviceConfigId: c.Key,
			})
		}
	case *types.VirtualDisk:
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

			err := dm.createVirtualDisk(spec.FileOperation, &types.CreateVirtualDisk_Task{
				Datacenter: &dc.Self,
				Name:       info.FileName,
			})
			if err != nil {
				return err
			}

			Map.Update(vm, []types.PropertyChange{
				{Name: "summary.config.numVirtualDisks", Val: vm.Summary.Config.NumVirtualDisks + 1},
			})

			p, _ := parseDatastorePath(info.FileName)

			host := Map.Get(*vm.Runtime.Host).(*HostSystem)

			entity := Map.FindByName(p.Datastore, host.Datastore)
			ref := entity.Reference()
			info.Datastore = &ref

			ds := entity.(*Datastore)

			// XXX: compare disk size and free space until windows stat is supported
			ds.Summary.FreeSpace -= getDiskSize(x)
			ds.Info.GetDatastoreInfo().FreeSpace = ds.Summary.FreeSpace

			vm.updateDiskLayouts()
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
	}

	return nil
}

func (vm *VirtualMachine) removeDevice(devices object.VirtualDeviceList, spec *types.VirtualDeviceConfigSpec) object.VirtualDeviceList {
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

					host := Map.Get(*vm.Runtime.Host).(*HostSystem)

					ds := Map.FindByName(p.Datastore, host.Datastore).(*Datastore)

					ds.Summary.FreeSpace += getDiskSize(device)
					ds.Info.GetDatastoreInfo().FreeSpace = ds.Summary.FreeSpace
				}

				if file != "" {
					dc := Map.getEntityDatacenter(Map.Get(*vm.Parent).(mo.Entity))
					dm := Map.VirtualDiskManager()

					dm.DeleteVirtualDiskTask(&types.DeleteVirtualDisk_Task{
						Name:       file,
						Datacenter: &dc.Self,
					})
				}
			}
			Map.Update(vm, []types.PropertyChange{
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

			networks := vm.Network
			RemoveReference(&networks, net)
			Map.Update(vm, []types.PropertyChange{
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

func (vm *VirtualMachine) configureDevices(spec *types.VirtualMachineConfigSpec) types.BaseMethodFault {
	devices := object.VirtualDeviceList(vm.Config.Hardware.Device)

	for i, change := range spec.DeviceChange {
		dspec := change.GetVirtualDeviceConfigSpec()
		device := dspec.Device.GetVirtualDevice()
		invalid := &types.InvalidDeviceSpec{DeviceIndex: int32(i)}

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

			err := vm.configureDevice(devices, dspec)
			if err != nil {
				return err
			}

			devices = append(devices, dspec.Device)
		case types.VirtualDeviceConfigSpecOperationEdit:
			rspec := *dspec
			rspec.Device = devices.FindByKey(device.Key)
			if rspec.Device == nil {
				return invalid
			}
			devices = vm.removeDevice(devices, &rspec)
			device.DeviceInfo = nil // regenerate summary + label

			err := vm.configureDevice(devices, dspec)
			if err != nil {
				return err
			}

			devices = append(devices, dspec.Device)
		case types.VirtualDeviceConfigSpecOperationRemove:
			devices = vm.removeDevice(devices, dspec)
		}
	}

	Map.Update(vm, []types.PropertyChange{
		{Name: "config.hardware.device", Val: []types.BaseVirtualDevice(devices)},
	})

	vm.updateDiskLayouts()

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
		c.run.start(c.VirtualMachine)
		c.ctx.postEvent(
			&types.VmStartingEvent{VmEvent: event},
			&types.VmPoweredOnEvent{VmEvent: event},
		)
	case types.VirtualMachinePowerStatePoweredOff:
		c.run.stop(c.VirtualMachine)
		c.ctx.postEvent(
			&types.VmStoppingEvent{VmEvent: event},
			&types.VmPoweredOffEvent{VmEvent: event},
		)
	case types.VirtualMachinePowerStateSuspended:
		c.run.pause(c.VirtualMachine)
		c.ctx.postEvent(
			&types.VmSuspendingEvent{VmEvent: event},
			&types.VmSuspendedEvent{VmEvent: event},
		)
	}

	Map.Update(c.VirtualMachine, []types.PropertyChange{
		{Name: "runtime.powerState", Val: c.state},
		{Name: "summary.runtime.powerState", Val: c.state},
		{Name: "summary.runtime.bootTime", Val: boot},
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
			Returnval: task.Run(),
		},
	}
}

func (vm *VirtualMachine) PowerOffVMTask(ctx *Context, c *types.PowerOffVM_Task) soap.HasFault {
	runner := &powerVMTask{vm, types.VirtualMachinePowerStatePoweredOff, ctx}
	task := CreateTask(runner.Reference(), "powerOff", runner.Run)

	return &methods.PowerOffVM_TaskBody{
		Res: &types.PowerOffVM_TaskResponse{
			Returnval: task.Run(),
		},
	}
}

func (vm *VirtualMachine) SuspendVMTask(ctx *Context, req *types.SuspendVM_Task) soap.HasFault {
	runner := &powerVMTask{vm, types.VirtualMachinePowerStateSuspended, ctx}
	task := CreateTask(runner.Reference(), "suspend", runner.Run)

	return &methods.SuspendVM_TaskBody{
		Res: &types.SuspendVM_TaskResponse{
			Returnval: task.Run(),
		},
	}
}

func (vm *VirtualMachine) ResetVMTask(ctx *Context, req *types.ResetVM_Task) soap.HasFault {
	task := CreateTask(vm, "reset", func(task *Task) (types.AnyType, types.BaseMethodFault) {
		res := vm.PowerOffVMTask(ctx, &types.PowerOffVM_Task{This: vm.Self})
		ctask := Map.Get(res.(*methods.PowerOffVM_TaskBody).Res.Returnval).(*Task)
		if ctask.Info.Error != nil {
			return nil, ctask.Info.Error.Fault
		}

		_ = vm.PowerOnVMTask(ctx, &types.PowerOnVM_Task{This: vm.Self})

		return nil, nil
	})

	return &methods.ResetVM_TaskBody{
		Res: &types.ResetVM_TaskResponse{
			Returnval: task.Run(),
		},
	}
}

func (vm *VirtualMachine) ReconfigVMTask(ctx *Context, req *types.ReconfigVM_Task) soap.HasFault {
	task := CreateTask(vm, "reconfigVm", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		err := vm.configure(&req.Spec)
		if err != nil {
			return nil, err
		}

		ctx.postEvent(&types.VmReconfiguredEvent{
			VmEvent:    vm.event(),
			ConfigSpec: req.Spec,
		})

		return nil, nil
	})

	return &methods.ReconfigVM_TaskBody{
		Res: &types.ReconfigVM_TaskResponse{
			Returnval: task.Run(),
		},
	}
}

func (vm *VirtualMachine) UpgradeVMTask(req *types.UpgradeVM_Task) soap.HasFault {
	body := &methods.UpgradeVM_TaskBody{}

	task := CreateTask(vm, "upgradeVm", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		if vm.Config.Version != esx.HardwareVersion {
			Map.Update(vm, []types.PropertyChange{{
				Name: "config.version", Val: esx.HardwareVersion,
			}})
		}
		return nil, nil
	})

	body.Res = &types.UpgradeVM_TaskResponse{
		Returnval: task.Run(),
	}

	return body
}

func (vm *VirtualMachine) DestroyTask(ctx *Context, req *types.Destroy_Task) soap.HasFault {
	task := CreateTask(vm, "destroy", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		r := vm.UnregisterVM(ctx, &types.UnregisterVM{
			This: req.This,
		})

		if r.Fault() != nil {
			return nil, r.Fault().VimFault().(types.BaseMethodFault)
		}

		// Remove all devices
		devices := object.VirtualDeviceList(vm.Config.Hardware.Device)
		spec, _ := devices.ConfigSpec(types.VirtualDeviceConfigSpecOperationRemove)
		vm.configureDevices(&types.VirtualMachineConfigSpec{DeviceChange: spec})

		// Delete VM files from the datastore (ignoring result for now)
		m := Map.FileManager()
		dc := Map.getEntityDatacenter(vm).Reference()

		_ = m.DeleteDatastoreFileTask(&types.DeleteDatastoreFile_Task{
			This:       m.Reference(),
			Name:       vm.Config.Files.LogDirectory,
			Datacenter: &dc,
		})

		vm.run.remove(vm)

		return nil, nil
	})

	return &methods.Destroy_TaskBody{
		Res: &types.Destroy_TaskResponse{
			Returnval: task.Run(),
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

	host := Map.Get(*vm.Runtime.Host).(*HostSystem)
	Map.RemoveReference(host, &host.Vm, vm.Self)

	switch pool := Map.Get(*vm.ResourcePool).(type) {
	case *ResourcePool:
		Map.RemoveReference(pool, &pool.Vm, vm.Self)
	case *VirtualApp:
		Map.RemoveReference(pool, &pool.Vm, vm.Self)
	}

	for i := range vm.Datastore {
		ds := Map.Get(vm.Datastore[i]).(*Datastore)
		Map.RemoveReference(ds, &ds.Vm, vm.Self)
	}

	ctx.postEvent(&types.VmRemovedEvent{VmEvent: vm.event()})
	Map.getEntityParent(vm, "Folder").(*Folder).removeChild(c.This)

	r.Res = new(types.UnregisterVMResponse)

	return r
}

func (vm *VirtualMachine) CloneVMTask(ctx *Context, req *types.CloneVM_Task) soap.HasFault {
	ctx.Caller = &vm.Self
	folder := Map.Get(req.Folder).(*Folder)
	host := Map.Get(*vm.Runtime.Host).(*HostSystem)
	event := vm.event()

	ctx.postEvent(&types.VmBeingClonedEvent{
		VmCloneEvent: types.VmCloneEvent{
			VmEvent: event,
		},
		DestFolder: folder.eventArgument(),
		DestName:   req.Name,
		DestHost:   *host.eventArgument(),
	})

	task := CreateTask(vm, "cloneVm", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		config := types.VirtualMachineConfigSpec{
			Name:    req.Name,
			GuestId: vm.Config.GuestId,
			Files: &types.VirtualMachineFileInfo{
				VmPathName: strings.Replace(vm.Config.Files.VmPathName, vm.Name, req.Name, -1),
			},
		}

		defaultDevices := object.VirtualDeviceList(esx.VirtualDevice)
		devices := vm.Config.Hardware.Device
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

		res := folder.CreateVMTask(ctx, &types.CreateVM_Task{
			This:   folder.Self,
			Config: config,
			Pool:   *vm.ResourcePool,
			Host:   vm.Runtime.Host,
		})

		ctask := Map.Get(res.(*methods.CreateVM_TaskBody).Res.Returnval).(*Task)
		if ctask.Info.Error != nil {
			return nil, ctask.Info.Error.Fault
		}

		ref := ctask.Info.Result.(types.ManagedObjectReference)
		clone := Map.Get(ref).(*VirtualMachine)
		clone.configureDevices(&types.VirtualMachineConfigSpec{DeviceChange: req.Spec.Location.DeviceChange})
		if req.Spec.Config != nil && req.Spec.Config.DeviceChange != nil {
			clone.configureDevices(&types.VirtualMachineConfigSpec{DeviceChange: req.Spec.Config.DeviceChange})
		}

		ctx.postEvent(&types.VmClonedEvent{
			VmCloneEvent: types.VmCloneEvent{VmEvent: clone.event()},
			SourceVm:     *event.Vm,
		})

		return ref, nil
	})

	return &methods.CloneVM_TaskBody{
		Res: &types.CloneVM_TaskResponse{
			Returnval: task.Run(),
		},
	}
}

func (vm *VirtualMachine) RelocateVMTask(req *types.RelocateVM_Task) soap.HasFault {
	task := CreateTask(vm, "relocateVm", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		var changes []types.PropertyChange

		if ref := req.Spec.Datastore; ref != nil {
			ds := Map.Get(*ref).(*Datastore)
			Map.RemoveReference(ds, &ds.Vm, *ref)

			// TODO: migrate vm.Config.Files, vm.Summary.Config.VmPathName, vm.Layout and vm.LayoutEx

			changes = append(changes, types.PropertyChange{Name: "datastore", Val: []types.ManagedObjectReference{*ref}})
		}

		if ref := req.Spec.Pool; ref != nil {
			pool := Map.Get(*ref).(*ResourcePool)
			Map.RemoveReference(pool, &pool.Vm, *ref)

			changes = append(changes, types.PropertyChange{Name: "resourcePool", Val: ref})
		}

		if ref := req.Spec.Host; ref != nil {
			host := Map.Get(*ref).(*HostSystem)
			Map.RemoveReference(host, &host.Vm, *ref)

			changes = append(changes,
				types.PropertyChange{Name: "runtime.host", Val: ref},
				types.PropertyChange{Name: "summary.runtime.host", Val: ref},
			)
		}

		Map.Update(vm, changes)

		return nil, nil
	})

	return &methods.RelocateVM_TaskBody{
		Res: &types.RelocateVM_TaskResponse{
			Returnval: task.Run(),
		},
	}
}

func (vm *VirtualMachine) CreateSnapshotTask(req *types.CreateSnapshot_Task) soap.HasFault {
	task := CreateTask(vm, "createSnapshot", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		var changes []types.PropertyChange

		if vm.Snapshot == nil {
			vm.Snapshot = &types.VirtualMachineSnapshotInfo{}
		}

		snapshot := &VirtualMachineSnapshot{}
		snapshot.Vm = vm.Reference()
		snapshot.Config = *vm.Config

		Map.Put(snapshot)

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
			parent := Map.Get(*cur).(*VirtualMachineSnapshot)
			parent.ChildSnapshot = append(parent.ChildSnapshot, snapshot.Self)

			ss := findSnapshotInTree(vm.Snapshot.RootSnapshotList, *cur)
			ss.ChildSnapshotList = append(ss.ChildSnapshotList, treeItem)
		} else {
			changes = append(changes, types.PropertyChange{
				Name: "snapshot.rootSnapshotList",
				Val:  append(vm.Snapshot.RootSnapshotList, treeItem),
			})
		}

		snapshot.createSnapshotFiles()

		changes = append(changes, types.PropertyChange{Name: "snapshot.currentSnapshot", Val: snapshot.Self})
		Map.Update(vm, changes)

		return nil, nil
	})

	return &methods.CreateSnapshot_TaskBody{
		Res: &types.CreateSnapshot_TaskResponse{
			Returnval: task.Run(),
		},
	}
}

func (vm *VirtualMachine) RevertToCurrentSnapshotTask(req *types.RevertToCurrentSnapshot_Task) soap.HasFault {
	body := &methods.RevertToCurrentSnapshot_TaskBody{}

	if vm.Snapshot == nil || vm.Snapshot.CurrentSnapshot == nil {
		body.Fault_ = Fault("snapshot not found", &types.NotFound{})

		return body
	}

	task := CreateTask(vm, "revertSnapshot", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		return nil, nil
	})

	body.Res = &types.RevertToCurrentSnapshot_TaskResponse{
		Returnval: task.Run(),
	}

	return body
}

func (vm *VirtualMachine) RemoveAllSnapshotsTask(ctx *Context, req *types.RemoveAllSnapshots_Task) soap.HasFault {
	task := CreateTask(vm, "RemoveAllSnapshots", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		if vm.Snapshot == nil {
			return nil, nil
		}

		refs := allSnapshotsInTree(vm.Snapshot.RootSnapshotList)

		Map.Update(vm, []types.PropertyChange{
			{Name: "snapshot", Val: nil},
		})

		for _, ref := range refs {
			Map.Get(ref).(*VirtualMachineSnapshot).removeSnapshotFiles(ctx)
			Map.Remove(ref)
		}

		return nil, nil
	})

	return &methods.RemoveAllSnapshots_TaskBody{
		Res: &types.RemoveAllSnapshots_TaskResponse{
			Returnval: task.Run(),
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
	vm.run.stop(vm)

	Map.Update(vm, []types.PropertyChange{
		{Name: "runtime.powerState", Val: types.VirtualMachinePowerStatePoweredOff},
		{Name: "summary.runtime.powerState", Val: types.VirtualMachinePowerStatePoweredOff},
	})

	r.Res = new(types.ShutdownGuestResponse)

	return r
}

func (vm *VirtualMachine) MarkAsTemplate(req *types.MarkAsTemplate) soap.HasFault {
	r := &methods.MarkAsTemplateBody{}

	if vm.Runtime.PowerState != types.VirtualMachinePowerStatePoweredOff {
		r.Fault_ = Fault("", &types.InvalidPowerState{
			RequestedState: types.VirtualMachinePowerStatePoweredOff,
			ExistingState:  vm.Runtime.PowerState,
		})
		return r
	}

	vm.Config.Template = true

	r.Res = &types.MarkAsTemplateResponse{}

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

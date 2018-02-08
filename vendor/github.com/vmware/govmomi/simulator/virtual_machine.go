/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"path"
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

	log *log.Logger
	out io.Closer
	sid int32
}

func NewVirtualMachine(parent types.ManagedObjectReference, spec *types.VirtualMachineConfigSpec) (*VirtualMachine, types.BaseMethodFault) {
	vm := &VirtualMachine{}
	vm.Parent = &parent

	if spec.Name == "" {
		return nil, &types.InvalidVmConfig{Property: "configSpec.name"}
	}

	if spec.Files == nil || spec.Files.VmPathName == "" {
		return nil, &types.InvalidVmConfig{Property: "configSpec.files.vmPathName"}
	}

	rspec := types.DefaultResourceConfigSpec()
	vm.Config = &types.VirtualMachineConfigInfo{
		ExtraConfig:      []types.BaseOptionValue{&types.OptionValue{Key: "govcsim", Value: "TRUE"}},
		Tools:            &types.ToolsConfigInfo{},
		MemoryAllocation: &rspec.MemoryAllocation,
		CpuAllocation:    &rspec.CpuAllocation,
	}
	vm.Summary.Guest = &types.VirtualMachineGuestSummary{}
	vm.Summary.Storage = &types.VirtualMachineStorageSummary{}
	vm.Summary.Vm = &vm.Self

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
		Version:           "vmx-11",
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
		return nil, err
	}

	vm.Runtime.PowerState = types.VirtualMachinePowerStatePoweredOff
	vm.Runtime.ConnectionState = types.VirtualMachineConnectionStateConnected
	vm.Summary.Runtime = vm.Runtime

	vm.Summary.QuickStats.GuestHeartbeatStatus = types.ManagedEntityStatusGray
	vm.Summary.OverallStatus = types.ManagedEntityStatusGreen
	vm.ConfigStatus = types.ManagedEntityStatusGreen

	return vm, nil
}

func (vm *VirtualMachine) apply(spec *types.VirtualMachineConfigSpec) {
	if spec.Files == nil {
		spec.Files = new(types.VirtualMachineFileInfo)
	}

	apply := []struct {
		src string
		dst *string
	}{
		{spec.Name, &vm.Name},
		{spec.Name, &vm.Config.Name},
		{spec.Name, &vm.Summary.Config.Name},
		{spec.GuestId, &vm.Config.GuestId},
		{spec.GuestId, &vm.Config.GuestFullName},
		{spec.GuestId, &vm.Summary.Guest.GuestId},
		{spec.GuestId, &vm.Summary.Config.GuestId},
		{spec.GuestId, &vm.Summary.Config.GuestFullName},
		{spec.Uuid, &vm.Config.Uuid},
		{spec.Version, &vm.Config.Version},
		{spec.Files.VmPathName, &vm.Config.Files.VmPathName},
		{spec.Files.VmPathName, &vm.Summary.Config.VmPathName},
		{spec.Files.SnapshotDirectory, &vm.Config.Files.SnapshotDirectory},
		{spec.Files.LogDirectory, &vm.Config.Files.LogDirectory},
	}

	for _, f := range apply {
		if f.src != "" {
			*f.dst = f.src
		}
	}

	if spec.MemoryMB != 0 {
		vm.Config.Hardware.MemoryMB = int32(spec.MemoryMB)
		vm.Summary.Config.MemorySizeMB = vm.Config.Hardware.MemoryMB
	}

	if spec.NumCPUs != 0 {
		vm.Config.Hardware.NumCPU = spec.NumCPUs
		vm.Summary.Config.NumCpu = vm.Config.Hardware.NumCPU
	}

	vm.Config.ExtraConfig = append(vm.Config.ExtraConfig, spec.ExtraConfig...)

	vm.Config.Modified = time.Now()

	vm.Summary.Config.Uuid = vm.Config.Uuid
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

func (vm *VirtualMachine) useDatastore(name string) *Datastore {
	host := Map.Get(*vm.Runtime.Host).(*HostSystem)

	ds := Map.FindByName(name, host.Datastore).(*Datastore)

	vm.Datastore = AddReference(ds.Self, vm.Datastore)

	return ds
}

func (vm *VirtualMachine) setLog(w io.WriteCloser) {
	vm.out = w
	vm.log = log.New(w, "vmx ", log.Flags())
}

func (vm *VirtualMachine) createFile(spec string, name string, register bool) (*os.File, types.BaseMethodFault) {
	p, fault := parseDatastorePath(spec)
	if fault != nil {
		return nil, fault
	}

	ds := vm.useDatastore(p.Datastore)

	file := path.Join(ds.Info.GetDatastoreInfo().Url, p.Path)

	if name != "" {
		if path.Ext(file) != "" {
			file = path.Dir(file)
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

func (vm *VirtualMachine) create(spec *types.VirtualMachineConfigSpec, register bool) types.BaseMethodFault {
	vm.apply(spec)

	files := []struct {
		spec string
		name string
		use  func(w io.WriteCloser)
	}{
		{vm.Config.Files.VmPathName, "", nil},
		{vm.Config.Files.VmPathName, fmt.Sprintf("%s.nvram", vm.Name), nil},
		{vm.Config.Files.LogDirectory, "vmware.log", vm.setLog},
	}

	for _, file := range files {
		f, err := vm.createFile(file.spec, file.name, register)
		if err != nil {
			return err
		}

		if file.use != nil {
			file.use(f)
		} else {
			_ = f.Close()
		}
	}

	vm.log.Print("created")

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

func (vm *VirtualMachine) configureDevice(devices object.VirtualDeviceList, device types.BaseVirtualDevice) types.BaseMethodFault {
	d := device.GetVirtualDevice()
	var controller types.BaseVirtualController

	if d.Key < 0 {
		// Choose a unique key
		if d.Key == -1 {
			d.Key = devices.NewKey()
		}

		d.Key *= -1

		for {
			if devices.FindByKey(d.Key) == nil {
				break
			}
			d.Key++
		}
	}

	label := devices.Name(device)
	summary := label
	dc := Map.getEntityDatacenter(Map.Get(*vm.Parent).(mo.Entity))
	dm := Map.VirtualDiskManager()

	switch x := device.(type) {
	case types.BaseVirtualEthernetCard:
		controller = devices.PickController((*types.VirtualPCIController)(nil))
		var net types.ManagedObjectReference

		switch b := d.Backing.(type) {
		case *types.VirtualEthernetCardNetworkBackingInfo:
			summary = b.DeviceName
			net = Map.FindByName(b.DeviceName, dc.Network).Reference()
			b.Network = &net
		case *types.VirtualEthernetCardDistributedVirtualPortBackingInfo:
			summary = fmt.Sprintf("DVSwitch: %s", b.Port.SwitchUuid)
			net.Type = "DistributedVirtualPortgroup"
			net.Value = b.Port.PortgroupKey
		}

		vm.Network = append(vm.Network, net)

		c := x.GetVirtualEthernetCard()
		if c.MacAddress == "" {
			c.MacAddress = vm.generateMAC()
		}
	case *types.VirtualDisk:
		switch b := d.Backing.(type) {
		case types.BaseVirtualDeviceFileBackingInfo:
			info := b.GetVirtualDeviceFileBackingInfo()

			if info.FileName == "" {
				filename, err := vm.genVmdkPath()
				if err != nil {
					return err
				}

				info.FileName = filename
			}

			err := dm.createVirtualDisk(&types.CreateVirtualDisk_Task{
				Datacenter: &dc.Self,
				Name:       info.FileName,
			})
			if err != nil {
				return err
			}

			p, _ := parseDatastorePath(info.FileName)

			info.Datastore = &types.ManagedObjectReference{
				Type:  "Datastore",
				Value: p.Datastore,
			}
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

func removeDevice(devices object.VirtualDeviceList, device types.BaseVirtualDevice) object.VirtualDeviceList {
	var result object.VirtualDeviceList

	for i, d := range devices {
		if d.GetVirtualDevice().Key == device.GetVirtualDevice().Key {
			result = append(result, devices[i+1:]...)
			break
		}

		result = append(result, d)
	}

	return result
}

func (vm *VirtualMachine) genVmdkPath() (string, types.BaseMethodFault) {
	vmdir := path.Dir(vm.Config.Files.VmPathName)

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
			if devices.FindByKey(device.Key) != nil {
				if vm.Self.Value != "" { // moid isn't set until CreateVM is done
					return invalid
				}

				// In this case, the CreateVM() spec included one of the default devices
				devices = removeDevice(devices, device)
			}

			err := vm.configureDevice(devices, dspec.Device)
			if err != nil {
				return err
			}

			devices = append(devices, dspec.Device)
		case types.VirtualDeviceConfigSpecOperationRemove:
			devices = removeDevice(devices, dspec.Device)
		}
	}

	vm.Config.Hardware.Device = []types.BaseVirtualDevice(devices)

	return nil
}

type powerVMTask struct {
	*VirtualMachine

	state types.VirtualMachinePowerState
}

func (c *powerVMTask) Run(task *Task) (types.AnyType, types.BaseMethodFault) {
	c.log.Printf("running power task: requesting %s, existing %s",
		c.state, c.VirtualMachine.Runtime.PowerState)

	if c.VirtualMachine.Runtime.PowerState == c.state {
		return nil, &types.InvalidPowerState{
			RequestedState: c.state,
			ExistingState:  c.VirtualMachine.Runtime.PowerState,
		}
	}

	c.VirtualMachine.Runtime.PowerState = c.state
	c.VirtualMachine.Summary.Runtime.PowerState = c.state

	bt := &c.VirtualMachine.Summary.Runtime.BootTime
	if c.state == types.VirtualMachinePowerStatePoweredOn {
		now := time.Now()
		*bt = &now
	} else {
		*bt = nil
	}

	return nil, nil
}

func (vm *VirtualMachine) PowerOnVMTask(c *types.PowerOnVM_Task) soap.HasFault {
	runner := &powerVMTask{vm, types.VirtualMachinePowerStatePoweredOn}
	task := CreateTask(runner.Reference(), "powerOn", runner.Run)

	return &methods.PowerOnVM_TaskBody{
		Res: &types.PowerOnVM_TaskResponse{
			Returnval: task.Run(),
		},
	}
}

func (vm *VirtualMachine) PowerOffVMTask(c *types.PowerOffVM_Task) soap.HasFault {
	runner := &powerVMTask{vm, types.VirtualMachinePowerStatePoweredOff}
	task := CreateTask(runner.Reference(), "powerOff", runner.Run)

	return &methods.PowerOffVM_TaskBody{
		Res: &types.PowerOffVM_TaskResponse{
			Returnval: task.Run(),
		},
	}
}

func (vm *VirtualMachine) ReconfigVMTask(req *types.ReconfigVM_Task) soap.HasFault {
	task := CreateTask(vm, "reconfigVm", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		err := vm.configure(&req.Spec)
		if err != nil {
			return nil, err
		}

		return nil, nil
	})

	return &methods.ReconfigVM_TaskBody{
		Res: &types.ReconfigVM_TaskResponse{
			Returnval: task.Run(),
		},
	}
}

func (vm *VirtualMachine) DestroyTask(req *types.Destroy_Task) soap.HasFault {
	task := CreateTask(vm, "destroy", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		r := vm.UnregisterVM(&types.UnregisterVM{
			This: req.This,
		})

		if r.Fault() != nil {
			return nil, r.Fault().VimFault().(types.BaseMethodFault)
		}

		// Delete VM files from the datastore (ignoring result for now)
		m := Map.FileManager()
		dc := Map.getEntityDatacenter(vm).Reference()

		_ = m.DeleteDatastoreFileTask(&types.DeleteDatastoreFile_Task{
			This:       m.Reference(),
			Name:       vm.Config.Files.LogDirectory,
			Datacenter: &dc,
		})

		return nil, nil
	})

	return &methods.Destroy_TaskBody{
		Res: &types.Destroy_TaskResponse{
			Returnval: task.Run(),
		},
	}
}

func (vm *VirtualMachine) UnregisterVM(c *types.UnregisterVM) soap.HasFault {
	r := &methods.UnregisterVMBody{}

	if vm.Runtime.PowerState == types.VirtualMachinePowerStatePoweredOn {
		r.Fault_ = Fault("", &types.InvalidPowerState{
			RequestedState: types.VirtualMachinePowerStatePoweredOff,
			ExistingState:  vm.Runtime.PowerState,
		})

		return r
	}

	_ = vm.out.Close() // Close log fd

	Map.getEntityParent(vm, "Folder").(*Folder).removeChild(c.This)

	host := Map.Get(*vm.Runtime.Host).(*HostSystem)
	host.Vm = RemoveReference(vm.Self, host.Vm)

	switch pool := Map.Get(*vm.ResourcePool).(type) {
	case *ResourcePool:
		pool.Vm = RemoveReference(vm.Self, pool.Vm)
	case *VirtualApp:
		pool.Vm = RemoveReference(vm.Self, pool.Vm)
	}

	for i := range vm.Datastore {
		ds := Map.Get(vm.Datastore[i]).(*Datastore)
		ds.Vm = RemoveReference(vm.Self, ds.Vm)
	}

	r.Res = new(types.UnregisterVMResponse)

	return r
}

func (vm *VirtualMachine) CloneVMTask(req *types.CloneVM_Task) soap.HasFault {
	task := CreateTask(vm, "cloneVm", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		folder := Map.Get(req.Folder).(*Folder)

		config := types.VirtualMachineConfigSpec{
			Name:    req.Name,
			GuestId: vm.Config.GuestId,
			Files: &types.VirtualMachineFileInfo{
				VmPathName: strings.Replace(vm.Config.Files.VmPathName, vm.Name, req.Name, -1),
			},
		}

		res := folder.CreateVMTask(&types.CreateVM_Task{
			This:   folder.Self,
			Config: config,
			Pool:   *vm.ResourcePool,
		})

		ctask := Map.Get(res.(*methods.CreateVM_TaskBody).Res.Returnval).(*Task)
		if ctask.Info.Error != nil {
			return nil, ctask.Info.Error.Fault
		}

		return ctask.Info.Result.(types.ManagedObjectReference), nil
	})

	return &methods.CloneVM_TaskBody{
		Res: &types.CloneVM_TaskResponse{
			Returnval: task.Run(),
		},
	}
}

func (vm *VirtualMachine) RelocateVMTask(req *types.RelocateVM_Task) soap.HasFault {
	task := CreateTask(vm, "relocateVm", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		if ref := req.Spec.Datastore; ref != nil {
			ds := Map.Get(*ref).(*Datastore)
			ds.Vm = RemoveReference(*ref, ds.Vm)

			vm.Datastore = []types.ManagedObjectReference{*ref}

			// TODO: migrate vm.Config.Files (and vm.Summary.Config.VmPathName)
		}

		if ref := req.Spec.Pool; ref != nil {
			pool := Map.Get(*ref).(*ResourcePool)
			pool.Vm = RemoveReference(*ref, pool.Vm)

			vm.ResourcePool = ref
		}

		if ref := req.Spec.Host; ref != nil {
			host := Map.Get(*ref).(*HostSystem)
			host.Vm = RemoveReference(*ref, host.Vm)

			vm.Runtime.Host = ref
		}

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
			vm.Snapshot.RootSnapshotList = append(vm.Snapshot.RootSnapshotList, treeItem)
		}

		vm.Snapshot.CurrentSnapshot = &snapshot.Self

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

func (vm *VirtualMachine) RemoveAllSnapshotsTask(req *types.RemoveAllSnapshots_Task) soap.HasFault {
	task := CreateTask(vm, "RemoveAllSnapshots", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		if vm.Snapshot == nil {
			return nil, nil
		}

		refs := allSnapshotsInTree(vm.Snapshot.RootSnapshotList)

		vm.Snapshot.CurrentSnapshot = nil
		vm.Snapshot.RootSnapshotList = nil

		for _, ref := range refs {
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

func (vm *VirtualMachine) ShutdownGuest(c *types.ShutdownGuest) soap.HasFault {
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

	r.Res = new(types.ShutdownGuestResponse)

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

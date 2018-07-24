/*
Copyright (c) 2015-2017 VMware, Inc. All Rights Reserved.

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

package object

import (
	"errors"
	"fmt"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strings"

	"github.com/vmware/govmomi/vim25/types"
)

// Type values for use in BootOrder
const (
	DeviceTypeNone     = "-"
	DeviceTypeCdrom    = "cdrom"
	DeviceTypeDisk     = "disk"
	DeviceTypeEthernet = "ethernet"
	DeviceTypeFloppy   = "floppy"
)

// VirtualDeviceList provides helper methods for working with a list of virtual devices.
type VirtualDeviceList []types.BaseVirtualDevice

// SCSIControllerTypes are used for adding a new SCSI controller to a VM.
func SCSIControllerTypes() VirtualDeviceList {
	// Return a mutable list of SCSI controller types, initialized with defaults.
	return VirtualDeviceList([]types.BaseVirtualDevice{
		&types.VirtualLsiLogicController{},
		&types.VirtualBusLogicController{},
		&types.ParaVirtualSCSIController{},
		&types.VirtualLsiLogicSASController{},
	}).Select(func(device types.BaseVirtualDevice) bool {
		c := device.(types.BaseVirtualSCSIController).GetVirtualSCSIController()
		c.SharedBus = types.VirtualSCSISharingNoSharing
		c.BusNumber = -1
		return true
	})
}

// EthernetCardTypes are used for adding a new ethernet card to a VM.
func EthernetCardTypes() VirtualDeviceList {
	return VirtualDeviceList([]types.BaseVirtualDevice{
		&types.VirtualE1000{},
		&types.VirtualE1000e{},
		&types.VirtualVmxnet2{},
		&types.VirtualVmxnet3{},
		&types.VirtualPCNet32{},
		&types.VirtualSriovEthernetCard{},
	}).Select(func(device types.BaseVirtualDevice) bool {
		c := device.(types.BaseVirtualEthernetCard).GetVirtualEthernetCard()
		c.GetVirtualDevice().Key = -1
		return true
	})
}

// Select returns a new list containing all elements of the list for which the given func returns true.
func (l VirtualDeviceList) Select(f func(device types.BaseVirtualDevice) bool) VirtualDeviceList {
	var found VirtualDeviceList

	for _, device := range l {
		if f(device) {
			found = append(found, device)
		}
	}

	return found
}

// SelectByType returns a new list with devices that are equal to or extend the given type.
func (l VirtualDeviceList) SelectByType(deviceType types.BaseVirtualDevice) VirtualDeviceList {
	dtype := reflect.TypeOf(deviceType)
	if dtype == nil {
		return nil
	}
	dname := dtype.Elem().Name()

	return l.Select(func(device types.BaseVirtualDevice) bool {
		t := reflect.TypeOf(device)

		if t == dtype {
			return true
		}

		_, ok := t.Elem().FieldByName(dname)

		return ok
	})
}

// SelectByBackingInfo returns a new list with devices matching the given backing info.
// If the value of backing is nil, any device with a backing of the same type will be returned.
func (l VirtualDeviceList) SelectByBackingInfo(backing types.BaseVirtualDeviceBackingInfo) VirtualDeviceList {
	t := reflect.TypeOf(backing)

	return l.Select(func(device types.BaseVirtualDevice) bool {
		db := device.GetVirtualDevice().Backing
		if db == nil {
			return false
		}

		if reflect.TypeOf(db) != t {
			return false
		}

		if reflect.ValueOf(backing).IsNil() {
			// selecting by backing type
			return true
		}

		switch a := db.(type) {
		case *types.VirtualEthernetCardNetworkBackingInfo:
			b := backing.(*types.VirtualEthernetCardNetworkBackingInfo)
			return a.DeviceName == b.DeviceName
		case *types.VirtualEthernetCardDistributedVirtualPortBackingInfo:
			b := backing.(*types.VirtualEthernetCardDistributedVirtualPortBackingInfo)
			return a.Port.SwitchUuid == b.Port.SwitchUuid &&
				a.Port.PortgroupKey == b.Port.PortgroupKey
		case *types.VirtualDiskFlatVer2BackingInfo:
			b := backing.(*types.VirtualDiskFlatVer2BackingInfo)
			if a.Parent != nil && b.Parent != nil {
				return a.Parent.FileName == b.Parent.FileName
			}
			return a.FileName == b.FileName
		case *types.VirtualSerialPortURIBackingInfo:
			b := backing.(*types.VirtualSerialPortURIBackingInfo)
			return a.ServiceURI == b.ServiceURI
		case types.BaseVirtualDeviceFileBackingInfo:
			b := backing.(types.BaseVirtualDeviceFileBackingInfo)
			return a.GetVirtualDeviceFileBackingInfo().FileName == b.GetVirtualDeviceFileBackingInfo().FileName
		default:
			return false
		}
	})
}

// Find returns the device matching the given name.
func (l VirtualDeviceList) Find(name string) types.BaseVirtualDevice {
	for _, device := range l {
		if l.Name(device) == name {
			return device
		}
	}
	return nil
}

// FindByKey returns the device matching the given key.
func (l VirtualDeviceList) FindByKey(key int32) types.BaseVirtualDevice {
	for _, device := range l {
		if device.GetVirtualDevice().Key == key {
			return device
		}
	}
	return nil
}

// FindIDEController will find the named IDE controller if given, otherwise will pick an available controller.
// An error is returned if the named controller is not found or not an IDE controller.  Or, if name is not
// given and no available controller can be found.
func (l VirtualDeviceList) FindIDEController(name string) (*types.VirtualIDEController, error) {
	if name != "" {
		d := l.Find(name)
		if d == nil {
			return nil, fmt.Errorf("device '%s' not found", name)
		}
		if c, ok := d.(*types.VirtualIDEController); ok {
			return c, nil
		}
		return nil, fmt.Errorf("%s is not an IDE controller", name)
	}

	c := l.PickController((*types.VirtualIDEController)(nil))
	if c == nil {
		return nil, errors.New("no available IDE controller")
	}

	return c.(*types.VirtualIDEController), nil
}

// CreateIDEController creates a new IDE controller.
func (l VirtualDeviceList) CreateIDEController() (types.BaseVirtualDevice, error) {
	ide := &types.VirtualIDEController{}
	ide.Key = l.NewKey()
	return ide, nil
}

// FindSCSIController will find the named SCSI controller if given, otherwise will pick an available controller.
// An error is returned if the named controller is not found or not an SCSI controller.  Or, if name is not
// given and no available controller can be found.
func (l VirtualDeviceList) FindSCSIController(name string) (*types.VirtualSCSIController, error) {
	if name != "" {
		d := l.Find(name)
		if d == nil {
			return nil, fmt.Errorf("device '%s' not found", name)
		}
		if c, ok := d.(types.BaseVirtualSCSIController); ok {
			return c.GetVirtualSCSIController(), nil
		}
		return nil, fmt.Errorf("%s is not an SCSI controller", name)
	}

	c := l.PickController((*types.VirtualSCSIController)(nil))
	if c == nil {
		return nil, errors.New("no available SCSI controller")
	}

	return c.(types.BaseVirtualSCSIController).GetVirtualSCSIController(), nil
}

// CreateSCSIController creates a new SCSI controller of type name if given, otherwise defaults to lsilogic.
func (l VirtualDeviceList) CreateSCSIController(name string) (types.BaseVirtualDevice, error) {
	ctypes := SCSIControllerTypes()

	if name == "scsi" || name == "" {
		name = ctypes.Type(ctypes[0])
	}

	found := ctypes.Select(func(device types.BaseVirtualDevice) bool {
		return l.Type(device) == name
	})

	if len(found) == 0 {
		return nil, fmt.Errorf("unknown SCSI controller type '%s'", name)
	}

	c, ok := found[0].(types.BaseVirtualSCSIController)
	if !ok {
		return nil, fmt.Errorf("invalid SCSI controller type '%s'", name)
	}

	scsi := c.GetVirtualSCSIController()
	scsi.BusNumber = l.newSCSIBusNumber()
	scsi.Key = l.NewKey()
	scsi.ScsiCtlrUnitNumber = 7
	return c.(types.BaseVirtualDevice), nil
}

var scsiBusNumbers = []int{0, 1, 2, 3}

// newSCSIBusNumber returns the bus number to use for adding a new SCSI bus device.
// -1 is returned if there are no bus numbers available.
func (l VirtualDeviceList) newSCSIBusNumber() int32 {
	var used []int

	for _, d := range l.SelectByType((*types.VirtualSCSIController)(nil)) {
		num := d.(types.BaseVirtualSCSIController).GetVirtualSCSIController().BusNumber
		if num >= 0 {
			used = append(used, int(num))
		} // else caller is creating a new vm using SCSIControllerTypes
	}

	sort.Ints(used)

	for i, n := range scsiBusNumbers {
		if i == len(used) || n != used[i] {
			return int32(n)
		}
	}

	return -1
}

// FindNVMEController will find the named NVME controller if given, otherwise will pick an available controller.
// An error is returned if the named controller is not found or not an NVME controller.  Or, if name is not
// given and no available controller can be found.
func (l VirtualDeviceList) FindNVMEController(name string) (*types.VirtualNVMEController, error) {
	if name != "" {
		d := l.Find(name)
		if d == nil {
			return nil, fmt.Errorf("device '%s' not found", name)
		}
		if c, ok := d.(*types.VirtualNVMEController); ok {
			return c, nil
		}
		return nil, fmt.Errorf("%s is not an NVME controller", name)
	}

	c := l.PickController((*types.VirtualNVMEController)(nil))
	if c == nil {
		return nil, errors.New("no available NVME controller")
	}

	return c.(*types.VirtualNVMEController), nil
}

// CreateNVMEController creates a new NVMWE controller.
func (l VirtualDeviceList) CreateNVMEController() (types.BaseVirtualDevice, error) {
	nvme := &types.VirtualNVMEController{}
	nvme.BusNumber = l.newNVMEBusNumber()
	nvme.Key = l.NewKey()

	return nvme, nil
}

var nvmeBusNumbers = []int{0, 1, 2, 3}

// newNVMEBusNumber returns the bus number to use for adding a new NVME bus device.
// -1 is returned if there are no bus numbers available.
func (l VirtualDeviceList) newNVMEBusNumber() int32 {
	var used []int

	for _, d := range l.SelectByType((*types.VirtualNVMEController)(nil)) {
		num := d.(types.BaseVirtualController).GetVirtualController().BusNumber
		if num >= 0 {
			used = append(used, int(num))
		} // else caller is creating a new vm using NVMEControllerTypes
	}

	sort.Ints(used)

	for i, n := range nvmeBusNumbers {
		if i == len(used) || n != used[i] {
			return int32(n)
		}
	}

	return -1
}

// FindDiskController will find an existing ide or scsi disk controller.
func (l VirtualDeviceList) FindDiskController(name string) (types.BaseVirtualController, error) {
	switch {
	case name == "ide":
		return l.FindIDEController("")
	case name == "scsi" || name == "":
		return l.FindSCSIController("")
	case name == "nvme":
		return l.FindNVMEController("")
	default:
		if c, ok := l.Find(name).(types.BaseVirtualController); ok {
			return c, nil
		}
		return nil, fmt.Errorf("%s is not a valid controller", name)
	}
}

// PickController returns a controller of the given type(s).
// If no controllers are found or have no available slots, then nil is returned.
func (l VirtualDeviceList) PickController(kind types.BaseVirtualController) types.BaseVirtualController {
	l = l.SelectByType(kind.(types.BaseVirtualDevice)).Select(func(device types.BaseVirtualDevice) bool {
		num := len(device.(types.BaseVirtualController).GetVirtualController().Device)

		switch device.(type) {
		case types.BaseVirtualSCSIController:
			return num < 15
		case *types.VirtualIDEController:
			return num < 2
		case *types.VirtualNVMEController:
			return num < 8
		default:
			return true
		}
	})

	if len(l) == 0 {
		return nil
	}

	return l[0].(types.BaseVirtualController)
}

// newUnitNumber returns the unit number to use for attaching a new device to the given controller.
func (l VirtualDeviceList) newUnitNumber(c types.BaseVirtualController) int32 {
	units := make([]bool, 30)

	switch sc := c.(type) {
	case types.BaseVirtualSCSIController:
		//  The SCSI controller sits on its own bus
		units[sc.GetVirtualSCSIController().ScsiCtlrUnitNumber] = true
	}

	key := c.GetVirtualController().Key

	for _, device := range l {
		d := device.GetVirtualDevice()

		if d.ControllerKey == key && d.UnitNumber != nil {
			units[int(*d.UnitNumber)] = true
		}
	}

	for unit, used := range units {
		if !used {
			return int32(unit)
		}
	}

	return -1
}

// NewKey returns the key to use for adding a new device to the device list.
// The device list we're working with here may not be complete (e.g. when
// we're only adding new devices), so any positive keys could conflict with device keys
// that are already in use. To avoid this type of conflict, we can use negative keys
// here, which will be resolved to positive keys by vSphere as the reconfiguration is done.
func (l VirtualDeviceList) NewKey() int32 {
	var key int32 = -200

	for _, device := range l {
		d := device.GetVirtualDevice()
		if d.Key < key {
			key = d.Key
		}
	}

	return key - 1
}

// AssignController assigns a device to a controller.
func (l VirtualDeviceList) AssignController(device types.BaseVirtualDevice, c types.BaseVirtualController) {
	d := device.GetVirtualDevice()
	d.ControllerKey = c.GetVirtualController().Key
	d.UnitNumber = new(int32)
	*d.UnitNumber = l.newUnitNumber(c)
	if d.Key == 0 {
		d.Key = -1
	}
}

// CreateDisk creates a new VirtualDisk device which can be added to a VM.
func (l VirtualDeviceList) CreateDisk(c types.BaseVirtualController, ds types.ManagedObjectReference, name string) *types.VirtualDisk {
	// If name is not specified, one will be chosen for you.
	// But if when given, make sure it ends in .vmdk, otherwise it will be treated as a directory.
	if len(name) > 0 && filepath.Ext(name) != ".vmdk" {
		name += ".vmdk"
	}

	device := &types.VirtualDisk{
		VirtualDevice: types.VirtualDevice{
			Backing: &types.VirtualDiskFlatVer2BackingInfo{
				DiskMode:        string(types.VirtualDiskModePersistent),
				ThinProvisioned: types.NewBool(true),
				VirtualDeviceFileBackingInfo: types.VirtualDeviceFileBackingInfo{
					FileName:  name,
					Datastore: &ds,
				},
			},
		},
	}

	l.AssignController(device, c)
	return device
}

// ChildDisk creates a new VirtualDisk device, linked to the given parent disk, which can be added to a VM.
func (l VirtualDeviceList) ChildDisk(parent *types.VirtualDisk) *types.VirtualDisk {
	disk := *parent
	backing := disk.Backing.(*types.VirtualDiskFlatVer2BackingInfo)
	p := new(DatastorePath)
	p.FromString(backing.FileName)
	p.Path = ""

	// Use specified disk as parent backing to a new disk.
	disk.Backing = &types.VirtualDiskFlatVer2BackingInfo{
		VirtualDeviceFileBackingInfo: types.VirtualDeviceFileBackingInfo{
			FileName:  p.String(),
			Datastore: backing.Datastore,
		},
		Parent:          backing,
		DiskMode:        backing.DiskMode,
		ThinProvisioned: backing.ThinProvisioned,
	}

	return &disk
}

func (l VirtualDeviceList) connectivity(device types.BaseVirtualDevice, v bool) error {
	c := device.GetVirtualDevice().Connectable
	if c == nil {
		return fmt.Errorf("%s is not connectable", l.Name(device))
	}

	c.Connected = v
	c.StartConnected = v

	return nil
}

// Connect changes the device to connected, returns an error if the device is not connectable.
func (l VirtualDeviceList) Connect(device types.BaseVirtualDevice) error {
	return l.connectivity(device, true)
}

// Disconnect changes the device to disconnected, returns an error if the device is not connectable.
func (l VirtualDeviceList) Disconnect(device types.BaseVirtualDevice) error {
	return l.connectivity(device, false)
}

// FindCdrom finds a cdrom device with the given name, defaulting to the first cdrom device if any.
func (l VirtualDeviceList) FindCdrom(name string) (*types.VirtualCdrom, error) {
	if name != "" {
		d := l.Find(name)
		if d == nil {
			return nil, fmt.Errorf("device '%s' not found", name)
		}
		if c, ok := d.(*types.VirtualCdrom); ok {
			return c, nil
		}
		return nil, fmt.Errorf("%s is not a cdrom device", name)
	}

	c := l.SelectByType((*types.VirtualCdrom)(nil))
	if len(c) == 0 {
		return nil, errors.New("no cdrom device found")
	}

	return c[0].(*types.VirtualCdrom), nil
}

// CreateCdrom creates a new VirtualCdrom device which can be added to a VM.
func (l VirtualDeviceList) CreateCdrom(c *types.VirtualIDEController) (*types.VirtualCdrom, error) {
	device := &types.VirtualCdrom{}

	l.AssignController(device, c)

	l.setDefaultCdromBacking(device)

	device.Connectable = &types.VirtualDeviceConnectInfo{
		AllowGuestControl: true,
		Connected:         true,
		StartConnected:    true,
	}

	return device, nil
}

// InsertIso changes the cdrom device backing to use the given iso file.
func (l VirtualDeviceList) InsertIso(device *types.VirtualCdrom, iso string) *types.VirtualCdrom {
	device.Backing = &types.VirtualCdromIsoBackingInfo{
		VirtualDeviceFileBackingInfo: types.VirtualDeviceFileBackingInfo{
			FileName: iso,
		},
	}

	return device
}

// EjectIso removes the iso file based backing and replaces with the default cdrom backing.
func (l VirtualDeviceList) EjectIso(device *types.VirtualCdrom) *types.VirtualCdrom {
	l.setDefaultCdromBacking(device)
	return device
}

func (l VirtualDeviceList) setDefaultCdromBacking(device *types.VirtualCdrom) {
	device.Backing = &types.VirtualCdromAtapiBackingInfo{
		VirtualDeviceDeviceBackingInfo: types.VirtualDeviceDeviceBackingInfo{
			DeviceName:    fmt.Sprintf("%s-%d-%d", DeviceTypeCdrom, device.ControllerKey, device.UnitNumber),
			UseAutoDetect: types.NewBool(false),
		},
	}
}

// FindFloppy finds a floppy device with the given name, defaulting to the first floppy device if any.
func (l VirtualDeviceList) FindFloppy(name string) (*types.VirtualFloppy, error) {
	if name != "" {
		d := l.Find(name)
		if d == nil {
			return nil, fmt.Errorf("device '%s' not found", name)
		}
		if c, ok := d.(*types.VirtualFloppy); ok {
			return c, nil
		}
		return nil, fmt.Errorf("%s is not a floppy device", name)
	}

	c := l.SelectByType((*types.VirtualFloppy)(nil))
	if len(c) == 0 {
		return nil, errors.New("no floppy device found")
	}

	return c[0].(*types.VirtualFloppy), nil
}

// CreateFloppy creates a new VirtualFloppy device which can be added to a VM.
func (l VirtualDeviceList) CreateFloppy() (*types.VirtualFloppy, error) {
	device := &types.VirtualFloppy{}

	c := l.PickController((*types.VirtualSIOController)(nil))
	if c == nil {
		return nil, errors.New("no available SIO controller")
	}

	l.AssignController(device, c)

	l.setDefaultFloppyBacking(device)

	device.Connectable = &types.VirtualDeviceConnectInfo{
		AllowGuestControl: true,
		Connected:         true,
		StartConnected:    true,
	}

	return device, nil
}

// InsertImg changes the floppy device backing to use the given img file.
func (l VirtualDeviceList) InsertImg(device *types.VirtualFloppy, img string) *types.VirtualFloppy {
	device.Backing = &types.VirtualFloppyImageBackingInfo{
		VirtualDeviceFileBackingInfo: types.VirtualDeviceFileBackingInfo{
			FileName: img,
		},
	}

	return device
}

// EjectImg removes the img file based backing and replaces with the default floppy backing.
func (l VirtualDeviceList) EjectImg(device *types.VirtualFloppy) *types.VirtualFloppy {
	l.setDefaultFloppyBacking(device)
	return device
}

func (l VirtualDeviceList) setDefaultFloppyBacking(device *types.VirtualFloppy) {
	device.Backing = &types.VirtualFloppyDeviceBackingInfo{
		VirtualDeviceDeviceBackingInfo: types.VirtualDeviceDeviceBackingInfo{
			DeviceName:    fmt.Sprintf("%s-%d", DeviceTypeFloppy, device.UnitNumber),
			UseAutoDetect: types.NewBool(false),
		},
	}
}

// FindSerialPort finds a serial port device with the given name, defaulting to the first serial port device if any.
func (l VirtualDeviceList) FindSerialPort(name string) (*types.VirtualSerialPort, error) {
	if name != "" {
		d := l.Find(name)
		if d == nil {
			return nil, fmt.Errorf("device '%s' not found", name)
		}
		if c, ok := d.(*types.VirtualSerialPort); ok {
			return c, nil
		}
		return nil, fmt.Errorf("%s is not a serial port device", name)
	}

	c := l.SelectByType((*types.VirtualSerialPort)(nil))
	if len(c) == 0 {
		return nil, errors.New("no serial port device found")
	}

	return c[0].(*types.VirtualSerialPort), nil
}

// CreateSerialPort creates a new VirtualSerialPort device which can be added to a VM.
func (l VirtualDeviceList) CreateSerialPort() (*types.VirtualSerialPort, error) {
	device := &types.VirtualSerialPort{
		YieldOnPoll: true,
	}

	c := l.PickController((*types.VirtualSIOController)(nil))
	if c == nil {
		return nil, errors.New("no available SIO controller")
	}

	l.AssignController(device, c)

	l.setDefaultSerialPortBacking(device)

	return device, nil
}

// ConnectSerialPort connects a serial port to a server or client uri.
func (l VirtualDeviceList) ConnectSerialPort(device *types.VirtualSerialPort, uri string, client bool, proxyuri string) *types.VirtualSerialPort {
	if strings.HasPrefix(uri, "[") {
		device.Backing = &types.VirtualSerialPortFileBackingInfo{
			VirtualDeviceFileBackingInfo: types.VirtualDeviceFileBackingInfo{
				FileName: uri,
			},
		}

		return device
	}

	direction := types.VirtualDeviceURIBackingOptionDirectionServer
	if client {
		direction = types.VirtualDeviceURIBackingOptionDirectionClient
	}

	device.Backing = &types.VirtualSerialPortURIBackingInfo{
		VirtualDeviceURIBackingInfo: types.VirtualDeviceURIBackingInfo{
			Direction:  string(direction),
			ServiceURI: uri,
			ProxyURI:   proxyuri,
		},
	}

	return device
}

// DisconnectSerialPort disconnects the serial port backing.
func (l VirtualDeviceList) DisconnectSerialPort(device *types.VirtualSerialPort) *types.VirtualSerialPort {
	l.setDefaultSerialPortBacking(device)
	return device
}

func (l VirtualDeviceList) setDefaultSerialPortBacking(device *types.VirtualSerialPort) {
	device.Backing = &types.VirtualSerialPortURIBackingInfo{
		VirtualDeviceURIBackingInfo: types.VirtualDeviceURIBackingInfo{
			Direction:  "client",
			ServiceURI: "localhost:0",
		},
	}
}

// CreateEthernetCard creates a new VirtualEthernetCard of the given name name and initialized with the given backing.
func (l VirtualDeviceList) CreateEthernetCard(name string, backing types.BaseVirtualDeviceBackingInfo) (types.BaseVirtualDevice, error) {
	ctypes := EthernetCardTypes()

	if name == "" {
		name = ctypes.deviceName(ctypes[0])
	}

	found := ctypes.Select(func(device types.BaseVirtualDevice) bool {
		return l.deviceName(device) == name
	})

	if len(found) == 0 {
		return nil, fmt.Errorf("unknown ethernet card type '%s'", name)
	}

	c, ok := found[0].(types.BaseVirtualEthernetCard)
	if !ok {
		return nil, fmt.Errorf("invalid ethernet card type '%s'", name)
	}

	c.GetVirtualEthernetCard().Backing = backing

	return c.(types.BaseVirtualDevice), nil
}

// PrimaryMacAddress returns the MacAddress field of the primary VirtualEthernetCard
func (l VirtualDeviceList) PrimaryMacAddress() string {
	eth0 := l.Find("ethernet-0")

	if eth0 == nil {
		return ""
	}

	return eth0.(types.BaseVirtualEthernetCard).GetVirtualEthernetCard().MacAddress
}

// convert a BaseVirtualDevice to a BaseVirtualMachineBootOptionsBootableDevice
var bootableDevices = map[string]func(device types.BaseVirtualDevice) types.BaseVirtualMachineBootOptionsBootableDevice{
	DeviceTypeNone: func(types.BaseVirtualDevice) types.BaseVirtualMachineBootOptionsBootableDevice {
		return &types.VirtualMachineBootOptionsBootableDevice{}
	},
	DeviceTypeCdrom: func(types.BaseVirtualDevice) types.BaseVirtualMachineBootOptionsBootableDevice {
		return &types.VirtualMachineBootOptionsBootableCdromDevice{}
	},
	DeviceTypeDisk: func(d types.BaseVirtualDevice) types.BaseVirtualMachineBootOptionsBootableDevice {
		return &types.VirtualMachineBootOptionsBootableDiskDevice{
			DeviceKey: d.GetVirtualDevice().Key,
		}
	},
	DeviceTypeEthernet: func(d types.BaseVirtualDevice) types.BaseVirtualMachineBootOptionsBootableDevice {
		return &types.VirtualMachineBootOptionsBootableEthernetDevice{
			DeviceKey: d.GetVirtualDevice().Key,
		}
	},
	DeviceTypeFloppy: func(types.BaseVirtualDevice) types.BaseVirtualMachineBootOptionsBootableDevice {
		return &types.VirtualMachineBootOptionsBootableFloppyDevice{}
	},
}

// BootOrder returns a list of devices which can be used to set boot order via VirtualMachine.SetBootOptions.
// The order can be any of "ethernet", "cdrom", "floppy" or "disk" or by specific device name.
// A value of "-" will clear the existing boot order on the VC/ESX side.
func (l VirtualDeviceList) BootOrder(order []string) []types.BaseVirtualMachineBootOptionsBootableDevice {
	var devices []types.BaseVirtualMachineBootOptionsBootableDevice

	for _, name := range order {
		if kind, ok := bootableDevices[name]; ok {
			if name == DeviceTypeNone {
				// Not covered in the API docs, nor obvious, but this clears the boot order on the VC/ESX side.
				devices = append(devices, new(types.VirtualMachineBootOptionsBootableDevice))
				continue
			}

			for _, device := range l {
				if l.Type(device) == name {
					devices = append(devices, kind(device))
				}
			}
			continue
		}

		if d := l.Find(name); d != nil {
			if kind, ok := bootableDevices[l.Type(d)]; ok {
				devices = append(devices, kind(d))
			}
		}
	}

	return devices
}

// SelectBootOrder returns an ordered list of devices matching the given bootable device order
func (l VirtualDeviceList) SelectBootOrder(order []types.BaseVirtualMachineBootOptionsBootableDevice) VirtualDeviceList {
	var devices VirtualDeviceList

	for _, bd := range order {
		for _, device := range l {
			if kind, ok := bootableDevices[l.Type(device)]; ok {
				if reflect.DeepEqual(kind(device), bd) {
					devices = append(devices, device)
				}
			}
		}
	}

	return devices
}

// TypeName returns the vmodl type name of the device
func (l VirtualDeviceList) TypeName(device types.BaseVirtualDevice) string {
	dtype := reflect.TypeOf(device)
	if dtype == nil {
		return ""
	}
	return dtype.Elem().Name()
}

var deviceNameRegexp = regexp.MustCompile(`(?:Virtual)?(?:Machine)?(\w+?)(?:Card|EthernetCard|Device|Controller)?$`)

func (l VirtualDeviceList) deviceName(device types.BaseVirtualDevice) string {
	name := "device"
	typeName := l.TypeName(device)

	m := deviceNameRegexp.FindStringSubmatch(typeName)
	if len(m) == 2 {
		name = strings.ToLower(m[1])
	}

	return name
}

// Type returns a human-readable name for the given device
func (l VirtualDeviceList) Type(device types.BaseVirtualDevice) string {
	switch device.(type) {
	case types.BaseVirtualEthernetCard:
		return DeviceTypeEthernet
	case *types.ParaVirtualSCSIController:
		return "pvscsi"
	case *types.VirtualLsiLogicSASController:
		return "lsilogic-sas"
	case *types.VirtualNVMEController:
		return "nvme"
	default:
		return l.deviceName(device)
	}
}

// Name returns a stable, human-readable name for the given device
func (l VirtualDeviceList) Name(device types.BaseVirtualDevice) string {
	var key string
	var UnitNumber int32
	d := device.GetVirtualDevice()
	if d.UnitNumber != nil {
		UnitNumber = *d.UnitNumber
	}

	dtype := l.Type(device)
	switch dtype {
	case DeviceTypeEthernet:
		key = fmt.Sprintf("%d", UnitNumber-7)
	case DeviceTypeDisk:
		key = fmt.Sprintf("%d-%d", d.ControllerKey, UnitNumber)
	default:
		key = fmt.Sprintf("%d", d.Key)
	}

	return fmt.Sprintf("%s-%s", dtype, key)
}

// ConfigSpec creates a virtual machine configuration spec for
// the specified operation, for the list of devices in the device list.
func (l VirtualDeviceList) ConfigSpec(op types.VirtualDeviceConfigSpecOperation) ([]types.BaseVirtualDeviceConfigSpec, error) {
	var fop types.VirtualDeviceConfigSpecFileOperation
	switch op {
	case types.VirtualDeviceConfigSpecOperationAdd:
		fop = types.VirtualDeviceConfigSpecFileOperationCreate
	case types.VirtualDeviceConfigSpecOperationEdit:
		fop = types.VirtualDeviceConfigSpecFileOperationReplace
	case types.VirtualDeviceConfigSpecOperationRemove:
		fop = types.VirtualDeviceConfigSpecFileOperationDestroy
	default:
		panic("unknown op")
	}

	var res []types.BaseVirtualDeviceConfigSpec
	for _, device := range l {
		config := &types.VirtualDeviceConfigSpec{
			Device:    device,
			Operation: op,
		}

		if disk, ok := device.(*types.VirtualDisk); ok {
			config.FileOperation = fop

			// Special case to attach an existing disk
			if op == types.VirtualDeviceConfigSpecOperationAdd && disk.CapacityInKB == 0 {
				childDisk := false
				if b, ok := disk.Backing.(*types.VirtualDiskFlatVer2BackingInfo); ok {
					childDisk = b.Parent != nil
				}

				if !childDisk {
					// Existing disk, clear file operation
					config.FileOperation = ""
				}
			}
		}

		res = append(res, config)
	}

	return res, nil
}

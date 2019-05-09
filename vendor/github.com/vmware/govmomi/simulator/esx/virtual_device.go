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

package esx

import "github.com/vmware/govmomi/vim25/types"

// VirtualDevice is the default set of VirtualDevice types created for a VirtualMachine
// Capture method:
//   govc vm.create foo
//   govc object.collect -s -dump vm/foo config.hardware.device
var VirtualDevice = []types.BaseVirtualDevice{
	&types.VirtualIDEController{
		VirtualController: types.VirtualController{
			VirtualDevice: types.VirtualDevice{
				DynamicData: types.DynamicData{},
				Key:         200,
				DeviceInfo: &types.Description{
					DynamicData: types.DynamicData{},
					Label:       "IDE 0",
					Summary:     "IDE 0",
				},
				Backing:       nil,
				Connectable:   (*types.VirtualDeviceConnectInfo)(nil),
				SlotInfo:      nil,
				ControllerKey: 0,
				UnitNumber:    (*int32)(nil),
			},
			BusNumber: 0,
			Device:    nil,
		},
	},
	&types.VirtualIDEController{
		VirtualController: types.VirtualController{
			VirtualDevice: types.VirtualDevice{
				DynamicData: types.DynamicData{},
				Key:         201,
				DeviceInfo: &types.Description{
					DynamicData: types.DynamicData{},
					Label:       "IDE 1",
					Summary:     "IDE 1",
				},
				Backing:       nil,
				Connectable:   (*types.VirtualDeviceConnectInfo)(nil),
				SlotInfo:      nil,
				ControllerKey: 0,
				UnitNumber:    (*int32)(nil),
			},
			BusNumber: 1,
			Device:    nil,
		},
	},
	&types.VirtualPS2Controller{
		VirtualController: types.VirtualController{
			VirtualDevice: types.VirtualDevice{
				DynamicData: types.DynamicData{},
				Key:         300,
				DeviceInfo: &types.Description{
					DynamicData: types.DynamicData{},
					Label:       "PS2 controller 0",
					Summary:     "PS2 controller 0",
				},
				Backing:       nil,
				Connectable:   (*types.VirtualDeviceConnectInfo)(nil),
				SlotInfo:      nil,
				ControllerKey: 0,
				UnitNumber:    (*int32)(nil),
			},
			BusNumber: 0,
			Device:    []int32{600, 700},
		},
	},
	&types.VirtualPCIController{
		VirtualController: types.VirtualController{
			VirtualDevice: types.VirtualDevice{
				DynamicData: types.DynamicData{},
				Key:         100,
				DeviceInfo: &types.Description{
					DynamicData: types.DynamicData{},
					Label:       "PCI controller 0",
					Summary:     "PCI controller 0",
				},
				Backing:       nil,
				Connectable:   (*types.VirtualDeviceConnectInfo)(nil),
				SlotInfo:      nil,
				ControllerKey: 0,
				UnitNumber:    (*int32)(nil),
			},
			BusNumber: 0,
			Device:    []int32{500, 12000},
		},
	},
	&types.VirtualSIOController{
		VirtualController: types.VirtualController{
			VirtualDevice: types.VirtualDevice{
				DynamicData: types.DynamicData{},
				Key:         400,
				DeviceInfo: &types.Description{
					DynamicData: types.DynamicData{},
					Label:       "SIO controller 0",
					Summary:     "SIO controller 0",
				},
				Backing:       nil,
				Connectable:   (*types.VirtualDeviceConnectInfo)(nil),
				SlotInfo:      nil,
				ControllerKey: 0,
				UnitNumber:    (*int32)(nil),
			},
			BusNumber: 0,
			Device:    nil,
		},
	},
	&types.VirtualKeyboard{
		VirtualDevice: types.VirtualDevice{
			DynamicData: types.DynamicData{},
			Key:         600,
			DeviceInfo: &types.Description{
				DynamicData: types.DynamicData{},
				Label:       "Keyboard ",
				Summary:     "Keyboard",
			},
			Backing:       nil,
			Connectable:   (*types.VirtualDeviceConnectInfo)(nil),
			SlotInfo:      nil,
			ControllerKey: 300,
			UnitNumber:    types.NewInt32(0),
		},
	},
	&types.VirtualPointingDevice{
		VirtualDevice: types.VirtualDevice{
			DynamicData: types.DynamicData{},
			Key:         700,
			DeviceInfo: &types.Description{
				DynamicData: types.DynamicData{},
				Label:       "Pointing device",
				Summary:     "Pointing device; Device",
			},
			Backing: &types.VirtualPointingDeviceDeviceBackingInfo{
				VirtualDeviceDeviceBackingInfo: types.VirtualDeviceDeviceBackingInfo{
					VirtualDeviceBackingInfo: types.VirtualDeviceBackingInfo{},
					DeviceName:               "",
					UseAutoDetect:            types.NewBool(false),
				},
				HostPointingDevice: "autodetect",
			},
			Connectable:   (*types.VirtualDeviceConnectInfo)(nil),
			SlotInfo:      nil,
			ControllerKey: 300,
			UnitNumber:    types.NewInt32(1),
		},
	},
	&types.VirtualMachineVideoCard{
		VirtualDevice: types.VirtualDevice{
			DynamicData: types.DynamicData{},
			Key:         500,
			DeviceInfo: &types.Description{
				DynamicData: types.DynamicData{},
				Label:       "Video card ",
				Summary:     "Video card",
			},
			Backing:       nil,
			Connectable:   (*types.VirtualDeviceConnectInfo)(nil),
			SlotInfo:      nil,
			ControllerKey: 100,
			UnitNumber:    types.NewInt32(0),
		},
		VideoRamSizeInKB:       4096,
		NumDisplays:            1,
		UseAutoDetect:          types.NewBool(false),
		Enable3DSupport:        types.NewBool(false),
		Use3dRenderer:          "automatic",
		GraphicsMemorySizeInKB: 262144,
	},
	&types.VirtualMachineVMCIDevice{
		VirtualDevice: types.VirtualDevice{
			DynamicData: types.DynamicData{},
			Key:         12000,
			DeviceInfo: &types.Description{
				DynamicData: types.DynamicData{},
				Label:       "VMCI device",
				Summary:     "Device on the virtual machine PCI bus that provides support for the virtual machine communication interface",
			},
			Backing:       nil,
			Connectable:   (*types.VirtualDeviceConnectInfo)(nil),
			SlotInfo:      nil,
			ControllerKey: 100,
			UnitNumber:    types.NewInt32(17),
		},
		Id:                             -1,
		AllowUnrestrictedCommunication: types.NewBool(false),
		FilterEnable:                   types.NewBool(true),
		FilterInfo:                     (*types.VirtualMachineVMCIDeviceFilterInfo)(nil),
	},
}

// EthernetCard template for types.VirtualEthernetCard
var EthernetCard = types.VirtualE1000{
	VirtualEthernetCard: types.VirtualEthernetCard{
		VirtualDevice: types.VirtualDevice{
			DynamicData: types.DynamicData{},
			Key:         4000,
			Backing: &types.VirtualEthernetCardNetworkBackingInfo{
				VirtualDeviceDeviceBackingInfo: types.VirtualDeviceDeviceBackingInfo{
					VirtualDeviceBackingInfo: types.VirtualDeviceBackingInfo{},
					DeviceName:               "VM Network",
					UseAutoDetect:            types.NewBool(false),
				},
				Network:           (*types.ManagedObjectReference)(nil),
				InPassthroughMode: types.NewBool(false),
			},
			Connectable: &types.VirtualDeviceConnectInfo{
				DynamicData:       types.DynamicData{},
				StartConnected:    true,
				AllowGuestControl: true,
				Connected:         false,
				Status:            "untried",
			},
			SlotInfo: &types.VirtualDevicePciBusSlotInfo{
				VirtualDeviceBusSlotInfo: types.VirtualDeviceBusSlotInfo{},
				PciSlotNumber:            32,
			},
			ControllerKey: 100,
			UnitNumber:    types.NewInt32(7),
		},
		AddressType:      "generated",
		MacAddress:       "",
		WakeOnLanEnabled: types.NewBool(true),
	},
}

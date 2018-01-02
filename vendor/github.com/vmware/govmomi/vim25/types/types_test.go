/*
Copyright (c) 2014-2015 VMware, Inc. All Rights Reserved.

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

package types

import (
	"testing"

	"github.com/vmware/govmomi/vim25/xml"
)

func TestVirtualMachineConfigSpec(t *testing.T) {
	spec := VirtualMachineConfigSpec{
		Name:     "vm-001",
		GuestId:  "otherGuest",
		Files:    &VirtualMachineFileInfo{VmPathName: "[datastore1]"},
		NumCPUs:  1,
		MemoryMB: 128,
		DeviceChange: []BaseVirtualDeviceConfigSpec{
			&VirtualDeviceConfigSpec{
				Operation: VirtualDeviceConfigSpecOperationAdd,
				Device: &VirtualLsiLogicController{VirtualSCSIController{
					SharedBus: VirtualSCSISharingNoSharing,
					VirtualController: VirtualController{
						BusNumber: 0,
						VirtualDevice: VirtualDevice{
							Key: 1000,
						},
					},
				}},
			},
			&VirtualDeviceConfigSpec{
				Operation:     VirtualDeviceConfigSpecOperationAdd,
				FileOperation: VirtualDeviceConfigSpecFileOperationCreate,
				Device: &VirtualDisk{
					VirtualDevice: VirtualDevice{
						Key:           0,
						ControllerKey: 1000,
						UnitNumber:    new(int32), // zero default value
						Backing: &VirtualDiskFlatVer2BackingInfo{
							DiskMode:        string(VirtualDiskModePersistent),
							ThinProvisioned: NewBool(true),
							VirtualDeviceFileBackingInfo: VirtualDeviceFileBackingInfo{
								FileName: "[datastore1]",
							},
						},
					},
					CapacityInKB: 4000000,
				},
			},
			&VirtualDeviceConfigSpec{
				Operation: VirtualDeviceConfigSpecOperationAdd,
				Device: &VirtualE1000{VirtualEthernetCard{
					VirtualDevice: VirtualDevice{
						Key: 0,
						DeviceInfo: &Description{
							Label:   "Network Adapter 1",
							Summary: "VM Network",
						},
						Backing: &VirtualEthernetCardNetworkBackingInfo{
							VirtualDeviceDeviceBackingInfo: VirtualDeviceDeviceBackingInfo{
								DeviceName: "VM Network",
							},
						},
					},
					AddressType: string(VirtualEthernetCardMacTypeGenerated),
				}},
			},
		},
		ExtraConfig: []BaseOptionValue{
			&OptionValue{Key: "bios.bootOrder", Value: "ethernet0"},
		},
	}

	_, err := xml.MarshalIndent(spec, "", " ")
	if err != nil {
		t.Fatal(err)
	}
}

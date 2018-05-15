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

// HostStorageDeviceInfo is the default template for the HostSystem config.storageDevice property.
// Capture method:
//   govc object.collect -s -dump HostSystem:ha-host config.storageDevice
var HostStorageDeviceInfo = types.HostStorageDeviceInfo{
	HostBusAdapter: []types.BaseHostHostBusAdapter{
		&types.HostParallelScsiHba{
			HostHostBusAdapter: types.HostHostBusAdapter{
				Key:    "key-vim.host.ParallelScsiHba-vmhba0",
				Device: "vmhba0",
				Bus:    3,
				Status: "unknown",
				Model:  "PVSCSI SCSI Controller",
				Driver: "pvscsi",
				Pci:    "0000:03:00.0",
			},
		},
		&types.HostBlockHba{
			HostHostBusAdapter: types.HostHostBusAdapter{
				Key:    "key-vim.host.BlockHba-vmhba1",
				Device: "vmhba1",
				Bus:    0,
				Status: "unknown",
				Model:  "PIIX4 for 430TX/440BX/MX IDE Controller",
				Driver: "vmkata",
				Pci:    "0000:00:07.1",
			},
		},
		&types.HostBlockHba{
			HostHostBusAdapter: types.HostHostBusAdapter{
				Key:    "key-vim.host.BlockHba-vmhba64",
				Device: "vmhba64",
				Bus:    0,
				Status: "unknown",
				Model:  "PIIX4 for 430TX/440BX/MX IDE Controller",
				Driver: "vmkata",
				Pci:    "0000:00:07.1",
			},
		},
	},
	ScsiLun: []types.BaseScsiLun{
		&types.ScsiLun{
			HostDevice: types.HostDevice{
				DeviceName: "/vmfs/devices/cdrom/mpx.vmhba1:C0:T0:L0",
				DeviceType: "cdrom",
			},
			Key:  "key-vim.host.ScsiLun-0005000000766d686261313a303a30",
			Uuid: "0005000000766d686261313a303a30",
			Descriptor: []types.ScsiLunDescriptor{
				{
					Quality: "lowQuality",
					Id:      "mpx.vmhba1:C0:T0:L0",
				},
				{
					Quality: "lowQuality",
					Id:      "vml.0005000000766d686261313a303a30",
				},
				{
					Quality: "lowQuality",
					Id:      "0005000000766d686261313a303a30",
				},
			},
			CanonicalName: "mpx.vmhba1:C0:T0:L0",
			DisplayName:   "Local NECVMWar CD-ROM (mpx.vmhba1:C0:T0:L0)",
			LunType:       "cdrom",
			Vendor:        "NECVMWar",
			Model:         "VMware IDE CDR00",
			Revision:      "1.00",
			ScsiLevel:     5,
			SerialNumber:  "unavailable",
			DurableName:   (*types.ScsiLunDurableName)(nil),
			AlternateName: []types.ScsiLunDurableName{
				{
					Namespace:   "GENERIC_VPD",
					NamespaceId: 0x5,
					Data:        []uint8{0x2d, 0x37, 0x39},
				},
				{
					Namespace:   "GENERIC_VPD",
					NamespaceId: 0x5,
					Data:        []uint8{0x30},
				},
			},
			StandardInquiry:  []uint8{0x30},
			QueueDepth:       0,
			OperationalState: []string{"ok"},
			Capabilities:     &types.ScsiLunCapabilities{},
			VStorageSupport:  "vStorageUnsupported",
			ProtocolEndpoint: types.NewBool(false),
		},
		&types.HostScsiDisk{
			ScsiLun: types.ScsiLun{
				HostDevice: types.HostDevice{
					DeviceName: "/vmfs/devices/disks/mpx.vmhba0:C0:T0:L0",
					DeviceType: "disk",
				},
				Key:  "key-vim.host.ScsiDisk-0000000000766d686261303a303a30",
				Uuid: "0000000000766d686261303a303a30",
				Descriptor: []types.ScsiLunDescriptor{
					{
						Quality: "lowQuality",
						Id:      "mpx.vmhba0:C0:T0:L0",
					},
					{
						Quality: "lowQuality",
						Id:      "vml.0000000000766d686261303a303a30",
					},
					{
						Quality: "lowQuality",
						Id:      "0000000000766d686261303a303a30",
					},
				},
				CanonicalName: "mpx.vmhba0:C0:T0:L0",
				DisplayName:   "Local VMware, Disk (mpx.vmhba0:C0:T0:L0)",
				LunType:       "disk",
				Vendor:        "VMware, ",
				Model:         "VMware Virtual S",
				Revision:      "1.0 ",
				ScsiLevel:     2,
				SerialNumber:  "unavailable",
				DurableName:   (*types.ScsiLunDurableName)(nil),
				AlternateName: []types.ScsiLunDurableName{
					{
						Namespace:   "GENERIC_VPD",
						NamespaceId: 0x5,
						Data:        []uint8{0x2d, 0x37, 0x39},
					},
					{
						Namespace:   "GENERIC_VPD",
						NamespaceId: 0x5,
						Data:        []uint8{0x30},
					},
				},
				StandardInquiry:  []uint8{0x30},
				QueueDepth:       1024,
				OperationalState: []string{"ok"},
				Capabilities:     &types.ScsiLunCapabilities{},
				VStorageSupport:  "vStorageUnsupported",
				ProtocolEndpoint: types.NewBool(false),
			},
			Capacity: types.HostDiskDimensionsLba{
				BlockSize: 512,
				Block:     67108864,
			},
			DevicePath:            "/vmfs/devices/disks/mpx.vmhba0:C0:T0:L0",
			Ssd:                   types.NewBool(true),
			LocalDisk:             types.NewBool(true),
			PhysicalLocation:      nil,
			EmulatedDIXDIFEnabled: types.NewBool(false),
			VsanDiskInfo:          (*types.VsanHostVsanDiskInfo)(nil),
			ScsiDiskType:          "native512",
		},
	},
	ScsiTopology: &types.HostScsiTopology{
		Adapter: []types.HostScsiTopologyInterface{
			{
				Key:     "key-vim.host.ScsiTopology.Interface-vmhba0",
				Adapter: "key-vim.host.ParallelScsiHba-vmhba0",
				Target: []types.HostScsiTopologyTarget{
					{
						Key:    "key-vim.host.ScsiTopology.Target-vmhba0:0:0",
						Target: 0,
						Lun: []types.HostScsiTopologyLun{
							{
								Key:     "key-vim.host.ScsiTopology.Lun-0000000000766d686261303a303a30",
								Lun:     0,
								ScsiLun: "key-vim.host.ScsiDisk-0000000000766d686261303a303a30",
							},
						},
						Transport: &types.HostParallelScsiTargetTransport{},
					},
				},
			},
			{
				Key:     "key-vim.host.ScsiTopology.Interface-vmhba1",
				Adapter: "key-vim.host.BlockHba-vmhba1",
				Target: []types.HostScsiTopologyTarget{
					{
						Key:    "key-vim.host.ScsiTopology.Target-vmhba1:0:0",
						Target: 0,
						Lun: []types.HostScsiTopologyLun{
							{
								Key:     "key-vim.host.ScsiTopology.Lun-0005000000766d686261313a303a30",
								Lun:     0,
								ScsiLun: "key-vim.host.ScsiLun-0005000000766d686261313a303a30",
							},
						},
						Transport: &types.HostBlockAdapterTargetTransport{},
					},
				},
			},
			{
				Key:     "key-vim.host.ScsiTopology.Interface-vmhba64",
				Adapter: "key-vim.host.BlockHba-vmhba64",
				Target:  nil,
			},
		},
	},
	MultipathInfo: &types.HostMultipathInfo{
		Lun: []types.HostMultipathInfoLogicalUnit{
			{
				Key: "key-vim.host.MultipathInfo.LogicalUnit-0005000000766d686261313a303a30",
				Id:  "0005000000766d686261313a303a30",
				Lun: "key-vim.host.ScsiLun-0005000000766d686261313a303a30",
				Path: []types.HostMultipathInfoPath{
					{
						Key:           "key-vim.host.MultipathInfo.Path-vmhba1:C0:T0:L0",
						Name:          "vmhba1:C0:T0:L0",
						PathState:     "active",
						State:         "active",
						IsWorkingPath: types.NewBool(true),
						Adapter:       "key-vim.host.BlockHba-vmhba1",
						Lun:           "key-vim.host.MultipathInfo.LogicalUnit-0005000000766d686261313a303a30",
						Transport:     &types.HostBlockAdapterTargetTransport{},
					},
				},
				Policy: &types.HostMultipathInfoFixedLogicalUnitPolicy{
					HostMultipathInfoLogicalUnitPolicy: types.HostMultipathInfoLogicalUnitPolicy{
						Policy: "VMW_PSP_FIXED",
					},
					Prefer: "vmhba1:C0:T0:L0",
				},
				StorageArrayTypePolicy: &types.HostMultipathInfoLogicalUnitStorageArrayTypePolicy{
					Policy: "VMW_SATP_LOCAL",
				},
			},
			{
				Key: "key-vim.host.MultipathInfo.LogicalUnit-0000000000766d686261303a303a30",
				Id:  "0000000000766d686261303a303a30",
				Lun: "key-vim.host.ScsiDisk-0000000000766d686261303a303a30",
				Path: []types.HostMultipathInfoPath{
					{
						Key:           "key-vim.host.MultipathInfo.Path-vmhba0:C0:T0:L0",
						Name:          "vmhba0:C0:T0:L0",
						PathState:     "active",
						State:         "active",
						IsWorkingPath: types.NewBool(true),
						Adapter:       "key-vim.host.ParallelScsiHba-vmhba0",
						Lun:           "key-vim.host.MultipathInfo.LogicalUnit-0000000000766d686261303a303a30",
						Transport:     &types.HostParallelScsiTargetTransport{},
					},
				},
				Policy: &types.HostMultipathInfoFixedLogicalUnitPolicy{
					HostMultipathInfoLogicalUnitPolicy: types.HostMultipathInfoLogicalUnitPolicy{
						Policy: "VMW_PSP_FIXED",
					},
					Prefer: "vmhba0:C0:T0:L0",
				},
				StorageArrayTypePolicy: &types.HostMultipathInfoLogicalUnitStorageArrayTypePolicy{
					Policy: "VMW_SATP_LOCAL",
				},
			},
		},
	},
	PlugStoreTopology: &types.HostPlugStoreTopology{
		Adapter: []types.HostPlugStoreTopologyAdapter{
			{
				Key:     "key-vim.host.PlugStoreTopology.Adapter-vmhba0",
				Adapter: "key-vim.host.ParallelScsiHba-vmhba0",
				Path:    []string{"key-vim.host.PlugStoreTopology.Path-vmhba0:C0:T0:L0"},
			},
			{
				Key:     "key-vim.host.PlugStoreTopology.Adapter-vmhba1",
				Adapter: "key-vim.host.BlockHba-vmhba1",
				Path:    []string{"key-vim.host.PlugStoreTopology.Path-vmhba1:C0:T0:L0"},
			},
			{
				Key:     "key-vim.host.PlugStoreTopology.Adapter-vmhba64",
				Adapter: "key-vim.host.BlockHba-vmhba64",
				Path:    nil,
			},
		},
		Path: []types.HostPlugStoreTopologyPath{
			{
				Key:           "key-vim.host.PlugStoreTopology.Path-vmhba0:C0:T0:L0",
				Name:          "vmhba0:C0:T0:L0",
				ChannelNumber: 0,
				TargetNumber:  0,
				LunNumber:     0,
				Adapter:       "key-vim.host.PlugStoreTopology.Adapter-vmhba0",
				Target:        "key-vim.host.PlugStoreTopology.Target-pscsi.0:0",
				Device:        "key-vim.host.PlugStoreTopology.Device-0000000000766d686261303a303a30",
			},
			{
				Key:           "key-vim.host.PlugStoreTopology.Path-vmhba1:C0:T0:L0",
				Name:          "vmhba1:C0:T0:L0",
				ChannelNumber: 0,
				TargetNumber:  0,
				LunNumber:     0,
				Adapter:       "key-vim.host.PlugStoreTopology.Adapter-vmhba1",
				Target:        "key-vim.host.PlugStoreTopology.Target-ide.0:0",
				Device:        "key-vim.host.PlugStoreTopology.Device-0005000000766d686261313a303a30",
			},
		},
		Target: []types.HostPlugStoreTopologyTarget{
			{
				Key:       "key-vim.host.PlugStoreTopology.Target-pscsi.0:0",
				Transport: &types.HostParallelScsiTargetTransport{},
			},
			{
				Key:       "key-vim.host.PlugStoreTopology.Target-ide.0:0",
				Transport: &types.HostBlockAdapterTargetTransport{},
			},
		},
		Device: []types.HostPlugStoreTopologyDevice{
			{
				Key:  "key-vim.host.PlugStoreTopology.Device-0005000000766d686261313a303a30",
				Lun:  "key-vim.host.ScsiLun-0005000000766d686261313a303a30",
				Path: []string{"key-vim.host.PlugStoreTopology.Path-vmhba1:C0:T0:L0"},
			},
			{
				Key:  "key-vim.host.PlugStoreTopology.Device-0000000000766d686261303a303a30",
				Lun:  "key-vim.host.ScsiDisk-0000000000766d686261303a303a30",
				Path: []string{"key-vim.host.PlugStoreTopology.Path-vmhba0:C0:T0:L0"},
			},
		},
		Plugin: []types.HostPlugStoreTopologyPlugin{
			{
				Key:         "key-vim.host.PlugStoreTopology.Plugin-NMP",
				Name:        "NMP",
				Device:      []string{"key-vim.host.PlugStoreTopology.Device-0005000000766d686261313a303a30", "key-vim.host.PlugStoreTopology.Device-0000000000766d686261303a303a30"},
				ClaimedPath: []string{"key-vim.host.PlugStoreTopology.Path-vmhba0:C0:T0:L0", "key-vim.host.PlugStoreTopology.Path-vmhba1:C0:T0:L0"},
			},
		},
	},
	SoftwareInternetScsiEnabled: false,
}

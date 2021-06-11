/*
Copyright (c) 2020 VMware, Inc. All Rights Reserved.

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
	"github.com/vmware/govmomi/simulator/esx"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type HostStorageSystem struct {
	mo.HostStorageSystem

	Host *mo.HostSystem
	HBA  []types.BaseHostHostBusAdapter
}

func NewHostStorageSystem(h *mo.HostSystem) *HostStorageSystem {
	s := &HostStorageSystem{Host: h}

	s.StorageDeviceInfo = &esx.HostStorageDeviceInfo

	s.HBA = fibreChannelHBA

	return s
}

// RescanAllHba swaps HostStorageSystem.HBA and StorageDeviceInfo.HostBusAdapter.
// This allows testing HBA with and without Fibre Channel data.
func (s *HostStorageSystem) RescanAllHba(ctx *Context, _ *types.RescanAllHba) soap.HasFault {
	hba := s.StorageDeviceInfo.HostBusAdapter
	s.StorageDeviceInfo.HostBusAdapter = s.HBA
	s.HBA = hba

	ctx.WithLock(s.Host, func() {
		s.Host.Config.StorageDevice.HostBusAdapter = s.StorageDeviceInfo.HostBusAdapter
	})

	return &methods.RescanAllHbaBody{
		Res: new(types.RescanAllHbaResponse),
	}
}

func (s *HostStorageSystem) RescanVmfs(*Context, *types.RescanVmfs) soap.HasFault {
	return &methods.RescanVmfsBody{Res: new(types.RescanVmfsResponse)}
}

func (s *HostStorageSystem) RefreshStorageSystem(*Context, *types.RefreshStorageSystem) soap.HasFault {
	return &methods.RefreshStorageSystemBody{Res: new(types.RefreshStorageSystemResponse)}
}

// HBA with FibreChannel data, see RescanAllHba()
var fibreChannelHBA = []types.BaseHostHostBusAdapter{
	&types.HostBlockHba{
		HostHostBusAdapter: types.HostHostBusAdapter{
			Key:    "key-vim.host.BlockHba-vmhba0",
			Device: "vmhba0",
			Bus:    0,
			Status: "unknown",
			Model:  "Lewisburg SATA AHCI Controller",
			Driver: "vmw_ahci",
			Pci:    "0000:00:11.5",
		},
	},
	&types.HostBlockHba{
		HostHostBusAdapter: types.HostHostBusAdapter{
			Key:    "key-vim.host.BlockHba-vmhba1",
			Device: "vmhba1",
			Bus:    0,
			Status: "unknown",
			Model:  "Lewisburg SATA AHCI Controller",
			Driver: "vmw_ahci",
			Pci:    "0000:00:17.0",
		},
	},
	&types.HostFibreChannelHba{
		HostHostBusAdapter: types.HostHostBusAdapter{
			Key:    "key-vim.host.FibreChannelHba-vmhba2",
			Device: "vmhba2",
			Bus:    59,
			Status: "online",
			Model:  "Emulex LightPulse LPe32000 PCIe Fibre Channel Adapter",
			Driver: "lpfc",
			Pci:    "0000:3b:00.0",
		},
		PortWorldWideName: 1152922127287604726,
		NodeWorldWideName: 2305843631894451702,
		PortType:          "unknown",
		Speed:             16,
	},
	&types.HostFibreChannelHba{
		HostHostBusAdapter: types.HostHostBusAdapter{
			Key:    "key-vim.host.FibreChannelHba-vmhba3",
			Device: "vmhba3",
			Bus:    95,
			Status: "online",
			Model:  "Emulex LightPulse LPe32000 PCIe Fibre Channel Adapter",
			Driver: "lpfc",
			Pci:    "0000:5f:00.0",
		},
		PortWorldWideName: 1152922127287604554,
		NodeWorldWideName: 2305843631894451530,
		PortType:          "unknown",
		Speed:             16,
	},
	&types.HostSerialAttachedHba{
		HostHostBusAdapter: types.HostHostBusAdapter{
			Key:    "key-vim.host.SerialAttachedHba-vmhba4",
			Device: "vmhba4",
			Bus:    24,
			Status: "unknown",
			Model:  "PERC H330 Adapter",
			Driver: "lsi_mr3",
			Pci:    "0000:18:00.0",
		},
		NodeWorldWideName: "5d0946606e78ac00",
	},
}

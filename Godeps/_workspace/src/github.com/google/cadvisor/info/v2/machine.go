// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package v2

import (
	// TODO(rjnagal): Move structs from v1.
	"github.com/google/cadvisor/info/v1"
)

type Attributes struct {
	// Kernel version.
	KernelVersion string `json:"kernel_version"`

	// OS image being used for cadvisor container, or host image if running on host directly.
	ContainerOsVersion string `json:"container_os_version"`

	// Docker version.
	DockerVersion string `json:"docker_version"`

	// cAdvisor version.
	CadvisorVersion string `json:"cadvisor_version"`

	// The number of cores in this machine.
	NumCores int `json:"num_cores"`

	// Maximum clock speed for the cores, in KHz.
	CpuFrequency uint64 `json:"cpu_frequency_khz"`

	// The amount of memory (in bytes) in this machine
	MemoryCapacity int64 `json:"memory_capacity"`

	// The machine id
	MachineID string `json:"machine_id"`

	// The system uuid
	SystemUUID string `json:"system_uuid"`

	// Filesystems on this machine.
	Filesystems []v1.FsInfo `json:"filesystems"`

	// Disk map
	DiskMap map[string]v1.DiskInfo `json:"disk_map"`

	// Network devices
	NetworkDevices []v1.NetInfo `json:"network_devices"`

	// Machine Topology
	// Describes cpu/memory layout and hierarchy.
	Topology []v1.Node `json:"topology"`

	// Cloud provider the machine belongs to
	CloudProvider v1.CloudProvider `json:"cloud_provider"`

	// Type of cloud instance (e.g. GCE standard) the machine is.
	InstanceType v1.InstanceType `json:"instance_type"`
}

func GetAttributes(mi *v1.MachineInfo, vi *v1.VersionInfo) Attributes {
	return Attributes{
		KernelVersion:      vi.KernelVersion,
		ContainerOsVersion: vi.ContainerOsVersion,
		DockerVersion:      vi.DockerVersion,
		CadvisorVersion:    vi.CadvisorVersion,
		NumCores:           mi.NumCores,
		CpuFrequency:       mi.CpuFrequency,
		MemoryCapacity:     mi.MemoryCapacity,
		MachineID:          mi.MachineID,
		SystemUUID:         mi.SystemUUID,
		Filesystems:        mi.Filesystems,
		DiskMap:            mi.DiskMap,
		NetworkDevices:     mi.NetworkDevices,
		Topology:           mi.Topology,
		CloudProvider:      mi.CloudProvider,
		InstanceType:       mi.InstanceType,
	}
}

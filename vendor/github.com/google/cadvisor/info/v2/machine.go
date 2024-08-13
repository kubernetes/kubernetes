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
	"time"

	v1 "github.com/google/cadvisor/info/v1"
)

type Attributes struct {
	// Kernel version.
	KernelVersion string `json:"kernel_version"`

	// OS image being used for cadvisor container, or host image if running on host directly.
	ContainerOsVersion string `json:"container_os_version"`

	// Docker version.
	DockerVersion string `json:"docker_version"`

	// Docker API version.
	DockerAPIVersion string `json:"docker_api_version"`

	// cAdvisor version.
	CadvisorVersion string `json:"cadvisor_version"`

	// The number of cores in this machine.
	NumCores int `json:"num_cores"`

	// Maximum clock speed for the cores, in KHz.
	CpuFrequency uint64 `json:"cpu_frequency_khz"`

	// The amount of memory (in bytes) in this machine
	MemoryCapacity uint64 `json:"memory_capacity"`

	// The machine id
	MachineID string `json:"machine_id"`

	// The system uuid
	SystemUUID string `json:"system_uuid"`

	// HugePages on this machine.
	HugePages []v1.HugePagesInfo `json:"hugepages"`

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
		DockerAPIVersion:   vi.DockerAPIVersion,
		CadvisorVersion:    vi.CadvisorVersion,
		NumCores:           mi.NumCores,
		CpuFrequency:       mi.CpuFrequency,
		MemoryCapacity:     mi.MemoryCapacity,
		MachineID:          mi.MachineID,
		SystemUUID:         mi.SystemUUID,
		HugePages:          mi.HugePages,
		Filesystems:        mi.Filesystems,
		DiskMap:            mi.DiskMap,
		NetworkDevices:     mi.NetworkDevices,
		Topology:           mi.Topology,
		CloudProvider:      mi.CloudProvider,
		InstanceType:       mi.InstanceType,
	}
}

// MachineStats contains usage statistics for the entire machine.
type MachineStats struct {
	// The time of this stat point.
	Timestamp time.Time `json:"timestamp"`
	// In nanoseconds (aggregated)
	Cpu *v1.CpuStats `json:"cpu,omitempty"`
	// In nanocores per second (instantaneous)
	CpuInst *CpuInstStats `json:"cpu_inst,omitempty"`
	// Memory statistics
	Memory *v1.MemoryStats `json:"memory,omitempty"`
	// Network statistics
	Network *NetworkStats `json:"network,omitempty"`
	// Filesystem statistics
	Filesystem []MachineFsStats `json:"filesystem,omitempty"`
	// Task load statistics
	Load *v1.LoadStats `json:"load_stats,omitempty"`
}

// MachineFsStats contains per filesystem capacity and usage information.
type MachineFsStats struct {
	// The block device name associated with the filesystem.
	Device string `json:"device"`

	// Type of filesystem.
	Type string `json:"type"`

	// Number of bytes that can be consumed on this filesystem.
	Capacity *uint64 `json:"capacity,omitempty"`

	// Number of bytes that is currently consumed on this filesystem.
	Usage *uint64 `json:"usage,omitempty"`

	// Number of bytes available for non-root user on this filesystem.
	Available *uint64 `json:"available,omitempty"`

	// Number of inodes that are available on this filesystem.
	InodesFree *uint64 `json:"inodes_free,omitempty"`

	// DiskStats for this device.
	DiskStats `json:"inline"`
}

// DiskStats contains per partition usage information.
// This information is only available at the machine level.
type DiskStats struct {
	// Number of reads completed
	// This is the total number of reads completed successfully.
	ReadsCompleted *uint64 `json:"reads_completed,omitempty"`

	// Number of reads merged
	// Reads and writes which are adjacent to each other may be merged for
	// efficiency.  Thus two 4K reads may become one 8K read before it is
	// ultimately handed to the disk, and so it will be counted (and queued)
	// as only one I/O.  This field lets you know how often this was done.
	ReadsMerged *uint64 `json:"reads_merged,omitempty"`

	// Number of sectors read
	// This is the total number of sectors read successfully.
	SectorsRead *uint64 `json:"sectors_read,omitempty"`

	// Time spent reading
	// This is the total number of milliseconds spent by all reads (as
	// measured from __make_request() to end_that_request_last()).
	ReadDuration *time.Duration `json:"read_duration,omitempty"`

	// Number of writes completed
	// This is the total number of writes completed successfully.
	WritesCompleted *uint64 `json:"writes_completed,omitempty"`

	// Number of writes merged
	// See the description of reads merged.
	WritesMerged *uint64 `json:"writes_merged,omitempty"`

	// Number of sectors written
	// This is the total number of sectors written successfully.
	SectorsWritten *uint64 `json:"sectors_written,omitempty"`

	// Time spent writing
	// This is the total number of milliseconds spent by all writes (as
	// measured from __make_request() to end_that_request_last()).
	WriteDuration *time.Duration `json:"write_duration,omitempty"`

	// Number of I/Os currently in progress
	// The only field that should go to zero. Incremented as requests are
	// given to appropriate struct request_queue and decremented as they finish.
	IoInProgress *uint64 `json:"io_in_progress,omitempty"`

	// Time spent doing I/Os
	// This field increases so long as field 9 is nonzero.
	IoDuration *time.Duration `json:"io_duration,omitempty"`

	// weighted time spent doing I/Os
	// This field is incremented at each I/O start, I/O completion, I/O
	// merge, or read of these stats by the number of I/Os in progress
	// (field 9) times the number of milliseconds spent doing I/O since the
	// last update of this field.  This can provide an easy measure of both
	// I/O completion time and the backlog that may be accumulating.
	WeightedIoDuration *time.Duration `json:"weighted_io_duration,omitempty"`
}

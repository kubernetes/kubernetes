// Copyright 2014 Google Inc. All Rights Reserved.
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

package v1

type FsInfo struct {
	// Block device associated with the filesystem.
	Device string `json:"device"`
	// DeviceMajor is the major identifier of the device, used for correlation with blkio stats
	DeviceMajor uint64 `json:"-"`
	// DeviceMinor is the minor identifier of the device, used for correlation with blkio stats
	DeviceMinor uint64 `json:"-"`

	// Total number of bytes available on the filesystem.
	Capacity uint64 `json:"capacity"`

	// Type of device.
	Type string `json:"type"`

	// Total number of inodes available on the filesystem.
	Inodes uint64 `json:"inodes"`

	// HasInodes when true, indicates that Inodes info will be available.
	HasInodes bool `json:"has_inodes"`
}

type Node struct {
	Id int `json:"node_id"`
	// Per-node memory
	Memory uint64  `json:"memory"`
	Cores  []Core  `json:"cores"`
	Caches []Cache `json:"caches"`
}

type Core struct {
	Id      int     `json:"core_id"`
	Threads []int   `json:"thread_ids"`
	Caches  []Cache `json:"caches"`
}

type Cache struct {
	// Size of memory cache in bytes.
	Size uint64 `json:"size"`
	// Type of memory cache: data, instruction, or unified.
	Type string `json:"type"`
	// Level (distance from cpus) in a multi-level cache hierarchy.
	Level int `json:"level"`
}

func (self *Node) FindCore(id int) (bool, int) {
	for i, n := range self.Cores {
		if n.Id == id {
			return true, i
		}
	}
	return false, -1
}

func (self *Node) AddThread(thread int, core int) {
	var coreIdx int
	if core == -1 {
		// Assume one hyperthread per core when topology data is missing.
		core = thread
	}
	ok, coreIdx := self.FindCore(core)

	if !ok {
		// New core
		core := Core{Id: core}
		self.Cores = append(self.Cores, core)
		coreIdx = len(self.Cores) - 1
	}
	self.Cores[coreIdx].Threads = append(self.Cores[coreIdx].Threads, thread)
}

func (self *Node) AddNodeCache(c Cache) {
	self.Caches = append(self.Caches, c)
}

func (self *Node) AddPerCoreCache(c Cache) {
	for idx := range self.Cores {
		self.Cores[idx].Caches = append(self.Cores[idx].Caches, c)
	}
}

type DiskInfo struct {
	// device name
	Name string `json:"name"`

	// Major number
	Major uint64 `json:"major"`

	// Minor number
	Minor uint64 `json:"minor"`

	// Size in bytes
	Size uint64 `json:"size"`

	// I/O Scheduler - one of "none", "noop", "cfq", "deadline"
	Scheduler string `json:"scheduler"`
}

type NetInfo struct {
	// Device name
	Name string `json:"name"`

	// Mac Address
	MacAddress string `json:"mac_address"`

	// Speed in MBits/s
	Speed int64 `json:"speed"`

	// Maximum Transmission Unit
	Mtu int64 `json:"mtu"`
}

type CloudProvider string

const (
	GCE             CloudProvider = "GCE"
	AWS                           = "AWS"
	Azure                         = "Azure"
	Baremetal                     = "Baremetal"
	UnknownProvider               = "Unknown"
)

type InstanceType string

const (
	NoInstance      InstanceType = "None"
	UnknownInstance              = "Unknown"
)

type InstanceID string

const (
	UnNamedInstance InstanceID = "None"
)

type MachineInfo struct {
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

	// The boot id
	BootID string `json:"boot_id"`

	// Filesystems on this machine.
	Filesystems []FsInfo `json:"filesystems"`

	// Disk map
	DiskMap map[string]DiskInfo `json:"disk_map"`

	// Network devices
	NetworkDevices []NetInfo `json:"network_devices"`

	// Machine Topology
	// Describes cpu/memory layout and hierarchy.
	Topology []Node `json:"topology"`

	// Cloud provider the machine belongs to.
	CloudProvider CloudProvider `json:"cloud_provider"`

	// Type of cloud instance (e.g. GCE standard) the machine is.
	InstanceType InstanceType `json:"instance_type"`

	// ID of cloud instance (e.g. instance-1) given to it by the cloud provider.
	InstanceID InstanceID `json:"instance_id"`
}

type VersionInfo struct {
	// Kernel version.
	KernelVersion string `json:"kernel_version"`

	// OS image being used for cadvisor container, or host image if running on host directly.
	ContainerOsVersion string `json:"container_os_version"`

	// Docker version.
	DockerVersion string `json:"docker_version"`

	// Docker API Version
	DockerAPIVersion string `json:"docker_api_version"`

	// cAdvisor version.
	CadvisorVersion string `json:"cadvisor_version"`
	// cAdvisor git revision.
	CadvisorRevision string `json:"cadvisor_revision"`
}

type MachineInfoFactory interface {
	GetMachineInfo() (*MachineInfo, error)
	GetVersionInfo() (*VersionInfo, error)
}

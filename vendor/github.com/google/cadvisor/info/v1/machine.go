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

import "time"

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
	Memory    uint64          `json:"memory"`
	HugePages []HugePagesInfo `json:"hugepages"`
	Cores     []Core          `json:"cores"`
	Caches    []Cache         `json:"caches"`
}

type Core struct {
	Id           int     `json:"core_id"`
	Threads      []int   `json:"thread_ids"`
	Caches       []Cache `json:"caches"`
	UncoreCaches []Cache `json:"uncore_caches"`
	SocketID     int     `json:"socket_id"`
}

type Cache struct {
	// Id of memory cache
	Id int `json:"id"`
	// Size of memory cache in bytes.
	Size uint64 `json:"size"`
	// Type of memory cache: data, instruction, or unified.
	Type string `json:"type"`
	// Level (distance from cpus) in a multi-level cache hierarchy.
	Level int `json:"level"`
}

func (n *Node) FindCore(id int) (bool, int) {
	for i, n := range n.Cores {
		if n.Id == id {
			return true, i
		}
	}
	return false, -1
}

// FindCoreByThread returns bool if found Core with same thread as provided and it's index in Node Core array.
// If it's not found, returns false and -1.
func (n *Node) FindCoreByThread(thread int) (bool, int) {
	for i, n := range n.Cores {
		for _, t := range n.Threads {
			if t == thread {
				return true, i
			}
		}
	}
	return false, -1
}

func (n *Node) AddThread(thread int, core int) {
	var coreIdx int
	if core == -1 {
		// Assume one hyperthread per core when topology data is missing.
		core = thread
	}
	ok, coreIdx := n.FindCore(core)

	if !ok {
		// New core
		core := Core{Id: core}
		n.Cores = append(n.Cores, core)
		coreIdx = len(n.Cores) - 1
	}
	n.Cores[coreIdx].Threads = append(n.Cores[coreIdx].Threads, thread)
}

func (n *Node) AddNodeCache(c Cache) {
	n.Caches = append(n.Caches, c)
}

func (n *Node) AddPerCoreCache(c Cache) {
	for idx := range n.Cores {
		n.Cores[idx].Caches = append(n.Cores[idx].Caches, c)
	}
}

type HugePagesInfo struct {
	// huge page size (in kB)
	PageSize uint64 `json:"page_size"`

	// number of huge pages
	NumPages uint64 `json:"num_pages"`
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
	AWS             CloudProvider = "AWS"
	Azure           CloudProvider = "Azure"
	UnknownProvider CloudProvider = "Unknown"
)

type InstanceType string

const (
	UnknownInstance = "Unknown"
)

type InstanceID string

const (
	UnNamedInstance InstanceID = "None"
)

type MachineInfo struct {
	// The time of this information point.
	Timestamp time.Time `json:"timestamp"`

	// Vendor id of CPU.
	CPUVendorID string `json:"vendor_id"`

	// The number of cores in this machine.
	NumCores int `json:"num_cores"`

	// The number of physical cores in this machine.
	NumPhysicalCores int `json:"num_physical_cores"`

	// The number of cpu sockets in this machine.
	NumSockets int `json:"num_sockets"`

	// Maximum clock speed for the cores, in KHz.
	CpuFrequency uint64 `json:"cpu_frequency_khz"`

	// The amount of memory (in bytes) in this machine
	MemoryCapacity uint64 `json:"memory_capacity"`

	// Memory capacity and number of DIMMs by memory type
	MemoryByType map[string]*MemoryInfo `json:"memory_by_type"`

	NVMInfo NVMInfo `json:"nvm"`

	// HugePages on this machine.
	HugePages []HugePagesInfo `json:"hugepages"`

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

func (m *MachineInfo) Clone() *MachineInfo {
	memoryByType := m.MemoryByType
	if len(m.MemoryByType) > 0 {
		memoryByType = make(map[string]*MemoryInfo)
		for memoryType, memoryInfo := range m.MemoryByType {
			memoryByType[memoryType] = memoryInfo
		}
	}
	diskMap := m.DiskMap
	if len(m.DiskMap) > 0 {
		diskMap = make(map[string]DiskInfo)
		for k, info := range m.DiskMap {
			diskMap[k] = info
		}
	}
	copy := MachineInfo{
		CPUVendorID:      m.CPUVendorID,
		Timestamp:        m.Timestamp,
		NumCores:         m.NumCores,
		NumPhysicalCores: m.NumPhysicalCores,
		NumSockets:       m.NumSockets,
		CpuFrequency:     m.CpuFrequency,
		MemoryCapacity:   m.MemoryCapacity,
		MemoryByType:     memoryByType,
		NVMInfo:          m.NVMInfo,
		HugePages:        m.HugePages,
		MachineID:        m.MachineID,
		SystemUUID:       m.SystemUUID,
		BootID:           m.BootID,
		Filesystems:      m.Filesystems,
		DiskMap:          diskMap,
		NetworkDevices:   m.NetworkDevices,
		Topology:         m.Topology,
		CloudProvider:    m.CloudProvider,
		InstanceType:     m.InstanceType,
		InstanceID:       m.InstanceID,
	}
	return &copy
}

type MemoryInfo struct {
	// The amount of memory (in bytes).
	Capacity uint64 `json:"capacity"`

	// Number of memory DIMMs.
	DimmCount uint `json:"dimm_count"`
}

type NVMInfo struct {
	// The total NVM capacity in bytes for memory mode.
	MemoryModeCapacity uint64 `json:"memory_mode_capacity"`

	//The total NVM capacity in bytes for app direct mode.
	AppDirectModeCapacity uint64 `json:"app direct_mode_capacity"`

	// Average power budget in watts for NVM devices configured in BIOS.
	AvgPowerBudget uint `json:"avg_power_budget"`
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

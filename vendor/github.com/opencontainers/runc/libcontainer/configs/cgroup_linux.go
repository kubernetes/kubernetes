package configs

import (
	systemdDbus "github.com/coreos/go-systemd/v22/dbus"
)

type FreezerState string

const (
	Undefined FreezerState = ""
	Frozen    FreezerState = "FROZEN"
	Thawed    FreezerState = "THAWED"
)

type Cgroup struct {
	// Deprecated, use Path instead
	Name string `json:"name,omitempty"`

	// name of parent of cgroup or slice
	// Deprecated, use Path instead
	Parent string `json:"parent,omitempty"`

	// Path specifies the path to cgroups that are created and/or joined by the container.
	// The path is assumed to be relative to the host system cgroup mountpoint.
	Path string `json:"path"`

	// ScopePrefix describes prefix for the scope name
	ScopePrefix string `json:"scope_prefix"`

	// Paths represent the absolute cgroups paths to join.
	// This takes precedence over Path.
	Paths map[string]string

	// Resources contains various cgroups settings to apply
	*Resources

	// SystemdProps are any additional properties for systemd,
	// derived from org.systemd.property.xxx annotations.
	// Ignored unless systemd is used for managing cgroups.
	SystemdProps []systemdDbus.Property `json:"-"`
}

type Resources struct {
	// Devices is the set of access rules for devices in the container.
	Devices []*DeviceRule `json:"devices"`

	// Memory limit (in bytes)
	Memory int64 `json:"memory"`

	// Memory reservation or soft_limit (in bytes)
	MemoryReservation int64 `json:"memory_reservation"`

	// Total memory usage (memory + swap); set `-1` to enable unlimited swap
	MemorySwap int64 `json:"memory_swap"`

	// Kernel memory limit (in bytes)
	KernelMemory int64 `json:"kernel_memory"`

	// Kernel memory limit for TCP use (in bytes)
	KernelMemoryTCP int64 `json:"kernel_memory_tcp"`

	// CPU shares (relative weight vs. other containers)
	CpuShares uint64 `json:"cpu_shares"`

	// CPU hardcap limit (in usecs). Allowed cpu time in a given period.
	CpuQuota int64 `json:"cpu_quota"`

	// CPU period to be used for hardcapping (in usecs). 0 to use system default.
	CpuPeriod uint64 `json:"cpu_period"`

	// How many time CPU will use in realtime scheduling (in usecs).
	CpuRtRuntime int64 `json:"cpu_rt_quota"`

	// CPU period to be used for realtime scheduling (in usecs).
	CpuRtPeriod uint64 `json:"cpu_rt_period"`

	// CPU to use
	CpusetCpus string `json:"cpuset_cpus"`

	// MEM to use
	CpusetMems string `json:"cpuset_mems"`

	// Process limit; set <= `0' to disable limit.
	PidsLimit int64 `json:"pids_limit"`

	// Specifies per cgroup weight, range is from 10 to 1000.
	BlkioWeight uint16 `json:"blkio_weight"`

	// Specifies tasks' weight in the given cgroup while competing with the cgroup's child cgroups, range is from 10 to 1000, cfq scheduler only
	BlkioLeafWeight uint16 `json:"blkio_leaf_weight"`

	// Weight per cgroup per device, can override BlkioWeight.
	BlkioWeightDevice []*WeightDevice `json:"blkio_weight_device"`

	// IO read rate limit per cgroup per device, bytes per second.
	BlkioThrottleReadBpsDevice []*ThrottleDevice `json:"blkio_throttle_read_bps_device"`

	// IO write rate limit per cgroup per device, bytes per second.
	BlkioThrottleWriteBpsDevice []*ThrottleDevice `json:"blkio_throttle_write_bps_device"`

	// IO read rate limit per cgroup per device, IO per second.
	BlkioThrottleReadIOPSDevice []*ThrottleDevice `json:"blkio_throttle_read_iops_device"`

	// IO write rate limit per cgroup per device, IO per second.
	BlkioThrottleWriteIOPSDevice []*ThrottleDevice `json:"blkio_throttle_write_iops_device"`

	// set the freeze value for the process
	Freezer FreezerState `json:"freezer"`

	// Hugetlb limit (in bytes)
	HugetlbLimit []*HugepageLimit `json:"hugetlb_limit"`

	// Whether to disable OOM Killer
	OomKillDisable bool `json:"oom_kill_disable"`

	// Tuning swappiness behaviour per cgroup
	MemorySwappiness *uint64 `json:"memory_swappiness"`

	// Set priority of network traffic for container
	NetPrioIfpriomap []*IfPrioMap `json:"net_prio_ifpriomap"`

	// Set class identifier for container's network packets
	NetClsClassid uint32 `json:"net_cls_classid_u"`

	// Used on cgroups v2:

	// CpuWeight sets a proportional bandwidth limit.
	CpuWeight uint64 `json:"cpu_weight"`
}

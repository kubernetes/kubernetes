package cgroups

import (
	systemdDbus "github.com/coreos/go-systemd/v22/dbus"
	devices "github.com/opencontainers/cgroups/devices/config"
)

type FreezerState string

const (
	Undefined FreezerState = ""
	Frozen    FreezerState = "FROZEN"
	Thawed    FreezerState = "THAWED"
)

// Cgroup holds properties of a cgroup on Linux.
type Cgroup struct {
	// Name specifies the name of the cgroup
	Name string `json:"name,omitempty"`

	// Parent specifies the name of parent of cgroup or slice
	Parent string `json:"parent,omitempty"`

	// Path specifies the path to cgroups that are created and/or joined by the container.
	// The path is assumed to be relative to the host system cgroup mountpoint.
	Path string `json:"path,omitempty"`

	// ScopePrefix describes prefix for the scope name.
	ScopePrefix string `json:"scope_prefix,omitempty"`

	// Resources contains various cgroups settings to apply.
	*Resources

	// Systemd tells if systemd should be used to manage cgroups.
	Systemd bool `json:"Systemd,omitempty"`

	// SystemdProps are any additional properties for systemd,
	// derived from org.systemd.property.xxx annotations.
	// Ignored unless systemd is used for managing cgroups.
	SystemdProps []systemdDbus.Property `json:"-"`

	// Rootless tells if rootless cgroups should be used.
	Rootless bool `json:"Rootless,omitempty"`

	// The host UID that should own the cgroup, or nil to accept
	// the default ownership.  This should only be set when the
	// cgroupfs is to be mounted read/write.
	// Not all cgroup manager implementations support changing
	// the ownership.
	OwnerUID *int `json:"owner_uid,omitempty"`
}

type Resources struct {
	// Devices is the set of access rules for devices in the container.
	Devices []*devices.Rule `json:"devices,omitempty"`

	// Memory limit (in bytes).
	Memory int64 `json:"memory,omitempty"`

	// Memory reservation or soft_limit (in bytes).
	MemoryReservation int64 `json:"memory_reservation,omitempty"`

	// Total memory usage (memory+swap); use -1 for unlimited swap.
	MemorySwap int64 `json:"memory_swap,omitempty"`

	// CPU shares (relative weight vs. other containers).
	CpuShares uint64 `json:"cpu_shares,omitempty"` //nolint:revive // Suppress "var-naming: struct field CpuShares should be CPUShares".

	// CPU hardcap limit (in usecs). Allowed cpu time in a given period.
	CpuQuota int64 `json:"cpu_quota,omitempty"` //nolint:revive // Suppress "var-naming: struct field CpuQuota should be CPUQuota".

	// CPU hardcap burst limit (in usecs). Allowed accumulated cpu time additionally for burst in a given period.
	CpuBurst *uint64 `json:"cpu_burst,omitempty"` //nolint:revive // Suppress "var-naming: struct field CpuBurst should be CPUBurst".

	// CPU period to be used for hardcapping (in usecs). 0 to use system default.
	CpuPeriod uint64 `json:"cpu_period,omitempty"` //nolint:revive // Suppress "var-naming: struct field CpuPeriod should be CPUPeriod".

	// How many time CPU will use in realtime scheduling (in usecs).
	CpuRtRuntime int64 `json:"cpu_rt_quota,omitempty"` //nolint:revive // Suppress "var-naming: struct field CpuRtRuntime should be CPURtRuntime".

	// CPU period to be used for realtime scheduling (in usecs).
	CpuRtPeriod uint64 `json:"cpu_rt_period,omitempty"` //nolint:revive // Suppress "var-naming: struct field CpuQuota should be CPUQuota".

	// Cpuset CPUs to use.
	CpusetCpus string `json:"cpuset_cpus,omitempty"`

	// Cpuset memory nodes to use.
	CpusetMems string `json:"cpuset_mems,omitempty"`

	// Cgroup's SCHED_IDLE value.
	CPUIdle *int64 `json:"cpu_idle,omitempty"`

	// Process limit; set < `0' to disable limit. `nil` means "keep current limit".
	PidsLimit *int64 `json:"pids_limit,omitempty"`

	// Specifies per cgroup weight, range is from 10 to 1000.
	BlkioWeight uint16 `json:"blkio_weight,omitempty"`

	// Tasks' weight in the given cgroup while competing with the cgroup's child cgroups, range is from 10 to 1000, cfq scheduler only.
	BlkioLeafWeight uint16 `json:"blkio_leaf_weight,omitempty"`

	// Weight per cgroup per device, can override BlkioWeight.
	BlkioWeightDevice []*WeightDevice `json:"blkio_weight_device,omitempty"`

	// IO read rate limit per cgroup per device, bytes per second.
	BlkioThrottleReadBpsDevice []*ThrottleDevice `json:"blkio_throttle_read_bps_device,omitempty"`

	// IO write rate limit per cgroup per device, bytes per second.
	BlkioThrottleWriteBpsDevice []*ThrottleDevice `json:"blkio_throttle_write_bps_device,omitempty"`

	// IO read rate limit per cgroup per device, IO per second.
	BlkioThrottleReadIOPSDevice []*ThrottleDevice `json:"blkio_throttle_read_iops_device,omitempty"`

	// IO write rate limit per cgroup per device, IO per second.
	BlkioThrottleWriteIOPSDevice []*ThrottleDevice `json:"blkio_throttle_write_iops_device,omitempty"`

	// Freeze value for the process.
	Freezer FreezerState `json:"freezer,omitempty"`

	// Hugetlb limit (in bytes).
	HugetlbLimit []*HugepageLimit `json:"hugetlb_limit,omitempty"`

	// Whether to disable OOM killer.
	OomKillDisable bool `json:"oom_kill_disable,omitempty"`

	// Tuning swappiness behaviour per cgroup.
	MemorySwappiness *uint64 `json:"memory_swappiness,omitempty"`

	// Set priority of network traffic for container.
	NetPrioIfpriomap []*IfPrioMap `json:"net_prio_ifpriomap,omitempty"`

	// Set class identifier for container's network packets.
	NetClsClassid uint32 `json:"net_cls_classid_u,omitempty"`

	// Rdma resource restriction configuration.
	Rdma map[string]LinuxRdma `json:"rdma,omitempty"`

	// Used on cgroups v2:

	// CpuWeight sets a proportional bandwidth limit.
	CpuWeight uint64 `json:"cpu_weight,omitempty"` //nolint:revive // Suppress "var-naming: struct field CpuWeight should be CPUWeight".

	// Unified is cgroupv2-only key-value map.
	Unified map[string]string `json:"unified,omitempty"`

	// SkipDevices allows to skip configuring device permissions.
	// Used by e.g. kubelet while creating a parent cgroup (kubepods)
	// common for many containers, and by runc update.
	//
	// NOTE it is impossible to start a container which has this flag set.
	SkipDevices bool `json:"-"`

	// SkipFreezeOnSet is a flag for cgroup manager to skip the cgroup
	// freeze when setting resources. Only applicable to systemd legacy
	// (i.e. cgroup v1) manager (which uses freeze by default to avoid
	// spurious permission errors caused by systemd inability to update
	// device rules in a non-disruptive manner).
	//
	// If not set, a few methods (such as looking into cgroup's
	// devices.list and querying the systemd unit properties) are used
	// during Set() to figure out whether the freeze is required. Those
	// methods may be relatively slow, thus this flag.
	SkipFreezeOnSet bool `json:"-"`

	// MemoryCheckBeforeUpdate is a flag for cgroup v2 managers to check
	// if the new memory limits (Memory and MemorySwap) being set are lower
	// than the current memory usage, and reject if so.
	MemoryCheckBeforeUpdate bool `json:"memory_check_before_update,omitempty"`
}

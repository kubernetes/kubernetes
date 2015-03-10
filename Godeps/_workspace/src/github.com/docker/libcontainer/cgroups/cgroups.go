package cgroups

import (
	"fmt"

	"github.com/docker/libcontainer/devices"
)

type FreezerState string

const (
	Undefined FreezerState = ""
	Frozen    FreezerState = "FROZEN"
	Thawed    FreezerState = "THAWED"
)

type NotFoundError struct {
	Subsystem string
}

func (e *NotFoundError) Error() string {
	return fmt.Sprintf("mountpoint for %s not found", e.Subsystem)
}

func NewNotFoundError(sub string) error {
	return &NotFoundError{
		Subsystem: sub,
	}
}

func IsNotFound(err error) bool {
	if err == nil {
		return false
	}

	_, ok := err.(*NotFoundError)
	return ok
}

type Cgroup struct {
	Name   string `json:"name,omitempty"`
	Parent string `json:"parent,omitempty"` // name of parent cgroup or slice

	AllowAllDevices   bool              `json:"allow_all_devices,omitempty"` // If this is true allow access to any kind of device within the container.  If false, allow access only to devices explicitly listed in the allowed_devices list.
	AllowedDevices    []*devices.Device `json:"allowed_devices,omitempty"`
	Memory            int64             `json:"memory,omitempty"`             // Memory limit (in bytes)
	MemoryReservation int64             `json:"memory_reservation,omitempty"` // Memory reservation or soft_limit (in bytes)
	MemorySwap        int64             `json:"memory_swap,omitempty"`        // Total memory usage (memory + swap); set `-1' to disable swap
	CpuShares         int64             `json:"cpu_shares,omitempty"`         // CPU shares (relative weight vs. other containers)
	CpuQuota          int64             `json:"cpu_quota,omitempty"`          // CPU hardcap limit (in usecs). Allowed cpu time in a given period.
	CpuPeriod         int64             `json:"cpu_period,omitempty"`         // CPU period to be used for hardcapping (in usecs). 0 to use system default.
	CpusetCpus        string            `json:"cpuset_cpus,omitempty"`        // CPU to use
	CpusetMems        string            `json:"cpuset_mems,omitempty"`        // MEM to use
	Freezer           FreezerState      `json:"freezer,omitempty"`            // set the freeze value for the process
	Slice             string            `json:"slice,omitempty"`              // Parent slice to use for systemd
}

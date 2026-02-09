package cgroups

import (
	"errors"
)

var (
	// ErrDevicesUnsupported is an error returned when a cgroup manager
	// is not configured to set device rules.
	ErrDevicesUnsupported = errors.New("cgroup manager is not configured to set device rules")

	// ErrRootless is returned by [Manager.Apply] when there is an error
	// creating cgroup directory, and cgroup.Rootless is set. In general,
	// this error is to be ignored.
	ErrRootless = errors.New("cgroup manager can not access cgroup (rootless container)")

	// DevicesSetV1 and DevicesSetV2 are functions to set devices for
	// cgroup v1 and v2, respectively. Unless
	// [github.com/opencontainers/cgroups/devices]
	// package is imported, it is set to nil, so cgroup managers can't
	// manage devices.
	DevicesSetV1 func(path string, r *Resources) error
	DevicesSetV2 func(path string, r *Resources) error
)

type Manager interface {
	// Apply creates a cgroup, if not yet created, and adds a process
	// with the specified pid into that cgroup.  A special value of -1
	// can be used to merely create a cgroup.
	Apply(pid int) error

	// AddPid adds a process with a given pid to an existing cgroup.
	// The subcgroup argument is either empty, or a path relative to
	// a cgroup under under the manager's cgroup.
	AddPid(subcgroup string, pid int) error

	// GetPids returns the PIDs of all processes inside the cgroup.
	GetPids() ([]int, error)

	// GetAllPids returns the PIDs of all processes inside the cgroup
	// any all its sub-cgroups.
	GetAllPids() ([]int, error)

	// GetStats returns cgroups statistics.
	GetStats() (*Stats, error)

	// Freeze sets the freezer cgroup to the specified state.
	Freeze(state FreezerState) error

	// Destroy removes cgroup.
	Destroy() error

	// Path returns a cgroup path to the specified controller/subsystem.
	// For cgroupv2, the argument is unused and can be empty.
	Path(string) string

	// Set sets cgroup resources parameters/limits. If the argument is nil,
	// the resources specified during Manager creation (or the previous call
	// to Set) are used.
	Set(r *Resources) error

	// GetPaths returns cgroup path(s) to save in a state file in order to
	// restore later.
	//
	// For cgroup v1, a key is cgroup subsystem name, and the value is the
	// path to the cgroup for this subsystem.
	//
	// For cgroup v2 unified hierarchy, a key is "", and the value is the
	// unified path.
	GetPaths() map[string]string

	// GetCgroups returns the cgroup data as configured.
	GetCgroups() (*Cgroup, error)

	// GetFreezerState retrieves the current FreezerState of the cgroup.
	GetFreezerState() (FreezerState, error)

	// Exists returns whether the cgroup path exists or not.
	Exists() bool

	// OOMKillCount reports OOM kill count for the cgroup.
	OOMKillCount() (uint64, error)
}

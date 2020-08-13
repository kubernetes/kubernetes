// +build linux

package cgroups

import (
	"github.com/opencontainers/runc/libcontainer/configs"
)

type Manager interface {
	// Applies cgroup configuration to the process with the specified pid
	Apply(pid int) error

	// Returns the PIDs inside the cgroup set
	GetPids() ([]int, error)

	// Returns the PIDs inside the cgroup set & all sub-cgroups
	GetAllPids() ([]int, error)

	// Returns statistics for the cgroup set
	GetStats() (*Stats, error)

	// Toggles the freezer cgroup according with specified state
	Freeze(state configs.FreezerState) error

	// Destroys the cgroup set
	Destroy() error

	// Path returns a cgroup path to the specified controller/subsystem.
	// For cgroupv2, the argument is unused and can be empty.
	Path(string) string

	// Sets the cgroup as configured.
	Set(container *configs.Config) error

	// GetPaths returns cgroup path(s) to save in a state file in order to restore later.
	//
	// For cgroup v1, a key is cgroup subsystem name, and the value is the path
	// to the cgroup for this subsystem.
	//
	// For cgroup v2 unified hierarchy, a key is "", and the value is the unified path.
	GetPaths() map[string]string

	// GetCgroups returns the cgroup data as configured.
	GetCgroups() (*configs.Cgroup, error)

	// GetFreezerState retrieves the current FreezerState of the cgroup.
	GetFreezerState() (configs.FreezerState, error)

	// Whether the cgroup path exists or not
	Exists() bool
}

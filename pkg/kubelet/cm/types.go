/*
Copyright 2016 The Kubernetes Authors.

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

package cm

import (
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
)

// ResourceConfig holds information about all the supported cgroup resource parameters.
type ResourceConfig struct {
	// Memory limit (in bytes).
	Memory *int64
	// CPU shares (relative weight vs. other containers).
	CpuShares *uint64
	// CPU hardcap limit (in usecs). Allowed cpu time in a given period.
	CpuQuota *int64
	// CPU quota period.
	CpuPeriod *uint64
	// HugePageLimit map from page size (in bytes) to limit (in bytes)
	HugePageLimit map[int64]int64
	// Maximum number of pids
	PodPidsLimit *int64
}

// CgroupName is the abstract name of a cgroup prior to any driver specific conversion.
type CgroupName string

// CgroupConfig holds the cgroup configuration information.
// This is common object which is used to specify
// cgroup information to both systemd and raw cgroup fs
// implementation of the Cgroup Manager interface.
type CgroupConfig struct {
	// Fully qualified name prior to any driver specific conversions.
	Name CgroupName
	// ResourceParameters contains various cgroups settings to apply.
	ResourceParameters *ResourceConfig
}

// MemoryStats holds the on-demand statistics from the memory cgroup
type MemoryStats struct {
	// Memory usage (in bytes).
	Usage int64
}

// ResourceStats holds on-demand statistics from various cgroup subsystems
type ResourceStats struct {
	// Memory statistics.
	MemoryStats *MemoryStats
}

// CgroupManager allows for cgroup management.
// Supports Cgroup Creation ,Deletion and Updates.
type CgroupManager interface {
	// Create creates and applies the cgroup configurations on the cgroup.
	// It just creates the leaf cgroups.
	// It expects the parent cgroup to already exist.
	Create(*CgroupConfig) error
	// Destroy the cgroup.
	Destroy(*CgroupConfig) error
	// Update cgroup configuration.
	Update(*CgroupConfig) error
	// Exists checks if the cgroup already exists
	Exists(name CgroupName) bool
	// Name returns the literal cgroupfs name on the host after any driver specific conversions.
	// We would expect systemd implementation to make appropriate name conversion.
	// For example, if we pass /foo/bar
	// then systemd should convert the name to something like
	// foo.slice/foo-bar.slice
	Name(name CgroupName) string
	// CgroupName converts the literal cgroupfs name on the host to an internal identifier.
	CgroupName(name string) CgroupName
	// Pids scans through all subsystems to find pids associated with specified cgroup.
	Pids(name CgroupName) []int
	// ReduceCPULimits reduces the CPU CFS values to the minimum amount of shares.
	ReduceCPULimits(cgroupName CgroupName) error
	// GetResourceStats returns statistics of the specified cgroup as read from the cgroup fs.
	GetResourceStats(name CgroupName) (*ResourceStats, error)
}

// QOSContainersInfo stores the names of containers per qos
type QOSContainersInfo struct {
	Guaranteed string
	BestEffort string
	Burstable  string
}

// PodContainerManager stores and manages pod level containers
// The Pod workers interact with the PodContainerManager to create and destroy
// containers for the pod.
type PodContainerManager interface {
	// GetPodContainerName returns the CgroupName identifier, and its literal cgroupfs form on the host.
	GetPodContainerName(*v1.Pod) (CgroupName, string)

	// EnsureExists takes a pod as argument and makes sure that
	// pod cgroup exists if qos cgroup hierarchy flag is enabled.
	// If the pod cgroup doesn't already exist this method creates it.
	EnsureExists(*v1.Pod) error

	// Exists returns true if the pod cgroup exists.
	Exists(*v1.Pod) bool

	// Destroy takes a pod Cgroup name as argument and destroys the pod's container.
	Destroy(name CgroupName) error

	// ReduceCPULimits reduces the CPU CFS values to the minimum amount of shares.
	ReduceCPULimits(name CgroupName) error

	// GetAllPodsFromCgroups enumerates the set of pod uids to their associated cgroup based on state of cgroupfs system.
	GetAllPodsFromCgroups() (map[types.UID]CgroupName, error)

	Update(*v1.Pod) error
}

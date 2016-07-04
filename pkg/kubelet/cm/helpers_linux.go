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
	"fmt"
	"path"

	libcontainercgroups "github.com/opencontainers/runc/libcontainer/cgroups"
	cgroupfs "github.com/opencontainers/runc/libcontainer/cgroups/fs"
	libcontainerconfigs "github.com/opencontainers/runc/libcontainer/configs"
)

// cgroupSubsystems holds information about the mounted cgroup subsytems
type cgroupSubsystems struct {
	// Cgroup subsystem mounts.
	// e.g.: "/sys/fs/cgroup/cpu" -> ["cpu", "cpuacct"]
	mounts []libcontainercgroups.Mount

	// Cgroup subsystem to their mount location.
	// e.g.: "cpu" -> "/sys/fs/cgroup/cpu"
	mountPoints map[string]string
}

// GetCgroupSubsystems returns information about the mounted cgroup subsystems
func getCgroupSubsystems() (*cgroupSubsystems, error) {
	// Get all cgroup mounts.
	allCgroups, err := libcontainercgroups.GetCgroupMounts()
	if err != nil {
		return &cgroupSubsystems{}, err
	}
	if len(allCgroups) == 0 {
		return &cgroupSubsystems{}, fmt.Errorf("failed to find cgroup mounts")
	}

	//TODO(@dubstack) should we trim to only the supported ones
	mountPoints := make(map[string]string, len(allCgroups))
	for _, mount := range allCgroups {
		for _, subsystem := range mount.Subsystems {
			mountPoints[subsystem] = mount.Mountpoint
		}
	}
	return &cgroupSubsystems{
		mounts:      allCgroups,
		mountPoints: mountPoints,
	}, nil
}

// getLibcontainerCgroupManager returns libcontainer's cgroups manager
// object with the specified cgroup configuration
func getLibcontainerCgroupManager(cgroupConfig *CgroupConfig, subsystems *cgroupSubsystems) (*cgroupfs.Manager, error) {
	// get cgroup name
	name := cgroupConfig.Name

	// Get map of all cgroup paths on the system for the particular cgroup
	cgroupPaths := make(map[string]string, len(subsystems.mountPoints))
	for key, val := range subsystems.mountPoints {
		cgroupPaths[key] = path.Join(val, name)
	}

	// Extract the cgroup resource parameters
	resourceConfig := cgroupConfig.ResourceParameters
	resources := &libcontainerconfigs.Resources{}
	resources.AllowAllDevices = true
	if resourceConfig.Memory != nil {
		resources.Memory = *resourceConfig.Memory
	}
	if resourceConfig.CpuShares != nil {
		resources.CpuShares = *resourceConfig.CpuShares
	}
	if resourceConfig.CpuQuota != nil {
		resources.CpuQuota = *resourceConfig.CpuQuota
	}
	// Initialize libcontainer's cgroup config
	libcontainerCgroupConfig := &libcontainerconfigs.Cgroup{
		Name:      path.Base(name),
		Parent:    path.Dir(name),
		Resources: resources,
	}
	return &cgroupfs.Manager{
		Cgroups: libcontainerCgroupConfig,
		Paths:   cgroupPaths,
	}, nil
}

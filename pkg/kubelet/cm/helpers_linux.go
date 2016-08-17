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

	libcontainercgroups "github.com/opencontainers/runc/libcontainer/cgroups"
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

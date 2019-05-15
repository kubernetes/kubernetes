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

package util

import (
	"path/filepath"

	libcontainercgroups "github.com/opencontainers/runc/libcontainer/cgroups"
	libcontainerutils "github.com/opencontainers/runc/libcontainer/utils"
)

// GetPids gets pids of the desired cgroup
// Forked from opencontainers/runc/libcontainer/cgroup/fs.Manager.GetPids()
func GetPids(cgroupPath string) ([]int, error) {
	dir, err := getCgroupPath(cgroupPath)
	if err != nil {
		return nil, err
	}
	return libcontainercgroups.GetPids(dir)
}

// getCgroupPath gets the file path to the "devices" subsystem of the desired cgroup.
// cgroupPath is the path in the cgroup hierarchy.
func getCgroupPath(cgroupPath string) (string, error) {
	cgroupPath = libcontainerutils.CleanPath(cgroupPath)

	mnt, root, err := libcontainercgroups.FindCgroupMountpointAndRoot("devices")
	// If we didn't mount the subsystem, there is no point we make the path.
	if err != nil {
		return "", err
	}

	// If the cgroup name/path is absolute do not look relative to the cgroup of the init process.
	if filepath.IsAbs(cgroupPath) {
		// Sometimes subsystems can be mounted together as 'cpu,cpuacct'.
		return filepath.Join(root, mnt, cgroupPath), nil
	}

	parentPath, err := getCgroupParentPath(mnt, root)
	if err != nil {
		return "", err
	}

	return filepath.Join(parentPath, cgroupPath), nil
}

// getCgroupParentPath gets the parent filepath to this cgroup, for resolving relative cgroup paths.
func getCgroupParentPath(mountpoint, root string) (string, error) {
	// Use GetThisCgroupDir instead of GetInitCgroupDir, because the creating
	// process could in container and shared pid namespace with host, and
	// /proc/1/cgroup could point to whole other world of cgroups.
	initPath, err := libcontainercgroups.GetOwnCgroup("devices")
	if err != nil {
		return "", err
	}
	// This is needed for nested containers, because in /proc/self/cgroup we
	// see paths from host, which don't exist in container.
	relDir, err := filepath.Rel(root, initPath)
	if err != nil {
		return "", err
	}
	return filepath.Join(mountpoint, relDir), nil
}

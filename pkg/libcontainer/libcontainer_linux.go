//go:build linux
// +build linux

/*
Copyright 2023 The Kubernetes Authors.

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

package libcontainer

/*
Copyright 2023 The Kubernetes Authors.

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

import (
	libcontainercgroups "github.com/opencontainers/runc/libcontainer/cgroups"
	libcontainerfscommon "github.com/opencontainers/runc/libcontainer/cgroups/fscommon"
	libcontainermanager "github.com/opencontainers/runc/libcontainer/cgroups/manager"
	libcontainersystemd "github.com/opencontainers/runc/libcontainer/cgroups/systemd"
	libcontainerconfigs "github.com/opencontainers/runc/libcontainer/configs"
	libcontainerutils "github.com/opencontainers/runc/libcontainer/utils"
)

var (
	IsCgroup2UnifiedMode        = libcontainercgroups.IsCgroup2UnifiedMode
	IsNotFound                  = libcontainercgroups.IsNotFound
	HugePageSizes               = libcontainercgroups.HugePageSizes
	GetPids                     = libcontainercgroups.GetPids
	FindCgroupMountpointAndRoot = libcontainercgroups.FindCgroupMountpointAndRoot
	GetOwnCgroup                = libcontainercgroups.GetOwnCgroup
	ParseCgroupFile             = libcontainercgroups.ParseCgroupFile
	NewNotFoundError            = libcontainercgroups.NewNotFoundError
	GetAllSubsystems            = libcontainercgroups.GetAllSubsystems
	PathExists                  = libcontainercgroups.PathExists

	New                  = libcontainermanager.New
	GetCgroupParamUint   = libcontainerfscommon.GetCgroupParamUint
	GetCgroupParamString = libcontainerfscommon.GetCgroupParamString

	ExpandSlice = libcontainersystemd.ExpandSlice

	CleanPath = libcontainerutils.CleanPath
)

type Mount libcontainercgroups.Mount

type Manager libcontainercgroups.Manager

type Cgroup libcontainerconfigs.Cgroup
type Resources libcontainerconfigs.Resources
type HugepageLimit libcontainerconfigs.HugepageLimit

func GetCgroupMounts(all bool) ([]Mount, error) {
	mounts, err := libcontainercgroups.GetCgroupMounts(all)
	if err != nil {
		return nil, err
	}
	var ret []Mount
	for _, m := range mounts {
		ret = append(ret, Mount{
			Mountpoint: m.Mountpoint,
			Root:       m.Root,
			Subsystems: m.Subsystems,
		})
	}
	return ret, nil
}

// Create a cgroup container manager.
func CreateContainerManager(containerName string) (Manager, error) {
	cg := &libcontainerconfigs.Cgroup{
		Parent: "/",
		Name:   containerName,
		Resources: &libcontainerconfigs.Resources{
			SkipDevices: true,
		},
		Systemd: false,
	}

	return libcontainermanager.New(cg)
}

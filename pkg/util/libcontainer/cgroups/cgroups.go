package cgroups

import (
	libcontainercgroups "github.com/opencontainers/runc/libcontainer/cgroups"
)

var (
	IsCgroup2UnifiedMode        = libcontainercgroups.IsCgroup2UnifiedMode
	ParseCgroupFile             = libcontainercgroups.ParseCgroupFile
	NewNotFoundError            = libcontainercgroups.NewNotFoundError
	IsNotFound                  = libcontainercgroups.IsNotFound
	HugePageSizes               = libcontainercgroups.HugePageSizes
	PathExists                  = libcontainercgroups.PathExists
	FindCgroupMountpointAndRoot = libcontainercgroups.FindCgroupMountpointAndRoot
	GetOwnCgroup                = libcontainercgroups.GetOwnCgroup
	GetPids                     = libcontainercgroups.GetPids
)

type (
	Manager = libcontainercgroups.Manager
)

type Mount struct {
	Mountpoint string
	Root       string
	Subsystems []string
}

func GetCgroupMounts(all bool) ([]Mount, error) {
	if mounts, err := libcontainercgroups.GetCgroupMounts(all); err != nil {
		return nil, err
	} else {
		var allMounts []Mount
		for _, mount := range mounts {
			allMounts = append(allMounts, Mount{
				Mountpoint: mount.Mountpoint,
				Root:       mount.Root,
				Subsystems: mount.Subsystems,
			})
		}
		return allMounts, nil
	}
}

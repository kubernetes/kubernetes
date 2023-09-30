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
	GetCgroupMounts             = libcontainercgroups.GetCgroupMounts
	GetAllSubsystems            = libcontainercgroups.GetAllSubsystems
	PathExists                  = libcontainercgroups.PathExists
	FindCgroupMountpointAndRoot = libcontainercgroups.FindCgroupMountpointAndRoot
	GetOwnCgroup                = libcontainercgroups.GetOwnCgroup
	GetPids                     = libcontainercgroups.GetPids
)

type (
	Mount   = libcontainercgroups.Mount
	Manager = libcontainercgroups.Manager
)

package configs

import (
	libcontainerconfigs "github.com/opencontainers/runc/libcontainer/configs"
)

var ()

type (
	Cgroup        = libcontainerconfigs.Cgroup
	Resources     = libcontainerconfigs.Resources
	HugepageLimit = libcontainerconfigs.HugepageLimit
)

package manager

import (
	libcontainermanager "github.com/opencontainers/runc/libcontainer/cgroups/manager"
)

var (
	New = libcontainermanager.New
)

package apparmor

import (
	libcontainerapparmor "github.com/opencontainers/runc/libcontainer/apparmor"
)

var (
	IsEnabled = libcontainerapparmor.IsEnabled
)

package userns

import (
	libcontaineruserns "github.com/opencontainers/runc/libcontainer/userns"
)

var (
	RunningInUserNS = libcontaineruserns.RunningInUserNS
)

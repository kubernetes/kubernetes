package apparmor

import "github.com/opencontainers/runc/libcontainer/apparmor"

// Shim interface allows injection of bad host configuration for testing purposes
type Shim interface {
	IsEnabled() bool
}

type shim struct{}

func (s shim) IsEnabled() bool {
	return apparmor.IsEnabled()
}

// +build !linux

package userns

import "github.com/opencontainers/runc/libcontainer/user"

// runningInUserNS is a stub for non-Linux systems
// Always returns false
func runningInUserNS() bool {
	return false
}

// uidMapInUserNS is a stub for non-Linux systems
// Always returns false
func uidMapInUserNS(uidmap []user.IDMap) bool {
	return false
}

//go:build !linux
// +build !linux

package userns

import "github.com/dims/libcontainer/user"

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

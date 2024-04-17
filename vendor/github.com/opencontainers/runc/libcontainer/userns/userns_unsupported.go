//go:build !linux
// +build !linux

package userns

// runningInUserNS is a stub for non-Linux systems
// Always returns false
func runningInUserNS() bool {
	return false
}

// uidMapInUserNS is a stub for non-Linux systems
// Always returns false
func uidMapInUserNS(uidMap string) bool {
	return false
}

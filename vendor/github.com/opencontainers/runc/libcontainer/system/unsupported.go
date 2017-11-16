// +build !linux

package system

// RunningInUserNS is a stub for non-Linux systems
// Always returns false
func RunningInUserNS() bool {
	return false
}

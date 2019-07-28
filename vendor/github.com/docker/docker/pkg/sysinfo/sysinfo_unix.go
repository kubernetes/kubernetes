// +build !linux,!windows

package sysinfo // import "github.com/docker/docker/pkg/sysinfo"

// New returns an empty SysInfo for non linux for now.
func New(quiet bool) *SysInfo {
	sysInfo := &SysInfo{}
	return sysInfo
}

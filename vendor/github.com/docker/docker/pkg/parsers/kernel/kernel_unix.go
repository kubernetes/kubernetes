// +build linux freebsd openbsd

// Package kernel provides helper function to get, parse and compare kernel
// versions for different platforms.
package kernel

import (
	"bytes"

	"github.com/sirupsen/logrus"
)

// GetKernelVersion gets the current kernel version.
func GetKernelVersion() (*VersionInfo, error) {
	uts, err := uname()
	if err != nil {
		return nil, err
	}

	// Remove the \x00 from the release for Atoi to parse correctly
	return ParseRelease(string(uts.Release[:bytes.IndexByte(uts.Release[:], 0)]))
}

// CheckKernelVersion checks if current kernel is newer than (or equal to)
// the given version.
func CheckKernelVersion(k, major, minor int) bool {
	if v, err := GetKernelVersion(); err != nil {
		logrus.Warnf("error getting kernel version: %s", err)
	} else {
		if CompareKernelVersion(*v, VersionInfo{Kernel: k, Major: major, Minor: minor}) < 0 {
			return false
		}
	}
	return true
}

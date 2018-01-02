// +build windows

package main

const (
	// identifies if test suite is running on a unix platform
	isUnixCli = false

	// this is the expected file permission set on windows: gh#11395
	expectedFileChmod = "-rwxr-xr-x"

	// On Windows, the busybox image doesn't have the `top` command, so we rely
	// on `sleep` with a high duration.
	defaultSleepImage = "busybox"
)

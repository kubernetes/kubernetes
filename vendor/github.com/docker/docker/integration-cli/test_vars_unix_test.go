// +build !windows

package main

const (
	// identifies if test suite is running on a unix platform
	isUnixCli = true

	expectedFileChmod = "-rw-r--r--"

	// On Unix variants, the busybox image comes with the `top` command which
	// runs indefinitely while still being interruptible by a signal.
	defaultSleepImage = "busybox"
)

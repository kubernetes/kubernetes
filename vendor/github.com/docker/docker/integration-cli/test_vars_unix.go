// +build !windows

package main

const (
	// identifies if test suite is running on a unix platform
	isUnixCli = true

	expectedFileChmod = "-rw-r--r--"
)

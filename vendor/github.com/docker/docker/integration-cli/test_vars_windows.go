// +build windows

package main

const (
	// identifies if test suite is running on a unix platform
	isUnixCli = false

	// this is the expected file permission set on windows: gh#11395
	expectedFileChmod = "-rwxr-xr-x"
)

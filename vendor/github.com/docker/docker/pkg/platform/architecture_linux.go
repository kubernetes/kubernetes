// Package platform provides helper function to get the runtime architecture
// for different platforms.
package platform

import (
	"golang.org/x/sys/unix"
)

// runtimeArchitecture gets the name of the current architecture (x86, x86_64, …)
func runtimeArchitecture() (string, error) {
	utsname := &unix.Utsname{}
	if err := unix.Uname(utsname); err != nil {
		return "", err
	}
	return charsToString(utsname.Machine), nil
}

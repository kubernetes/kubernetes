// +build freebsd darwin

// Package platform provides helper function to get the runtime architecture
// for different platforms.
package platform

import (
	"os/exec"
	"strings"
)

// runtimeArchitecture gets the name of the current architecture (x86, x86_64, i86pc, sun4v, ...)
func runtimeArchitecture() (string, error) {
	cmd := exec.Command("/usr/bin/uname", "-m")
	machine, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(machine)), nil
}

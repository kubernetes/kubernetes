// +build freebsd darwin

package operatingsystem // import "github.com/docker/docker/pkg/parsers/operatingsystem"

import (
	"errors"
	"os/exec"
)

// GetOperatingSystem gets the name of the current operating system.
func GetOperatingSystem() (string, error) {
	cmd := exec.Command("uname", "-s")
	osName, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return string(osName), nil
}

// IsContainerized returns true if we are running inside a container.
// No-op on FreeBSD and Darwin, always returns false.
func IsContainerized() (bool, error) {
	// TODO: Implement jail detection for freeBSD
	return false, errors.New("Cannot detect if we are in container")
}

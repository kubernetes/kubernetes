// +build darwin

// Package kernel provides helper function to get, parse and compare kernel
// versions for different platforms.
package kernel

import (
	"fmt"
	"os/exec"
	"strings"

	"github.com/mattn/go-shellwords"
)

// GetKernelVersion gets the current kernel version.
func GetKernelVersion() (*VersionInfo, error) {
	release, err := getRelease()
	if err != nil {
		return nil, err
	}

	return ParseRelease(release)
}

// getRelease uses `system_profiler SPSoftwareDataType` to get OSX kernel version
func getRelease() (string, error) {
	cmd := exec.Command("system_profiler", "SPSoftwareDataType")
	osName, err := cmd.Output()
	if err != nil {
		return "", err
	}

	var release string
	data := strings.Split(string(osName), "\n")
	for _, line := range data {
		if strings.Contains(line, "Kernel Version") {
			// It has the format like '      Kernel Version: Darwin 14.5.0'
			content := strings.SplitN(line, ":", 2)
			if len(content) != 2 {
				return "", fmt.Errorf("Kernel Version is invalid")
			}

			prettyNames, err := shellwords.Parse(content[1])
			if err != nil {
				return "", fmt.Errorf("Kernel Version is invalid: %s", err.Error())
			}

			if len(prettyNames) != 2 {
				return "", fmt.Errorf("Kernel Version needs to be 'Darwin x.x.x' ")
			}
			release = prettyNames[1]
		}
	}

	return release, nil
}

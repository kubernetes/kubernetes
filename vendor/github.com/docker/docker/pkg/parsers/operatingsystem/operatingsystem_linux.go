// Package operatingsystem provides helper function to get the operating system
// name for different platforms.
package operatingsystem // import "github.com/docker/docker/pkg/parsers/operatingsystem"

import (
	"bufio"
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"strings"

	"github.com/mattn/go-shellwords"
)

var (
	// file to use to detect if the daemon is running in a container
	proc1Cgroup = "/proc/1/cgroup"

	// file to check to determine Operating System
	etcOsRelease = "/etc/os-release"

	// used by stateless systems like Clear Linux
	altOsRelease = "/usr/lib/os-release"
)

// GetOperatingSystem gets the name of the current operating system.
func GetOperatingSystem() (string, error) {
	osReleaseFile, err := os.Open(etcOsRelease)
	if err != nil {
		if !os.IsNotExist(err) {
			return "", fmt.Errorf("Error opening %s: %v", etcOsRelease, err)
		}
		osReleaseFile, err = os.Open(altOsRelease)
		if err != nil {
			return "", fmt.Errorf("Error opening %s: %v", altOsRelease, err)
		}
	}
	defer osReleaseFile.Close()

	var prettyName string
	scanner := bufio.NewScanner(osReleaseFile)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "PRETTY_NAME=") {
			data := strings.SplitN(line, "=", 2)
			prettyNames, err := shellwords.Parse(data[1])
			if err != nil {
				return "", fmt.Errorf("PRETTY_NAME is invalid: %s", err.Error())
			}
			if len(prettyNames) != 1 {
				return "", fmt.Errorf("PRETTY_NAME needs to be enclosed by quotes if they have spaces: %s", data[1])
			}
			prettyName = prettyNames[0]
		}
	}
	if prettyName != "" {
		return prettyName, nil
	}
	// If not set, defaults to PRETTY_NAME="Linux"
	// c.f. http://www.freedesktop.org/software/systemd/man/os-release.html
	return "Linux", nil
}

// IsContainerized returns true if we are running inside a container.
func IsContainerized() (bool, error) {
	b, err := ioutil.ReadFile(proc1Cgroup)
	if err != nil {
		return false, err
	}
	for _, line := range bytes.Split(b, []byte{'\n'}) {
		if len(line) > 0 && !bytes.HasSuffix(line, []byte{'/'}) && !bytes.HasSuffix(line, []byte("init.scope")) {
			return true, nil
		}
	}
	return false, nil
}

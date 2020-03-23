/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package e2enode

import (
	"fmt"
	"strings"

	"github.com/blang/semver"
	systemdutil "github.com/coreos/go-systemd/util"
)

// getDockerAPIVersion returns the Docker's API version.
func getDockerAPIVersion() (semver.Version, error) {
	output, err := runCommand("docker", "version", "-f", "{{.Server.APIVersion}}")
	if err != nil {
		return semver.Version{}, fmt.Errorf("failed to get docker server version: %v", err)
	}
	return semver.MustParse(strings.TrimSpace(output) + ".0"), nil
}

// isSharedPIDNamespaceSupported returns true if the Docker version is 1.13.1+
// (API version 1.26+), and false otherwise.
func isSharedPIDNamespaceSupported() (bool, error) {
	version, err := getDockerAPIVersion()
	if err != nil {
		return false, err
	}
	return version.GTE(semver.MustParse("1.26.0")), nil
}

// isDockerLiveRestoreSupported returns true if live-restore is supported in
// the current Docker version.
func isDockerLiveRestoreSupported() (bool, error) {
	version, err := getDockerAPIVersion()
	if err != nil {
		return false, err
	}
	return version.GTE(semver.MustParse("1.26.0")), nil
}

// getDockerInfo returns the Info struct for the running Docker daemon.
func getDockerInfo(key string) (string, error) {
	output, err := runCommand("docker", "info", "-f", "{{."+key+"}}")
	if err != nil {
		return "", fmt.Errorf("failed to get docker info: %v", err)
	}
	return strings.TrimSpace(output), nil
}

// isDockerLiveRestoreEnabled returns true if live-restore is enabled in the
// Docker.
func isDockerLiveRestoreEnabled() (bool, error) {
	info, err := getDockerInfo("LiveRestoreEnabled")
	if err != nil {
		return false, err
	}
	return info == "true", nil
}

// getDockerLoggingDriver returns the name of the logging driver.
func getDockerLoggingDriver() (string, error) {
	info, err := getDockerInfo("LoggingDriver")
	if err != nil {
		return "", err
	}
	return info, nil
}

// isDockerSELinuxSupportEnabled checks whether the Docker daemon was started
// with SELinux support enabled.
func isDockerSELinuxSupportEnabled() (bool, error) {
	info, err := getDockerInfo("SecurityOptions")
	if err != nil {
		return false, err
	}
	return strings.Contains(info, "name=selinux"), nil
}

// startDockerDaemon starts the Docker daemon.
func startDockerDaemon() error {
	switch {
	case systemdutil.IsRunningSystemd():
		_, err := runCommand("systemctl", "start", "docker")
		return err
	default:
		_, err := runCommand("service", "docker", "start")
		return err
	}
}

// stopDockerDaemon stops the Docker daemon.
func stopDockerDaemon() error {
	switch {
	case systemdutil.IsRunningSystemd():
		_, err := runCommand("systemctl", "stop", "docker")
		return err
	default:
		_, err := runCommand("service", "docker", "stop")
		return err
	}
}

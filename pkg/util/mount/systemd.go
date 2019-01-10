// +build linux

/*
Copyright 2018 The Kubernetes Authors.

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

package mount

import (
	"github.com/golang/glog"
)

const (
	defaultSystemdRunPath = "systemd-run"
)

// DetectSystemd returns true if OS runs with systemd as init. When not sure
// (permission errors, ...), it returns false.
// There may be different ways how to detect systemd, this one makes sure that
// systemd-runs works.
func DetectSystemd(exec Exec, systemdRunPath string) bool {
	if systemdRunPath == "" {
		systemdRunPath = defaultSystemdRunPath
	}
	// Try to run systemd-run --scope /bin/true, that should be enough
	// to make sure that systemd is really running and not just installed,
	// which happens when running in a container with a systemd-based image
	// but with different pid 1.
	output, err := exec.Run(systemdRunPath, "--description=Kubernetes systemd probe", "--scope", "true")
	if err != nil {
		glog.V(4).Infof("Cannot run systemd-run, assuming non-systemd OS")
		glog.V(4).Infof("systemd-run failed with: %v", err)
		glog.V(4).Infof("systemd-run output: %s", string(output))
		return false
	}
	return true
}

// AddSystemdScope adds "system-run --scope" to given command line. This helps
// to run command in isolated scope.
func AddSystemdScope(systemdRunPath string, description, command string, args []string) (string, []string) {
	if systemdRunPath == "" {
		systemdRunPath = defaultSystemdRunPath
	}
	systemdRunArgs := []string{"--description", description, "--scope", "--", command}
	return systemdRunPath, append(systemdRunArgs, args...)
}

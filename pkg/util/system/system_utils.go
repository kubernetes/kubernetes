/*
Copyright 2016 The Kubernetes Authors.

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

package system

import (
	"fmt"
	"os/exec"
)

// detectSystemd returns true if OS runs with systemd as init. When not sure
// (permission errors, ...), it returns false.
// There may be different ways how to detect systemd, this one makes sure that
// systemd-runs (needed by Mount()) works.
func DetectSystemd() bool {
	if _, err := exec.LookPath("systemd-run"); err != nil {
		return false
	}
	// Try to run systemd-run --scope /bin/true, that should be enough
	// to make sure that systemd is really running and not just installed,
	// which happens when running in a container with a systemd-based image
	// but with different pid 1.
	cmd := exec.Command("systemd-run", "--description=Kubernetes systemd probe", "--scope", "true")
	_, err := cmd.CombinedOutput()
	if err != nil {
		return false
	}
	return true
}


// AddSystemdScope adds "system-run --scope" to given command line
// implementation is shared with NsEnterMounter
func AddSystemdScope(systemdRunPath, mountName, command string, args []string) (string, []string) {
	descriptionArg := fmt.Sprintf("--description=Kubernetes transient %s for %s", command, mountName)
	systemdRunArgs := []string{descriptionArg, "--scope", "--", command}
	return systemdRunPath, append(systemdRunArgs, args...)
}

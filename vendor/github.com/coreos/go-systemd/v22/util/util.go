// Copyright 2015 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package util contains utility functions related to systemd that applications
// can use to check things like whether systemd is running.  Note that some of
// these functions attempt to manually load systemd libraries at runtime rather
// than linking against them.
package util

import (
	"fmt"
	"os"
	"strings"
)

var ErrNoCGO = fmt.Errorf("go-systemd built with CGO disabled")

// GetRunningSlice attempts to retrieve the name of the systemd slice in which
// the current process is running.
// This function is a wrapper around the libsystemd C library; if it cannot be
// opened, an error is returned.
func GetRunningSlice() (string, error) {
	return getRunningSlice()
}

// RunningFromSystemService tries to detect whether the current process has
// been invoked from a system service. The condition for this is whether the
// process is _not_ a user process. User processes are those running in session
// scopes or under per-user `systemd --user` instances.
//
// To avoid false positives on systems without `pam_systemd` (which is
// responsible for creating user sessions), this function also uses a heuristic
// to detect whether it's being invoked from a session leader process. This is
// the case if the current process is executed directly from a service file
// (e.g. with `ExecStart=/this/cmd`). Note that this heuristic will fail if the
// command is instead launched in a subshell or similar so that it is not
// session leader (e.g. `ExecStart=/bin/bash -c "/this/cmd"`)
//
// This function is a wrapper around the libsystemd C library; if this is
// unable to successfully open a handle to the library for any reason (e.g. it
// cannot be found), an error will be returned.
func RunningFromSystemService() (bool, error) {
	return runningFromSystemService()
}

// CurrentUnitName attempts to retrieve the name of the systemd system unit
// from which the calling process has been invoked. It wraps the systemd
// `sd_pid_get_unit` call, with the same caveat: for processes not part of a
// systemd system unit, this function will return an error.
func CurrentUnitName() (string, error) {
	return currentUnitName()
}

// IsRunningSystemd checks whether the host was booted with systemd as its init
// system. This functions similarly to systemd's `sd_booted(3)`: internally, it
// checks whether /run/systemd/system/ exists and is a directory.
// http://www.freedesktop.org/software/systemd/man/sd_booted.html
func IsRunningSystemd() bool {
	fi, err := os.Lstat("/run/systemd/system")
	if err != nil {
		return false
	}
	return fi.IsDir()
}

// GetMachineID returns a host's 128-bit machine ID as a string. This functions
// similarly to systemd's `sd_id128_get_machine`: internally, it simply reads
// the contents of /etc/machine-id
// http://www.freedesktop.org/software/systemd/man/sd_id128_get_machine.html
func GetMachineID() (string, error) {
	machineID, err := os.ReadFile("/etc/machine-id")
	if err != nil {
		return "", fmt.Errorf("failed to read /etc/machine-id: %v", err)
	}
	return strings.TrimSpace(string(machineID)), nil
}

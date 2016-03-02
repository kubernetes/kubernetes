/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package rkt

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"

	"github.com/coreos/go-systemd/dbus"
)

// systemdVersion is a type wraps the int to implement kubecontainer.Version interface.
type systemdVersion int

func (s systemdVersion) String() string {
	return fmt.Sprintf("%d", s)
}

func (s systemdVersion) Compare(other string) (int, error) {
	v, err := strconv.Atoi(other)
	if err != nil {
		return -1, err
	}
	if int(s) < v {
		return -1, nil
	} else if int(s) > v {
		return 1, nil
	}
	return 0, nil
}

// systemdInterface is an abstraction of the go-systemd/dbus to make
// it mockable for testing.
// TODO(yifan): Eventually we should move these functionalities to:
// 1. a package for launching/stopping rkt pods.
// 2. rkt api-service interface for listing pods.
// See https://github.com/coreos/rkt/issues/1769.
type systemdInterface interface {
	// Version returns the version of the systemd.
	Version() (systemdVersion, error)
	// ListUnits lists all the loaded units.
	ListUnits() ([]dbus.UnitStatus, error)
	// StopUnits stops the unit with the given name.
	StopUnit(name string, mode string, ch chan<- string) (int, error)
	// StopUnits restarts the unit with the given name.
	RestartUnit(name string, mode string, ch chan<- string) (int, error)
	// Reload is equivalent to 'systemctl daemon-reload'.
	Reload() error
}

// systemd implements the systemdInterface using dbus and systemctl.
// All the functions other then Version() are already implemented by go-systemd/dbus.
type systemd struct {
	*dbus.Conn
}

// newSystemd creates a systemd object that implements systemdInterface.
func newSystemd() (*systemd, error) {
	dbusConn, err := dbus.New()
	if err != nil {
		return nil, err
	}
	return &systemd{dbusConn}, nil
}

// Version returns the version of the systemd.
func (s *systemd) Version() (systemdVersion, error) {
	output, err := exec.Command("systemctl", "--version").Output()
	if err != nil {
		return -1, err
	}
	// Example output of 'systemctl --version':
	//
	// systemd 215
	// +PAM +AUDIT +SELINUX +IMA +SYSVINIT +LIBCRYPTSETUP +GCRYPT +ACL +XZ -SECCOMP -APPARMOR
	//
	lines := strings.Split(string(output), "\n")
	tuples := strings.Split(lines[0], " ")
	if len(tuples) != 2 {
		return -1, fmt.Errorf("rkt: Failed to parse version %v", lines)
	}
	result, err := strconv.Atoi(string(tuples[1]))
	if err != nil {
		return -1, err
	}
	return systemdVersion(result), nil
}

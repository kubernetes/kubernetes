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

package initsystem

import (
	"fmt"
	"os/exec"
	"strings"
)

type InitSystem interface {
	// ServiceStart tries to start a specific service
	ServiceStart(service string) error

	// ServiceStop tries to stop a specific service
	ServiceStop(service string) error

	// ServiceExists ensures the service is defined for this init system.
	ServiceExists(service string) bool

	// ServiceIsEnabled ensures the service is enabled to start on each boot.
	ServiceIsEnabled(service string) bool

	// ServiceIsActive ensures the service is running, or attempting to run. (crash looping in the case of kubelet)
	ServiceIsActive(service string) bool
}

type SystemdInitSystem struct{}

func (sysd SystemdInitSystem) ServiceStart(service string) error {
	args := []string{"start", service}
	_, err := exec.Command("systemctl", args...).Output()
	return err
}

func (sysd SystemdInitSystem) ServiceStop(service string) error {
	args := []string{"stop", service}
	_, err := exec.Command("systemctl", args...).Output()
	return err
}

func (sysd SystemdInitSystem) ServiceExists(service string) bool {
	args := []string{"status", service}
	outBytes, _ := exec.Command("systemctl", args...).Output()
	output := string(outBytes)
	if strings.Contains(output, "Loaded: not-found") {
		return false
	}
	return true
}

func (sysd SystemdInitSystem) ServiceIsEnabled(service string) bool {
	args := []string{"is-enabled", service}
	_, err := exec.Command("systemctl", args...).Output()
	if err != nil {
		return false
	}
	return true
}

// ServiceIsActive will check is the service is "active". In the case of
// crash looping services (kubelet in our case) status will return as
// "activating", so we will consider this active as well.
func (sysd SystemdInitSystem) ServiceIsActive(service string) bool {
	args := []string{"is-active", service}
	// Ignoring error here, command returns non-0 if in "activating" status:
	outBytes, _ := exec.Command("systemctl", args...).Output()
	output := strings.TrimSpace(string(outBytes))
	if output == "active" || output == "activating" {
		return true
	}
	return false
}

// getInitSystem returns an InitSystem for the current system, or nil
// if we cannot detect a supported init system for pre-flight checks.
// This indicates we will skip init system checks, not an error.
func GetInitSystem() (InitSystem, error) {
	// Assume existence of systemctl in path implies this is a systemd system:
	_, err := exec.LookPath("systemctl")
	if err == nil {
		return &SystemdInitSystem{}, nil
	}
	return nil, fmt.Errorf("no supported init system detected, skipping checking for services")
}

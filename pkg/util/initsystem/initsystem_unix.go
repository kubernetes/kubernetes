// +build !windows

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

package initsystem

import (
	"fmt"
	"os/exec"
	"strings"
)

type OpenRCInitSystem struct{}

func (openrc OpenRCInitSystem) ServiceStart(service string) error {
	args := []string{service, "start"}
	return exec.Command("rc-service", args...).Run()
}

func (openrc OpenRCInitSystem) ServiceStop(service string) error {
	args := []string{service, "stop"}
	return exec.Command("rc-service", args...).Run()
}

func (openrc OpenRCInitSystem) ServiceRestart(service string) error {
	args := []string{service, "restart"}
	return exec.Command("rc-service", args...).Run()
}

// openrc writes to stderr if a service is not found or not enabled
// this is in contrast to systemd which only writes to stdout.
// Hence, we use the Combinedoutput, and ignore the error.
func (openrc OpenRCInitSystem) ServiceExists(service string) bool {
	args := []string{service, "status"}
	outBytes, _ := exec.Command("rc-service", args...).CombinedOutput()
	if strings.Contains(string(outBytes), "does not exist") {
		return false
	}
	return true
}

func (openrc OpenRCInitSystem) ServiceIsEnabled(service string) bool {
	args := []string{"show", "default"}
	outBytes, _ := exec.Command("rc-update", args...).Output()
	if strings.Contains(string(outBytes), service) {
		return true
	}
	return false
}

func (openrc OpenRCInitSystem) ServiceIsActive(service string) bool {
	args := []string{service, "status"}
	outBytes, _ := exec.Command("rc-service", args...).Output()
	if strings.Contains(string(outBytes), "stopped") {
		return false
	}
	return true
}

func (openrc OpenRCInitSystem) EnableCommand(service string) string {
	return fmt.Sprintf("rc-update add %s default", service)
}

type SystemdInitSystem struct{}

func (sysd SystemdInitSystem) EnableCommand(service string) string {
	return fmt.Sprintf("systemctl enable %s.service", service)
}

func (sysd SystemdInitSystem) reloadSystemd() error {
	if err := exec.Command("systemctl", "daemon-reload").Run(); err != nil {
		return fmt.Errorf("failed to reload systemd: %v", err)
	}
	return nil
}

func (sysd SystemdInitSystem) ServiceStart(service string) error {
	// Before we try to start any service, make sure that systemd is ready
	if err := sysd.reloadSystemd(); err != nil {
		return err
	}
	args := []string{"start", service}
	return exec.Command("systemctl", args...).Run()
}

func (sysd SystemdInitSystem) ServiceRestart(service string) error {
	// Before we try to restart any service, make sure that systemd is ready
	if err := sysd.reloadSystemd(); err != nil {
		return err
	}
	args := []string{"restart", service}
	return exec.Command("systemctl", args...).Run()
}

func (sysd SystemdInitSystem) ServiceStop(service string) error {
	args := []string{"stop", service}
	return exec.Command("systemctl", args...).Run()
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
	err := exec.Command("systemctl", args...).Run()
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

// GetInitSystem returns an InitSystem for the current system, or nil
// if we cannot detect a supported init system.
// This indicates we will skip init system checks, not an error.
func GetInitSystem() (InitSystem, error) {
	// Assume existence of systemctl in path implies this is a systemd system:
	_, err := exec.LookPath("systemctl")
	if err == nil {
		return &SystemdInitSystem{}, nil
	}
	_, err = exec.LookPath("openrc")
	if err == nil {
		return &OpenRCInitSystem{}, nil
	}

	return nil, fmt.Errorf("no supported init system detected, skipping checking for services")
}

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

	// ServiceRestart tries to reload the environment and restart the specific service
	ServiceRestart(service string) error

	// ServiceExists ensures the service is defined for this init system.
	ServiceExists(service string) bool

	// ServiceIsEnabled ensures the service is enabled to start on each boot.
	ServiceIsEnabled(service string) bool

	// ServiceIsActive ensures the service is running, or attempting to run. (crash looping in the case of kubelet)
	ServiceIsActive(service string) bool
}

type SystemdInitSystem struct{}

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

// WindowsInitSystem is the windows implementation of InitSystem
type WindowsInitSystem struct{}

func (sysd WindowsInitSystem) ServiceStart(service string) error {
	args := []string{"Start-Service", service}
	err := exec.Command("powershell", args...).Run()
	return err
}

func (sysd WindowsInitSystem) ServiceRestart(service string) error {
	if err := sysd.ServiceStop(service); err != nil {
		return fmt.Errorf("couldn't stop service: %v", err)
	}
	if err := sysd.ServiceStart(service); err != nil {
		return fmt.Errorf("couldn't start service: %v", err)
	}
	return nil
}

func (sysd WindowsInitSystem) ServiceStop(service string) error {
	args := []string{"Stop-Service", service}
	err := exec.Command("powershell", args...).Run()
	return err
}

func (sysd WindowsInitSystem) ServiceExists(service string) bool {
	args := []string{"Get-Service", service}
	err := exec.Command("powershell", args...).Run()
	if err != nil {
		return false
	}
	return true

}

func (sysd WindowsInitSystem) ServiceIsEnabled(service string) bool {
	args := []string{"Get-Service", service + "| select -property starttype"}
	outBytes, _ := exec.Command("powershell", args...).Output()
	output := strings.TrimSpace(string(outBytes))
	if strings.Contains(output, "Automatic") {
		return true
	}
	return false
}

func (sysd WindowsInitSystem) ServiceIsActive(service string) bool {
	args := []string{"Get-Service", service + "| select -property status"}
	outBytes, _ := exec.Command("powershell", args...).Output()
	output := strings.TrimSpace(string(outBytes))
	if strings.Contains(output, "Running") {
		return true
	}
	return false
}

// GetInitSystem returns an InitSystem for the current system, or nil
// if we cannot detect a supported init system for pre-flight checks.
// This indicates we will skip init system checks, not an error.
func GetInitSystem() (InitSystem, error) {
	// Assume existence of systemctl in path implies this is a systemd system:
	_, err := exec.LookPath("systemctl")
	if err == nil {
		return &SystemdInitSystem{}, nil
	}
	_, err = exec.LookPath("wininit.exe")
	if err == nil {
		return &WindowsInitSystem{}, nil
	}
	return nil, fmt.Errorf("no supported init system detected, skipping checking for services")
}

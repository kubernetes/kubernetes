//go:build windows
// +build windows

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
	"time"

	"golang.org/x/sys/windows/svc"
	"golang.org/x/sys/windows/svc/mgr"
)

// WindowsInitSystem is the windows implementation of InitSystem
type WindowsInitSystem struct{}

// EnableCommand return a string describing how to enable a service
func (sysd WindowsInitSystem) EnableCommand(service string) string {
	return fmt.Sprintf("Set-Service '%s' -StartupType Automatic", service)
}

// ServiceStart tries to start a specific service
// Following Windows documentation: https://docs.microsoft.com/en-us/windows/desktop/Services/starting-a-service
func (sysd WindowsInitSystem) ServiceStart(service string) error {
	m, err := mgr.Connect()
	if err != nil {
		return err
	}
	defer m.Disconnect()

	s, err := m.OpenService(service)
	if err != nil {
		return fmt.Errorf("could not access service %s: %v", service, err)
	}
	defer s.Close()

	// Check if service is already started
	status, err := s.Query()
	if err != nil {
		return fmt.Errorf("could not query service %s: %v", service, err)
	}

	if status.State != svc.Stopped && status.State != svc.StopPending {
		return nil
	}

	timeout := time.Now().Add(10 * time.Second)
	for status.State != svc.Stopped {
		if timeout.Before(time.Now()) {
			return fmt.Errorf("timeout waiting for %s service to stop", service)
		}
		time.Sleep(300 * time.Millisecond)
		status, err = s.Query()
		if err != nil {
			return fmt.Errorf("could not retrieve %s service status: %v", service, err)
		}
	}

	// Start the service
	err = s.Start("is", "manual-started")
	if err != nil {
		return fmt.Errorf("could not start service %s: %v", service, err)
	}

	// Check that the start was successful
	status, err = s.Query()
	if err != nil {
		return fmt.Errorf("could not query service %s: %v", service, err)
	}
	timeout = time.Now().Add(10 * time.Second)
	for status.State != svc.Running {
		if timeout.Before(time.Now()) {
			return fmt.Errorf("timeout waiting for %s service to start", service)
		}
		time.Sleep(300 * time.Millisecond)
		status, err = s.Query()
		if err != nil {
			return fmt.Errorf("could not retrieve %s service status: %v", service, err)
		}
	}
	return nil
}

// ServiceRestart tries to reload the environment and restart the specific service
func (sysd WindowsInitSystem) ServiceRestart(service string) error {
	if err := sysd.ServiceStop(service); err != nil {
		return fmt.Errorf("couldn't stop service %s: %v", service, err)
	}
	if err := sysd.ServiceStart(service); err != nil {
		return fmt.Errorf("couldn't start service %s: %v", service, err)
	}

	return nil
}

// ServiceStop tries to stop a specific service
// Following Windows documentation: https://docs.microsoft.com/en-us/windows/desktop/Services/stopping-a-service
func (sysd WindowsInitSystem) ServiceStop(service string) error {
	m, err := mgr.Connect()
	if err != nil {
		return err
	}
	defer m.Disconnect()

	s, err := m.OpenService(service)
	if err != nil {
		return fmt.Errorf("could not access service %s: %v", service, err)
	}
	defer s.Close()

	// Check if service is already stopped
	status, err := s.Query()
	if err != nil {
		return fmt.Errorf("could not query service %s: %v", service, err)
	}

	if status.State == svc.Stopped {
		return nil
	}

	// If StopPending, check that service eventually stops
	if status.State == svc.StopPending {
		timeout := time.Now().Add(10 * time.Second)
		for status.State != svc.Stopped {
			if timeout.Before(time.Now()) {
				return fmt.Errorf("timeout waiting for %s service to stop", service)
			}
			time.Sleep(300 * time.Millisecond)
			status, err = s.Query()
			if err != nil {
				return fmt.Errorf("could not retrieve %s service status: %v", service, err)
			}
		}
		return nil
	}

	// Stop the service
	status, err = s.Control(svc.Stop)
	if err != nil {
		return fmt.Errorf("could not stop service %s: %v", service, err)
	}

	// Check that the stop was successful
	status, err = s.Query()
	if err != nil {
		return fmt.Errorf("could not query service %s: %v", service, err)
	}
	timeout := time.Now().Add(10 * time.Second)
	for status.State != svc.Stopped {
		if timeout.Before(time.Now()) {
			return fmt.Errorf("timeout waiting for %s service to stop", service)
		}
		time.Sleep(300 * time.Millisecond)
		status, err = s.Query()
		if err != nil {
			return fmt.Errorf("could not retrieve %s service status: %v", service, err)
		}
	}
	return nil
}

// ServiceExists ensures the service is defined for this init system.
func (sysd WindowsInitSystem) ServiceExists(service string) bool {
	m, err := mgr.Connect()
	if err != nil {
		return false
	}
	defer m.Disconnect()
	s, err := m.OpenService(service)
	if err != nil {
		return false
	}
	defer s.Close()

	return true
}

// ServiceIsEnabled ensures the service is enabled to start on each boot.
func (sysd WindowsInitSystem) ServiceIsEnabled(service string) bool {
	m, err := mgr.Connect()
	if err != nil {
		return false
	}
	defer m.Disconnect()

	s, err := m.OpenService(service)
	if err != nil {
		return false
	}
	defer s.Close()

	c, err := s.Config()
	if err != nil {
		return false
	}

	return c.StartType != mgr.StartDisabled
}

// ServiceIsActive ensures the service is running, or attempting to run. (crash looping in the case of kubelet)
func (sysd WindowsInitSystem) ServiceIsActive(service string) bool {
	m, err := mgr.Connect()
	if err != nil {
		return false
	}
	defer m.Disconnect()
	s, err := m.OpenService(service)
	if err != nil {
		return false
	}
	defer s.Close()

	status, err := s.Query()
	if err != nil {
		return false
	}
	return status.State == svc.Running
}

// GetInitSystem returns an InitSystem for the current system, or nil
// if we cannot detect a supported init system.
// This indicates we will skip init system checks, not an error.
func GetInitSystem() (InitSystem, error) {
	m, err := mgr.Connect()
	if err != nil {
		return nil, fmt.Errorf("no supported init system detected: %v", err)
	}
	defer m.Disconnect()
	return &WindowsInitSystem{}, nil
}

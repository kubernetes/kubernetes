/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package tests

import (
	"fmt"

	"github.com/docker/machine/libmachine/drivers"
	"github.com/docker/machine/libmachine/mcnflag"
	"github.com/docker/machine/libmachine/state"
)

// MockDriver is a struct used to mock out libmachine.Driver
type MockDriver struct {
	drivers.BaseDriver
	CurrentState state.State
	RemoveError  bool
	HostError    bool
	Port         int
}

// Create creates a MockDriver instance
func (driver *MockDriver) Create() error {
	driver.CurrentState = state.Running
	return nil
}

func (driver *MockDriver) GetIP() (string, error) {
	if driver.BaseDriver.IPAddress != "" {
		return driver.BaseDriver.IPAddress, nil
	}
	return "127.0.0.1", nil
}

// GetCreateFlags returns the flags used to create a MockDriver
func (driver *MockDriver) GetCreateFlags() []mcnflag.Flag {
	return []mcnflag.Flag{}
}

func (driver *MockDriver) GetSSHPort() (int, error) {
	return driver.Port, nil
}

// GetSSHHostname returns the hostname for SSH
func (driver *MockDriver) GetSSHHostname() (string, error) {
	if driver.HostError {
		return "", fmt.Errorf("Error getting host!")
	}
	return "localhost", nil
}

// GetSSHHostname returns the hostname for SSH
func (driver *MockDriver) GetSSHKeyPath() string {
	return driver.BaseDriver.SSHKeyPath
}

// GetState returns the state of the driver
func (driver *MockDriver) GetState() (state.State, error) {
	return driver.CurrentState, nil
}

// GetURL returns the URL of the driver
func (driver *MockDriver) GetURL() (string, error) {
	return "", nil
}

// Kill kills the machine
func (driver *MockDriver) Kill() error {
	driver.CurrentState = state.Stopped
	return nil
}

// Remove removes the machine
func (driver *MockDriver) Remove() error {
	if driver.RemoveError {
		return fmt.Errorf("Error deleting machine.")
	}
	return nil
}

// Restart restarts the machine
func (driver *MockDriver) Restart() error {
	driver.CurrentState = state.Running
	return nil
}

// SetConfigFromFlags sets the machine config
func (driver *MockDriver) SetConfigFromFlags(opts drivers.DriverOptions) error {
	return nil
}

// Start starts the machine
func (driver *MockDriver) Start() error {
	driver.CurrentState = state.Running
	return nil
}

// Stop stops the machine
func (driver *MockDriver) Stop() error {
	driver.CurrentState = state.Stopped
	return nil
}

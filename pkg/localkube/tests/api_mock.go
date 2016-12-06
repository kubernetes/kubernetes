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
	"encoding/json"
	"fmt"

	"github.com/docker/machine/libmachine"
	"github.com/docker/machine/libmachine/auth"
	"github.com/docker/machine/libmachine/host"
	"github.com/docker/machine/libmachine/mcnerror"
	"github.com/docker/machine/libmachine/state"
	"github.com/pkg/errors"
)

// MockAPI is a struct used to mock out libmachine.API
type MockAPI struct {
	Hosts       map[string]*host.Host
	CreateError bool
	RemoveError bool
	SaveCalled  bool
}

func NewMockAPI() *MockAPI {
	m := MockAPI{
		Hosts: make(map[string]*host.Host),
	}
	return &m
}

// Close closes the API.
func (api *MockAPI) Close() error {
	return nil
}

// NewHost creates a new host.Host instance.
func (api *MockAPI) NewHost(driverName string, rawDriver []byte) (*host.Host, error) {
	var driver MockDriver
	if err := json.Unmarshal(rawDriver, &driver); err != nil {
		return nil, errors.Wrap(err, "Error unmarshalling json")
	}
	h := &host.Host{
		DriverName:  driverName,
		RawDriver:   rawDriver,
		Driver:      &MockDriver{},
		Name:        driver.GetMachineName(),
		HostOptions: &host.Options{AuthOptions: &auth.Options{}},
	}
	return h, nil
}

// Create creates the actual host.
func (api *MockAPI) Create(h *host.Host) error {
	if api.CreateError {
		return fmt.Errorf("Error creating host.")
	}
	return h.Driver.Create()
}

// Exists determines if the host already exists.
func (api *MockAPI) Exists(name string) (bool, error) {
	_, ok := api.Hosts[name]
	return ok, nil
}

// List the existing hosts.
func (api *MockAPI) List() ([]string, error) {
	return []string{}, nil
}

// Load loads a host from disk.
func (api *MockAPI) Load(name string) (*host.Host, error) {
	h, ok := api.Hosts[name]
	if !ok {
		return nil, mcnerror.ErrHostDoesNotExist{
			Name: name,
		}

	}
	return h, nil
}

// Remove a host.
func (api *MockAPI) Remove(name string) error {
	if api.RemoveError {
		return fmt.Errorf("Error removing %s", name)
	}

	delete(api.Hosts, name)
	return nil
}

// Save saves a host to disk.
func (api *MockAPI) Save(host *host.Host) error {
	api.Hosts[host.Name] = host
	api.SaveCalled = true
	return nil
}

// GetMachinesDir returns the directory to store machines in.
func (api MockAPI) GetMachinesDir() string {
	return ""
}

// State returns the state of a host.
func State(api libmachine.API, name string) state.State {
	host, _ := api.Load(name)
	machineState, _ := host.Driver.GetState()
	return machineState
}

// Exists tells whether a named host exists.
func Exists(api libmachine.API, name string) bool {
	exists, _ := api.Exists(name)
	return exists
}

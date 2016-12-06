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
)

// MockHost used for testing. When commands are run, the output from CommandOutput
// is used, if present. Then the output from Error is used, if present. Finally,
// "", nil is returned.
type MockHost struct {
	CommandOutput map[string]string
	Error         string
	Commands      map[string]int
	Driver        drivers.Driver
}

func NewMockHost() *MockHost {
	return &MockHost{
		CommandOutput: make(map[string]string),
		Commands:      make(map[string]int),
		Driver:        &MockDriver{},
	}
}

func (m MockHost) RunSSHCommand(cmd string) (string, error) {
	m.Commands[cmd] = 1
	output, ok := m.CommandOutput[cmd]
	if ok {
		return output, nil
	}
	if m.Error != "" {
		return "", fmt.Errorf(m.Error)
	}
	return "", nil
}

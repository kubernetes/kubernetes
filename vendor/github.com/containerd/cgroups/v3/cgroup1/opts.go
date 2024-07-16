/*
   Copyright The containerd Authors.

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

package cgroup1

import (
	"errors"
)

var (
	// ErrIgnoreSubsystem allows the specific subsystem to be skipped
	ErrIgnoreSubsystem = errors.New("skip subsystem")
	// ErrDevicesRequired is returned when the devices subsystem is required but
	// does not exist or is not active
	ErrDevicesRequired = errors.New("devices subsystem is required")
)

// InitOpts allows configuration for the creation or loading of a cgroup
type InitOpts func(*InitConfig) error

// InitConfig provides configuration options for the creation
// or loading of a cgroup and its subsystems
type InitConfig struct {
	// InitCheck can be used to check initialization errors from the subsystem
	InitCheck InitCheck
	hierarchy Hierarchy
}

func newInitConfig() *InitConfig {
	return &InitConfig{
		InitCheck: RequireDevices,
		hierarchy: Default,
	}
}

// InitCheck allows subsystems errors to be checked when initialized or loaded
type InitCheck func(Subsystem, Path, error) error

// AllowAny allows any subsystem errors to be skipped
func AllowAny(_ Subsystem, _ Path, _ error) error {
	return ErrIgnoreSubsystem
}

// RequireDevices requires the device subsystem but no others
func RequireDevices(s Subsystem, _ Path, _ error) error {
	if s.Name() == Devices {
		return ErrDevicesRequired
	}
	return ErrIgnoreSubsystem
}

// WithHiearchy sets a list of cgroup subsystems.
// The default list is coming from /proc/self/mountinfo.
func WithHiearchy(h Hierarchy) InitOpts {
	return func(c *InitConfig) error {
		c.hierarchy = h
		return nil
	}
}

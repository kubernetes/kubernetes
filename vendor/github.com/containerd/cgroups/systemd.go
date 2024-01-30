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

package cgroups

import (
	"context"
	"path/filepath"
	"strings"
	"sync"

	systemdDbus "github.com/coreos/go-systemd/v22/dbus"
	"github.com/godbus/dbus/v5"
	specs "github.com/opencontainers/runtime-spec/specs-go"
)

const (
	SystemdDbus  Name = "systemd"
	defaultSlice      = "system.slice"
)

var (
	canDelegate bool
	once        sync.Once
)

func Systemd() ([]Subsystem, error) {
	root, err := v1MountPoint()
	if err != nil {
		return nil, err
	}
	defaultSubsystems, err := defaults(root)
	if err != nil {
		return nil, err
	}
	s, err := NewSystemd(root)
	if err != nil {
		return nil, err
	}
	// make sure the systemd controller is added first
	return append([]Subsystem{s}, defaultSubsystems...), nil
}

func Slice(slice, name string) Path {
	if slice == "" {
		slice = defaultSlice
	}
	return func(subsystem Name) (string, error) {
		return filepath.Join(slice, name), nil
	}
}

func NewSystemd(root string) (*SystemdController, error) {
	return &SystemdController{
		root: root,
	}, nil
}

type SystemdController struct {
	mu   sync.Mutex
	root string
}

func (s *SystemdController) Name() Name {
	return SystemdDbus
}

func (s *SystemdController) Create(path string, _ *specs.LinuxResources) error {
	ctx := context.TODO()
	conn, err := systemdDbus.NewWithContext(ctx)
	if err != nil {
		return err
	}
	defer conn.Close()
	slice, name := splitName(path)
	// We need to see if systemd can handle the delegate property
	// Systemd will return an error if it cannot handle delegate regardless
	// of its bool setting.
	checkDelegate := func() {
		canDelegate = true
		dlSlice := newProperty("Delegate", true)
		if _, err := conn.StartTransientUnitContext(ctx, slice, "testdelegate", []systemdDbus.Property{dlSlice}, nil); err != nil {
			if dbusError, ok := err.(dbus.Error); ok {
				// Starting with systemd v237, Delegate is not even a property of slices anymore,
				// so the D-Bus call fails with "InvalidArgs" error.
				if strings.Contains(dbusError.Name, "org.freedesktop.DBus.Error.PropertyReadOnly") || strings.Contains(dbusError.Name, "org.freedesktop.DBus.Error.InvalidArgs") {
					canDelegate = false
				}
			}
		}

		_, _ = conn.StopUnitContext(ctx, slice, "testDelegate", nil)
	}
	once.Do(checkDelegate)
	properties := []systemdDbus.Property{
		systemdDbus.PropDescription("cgroup " + name),
		systemdDbus.PropWants(slice),
		newProperty("DefaultDependencies", false),
		newProperty("MemoryAccounting", true),
		newProperty("CPUAccounting", true),
		newProperty("BlockIOAccounting", true),
	}

	// If we can delegate, we add the property back in
	if canDelegate {
		properties = append(properties, newProperty("Delegate", true))
	}

	ch := make(chan string)
	_, err = conn.StartTransientUnitContext(ctx, name, "replace", properties, ch)
	if err != nil {
		return err
	}
	<-ch
	return nil
}

func (s *SystemdController) Delete(path string) error {
	ctx := context.TODO()
	conn, err := systemdDbus.NewWithContext(ctx)
	if err != nil {
		return err
	}
	defer conn.Close()
	_, name := splitName(path)
	ch := make(chan string)
	_, err = conn.StopUnitContext(ctx, name, "replace", ch)
	if err != nil {
		return err
	}
	<-ch
	return nil
}

func newProperty(name string, units interface{}) systemdDbus.Property {
	return systemdDbus.Property{
		Name:  name,
		Value: dbus.MakeVariant(units),
	}
}

func splitName(path string) (slice string, unit string) {
	slice, unit = filepath.Split(path)
	return strings.TrimSuffix(slice, "/"), unit
}

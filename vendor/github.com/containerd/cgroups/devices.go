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
	"fmt"
	"os"
	"path/filepath"

	specs "github.com/opencontainers/runtime-spec/specs-go"
)

const (
	allowDeviceFile = "devices.allow"
	denyDeviceFile  = "devices.deny"
	wildcard        = -1
)

func NewDevices(root string) *devicesController {
	return &devicesController{
		root: filepath.Join(root, string(Devices)),
	}
}

type devicesController struct {
	root string
}

func (d *devicesController) Name() Name {
	return Devices
}

func (d *devicesController) Path(path string) string {
	return filepath.Join(d.root, path)
}

func (d *devicesController) Create(path string, resources *specs.LinuxResources) error {
	if err := os.MkdirAll(d.Path(path), defaultDirPerm); err != nil {
		return err
	}
	for _, device := range resources.Devices {
		file := denyDeviceFile
		if device.Allow {
			file = allowDeviceFile
		}
		if device.Type == "" {
			device.Type = "a"
		}
		if err := retryingWriteFile(
			filepath.Join(d.Path(path), file),
			[]byte(deviceString(device)),
			defaultFilePerm,
		); err != nil {
			return err
		}
	}
	return nil
}

func (d *devicesController) Update(path string, resources *specs.LinuxResources) error {
	return d.Create(path, resources)
}

func deviceString(device specs.LinuxDeviceCgroup) string {
	return fmt.Sprintf("%s %s:%s %s",
		device.Type,
		deviceNumber(device.Major),
		deviceNumber(device.Minor),
		device.Access,
	)
}

func deviceNumber(number *int64) string {
	if number == nil || *number == wildcard {
		return "*"
	}
	return fmt.Sprint(*number)
}

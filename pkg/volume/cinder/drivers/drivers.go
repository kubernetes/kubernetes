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

package drivers

import "strings"

type Driver interface {
	// AttachDisk attaches the volume to the host and returns the devicePath.
	AttachDisk(connInfo ConnectionInfo) (string, error)
	// DetachDisk detaches the volume from the host.
	DetachDisk(connInfo ConnectionInfo) error
	// IsAttached returns whether the volume is already attached to the host and
	// the devicePath if it is.
	IsAttached(connInfo ConnectionInfo) (string, bool, error)
}

type ConnectionInfo struct {
	DriverVolumeType string `json:"driver_volume_type"`
	Data             struct {
		Name string `json:"name"`
	} `json:"data"`
}

var drivers = map[string]Driver{}

func RegisterDriver(name string, driver Driver) {
	if _, alreadyExists := drivers[name]; alreadyExists {
		panic("Attempted to register driver twice")
	}
	drivers[strings.ToLower(name)] = driver
}

func GetDriver(name string) Driver {
	return drivers[strings.ToLower(name)]
}

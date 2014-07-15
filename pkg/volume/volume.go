/*
Copyright 2014 Google Inc. All rights reserved.

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

package volumes

import (
	"errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)


// All volume types are expected to implement this interface
type Interface interface {
	// Prepares and mounts/unpacks the volume to a directory path.
	SetUp()
	// Returns the directory path the volume is mounted to.
	GetPath() string
	// Unmounts the volume and removes traces of the SetUp procedure.
	TearDown()
}

// Host Directory Volumes represent a bare host directory mount.
type HostDirectoryVolume struct {
	Path string
}

// Simple host directory mounts require no setup or cleanup, but still
// need to fulfill the interface definitions.
func (hostVol *HostDirectoryVolume) SetUp() {}

func (hostVol *HostDirectoryVolume) TearDown() {}

func (hostVol *HostDirectoryVolume) GetPath() string {
	return hostVol.Path
}

// Interprets API volume as a HostDirectory
func createHostDirectoryVolume(volume *api.Volume) *HostDirectoryVolume {
	return &HostDirectoryVolume{volume.HostDirectory.Path}
}

// Interprets parameters passed in the API as an internal structure
// with utility procedures for mounting.
func CreateVolume(volume *api.Volume) (Interface, error) {
	// TODO(jonesdl) We should probably not check every
	// pointer and directly resolve these types instead.
	if volume.HostDirectory != nil {
		return createHostDirectoryVolume(volume), nil
	} else {
		return nil, errors.New("Unsupported volume type.")
	}
}

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

package volume

import (
	"errors"
	"os"
	"path"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

var ErrUnsupportedVolumeType = errors.New("unsupported volume type")

// Interface is a directory used by pods or hosts. Interface implementations
// must be idempotent.
type Interface interface {
	// SetUp prepares and mounts/unpacks the volume to a directory path.
	SetUp() error

	// GetPath returns the directory path the volume is mounted to.
	GetPath() string

	// TearDown unmounts the volume and removes traces of the SetUp procedure.
	TearDown() error
}

// Host Directory volumes represent a bare host directory mount.
// The directory in Path will be directly exposed to the container.
type HostDirectory struct {
	Path string
}

// Host directory mounts require no setup or cleanup, but still
// need to fulfill the interface definitions.
func (hostVol *HostDirectory) SetUp() error {
	return nil
}

func (hostVol *HostDirectory) TearDown() error {
	return nil
}

func (hostVol *HostDirectory) GetPath() string {
	return hostVol.Path
}

// EmptyDirectory volumes are temporary directories exposed to the pod.
// These do not persist beyond the lifetime of a pod.
type EmptyDirectory struct {
	Name    string
	PodID   string
	RootDir string
}

// SetUp creates the new directory.
func (emptyDir *EmptyDirectory) SetUp() error {
	path := emptyDir.GetPath()
	err := os.MkdirAll(path, 0750)
	if err != nil {
		return err
	}
	return nil
}

// TODO(jonesdl) when we can properly invoke TearDown(), we should delete
// the directory created by SetUp.
func (emptyDir *EmptyDirectory) TearDown() error {
	return nil
}

func (emptyDir *EmptyDirectory) GetPath() string {
	return path.Join(emptyDir.RootDir, emptyDir.PodID, "volumes", "empty", emptyDir.Name)
}

// Interprets API volume as a HostDirectory
func createHostDirectory(volume *api.Volume) *HostDirectory {
	return &HostDirectory{volume.Source.HostDirectory.Path}
}

// Interprets API volume as an EmptyDirectory
func createEmptyDirectory(volume *api.Volume, podID string, rootDir string) *EmptyDirectory {
	return &EmptyDirectory{volume.Name, podID, rootDir}
}

// CreateVolume returns an Interface capable of mounting a volume described by an
// *api.Volume and whether or not it is mounted, or an error.
func CreateVolume(volume *api.Volume, podID string, rootDir string) (Interface, error) {
	source := volume.Source
	// TODO(jonesdl) We will want to throw an error here when we no longer
	// support the default behavior.
	if source == nil {
		return nil, nil
	}
	var vol Interface
	// TODO(jonesdl) We should probably not check every pointer and directly
	// resolve these types instead.
	if source.HostDirectory != nil {
		vol = createHostDirectory(volume)
	} else if source.EmptyDirectory != nil {
		vol = createEmptyDirectory(volume, podID, rootDir)
	} else {
		return nil, ErrUnsupportedVolumeType
	}
	return vol, nil
}

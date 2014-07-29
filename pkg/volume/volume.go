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

// Interface is a directory used by pods or hosts. All volume interface implementations
// must be idempotent.
type Interface interface {
	// GetPath returns the directory path the volume is mounted to.
	GetPath() string
}

// The Builder interface provides the method to set up/mount the volume.
type Builder interface {
	Interface
	// SetUp prepares and mounts/unpacks the volume to a directory path.
	SetUp() error
}

// The Cleaner interface provides the method to cleanup/unmount the volumes.
type Cleaner interface {
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
type EmptyDirectoryBuilder struct {
	Name    string
	PodID   string
	RootDir string
}

// SetUp creates the new directory.
func (emptyDir *EmptyDirectoryBuilder) SetUp() error {
	path := emptyDir.GetPath()
	err := os.MkdirAll(path, 0750)
	if err != nil {
		return err
	}
	return nil
}

func (emptyDir *EmptyDirectoryBuilder) GetPath() string {
	return path.Join(emptyDir.RootDir, emptyDir.PodID, "volumes", "empty", emptyDir.Name)
}

// EmptyDirectoryCleaners only need to know what path they are cleaning
type EmptyDirectoryCleaner struct {
	Path string
}

// Simply delete everything in the directory.
func (emptyDir *EmptyDirectoryCleaner) TearDown() error {
	return os.RemoveAll(emptyDir.Path)
}

// Interprets API volume as a HostDirectory
func CreateHostDirectoryBuilder(volume *api.Volume) *HostDirectory {
	return &HostDirectory{volume.Source.HostDirectory.Path}
}

// Interprets API volume as an EmptyDirectoryBuilder
func CreateEmptyDirectoryBuilder(volume *api.Volume, podID string, rootDir string) *EmptyDirectoryBuilder {
	return &EmptyDirectoryBuilder{volume.Name, podID, rootDir}
}

// CreateVolumeBuilder returns a Builder capable of mounting a volume described by an
// *api.Volume, or an error.
func CreateVolumeBuilder(volume *api.Volume, podID string, rootDir string) (Builder, error) {
	source := volume.Source
	// TODO(jonesdl) We will want to throw an error here when we no longer
	// support the default behavior.
	if source == nil {
		return nil, nil
	}
	var vol Builder
	// TODO(jonesdl) We should probably not check every pointer and directly
	// resolve these types instead.
	if source.HostDirectory != nil {
		vol = CreateHostDirectoryBuilder(volume)
	} else if source.EmptyDirectory != nil {
		vol = CreateEmptyDirectoryBuilder(volume, podID, rootDir)
	} else {
		return nil, ErrUnsupportedVolumeType
	}
	return vol, nil
}

// CreateVolumeCleaner returns a Cleaner capable of tearing down a volume.
func CreateVolumeCleaner(kind string, path string) (Cleaner, error) {
	switch kind {
	case "empty":
		return &EmptyDirectoryCleaner{path}, nil
	default:
		return nil, ErrUnsupportedVolumeType
	}
}

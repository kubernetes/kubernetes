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
	"io/ioutil"
	"os"
	"path"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/golang/glog"
)

var ErrUnsupportedVolumeType = errors.New("unsupported volume type")

// Interface is a directory used by pods or hosts.
// All method implementations of methods in the volume interface must be idempotent
type Interface interface {
	// GetPath returns the directory path the volume is mounted to.
	GetPath() string
}

// The Builder interface provides the method to set up/mount the volume.
type Builder interface {
	// Uses Interface to provide the path for Docker binds.
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

func (emptyDir *EmptyDirectory) GetPath() string {
	return path.Join(emptyDir.RootDir, emptyDir.PodID, "volumes", "empty", emptyDir.Name)
}

func (emptyDir *EmptyDirectory) renameDirectory() (string, error) {
	oldPath := emptyDir.GetPath()
	newPath, err := ioutil.TempDir(path.Dir(oldPath), emptyDir.Name+".deleting~")
	if err != nil {
		return "", err
	}
	err = os.Rename(oldPath, newPath)
	if err != nil {
		return "", err
	}
	return newPath, nil
}

// Simply delete everything in the directory.
func (emptyDir *EmptyDirectory) TearDown() error {
	tmpDir, err := emptyDir.renameDirectory()
	if err != nil {
		return err
	}
	err = os.RemoveAll(tmpDir)
	if err != nil {
		return err
	}
	return nil
}

// Interprets API volume as a HostDirectory
func createHostDirectory(volume *api.Volume) *HostDirectory {
	return &HostDirectory{volume.Source.HostDirectory.Path}
}

// Interprets API volume as an EmptyDirectory
func createEmptyDirectory(volume *api.Volume, podID string, rootDir string) *EmptyDirectory {
	return &EmptyDirectory{volume.Name, podID, rootDir}
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
		vol = createHostDirectory(volume)
	} else if source.EmptyDirectory != nil {
		vol = createEmptyDirectory(volume, podID, rootDir)
	} else {
		return nil, ErrUnsupportedVolumeType
	}
	return vol, nil
}

// CreateVolumeCleaner returns a Cleaner capable of tearing down a volume.
func CreateVolumeCleaner(kind string, name string, podID string, rootDir string) (Cleaner, error) {
	switch kind {
	case "empty":
		return &EmptyDirectory{name, podID, rootDir}, nil
	default:
		return nil, ErrUnsupportedVolumeType
	}
}

// Examines directory structure to determine volumes that are presently
// active and mounted. Returns a map of Cleaner types.
func GetCurrentVolumes(rootDirectory string) map[string]Cleaner {
	currentVolumes := make(map[string]Cleaner)
	mountPath := rootDirectory
	podIDDirs, err := ioutil.ReadDir(mountPath)
	if err != nil {
		glog.Errorf("Could not read directory: %s, (%s)", mountPath, err)
	}
	// Volume information is extracted from the directory structure:
	// (ROOT_DIR)/(POD_ID)/volumes/(VOLUME_KIND)/(VOLUME_NAME)
	for _, podIDDir := range podIDDirs {
		if !podIDDir.IsDir() {
			continue
		}
		podID := podIDDir.Name()
		podIDPath := path.Join(mountPath, podID, "volumes")
		volumeKindDirs, err := ioutil.ReadDir(podIDPath)
		if err != nil {
			glog.Errorf("Could not read directory: %s, (%s)", podIDPath, err)
		}
		for _, volumeKindDir := range volumeKindDirs {
			volumeKind := volumeKindDir.Name()
			volumeKindPath := path.Join(podIDPath, volumeKind)
			volumeNameDirs, err := ioutil.ReadDir(volumeKindPath)
			if err != nil {
				glog.Errorf("Could not read directory: %s, (%s)", volumeKindPath, err)
			}
			for _, volumeNameDir := range volumeNameDirs {
				volumeName := volumeNameDir.Name()
				identifier := path.Join(podID, volumeName)
				// TODO(thockin) This should instead return a reference to an extant volume object
				cleaner, err := CreateVolumeCleaner(volumeKind, volumeName, podID, rootDirectory)
				if err != nil {
					glog.Errorf("Could not create volume cleaner: %s, (%s)", volumeNameDirs, err)
					continue
				}
				currentVolumes[identifier] = cleaner
			}
		}
	}
	return currentVolumes
}

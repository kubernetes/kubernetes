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
	"fmt"
	"os"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/golang/glog"
)


// All volume types are expected to implement this interface
type Interface interface {
	// Prepares and mounts/unpacks the volume to a directory path.
	// This procedure must be idempotent.
	SetUp()
	// Returns the directory path the volume is mounted to.
	GetPath() string
	// Unmounts the volume and removes traces of the SetUp procedure.
	// This procedure must be idempotent.
	TearDown()
}

// Host Directory volumes represent a bare host directory mount.
// The directory in Path will be directly exposed to the container.
type HostDirectory struct {
	Path string
}

// Host directory mounts require no setup or cleanup, but still
// need to fulfill the interface definitions.
func (hostVol *HostDirectory) SetUp() {}

func (hostVol *HostDirectory) TearDown() {}

func (hostVol *HostDirectory) GetPath() string {
	return hostVol.Path
}

// EmptyDirectory volumes are temporary directories exposed to the pod.
// These do not persist beyond the lifetime of a pod.
type EmptyDirectory struct {
	Name string
	PodID string
}

// SetUp creates the new directory.
func (emptyDir *EmptyDirectory) SetUp() {
	if _, err := os.Stat(emptyDir.GetPath()); os.IsNotExist(err) {
		os.MkdirAll(emptyDir.GetPath(), 0750)
	} else {
		glog.Warningf("Directory already exists: (%v)", emptyDir.GetPath())
	}
}

// TODO(jonesdl) when we can properly invoke TearDown(), we should delete
// the directory created by SetUp.
func (emptyDir *EmptyDirectory) TearDown() {}

func (emptyDir *EmptyDirectory) GetPath() string {
	// TODO(jonesdl) We will want to add a flag to designate a root
	// directory for kubelet to write to. For now this will just be /exports
	return fmt.Sprintf("/exports/%v/%v", emptyDir.PodID, emptyDir.Name)
}

// Interprets API volume as a HostDirectory
func createHostDirectory(volume *api.Volume) *HostDirectory {
	return &HostDirectory{volume.Source.HostDirectory.Path}
}

// Interprets API volume as an EmptyDirectory
func createEmptyDirectory(volume *api.Volume, podID string) *EmptyDirectory {
	return &EmptyDirectory{volume.Name, podID}
}

// CreateVolume returns an Interface capable of mounting a volume described by an
// *api.Volume and whether or not it is mounted, or an error.
func CreateVolume(volume *api.Volume, podID string) (Interface, error) {
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
		vol = createEmptyDirectory(volume, podID)
	} else {
		return nil, errors.New("Unsupported volume type.")
	}
	return vol, nil
}

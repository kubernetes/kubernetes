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
	"syscall"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/golang/glog"
)

var ErrUnsupportedVolumeType = errors.New("unsupported volume type")

// Interface is a directory used by pods or hosts.
// All method implementations of methods in the volume interface must be idempotent.
type Interface interface {
	// GetPath returns the directory path the volume is mounted to.
	GetPath() string
}

// Builder interface provides method to set up/mount the volume.
type Builder interface {
	// Uses Interface to provide the path for Docker binds.
	Interface
	// SetUp prepares and mounts/unpacks the volume to a directory path.
	SetUp() error
}

// Cleaner interface provides method to cleanup/unmount the volumes.
type Cleaner interface {
	// TearDown unmounts the volume and removes traces of the SetUp procedure.
	TearDown() error
}

// The APIDiskUtil interface provides the methods to attach and detach persistent disks
// on a cloud platform.
type APIDiskUtil interface {
	// Establishes a connection and verifies that the kubelet is running
	// in the correct cloud platform.
	// Connect must be called at least once before attaching/detaching a disk.
	Connect() error
	// Attaches the disk to the kubelet and returns its device path
	AttachDisk(PD *PersistentDisk) (string, error)
	// Detaches the disk from the kubelet.
	DetachDisk(PD *PersistentDisk) error
}

// Host Directory volumes represent a bare host directory mount.
// The directory in Path will be directly exposed to the container.
type HostDirectory struct {
	Path string
}

// SetUp implements interface definitions, even though host directory
// mounts don't require any setup or cleanup.
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

// SetUp creates new directory.
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

func renameDirectory(vol Interface) (string, error) {
	oldPath := vol.GetPath()
	name := path.Base(oldPath)
	newPath, err := ioutil.TempDir(path.Dir(oldPath), name+".deleting~")
	if err != nil {
		return "", err
	}
	err = os.Rename(oldPath, newPath)
	if err != nil {
		return "", err
	}
	return newPath, nil
}

// TearDown simply deletes everything in the directory.
func (emptyDir *EmptyDirectory) TearDown() error {
	tmpDir, err := renameDirectory(emptyDir)
	if err != nil {
		return err
	}
	err = os.RemoveAll(tmpDir)
	if err != nil {
		return err
	}
	return nil
}

// Google Compute Engine Persistent Disks can only be used when running Kubernetes
// on a GCE cloud. PDs must be created in GCE prior to mounting in Kubernetes.
// A PD can only be mounted as Read-Write once, but can be mounted
// as read-only multiple times.
type PersistentDisk struct {
	Name    string
	PodID   string
	RootDir string
	// Unique identifier of the PD, used to find the resource in an API.
	PDName string
	// Filesystem type, optional.
	FSType string
	// Specifies whether the disk will be attached as ReadOnly.
	ReadOnly bool
	util     APIDiskUtil
}

func (PD *PersistentDisk) GetPath() string {
	return path.Join(PD.RootDir, PD.PodID, "volumes", "pd", PD.Name)
}

func (PD *PersistentDisk) SetUp() error {
	if _, err := os.Stat(PD.GetPath()); !os.IsNotExist(err) {
		return nil
	}
	if err := PD.util.Connect(); err != nil {
		return err
	}
	devicePath, err := PD.util.AttachDisk(PD)
	if err != nil {
		return err
	}
	globalPDPath := path.Join(PD.RootDir, "global", "pd", PD.PDName)
	// Only mount the PD globally once.
	if _, err = os.Stat(globalPDPath); os.IsNotExist(err) {
		err = os.MkdirAll(globalPDPath, 0750)
		if err != nil {
			return err
		}
		err = syscall.Mount(devicePath, globalPDPath, PD.FSType, 0, "")
		if err != nil {
			return err
		}
	}
	//Perform a bind mount to the full path to allow duplicate mounts of the same PD.
	if _, err = os.Stat(PD.GetPath()); os.IsNotExist(err) {
		err = os.MkdirAll(PD.GetPath(), 0750)
		if err != nil {
			return err
		}
		err = syscall.Mount(globalPDPath, PD.GetPath(), "", syscall.MS_BIND, "")
		if err != nil {
			return err
		}
	}
	return nil
}

func (PD *PersistentDisk) TearDown() error {
	if err := syscall.Unmount(PD.GetPath(), 0); err != nil {
		return err
	}
	tmpDir, err := renameDirectory(PD)
	if err != nil {
		return err
	}
	if err := os.RemoveAll(tmpDir); err != nil {
		return err
	}
	if err := PD.util.Connect(); err != nil {
		return err
	}
	if err := PD.util.DetachDisk(PD); err != nil {
		return err
	}
	return nil
}

// createHostDirectory interprets API volume as a HostDirectory.
func createHostDirectory(volume *api.Volume) *HostDirectory {
	return &HostDirectory{volume.Source.HostDirectory.Path}
}

// createEmptyDirectory interprets API volume as an EmptyDirectory.
func createEmptyDirectory(volume *api.Volume, podID string, rootDir string) *EmptyDirectory {
	return &EmptyDirectory{volume.Name, podID, rootDir}
}

// Interprets API volume as a PersistentDisk
func createPersistentDisk(volume *api.Volume, podID string, rootDir string) (*PersistentDisk, error) {
	PDName := volume.Source.PersistentDisk.PDName
	FSType := volume.Source.PersistentDisk.FSType
	readOnly := volume.Source.PersistentDisk.ReadOnly
	util, err := newDiskUtil(volume.Source.PersistentDisk.Platform)
	mounter := &DiskMounter{}
	if err != nil {
		return nil, err
	}
	return &PersistentDisk{volume.Name, podID, rootDir, PDName, FSType, readOnly, util, mounter}, nil
}

func newDiskUtil(Platform string) (APIDiskUtil, error) {
	switch Platform {
	case "gce":
		util := &GCEDiskUtil{}
		return util, nil
	default:
		return nil, ErrUnsupportedVolumeType
	}
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
	var err error
	// TODO(jonesdl) We should probably not check every pointer and directly
	// resolve these types instead.
	if source.HostDirectory != nil {
		vol = createHostDirectory(volume)
	} else if source.EmptyDirectory != nil {
		vol = createEmptyDirectory(volume, podID, rootDir)
	} else if source.PersistentDisk != nil {
		vol, err = createPersistentDisk(volume, podID, rootDir)
		if err != nil {
			return nil, err
		}
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

// GetCurrentVolumes examines directory structure to determine volumes that are
// presently active and mounted. Returns a map of Cleaner types.
func GetCurrentVolumes(rootDirectory string) map[string]Cleaner {
	currentVolumes := make(map[string]Cleaner)
	podIDDirs, err := ioutil.ReadDir(rootDirectory)
	if err != nil {
		glog.Errorf("Could not read directory: %s, (%s)", rootDirectory, err)
	}
	// Volume information is extracted from the directory structure:
	// (ROOT_DIR)/(POD_ID)/volumes/(VOLUME_KIND)/(VOLUME_NAME)
	for _, podIDDir := range podIDDirs {
		if !podIDDir.IsDir() {
			continue
		}
		podID := podIDDir.Name()
		podIDPath := path.Join(rootDirectory, podID, "volumes")
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

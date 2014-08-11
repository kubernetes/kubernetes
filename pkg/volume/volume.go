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
	// TODO(jonesdl) Connect() can probably be removed from the interface and used
	// as an implementation detail of attach/detach
	Connect() error
	// Attaches the disk to the kubelet's host machine.
	AttachDisk(PD *PersistentDisk) error
	// Detaches the disk from the kubelet's host machine.
	DetachDisk(PD *PersistentDisk, devicePath string) error
}

// Mounters wrap os/system specific calls to perform mounts.
type Mounter interface {
	Mount(string, string, string, uintptr, string) error
	Unmount(string, int) error
	// RefCount returns the device path for the source disk of the PD, and
	// the number of references to that target disk.
	RefCount(PD *PersistentDisk) (string, int, error)
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

// renameDirectory moves the path of the volume mount to a temporary
// directory, suffixed with ".deleting~".
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

// PersistentDisk volumes are disk resources provided by a cloud platform
// that are attached to the kubelet's host machine and exposed to the pod.
type PersistentDisk struct {
	Name    string
	PodID   string
	RootDir string
	// Unique identifier of the PD, used to find the disk resource in the provider.
	PDName string
	// Filesystem type, optional.
	FSType string
	// Specifies the partition to mount
	Partition string
	// Specifies the cloud platform provider.
	Provider string
	// Specifies whether the disk will be attached as ReadOnly.
	ReadOnly bool
	// Utility interface that provides API calls to the provider to attach/detach disks.
	util APIDiskUtil
	// Mounter interface that provides system calls to mount the disks.
	mounter Mounter
}

func (PD *PersistentDisk) GetPath() string {
	return path.Join(PD.RootDir, PD.PodID, "volumes", PD.Provider+"-pd", PD.Name)
}

// Attaches the disk and bind mounts to the volume path.
func (PD *PersistentDisk) SetUp() error {
	if _, err := os.Stat(PD.GetPath()); !os.IsNotExist(err) {
		return nil
	}
	if err := PD.util.Connect(); err != nil {
		return err
	}
	err := PD.util.AttachDisk(PD)
	if err != nil {
		return err
	}
	flags := uintptr(0)
	if PD.ReadOnly {
		flags = MOUNT_MS_RDONLY
	}
	//Perform a bind mount to the full path to allow duplicate mounts of the same PD.
	if _, err = os.Stat(PD.GetPath()); os.IsNotExist(err) {
		err = os.MkdirAll(PD.GetPath(), 0750)
		if err != nil {
			return err
		}
		globalPDPath := path.Join(PD.RootDir, "global", "pd", PD.PDName)
		err = PD.mounter.Mount(globalPDPath, PD.GetPath(), "", MOUNT_MS_BIND|flags, "")
		if err != nil {
			return err
		}
	}
	return nil
}

// Unmounts the bind mount, and detaches the disk only if the PD
// resource was the last reference to that disk on the kubelet.
func (PD *PersistentDisk) TearDown() error {
	devicePath, refCount, err := PD.mounter.RefCount(PD)
	if err != nil {
		return err
	}
	if err := PD.mounter.Unmount(PD.GetPath(), 0); err != nil {
		return err
	}
	refCount--
	tmpDir, err := renameDirectory(PD)
	if err != nil {
		return err
	}
	if err := os.RemoveAll(tmpDir); err != nil {
		return err
	}
	if err != nil {
		return err
	}
	// Detach the disk on the last reference.
	if refCount == 1 {
		if err := PD.util.Connect(); err != nil {
			return err
		}
		if err := PD.util.DetachDisk(PD, devicePath); err != nil {
			return err
		}
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
	partition := volume.Source.PersistentDisk.Partition
	readOnly := volume.Source.PersistentDisk.ReadOnly
	provider := volume.Source.PersistentDisk.Provider
	util, err := newDiskUtil(provider)
	mounter := &DiskMounter{}
	if err != nil {
		return nil, err
	}
	return &PersistentDisk{volume.Name, podID, rootDir, PDName, FSType, partition, provider, readOnly, util, mounter}, nil
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
	case "gce-pd":
		return &PersistentDisk{name, podID, rootDir, "", "", "", "gce", false, &GCEDiskUtil{}, &DiskMounter{}}, nil
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
					glog.Errorf("Could not create volume cleaner: %s, (%s)", volumeNameDir.Name(), err)
					continue
				}
				currentVolumes[identifier] = cleaner
			}
		}
	}
	return currentVolumes
}

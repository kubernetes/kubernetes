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
	"io/ioutil"
	"os"
	"path"
	"strconv"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/exec"
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

type gcePersistentDiskUtil interface {
	// Attaches the disk to the kubelet's host machine.
	AttachDisk(PD *GCEPersistentDisk) error
	// Detaches the disk from the kubelet's host machine.
	DetachDisk(PD *GCEPersistentDisk, devicePath string) error
}

// Mounters wrap os/system specific calls to perform mounts.
type mounter interface {
	Mount(source string, target string, fstype string, flags uintptr, data string) error
	Unmount(target string, flags int) error
	// RefCount returns the device path for the source disk of a volume, and
	// the number of references to that target disk.
	RefCount(vol Interface) (string, int, error)
}

// HostDir volumes represent a bare host directory mount.
// The directory in Path will be directly exposed to the container.
type HostDir struct {
	Path string
}

// SetUp implements interface definitions, even though host directory
// mounts don't require any setup or cleanup.
func (hostVol *HostDir) SetUp() error {
	return nil
}

func (hostVol *HostDir) GetPath() string {
	return hostVol.Path
}

type execInterface interface {
	ExecCommand(cmd []string, dir string) ([]byte, error)
}

type GitDir struct {
	Source   string
	Revision string
	PodID    string
	RootDir  string
	Name     string
	exec     exec.Interface
}

func newGitRepo(volume *api.Volume, podID, rootDir string) *GitDir {
	return &GitDir{
		Source:   volume.Source.GitRepo.Repository,
		Revision: volume.Source.GitRepo.Revision,
		PodID:    podID,
		RootDir:  rootDir,
		Name:     volume.Name,
		exec:     exec.New(),
	}
}

func (g *GitDir) ExecCommand(command string, args []string, dir string) ([]byte, error) {
	cmd := g.exec.Command(command, args...)
	cmd.SetDir(dir)
	return cmd.CombinedOutput()
}

func (g *GitDir) SetUp() error {
	volumePath := g.GetPath()
	if err := os.MkdirAll(volumePath, 0750); err != nil {
		return err
	}
	if _, err := g.ExecCommand("git", []string{"clone", g.Source}, g.GetPath()); err != nil {
		return err
	}
	files, err := ioutil.ReadDir(g.GetPath())
	if err != nil {
		return err
	}
	if len(g.Revision) == 0 {
		return nil
	}

	if len(files) != 1 {
		return fmt.Errorf("Unexpected directory contents: %v", files)
	}
	dir := path.Join(g.GetPath(), files[0].Name())
	if _, err := g.ExecCommand("git", []string{"checkout", g.Revision}, dir); err != nil {
		return err
	}
	if _, err := g.ExecCommand("git", []string{"reset", "--hard"}, dir); err != nil {
		return err
	}
	return nil
}

func (g *GitDir) GetPath() string {
	return path.Join(g.RootDir, g.PodID, "volumes", "git", g.Name)
}

// TearDown simply deletes everything in the directory.
func (g *GitDir) TearDown() error {
	tmpDir, err := renameDirectory(g.GetPath(), g.Name+"~deleting")
	if err != nil {
		return err
	}
	err = os.RemoveAll(tmpDir)
	if err != nil {
		return err
	}
	return nil
}

// EmptyDir volumes are temporary directories exposed to the pod.
// These do not persist beyond the lifetime of a pod.
type EmptyDir struct {
	Name    string
	PodID   string
	RootDir string
}

// SetUp creates new directory.
func (emptyDir *EmptyDir) SetUp() error {
	path := emptyDir.GetPath()
	return os.MkdirAll(path, 0750)
}

func (emptyDir *EmptyDir) GetPath() string {
	return path.Join(emptyDir.RootDir, emptyDir.PodID, "volumes", "empty", emptyDir.Name)
}

func renameDirectory(oldPath, newName string) (string, error) {
	newPath, err := ioutil.TempDir(path.Dir(oldPath), newName)
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
func (emptyDir *EmptyDir) TearDown() error {
	tmpDir, err := renameDirectory(emptyDir.GetPath(), emptyDir.Name+".deleting~")
	if err != nil {
		return err
	}
	err = os.RemoveAll(tmpDir)
	if err != nil {
		return err
	}
	return nil
}

// createHostDir interprets API volume as a HostDir.
func createHostDir(volume *api.Volume) *HostDir {
	return &HostDir{volume.Source.HostDir.Path}
}

// GCEPersistentDisk volumes are disk resources provided by Google Compute Engine
// that are attached to the kubelet's host machine and exposed to the pod.
type GCEPersistentDisk struct {
	Name    string
	PodID   string
	RootDir string
	// Unique identifier of the PD, used to find the disk resource in the provider.
	PDName string
	// Filesystem type, optional.
	FSType string
	// Specifies the partition to mount
	Partition string
	// Specifies whether the disk will be attached as ReadOnly.
	ReadOnly bool
	// Utility interface that provides API calls to the provider to attach/detach disks.
	util gcePersistentDiskUtil
	// Mounter interface that provides system calls to mount the disks.
	mounter mounter
}

func (PD *GCEPersistentDisk) GetPath() string {
	return path.Join(PD.RootDir, PD.PodID, "volumes", "gce-pd", PD.Name)
}

// Attaches the disk and bind mounts to the volume path.
func (PD *GCEPersistentDisk) SetUp() error {
	// TODO: handle failed mounts here.
	if _, err := os.Stat(PD.GetPath()); !os.IsNotExist(err) {
		return nil
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
		globalPDPath := makeGlobalPDName(PD.RootDir, PD.PDName, PD.ReadOnly)
		err = PD.mounter.Mount(globalPDPath, PD.GetPath(), "", MOUNT_MS_BIND|flags, "")
		if err != nil {
			os.RemoveAll(PD.GetPath())
			return err
		}
	}
	return nil
}

// Unmounts the bind mount, and detaches the disk only if the PD
// resource was the last reference to that disk on the kubelet.
func (PD *GCEPersistentDisk) TearDown() error {
	devicePath, refCount, err := PD.mounter.RefCount(PD)
	if err != nil {
		return err
	}
	if err := PD.mounter.Unmount(PD.GetPath(), 0); err != nil {
		return err
	}
	refCount--
	if err := os.RemoveAll(PD.GetPath()); err != nil {
		return err
	}
	if err != nil {
		return err
	}
	// If refCount is 1, then all bind mounts have been removed, and the
	// remaining reference is the global mount. It is safe to detach.
	if refCount == 1 {
		if err := PD.util.DetachDisk(PD, devicePath); err != nil {
			return err
		}
	}
	return nil
}

//TODO(jonesdl) prevent name collisions by using designated pod space as well.
// Ex. (ROOT_DIR)/pods/...
func makeGlobalPDName(rootDir, devName string, readOnly bool) string {
	var mode string
	if readOnly {
		mode = "ro"
	} else {
		mode = "rw"
	}
	return path.Join(rootDir, "global", "pd", mode, devName)
}

// createEmptyDir interprets API volume as an EmptyDir.
func createEmptyDir(volume *api.Volume, podID string, rootDir string) *EmptyDir {
	return &EmptyDir{volume.Name, podID, rootDir}
}

// Interprets API volume as a PersistentDisk
func createGCEPersistentDisk(volume *api.Volume, podID string, rootDir string) (*GCEPersistentDisk, error) {
	PDName := volume.Source.GCEPersistentDisk.PDName
	FSType := volume.Source.GCEPersistentDisk.FSType
	partition := strconv.Itoa(volume.Source.GCEPersistentDisk.Partition)
	if partition == "0" {
		partition = ""
	}
	readOnly := volume.Source.GCEPersistentDisk.ReadOnly
	// TODO: move these up into the Kubelet.
	util := &GCEDiskUtil{}
	mounter := &DiskMounter{}
	return &GCEPersistentDisk{
		Name:      volume.Name,
		PodID:     podID,
		RootDir:   rootDir,
		PDName:    PDName,
		FSType:    FSType,
		Partition: partition,
		ReadOnly:  readOnly,
		util:      util,
		mounter:   mounter}, nil
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
	if source.HostDir != nil {
		vol = createHostDir(volume)
	} else if source.EmptyDir != nil {
		vol = createEmptyDir(volume, podID, rootDir)
	} else if source.GCEPersistentDisk != nil {
		vol, err = createGCEPersistentDisk(volume, podID, rootDir)
		if err != nil {
			return nil, err
		}
	} else if source.GitRepo != nil {
		vol = newGitRepo(volume, podID, rootDir)
	} else {
		return nil, ErrUnsupportedVolumeType
	}
	return vol, nil
}

// CreateVolumeCleaner returns a Cleaner capable of tearing down a volume.
func CreateVolumeCleaner(kind string, name string, podID string, rootDir string) (Cleaner, error) {
	switch kind {
	case "empty":
		return &EmptyDir{name, podID, rootDir}, nil
	case "gce-pd":
		return &GCEPersistentDisk{
			Name:    name,
			PodID:   podID,
			RootDir: rootDir,
			util:    &GCEDiskUtil{},
			mounter: &DiskMounter{}}, nil
	case "git":
		return &GitDir{
			Name:    name,
			PodID:   podID,
			RootDir: rootDir,
		}, nil
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
		if _, err := os.Stat(podIDPath); os.IsNotExist(err) {
			continue
		}
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

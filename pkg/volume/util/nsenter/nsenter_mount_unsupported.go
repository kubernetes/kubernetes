// +build !linux

/*
Copyright 2014 The Kubernetes Authors.

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

package nsenter

import (
	"errors"
	"os"

	"k8s.io/utils/nsenter"

	"k8s.io/kubernetes/pkg/util/mount"
)

// Mounter provides the mount.Interface implementation for unsupported
// platforms.
type Mounter struct{}

// NewMounter returns a new Mounter for the current system
func NewMounter(rootDir string, ne *nsenter.Nsenter) *Mounter {
	return &Mounter{}
}

var _ = mount.Interface(&Mounter{})

// Mount mounts the source to the target. It is a noop for unsupported systems
func (*Mounter) Mount(source string, target string, fstype string, options []string) error {
	return nil
}

// Unmount unmounts the target path from the system. it is a noop for unsupported
// systems
func (*Mounter) Unmount(target string) error {
	return nil
}

// List returns a list of all mounted filesystems. It is a noop for unsupported systems
func (*Mounter) List() ([]mount.MountPoint, error) {
	return []mount.MountPoint{}, nil
}

// IsMountPointMatch tests if dir and mp are the same path
func (*Mounter) IsMountPointMatch(mp mount.MountPoint, dir string) bool {
	return (mp.Path == dir)
}

// IsLikelyNotMountPoint determines if a directory is not a mountpoint.
// It is a noop on unsupported systems
func (*Mounter) IsLikelyNotMountPoint(file string) (bool, error) {
	return true, nil
}

// DeviceOpened checks if block device in use. I tis a noop for unsupported systems
func (*Mounter) DeviceOpened(pathname string) (bool, error) {
	return false, nil
}

// PathIsDevice checks if pathname refers to a device. It is a noop for unsupported
// systems
func (*Mounter) PathIsDevice(pathname string) (bool, error) {
	return true, nil
}

// GetDeviceNameFromMount finds the device name from its global mount point using the
// given mountpath and plugin location. It is a noop of unsupported platforms
func (*Mounter) GetDeviceNameFromMount(mountPath, pluginMountDir string) (string, error) {
	return "", nil
}

// MakeRShared checks if path is shared and bind-mounts it as rshared if needed.
// It is a noop on unsupported platforms
func (*Mounter) MakeRShared(path string) error {
	return nil
}

// GetFileType checks for file/directory/socket/block/character devices.
// Always returns an error and "fake" filetype on unsupported platforms
func (*Mounter) GetFileType(_ string) (mount.FileType, error) {
	return mount.FileType("fake"), errors.New("not implemented")
}

// MakeDir creates a new directory. Noop on unsupported platforms
func (*Mounter) MakeDir(pathname string) error {
	return nil
}

// MakeFile creats an empty file. Noop on unsupported platforms
func (*Mounter) MakeFile(pathname string) error {
	return nil
}

// ExistsPath checks if pathname exists. Always returns an error on unsupported
// platforms
func (*Mounter) ExistsPath(pathname string) (bool, error) {
	return true, errors.New("not implemented")
}

// EvalHostSymlinks returns the path name after evaluating symlinks. Always
// returns an error on unsupported platforms
func (*Mounter) EvalHostSymlinks(pathname string) (string, error) {
	return "", errors.New("not implemented")
}

// GetMountRefs finds all mount references to the path, returns a
// list of paths. Always returns an error on unsupported platforms
func (*Mounter) GetMountRefs(pathname string) ([]string, error) {
	return nil, errors.New("not implemented")
}

// GetFSGroup returns FSGroup of pathname. Always returns an error on unsupported platforms
func (*Mounter) GetFSGroup(pathname string) (int64, error) {
	return -1, errors.New("not implemented")
}

// GetSELinuxSupport tests if pathname is on a mount that supports SELinux.
// Always returns an error on unsupported platforms
func (*Mounter) GetSELinuxSupport(pathname string) (bool, error) {
	return false, errors.New("not implemented")
}

// GetMode returns permissions of pathname. Always returns an error on unsupported platforms
func (*Mounter) GetMode(pathname string) (os.FileMode, error) {
	return 0, errors.New("not implemented")
}

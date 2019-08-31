// +build !linux,!windows,!providerless

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

package mount

import (
	"errors"
	"os"
)

// Mounter provides the default unsupported implementation of mount.Interface
type Mounter struct {
	mounterPath string
}

var errUnsupported = errors.New(" util/mount on this platform is not supported")

// New returns a mount.Interface for the current system.
// It provides options to override the default mounter behavior.
// mounterPath allows using an alternative to `/bin/mount` for mounting.
func New(mounterPath string) Interface {
	return &Mounter{
		mounterPath: mounterPath,
	}
}

// Mount returns an error
func (mounter *Mounter) Mount(source string, target string, fstype string, options []string) error {
	return errUnsupported
}

// Unmount returns an error
func (mounter *Mounter) Unmount(target string) error {
	return errUnsupported
}

// List returns an error
func (mounter *Mounter) List() ([]MntPoint, error) {
	return []MntPoint{}, errUnsupported
}

// IsMountPointMatch returns an error
func (mounter *Mounter) IsMountPointMatch(mp MntPoint, dir string) bool {
	return (mp.Path == dir)
}

// IsNotMountPoint returns an error
func (mounter *Mounter) IsNotMountPoint(dir string) (bool, error) {
	return IsNotMountPoint(mounter, dir)
}

// IsLikelyNotMountPoint returns an error
func (mounter *Mounter) IsLikelyNotMountPoint(file string) (bool, error) {
	return true, errUnsupported
}

// GetDeviceNameFromMount returns an error
func (mounter *Mounter) GetDeviceNameFromMount(mountPath, pluginDir string) (string, error) {
	return "", errUnsupported
}

func getDeviceNameFromMount(mounter Interface, mountPath, pluginDir string) (string, error) {
	return "", errUnsupported
}

// DeviceOpened returns an error
func (mounter *Mounter) DeviceOpened(pathname string) (bool, error) {
	return false, errUnsupported
}

// PathIsDevice returns an error
func (mounter *Mounter) PathIsDevice(pathname string) (bool, error) {
	return true, errUnsupported
}

// MakeRShared returns an error
func (mounter *Mounter) MakeRShared(path string) error {
	return errUnsupported
}

func (mounter *SafeFormatAndMount) formatAndMount(source string, target string, fstype string, options []string) error {
	return mounter.Interface.Mount(source, target, fstype, options)
}

func (mounter *SafeFormatAndMount) diskLooksUnformatted(disk string) (bool, error) {
	return true, errUnsupported
}

// GetFileType returns an error
func (mounter *Mounter) GetFileType(pathname string) (FileType, error) {
	return FileType("fake"), errUnsupported
}

// MakeDir returns an error
func (mounter *Mounter) MakeDir(pathname string) error {
	return errUnsupported
}

// MakeFile returns an error
func (mounter *Mounter) MakeFile(pathname string) error {
	return errUnsupported
}

// ExistsPath returns an error
func (mounter *Mounter) ExistsPath(pathname string) (bool, error) {
	return true, errors.New("not implemented")
}

// EvalHostSymlinks returns an error
func (mounter *Mounter) EvalHostSymlinks(pathname string) (string, error) {
	return "", errUnsupported
}

// PrepareSafeSubpath returns an error
func (mounter *Mounter) PrepareSafeSubpath(subPath Subpath) (newHostPath string, cleanupAction func(), err error) {
	return subPath.Path, nil, errUnsupported
}

// CleanSubPaths returns an error
func (mounter *Mounter) CleanSubPaths(podDir string, volumeName string) error {
	return errUnsupported
}

// SafeMakeDir returns an error
func (mounter *Mounter) SafeMakeDir(pathname string, base string, perm os.FileMode) error {
	return errUnsupported
}

// GetMountRefs returns an error
func (mounter *Mounter) GetMountRefs(pathname string) ([]string, error) {
	return nil, errors.New("not implemented")
}

// GetFSGroup returns an error
func (mounter *Mounter) GetFSGroup(pathname string) (int64, error) {
	return -1, errors.New("not implemented")
}

// GetSELinuxSupport returns an error
func (mounter *Mounter) GetSELinuxSupport(pathname string) (bool, error) {
	return false, errors.New("not implemented")
}

// GetMode returns an error
func (mounter *Mounter) GetMode(pathname string) (os.FileMode, error) {
	return 0, errors.New("not implemented")
}

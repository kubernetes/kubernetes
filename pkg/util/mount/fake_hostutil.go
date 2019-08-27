/*
Copyright 2015 The Kubernetes Authors.

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
	"sync"
)

// FakeHostUtil is a fake mount.HostUtils implementation for testing
type FakeHostUtil struct {
	MountPoints []MountPoint
	Filesystem  map[string]FileType

	mutex sync.Mutex
}

var _ HostUtils = &FakeHostUtil{}

// DeviceOpened checks if block device referenced by pathname is in use by
// checking if is listed as a device in the in-memory mountpoint table.
func (hu *FakeHostUtil) DeviceOpened(pathname string) (bool, error) {
	hu.mutex.Lock()
	defer hu.mutex.Unlock()

	for _, mp := range hu.MountPoints {
		if mp.Device == pathname {
			return true, nil
		}
	}
	return false, nil
}

// PathIsDevice always returns true
func (hu *FakeHostUtil) PathIsDevice(pathname string) (bool, error) {
	return true, nil
}

// GetDeviceNameFromMount given a mount point, find the volume id
func (hu *FakeHostUtil) GetDeviceNameFromMount(mounter Interface, mountPath, pluginMountDir string) (string, error) {
	return getDeviceNameFromMount(mounter, mountPath, pluginMountDir)
}

// MakeRShared checks if path is shared and bind-mounts it as rshared if needed.
// No-op for testing
func (hu *FakeHostUtil) MakeRShared(path string) error {
	return nil
}

// GetFileType checks for file/directory/socket/block/character devices.
// Defaults to Directory if otherwise unspecified.
func (hu *FakeHostUtil) GetFileType(pathname string) (FileType, error) {
	if t, ok := hu.Filesystem[pathname]; ok {
		return t, nil
	}
	return FileType("Directory"), nil
}

// PathExists checks if pathname exists.
func (hu *FakeHostUtil) PathExists(pathname string) (bool, error) {
	if _, ok := hu.Filesystem[pathname]; ok {
		return true, nil
	}
	return false, nil
}

// EvalHostSymlinks returns the path name after evaluating symlinks.
// No-op for testing
func (hu *FakeHostUtil) EvalHostSymlinks(pathname string) (string, error) {
	return pathname, nil
}

// GetOwner returns the integer ID for the user and group of the given path
// Not implemented for testing
func (hu *FakeHostUtil) GetOwner(pathname string) (int64, int64, error) {
	return -1, -1, errors.New("GetOwner not implemented")
}

// GetSELinuxSupport tests if pathname is on a mount that supports SELinux.
// Not implemented for testing
func (hu *FakeHostUtil) GetSELinuxSupport(pathname string) (bool, error) {
	return false, errors.New("GetSELinuxSupport not implemented")
}

// GetMode returns permissions of pathname.
// Not implemented for testing
func (hu *FakeHostUtil) GetMode(pathname string) (os.FileMode, error) {
	return 0, errors.New("not implemented")
}

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
	"path/filepath"
	"sync"

	"k8s.io/klog"
)

// FakeMounter implements mount.Interface for tests.
type FakeMounter struct {
	MountPoints []MountPoint
	Log         []FakeAction
	// Error to return for a path when calling IsLikelyNotMountPoint
	MountCheckErrors map[string]error
	// Some tests run things in parallel, make sure the mounter does not produce
	// any golang's DATA RACE warnings.
	mutex sync.Mutex
}

var _ Interface = &FakeMounter{}

const (
	// FakeActionMount is the string for specifying mount as FakeAction.Action
	FakeActionMount = "mount"
	// FakeActionUnmount is the string for specifying unmount as FakeAction.Action
	FakeActionUnmount = "unmount"
)

// FakeAction objects are logged every time a fake mount or unmount is called.
type FakeAction struct {
	Action string // "mount" or "unmount"
	Target string // applies to both mount and unmount actions
	Source string // applies only to "mount" actions
	FSType string // applies only to "mount" actions
}

// ResetLog clears all the log entries in FakeMounter
func (f *FakeMounter) ResetLog() {
	f.mutex.Lock()
	defer f.mutex.Unlock()

	f.Log = []FakeAction{}
}

// Mount records the mount event and updates the in-memory mount points for FakeMounter
func (f *FakeMounter) Mount(source string, target string, fstype string, options []string) error {
	f.mutex.Lock()
	defer f.mutex.Unlock()

	opts := []string{}

	for _, option := range options {
		// find 'bind' option
		if option == "bind" {
			// This is a bind-mount. In order to mimic linux behaviour, we must
			// use the original device of the bind-mount as the real source.
			// E.g. when mounted /dev/sda like this:
			//      $ mount /dev/sda /mnt/test
			//      $ mount -o bind /mnt/test /mnt/bound
			// then /proc/mount contains:
			// /dev/sda /mnt/test
			// /dev/sda /mnt/bound
			// (and not /mnt/test /mnt/bound)
			// I.e. we must use /dev/sda as source instead of /mnt/test in the
			// bind mount.
			for _, mnt := range f.MountPoints {
				if source == mnt.Path {
					source = mnt.Device
					break
				}
			}
		}
		// reuse MountPoint.Opts field to mark mount as readonly
		opts = append(opts, option)
	}

	// If target is a symlink, get its absolute path
	absTarget, err := filepath.EvalSymlinks(target)
	if err != nil {
		absTarget = target
	}
	f.MountPoints = append(f.MountPoints, MountPoint{Device: source, Path: absTarget, Type: fstype, Opts: opts})
	klog.V(5).Infof("Fake mounter: mounted %s to %s", source, absTarget)
	f.Log = append(f.Log, FakeAction{Action: FakeActionMount, Target: absTarget, Source: source, FSType: fstype})
	return nil
}

// Unmount records the unmount event and updates the in-memory mount points for FakeMounter
func (f *FakeMounter) Unmount(target string) error {
	f.mutex.Lock()
	defer f.mutex.Unlock()

	// If target is a symlink, get its absolute path
	absTarget, err := filepath.EvalSymlinks(target)
	if err != nil {
		absTarget = target
	}

	newMountpoints := []MountPoint{}
	for _, mp := range f.MountPoints {
		if mp.Path == absTarget {
			klog.V(5).Infof("Fake mounter: unmounted %s from %s", mp.Device, absTarget)
			// Don't copy it to newMountpoints
			continue
		}
		newMountpoints = append(newMountpoints, MountPoint{Device: mp.Device, Path: mp.Path, Type: mp.Type})
	}
	f.MountPoints = newMountpoints
	f.Log = append(f.Log, FakeAction{Action: FakeActionUnmount, Target: absTarget})
	delete(f.MountCheckErrors, target)
	return nil
}

// List returns all the in-memory mountpoints for FakeMounter
func (f *FakeMounter) List() ([]MountPoint, error) {
	f.mutex.Lock()
	defer f.mutex.Unlock()

	return f.MountPoints, nil
}

// IsMountPointMatch tests if dir and mp are the same path
func (f *FakeMounter) IsMountPointMatch(mp MountPoint, dir string) bool {
	return mp.Path == dir
}

// IsLikelyNotMountPoint determines whether a path is a mountpoint by checking
// if the absolute path to file is in the in-memory mountpoints
func (f *FakeMounter) IsLikelyNotMountPoint(file string) (bool, error) {
	f.mutex.Lock()
	defer f.mutex.Unlock()

	err := f.MountCheckErrors[file]
	if err != nil {
		return false, err
	}

	_, err = os.Stat(file)
	if err != nil {
		return true, err
	}

	// If file is a symlink, get its absolute path
	absFile, err := filepath.EvalSymlinks(file)
	if err != nil {
		absFile = file
	}

	for _, mp := range f.MountPoints {
		if mp.Path == absFile {
			klog.V(5).Infof("isLikelyNotMountPoint for %s: mounted %s, false", file, mp.Path)
			return false, nil
		}
	}
	klog.V(5).Infof("isLikelyNotMountPoint for %s: true", file)
	return true, nil
}

// GetMountRefs finds all mount references to the path, returns a
// list of paths.
func (f *FakeMounter) GetMountRefs(pathname string) ([]string, error) {
	realpath, err := filepath.EvalSymlinks(pathname)
	if err != nil {
		// Ignore error in FakeMounter, because we actually didn't create files.
		realpath = pathname
	}
	return getMountRefsByDev(f, realpath)
}

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

// MakeDir creates a new directory.
// No-op for testing
func (hu *FakeHostUtil) MakeDir(pathname string) error {
	return nil
}

// MakeFile creates a new file.
// No-op for testing
func (hu *FakeHostUtil) MakeFile(pathname string) error {
	return nil
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

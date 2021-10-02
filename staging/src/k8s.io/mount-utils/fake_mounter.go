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
	"os"
	"path/filepath"
	"sync"

	"k8s.io/klog/v2"
)

// FakeMounter implements mount.Interface for tests.
type FakeMounter struct {
	MountPoints []MountPoint
	log         []FakeAction
	// Error to return for a path when calling IsLikelyNotMountPoint
	MountCheckErrors map[string]error
	// Some tests run things in parallel, make sure the mounter does not produce
	// any golang's DATA RACE warnings.
	mutex       sync.Mutex
	UnmountFunc UnmountFunc
}

// UnmountFunc is a function callback to be executed during the Unmount() call.
type UnmountFunc func(path string) error

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

// NewFakeMounter returns a FakeMounter struct that implements Interface and is
// suitable for testing purposes.
func NewFakeMounter(mps []MountPoint) *FakeMounter {
	return &FakeMounter{
		MountPoints: mps,
	}
}

// ResetLog clears all the log entries in FakeMounter
func (f *FakeMounter) ResetLog() {
	f.mutex.Lock()
	defer f.mutex.Unlock()

	f.log = []FakeAction{}
}

// GetLog returns the slice of FakeActions taken by the mounter
func (f *FakeMounter) GetLog() []FakeAction {
	f.mutex.Lock()
	defer f.mutex.Unlock()

	return f.log
}

// Mount records the mount event and updates the in-memory mount points for FakeMounter
func (f *FakeMounter) Mount(source string, target string, fstype string, options []string) error {
	return f.MountSensitive(source, target, fstype, options, nil /* sensitiveOptions */)
}

// Mount records the mount event and updates the in-memory mount points for FakeMounter
// sensitiveOptions to be passed in a separate parameter from the normal
// mount options and ensures the sensitiveOptions are never logged. This
// method should be used by callers that pass sensitive material (like
// passwords) as mount options.
func (f *FakeMounter) MountSensitive(source string, target string, fstype string, options []string, sensitiveOptions []string) error {
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
	f.MountPoints = append(f.MountPoints, MountPoint{Device: source, Path: absTarget, Type: fstype, Opts: append(opts, sensitiveOptions...)})
	klog.V(5).Infof("Fake mounter: mounted %s to %s", source, absTarget)
	f.log = append(f.log, FakeAction{Action: FakeActionMount, Target: absTarget, Source: source, FSType: fstype})
	return nil
}

func (f *FakeMounter) MountSensitiveWithoutSystemd(source string, target string, fstype string, options []string, sensitiveOptions []string) error {
	return f.MountSensitive(source, target, fstype, options, nil /* sensitiveOptions */)
}

func (f *FakeMounter) MountSensitiveWithoutSystemdWithMountFlags(source string, target string, fstype string, options []string, sensitiveOptions []string, mountFlags []string) error {
	return f.MountSensitive(source, target, fstype, options, nil /* sensitiveOptions */)
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
			if f.UnmountFunc != nil {
				err := f.UnmountFunc(absTarget)
				if err != nil {
					return err
				}
			}
			klog.V(5).Infof("Fake mounter: unmounted %s from %s", mp.Device, absTarget)
			// Don't copy it to newMountpoints
			continue
		}
		newMountpoints = append(newMountpoints, MountPoint{Device: mp.Device, Path: mp.Path, Type: mp.Type})
	}
	f.MountPoints = newMountpoints
	f.log = append(f.log, FakeAction{Action: FakeActionUnmount, Target: absTarget})
	delete(f.MountCheckErrors, target)
	return nil
}

// List returns all the in-memory mountpoints for FakeMounter
func (f *FakeMounter) List() ([]MountPoint, error) {
	f.mutex.Lock()
	defer f.mutex.Unlock()

	return f.MountPoints, nil
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

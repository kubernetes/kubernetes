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
	"sync"

	"github.com/golang/glog"
)

// FakeMounter implements mount.Interface for tests.
type FakeMounter struct {
	MountPoints []MountPoint
	Log         []FakeAction
	// Some tests run things in parallel, make sure the mounter does not produce
	// any golang's DATA RACE warnings.
	mutex sync.Mutex
}

var _ Interface = &FakeMounter{}

// Values for FakeAction.Action
const FakeActionMount = "mount"
const FakeActionUnmount = "unmount"

// FakeAction objects are logged every time a fake mount or unmount is called.
type FakeAction struct {
	Action string // "mount" or "unmount"
	Target string // applies to both mount and unmount actions
	Source string // applies only to "mount" actions
	FSType string // applies only to "mount" actions
}

func (f *FakeMounter) ResetLog() {
	f.mutex.Lock()
	defer f.mutex.Unlock()

	f.Log = []FakeAction{}
}

func (f *FakeMounter) Mount(source string, target string, fstype string, options []string) error {
	f.mutex.Lock()
	defer f.mutex.Unlock()

	// find 'bind' option
	for _, option := range options {
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
			break
		}
	}

	f.MountPoints = append(f.MountPoints, MountPoint{Device: source, Path: target, Type: fstype})
	glog.V(5).Infof("Fake mounter: mounted %s to %s", source, target)
	f.Log = append(f.Log, FakeAction{Action: FakeActionMount, Target: target, Source: source, FSType: fstype})
	return nil
}

func (f *FakeMounter) Unmount(target string) error {
	f.mutex.Lock()
	defer f.mutex.Unlock()

	newMountpoints := []MountPoint{}
	for _, mp := range f.MountPoints {
		if mp.Path == target {
			glog.V(5).Infof("Fake mounter: unmounted %s from %s", mp.Device, target)
			// Don't copy it to newMountpoints
			continue
		}
		newMountpoints = append(newMountpoints, MountPoint{Device: mp.Device, Path: mp.Path, Type: mp.Type})
	}
	f.MountPoints = newMountpoints
	f.Log = append(f.Log, FakeAction{Action: FakeActionUnmount, Target: target})
	return nil
}

func (f *FakeMounter) List() ([]MountPoint, error) {
	f.mutex.Lock()
	defer f.mutex.Unlock()

	return f.MountPoints, nil
}

func (f *FakeMounter) IsLikelyNotMountPoint(file string) (bool, error) {
	f.mutex.Lock()
	defer f.mutex.Unlock()

	for _, mp := range f.MountPoints {
		if mp.Path == file {
			glog.V(5).Infof("isLikelyMountPoint for %s: mounted %s, false", file, mp.Path)
			return false, nil
		}
	}
	glog.V(5).Infof("isLikelyMountPoint for %s: true", file)
	return true, nil
}

func (f *FakeMounter) DeviceOpened(pathname string) (bool, error) {
	f.mutex.Lock()
	defer f.mutex.Unlock()

	for _, mp := range f.MountPoints {
		if mp.Device == pathname {
			return true, nil
		}
	}
	return false, nil
}

func (f *FakeMounter) PathIsDevice(pathname string) (bool, error) {
	return true, nil
}

func (f *FakeMounter) GetDeviceNameFromMount(mountPath, pluginDir string) (string, error) {
	return getDeviceNameFromMount(f, mountPath, pluginDir)
}

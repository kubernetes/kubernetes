/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

// FakeMounter implements mount.Interface for tests.
type FakeMounter struct {
	MountPoints []MountPoint
	Log         []FakeAction
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
	f.Log = []FakeAction{}
}

func (f *FakeMounter) Mount(source string, target string, fstype string, options []string) error {
	f.MountPoints = append(f.MountPoints, MountPoint{Device: source, Path: target, Type: fstype})
	f.Log = append(f.Log, FakeAction{Action: FakeActionMount, Target: target, Source: source, FSType: fstype})
	return nil
}

func (f *FakeMounter) Unmount(target string) error {
	newMountpoints := []MountPoint{}
	for _, mp := range f.MountPoints {
		if mp.Path != target {
			newMountpoints = append(newMountpoints, MountPoint{Device: mp.Device, Path: mp.Path, Type: mp.Type})
		}
	}
	f.MountPoints = newMountpoints
	f.Log = append(f.Log, FakeAction{Action: FakeActionUnmount, Target: target})
	return nil
}

func (f *FakeMounter) List() ([]MountPoint, error) {
	return f.MountPoints, nil
}

func (f *FakeMounter) IsMountPoint(file string) (bool, error) {
	for _, mp := range f.MountPoints {
		if mp.Path == file {
			return true, nil
		}
	}
	return false, nil
}

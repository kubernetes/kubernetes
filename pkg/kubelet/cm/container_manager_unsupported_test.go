// +build !linux,!windows

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

package cm

import (
	"fmt"

	"k8s.io/kubernetes/pkg/util/mount"
)

type fakeMountInterface struct {
	mountPoints []mount.MountPoint
}

func (mi *fakeMountInterface) Mount(source string, target string, fstype string, options []string) error {
	return fmt.Errorf("unsupported")
}

func (mi *fakeMountInterface) Unmount(target string) error {
	return fmt.Errorf("unsupported")
}

func (mi *fakeMountInterface) List() ([]mount.MountPoint, error) {
	return mi.mountPoints, nil
}

func (f *fakeMountInterface) IsMountPointMatch(mp mount.MountPoint, dir string) bool {
	return (mp.Path == dir)
}

func (f *fakeMountInterface) IsNotMountPoint(dir string) (bool, error) {
	return false, fmt.Errorf("unsupported")
}

func (mi *fakeMountInterface) IsLikelyNotMountPoint(file string) (bool, error) {
	return false, fmt.Errorf("unsupported")
}

func (mi *fakeMountInterface) DeviceOpened(pathname string) (bool, error) {
	for _, mp := range mi.mountPoints {
		if mp.Device == pathname {
			return true, nil
		}
	}
	return false, nil
}

func (mi *fakeMountInterface) PathIsDevice(pathname string) (bool, error) {
	return true, nil
}

func (mi *fakeMountInterface) GetDeviceNameFromMount(mountPath, pluginDir string) (string, error) {
	return "", nil
}

func (mi *fakeMountInterface) MakeRShared(path string) error {
	return nil
}

func fakeContainerMgrMountInt() mount.Interface {
	return &fakeMountInterface{
		[]mount.MountPoint{
			{
				Device: "cgroup",
				Type:   "cgroup",
				Opts:   []string{"rw", "relatime", "cpuset"},
			},
			{
				Device: "cgroup",
				Type:   "cgroup",
				Opts:   []string{"rw", "relatime", "cpu"},
			},
			{
				Device: "cgroup",
				Type:   "cgroup",
				Opts:   []string{"rw", "relatime", "cpuacct"},
			},
			{
				Device: "cgroup",
				Type:   "cgroup",
				Opts:   []string{"rw", "relatime", "memory"},
			},
		},
	}
}

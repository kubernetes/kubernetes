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

package mount

type NsenterMounter struct{}

func NewNsenterMounter() *NsenterMounter {
	return &NsenterMounter{}
}

var _ = Interface(&NsenterMounter{})

func (*NsenterMounter) Mount(source string, target string, fstype string, options []string) error {
	return nil
}

func (*NsenterMounter) Unmount(target string) error {
	return nil
}

func (*NsenterMounter) List() ([]MountPoint, error) {
	return []MountPoint{}, nil
}

func (m *NsenterMounter) IsNotMountPoint(dir string) (bool, error) {
	return IsNotMountPoint(m, dir)
}

func (*NsenterMounter) IsMountPointMatch(mp MountPoint, dir string) bool {
	return (mp.Path == dir)
}

func (*NsenterMounter) IsLikelyNotMountPoint(file string) (bool, error) {
	return true, nil
}

func (*NsenterMounter) DeviceOpened(pathname string) (bool, error) {
	return false, nil
}

func (*NsenterMounter) PathIsDevice(pathname string) (bool, error) {
	return true, nil
}

func (*NsenterMounter) GetDeviceNameFromMount(mountPath, pluginDir string) (string, error) {
	return "", nil
}

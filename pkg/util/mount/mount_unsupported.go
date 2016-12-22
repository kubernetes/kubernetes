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

type Mounter struct {
	mounterPath string
}

func (mounter *Mounter) Mount(source string, target string, fstype string, options []string) error {
	return nil
}

func (mounter *Mounter) Unmount(target string) error {
	return nil
}

func (mounter *Mounter) List() ([]MountPoint, error) {
	return []MountPoint{}, nil
}

func (mounter *Mounter) IsLikelyNotMountPoint(file string) (bool, error) {
	return true, nil
}

func (mounter *Mounter) GetDeviceNameFromMount(mountPath, pluginDir string) (string, error) {
	return "", nil
}

func (mounter *Mounter) DeviceOpened(pathname string) (bool, error) {
	return false, nil
}

func (mounter *Mounter) PathIsDevice(pathname string) (bool, error) {
	return true, nil
}

func (mounter *SafeFormatAndMount) formatAndMount(source string, target string, fstype string, options []string) error {
	return nil
}

func (mounter *SafeFormatAndMount) diskLooksUnformatted(disk string) (bool, error) {
	return true, nil
}

func IsNotMountPoint(file string) (bool, error) {
	return true, nil
}

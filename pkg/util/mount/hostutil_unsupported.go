// +build !linux,!windows

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
	"os"
)

type hostUtil struct{}

// NewHostUtil returns a struct that implements the HostUtils interface on
// unsupported platforms
func NewHostUtil() HostUtils {
	return &hostUtil{}
}

// DeviceOpened determines if the device is in use elsewhere
func (hu *hostUtil) DeviceOpened(pathname string) (bool, error) {
	return false, errUnsupported
}

// PathIsDevice determines if a path is a device.
func (hu *hostUtil) PathIsDevice(pathname string) (bool, error) {
	return true, errUnsupported
}

// GetDeviceNameFromMount finds the device name by checking the mount path
// to get the global mount path within its plugin directory
func (hu *hostUtil) GetDeviceNameFromMount(mounter Interface, mountPath, pluginMountDir string) (string, error) {
	return "", errUnsupported
}

func (hu *hostUtil) MakeRShared(path string) error {
	return errUnsupported
}

func (hu *hostUtil) GetFileType(pathname string) (FileType, error) {
	return FileType("fake"), errUnsupported
}

func (hu *hostUtil) MakeFile(pathname string) error {
	return errUnsupported
}

func (hu *hostUtil) MakeDir(pathname string) error {
	return errUnsupported
}

func (hu *hostUtil) PathExists(pathname string) (bool, error) {
	return true, errUnsupported
}

// EvalHostSymlinks returns the path name after evaluating symlinks
func (hu *hostUtil) EvalHostSymlinks(pathname string) (string, error) {
	return "", errUnsupported
}

// GetOwner returns the integer ID for the user and group of the given path
func (hu *hostUtil) GetOwner(pathname string) (int64, int64, error) {
	return -1, -1, errUnsupported
}

func (hu *hostUtil) GetSELinuxSupport(pathname string) (bool, error) {
	return false, errUnsupported
}

func (hu *hostUtil) GetMode(pathname string) (os.FileMode, error) {
	return 0, errUnsupported
}

//go:build !linux && !windows
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

package hostutil

import (
	"errors"
	"os"

	"k8s.io/mount-utils"
)

// HostUtil is an HostUtils implementation that allows compilation on
// unsupported platforms
type HostUtil struct{}

// NewHostUtil returns a struct that implements the HostUtils interface on
// unsupported platforms
func NewHostUtil() *HostUtil {
	return &HostUtil{}
}

var errUnsupported = errors.New("volume/util/hostutil on this platform is not supported")

// DeviceOpened always returns an error on unsupported platforms
func (hu *HostUtil) DeviceOpened(pathname string) (bool, error) {
	return false, errUnsupported
}

// PathIsDevice always returns an error on unsupported platforms
func (hu *HostUtil) PathIsDevice(pathname string) (bool, error) {
	return true, errUnsupported
}

// GetDeviceNameFromMount always returns an error on unsupported platforms
func (hu *HostUtil) GetDeviceNameFromMount(mounter mount.Interface, mountPath, pluginMountDir string) (string, error) {
	return getDeviceNameFromMount(mounter, mountPath, pluginMountDir)
}

// MakeRShared always returns an error on unsupported platforms
func (hu *HostUtil) MakeRShared(path string) error {
	return errUnsupported
}

// GetFileType always returns an error on unsupported platforms
func (hu *HostUtil) GetFileType(pathname string) (FileType, error) {
	return FileType("fake"), errUnsupported
}

// MakeFile always returns an error on unsupported platforms
func (hu *HostUtil) MakeFile(pathname string) error {
	return errUnsupported
}

// MakeDir always returns an error on unsupported platforms
func (hu *HostUtil) MakeDir(pathname string) error {
	return errUnsupported
}

// PathExists always returns an error on unsupported platforms
func (hu *HostUtil) PathExists(pathname string) (bool, error) {
	return true, errUnsupported
}

// EvalHostSymlinks always returns an error on unsupported platforms
func (hu *HostUtil) EvalHostSymlinks(pathname string) (string, error) {
	return "", errUnsupported
}

// GetOwner always returns an error on unsupported platforms
func (hu *HostUtil) GetOwner(pathname string) (int64, int64, error) {
	return -1, -1, errUnsupported
}

// GetSELinuxSupport always returns an error on unsupported platforms
func (hu *HostUtil) GetSELinuxSupport(pathname string) (bool, error) {
	return false, errUnsupported
}

// GetMode always returns an error on unsupported platforms
func (hu *HostUtil) GetMode(pathname string) (os.FileMode, error) {
	return 0, errUnsupported
}

func getDeviceNameFromMount(mounter mount.Interface, mountPath, pluginMountDir string) (string, error) {
	return "", errUnsupported
}

// GetSELinuxMountContext returns value of -o context=XYZ mount option on
// given mount point.
func (hu *HostUtil) GetSELinuxMountContext(pathname string) (string, error) {
	return "", errUnsupported
}

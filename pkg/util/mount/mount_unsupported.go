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
	"errors"
	"os"
)

type Mounter struct {
	mounterPath string
}

var unsupportedErr = errors.New("util/mount on this platform is not supported")

// New returns a mount.Interface for the current system.
// It provides options to override the default mounter behavior.
// mounterPath allows using an alternative to `/bin/mount` for mounting.
func New(mounterPath string) Interface {
	return &Mounter{
		mounterPath: mounterPath,
	}
}

func (mounter *Mounter) Mount(source string, target string, fstype string, options []string) error {
	return unsupportedErr
}

func (mounter *Mounter) Unmount(target string) error {
	return unsupportedErr
}

func (mounter *Mounter) List() ([]MountPoint, error) {
	return []MountPoint{}, unsupportedErr
}

func (mounter *Mounter) IsMountPointMatch(mp MountPoint, dir string) bool {
	return (mp.Path == dir)
}

func (mounter *Mounter) IsLikelyNotMountPoint(file string) (bool, error) {
	return true, unsupportedErr
}

func (mounter *Mounter) GetMountRefs(pathname string) ([]string, error) {
	return nil, errors.New("not implemented")
}

func (mounter *SafeFormatAndMount) formatAndMount(source string, target string, fstype string, options []string) error {
	return mounter.Interface.Mount(source, target, fstype, options)
}

func (mounter *SafeFormatAndMount) diskLooksUnformatted(disk string) (bool, error) {
	return true, unsupportedErr
}

func getDeviceNameFromMount(mounter Interface, mountPath, pluginMountDir string) (string, error) {
	return "", unsupportedErr
}

type hostUtil struct{}

func NewHostUtil() HostUtils {
	return &hostUtil{}
}

// DeviceOpened determines if the device is in use elsewhere
func (hu *hostUtil) DeviceOpened(pathname string) (bool, error) {
	return false, unsupportedErr
}

// PathIsDevice determines if a path is a device.
func (hu *hostUtil) PathIsDevice(pathname string) (bool, error) {
	return true, unsupportedErr
}

// GetDeviceNameFromMount finds the device name by checking the mount path
// to get the global mount path within its plugin directory
func (hu *hostUtil) GetDeviceNameFromMount(mounter Interface, mountPath, pluginMountDir string) (string, error) {
	return "", unsupportedErr
}

func (hu *hostUtil) MakeRShared(path string) error {
	return unsupportedErr
}

func (hu *hostUtil) GetFileType(pathname string) (FileType, error) {
	return FileType("fake"), unsupportedErr
}

func (hu *hostUtil) MakeFile(pathname string) error {
	return unsupportedErr
}

func (hu *hostUtil) MakeDir(pathname string) error {
	return unsupportedErr
}

func (hu *hostUtil) ExistsPath(pathname string) (bool, error) {
	return true, unsupportedErr
}

// EvalHostSymlinks returns the path name after evaluating symlinks
func (hu *hostUtil) EvalHostSymlinks(pathname string) (string, error) {
	return "", unsupportedErr
}

func (hu *hostUtil) GetFSGroup(pathname string) (int64, error) {
	return -1, unsupportedErr
}

func (hu *hostUtil) GetSELinuxSupport(pathname string) (bool, error) {
	return false, unsupportedErr
}

func (hu *hostUtil) GetMode(pathname string) (os.FileMode, error) {
	return 0, unsupportedErr
}

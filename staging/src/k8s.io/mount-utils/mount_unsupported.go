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

package mount

import (
	"errors"
)

// Mounter implements mount.Interface for unsupported platforms
type Mounter struct {
	mounterPath string
}

var errUnsupported = errors.New("util/mount on this platform is not supported")

// New returns a mount.Interface for the current system.
// It provides options to override the default mounter behavior.
// mounterPath allows using an alternative to `/bin/mount` for mounting.
func New(mounterPath string) Interface {
	return &Mounter{
		mounterPath: mounterPath,
	}
}

// Mount always returns an error on unsupported platforms
func (mounter *Mounter) Mount(source string, target string, fstype string, options []string) error {
	return errUnsupported
}

// MountSensitive always returns an error on unsupported platforms
func (mounter *Mounter) MountSensitive(source string, target string, fstype string, options []string, sensitiveOptions []string) error {
	return errUnsupported
}

// MountSensitiveWithoutSystemd always returns an error on unsupported platforms
func (mounter *Mounter) MountSensitiveWithoutSystemd(source string, target string, fstype string, options []string, sensitiveOptions []string) error {
	return errUnsupported
}

// MountSensitiveWithoutSystemdWithMountFlags always returns an error on unsupported platforms
func (mounter *Mounter) MountSensitiveWithoutSystemdWithMountFlags(source string, target string, fstype string, options []string, sensitiveOptions []string, mountFlags []string) error {
	return errUnsupported
}

// Unmount always returns an error on unsupported platforms
func (mounter *Mounter) Unmount(target string) error {
	return errUnsupported
}

// List always returns an error on unsupported platforms
func (mounter *Mounter) List() ([]MountPoint, error) {
	return []MountPoint{}, errUnsupported
}

// IsLikelyNotMountPoint always returns an error on unsupported platforms
func (mounter *Mounter) IsLikelyNotMountPoint(file string) (bool, error) {
	return true, errUnsupported
}

// canSafelySkipMountPointCheck always returns false on unsupported platforms
func (mounter *Mounter) canSafelySkipMountPointCheck() bool {
	return false
}

// GetMountRefs always returns an error on unsupported platforms
func (mounter *Mounter) GetMountRefs(pathname string) ([]string, error) {
	return nil, errUnsupported
}

func (mounter *SafeFormatAndMount) formatAndMountSensitive(source string, target string, fstype string, options []string, sensitiveOptions []string) error {
	return mounter.Interface.Mount(source, target, fstype, options)
}

func (mounter *SafeFormatAndMount) diskLooksUnformatted(disk string) (bool, error) {
	return true, errUnsupported
}

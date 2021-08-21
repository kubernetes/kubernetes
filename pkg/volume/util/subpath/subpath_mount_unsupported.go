// +build !linux,!windows

/*
Copyright 2021 The Kubernetes Authors.

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

package subpath

import (
	"errors"

	"k8s.io/utils/mount"
)

// Mounter implements mount.Interface for unsupported platforms
type Mounter struct {
	mount.Interface
	mounterPath string
}

var errUtilsMountUnsupported = errors.New("utils/mount on this platform is not supported")

// NewMounter returns a MountInterface for the current system.
// It provides options to override the default mounter behavior.
// mounterPath allows using an alternative to `/bin/mount` for mounting.
func NewMounter(mounter mount.Interface, mounterPath string) MountInterface {
	return &Mounter{
		Interface:   mounter,
		mounterPath: mounterPath,
	}
}

// MountSensitiveWithFlags is the same as MountSensitive() with additional mount flags
func (mounter *Mounter) MountSensitiveWithFlags(source string, target string, fstype string, options []string, sensitiveOptions []string, mountFlags []string) error {
	return errUtilsMountUnsupported
}

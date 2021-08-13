// +build windows

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
	"k8s.io/utils/mount"
)

// Mounter provides the subpath implementation of mount.Interface
// for the windows platform.  This implementation assumes that the
// kubelet is running in the host's root mount namespace.
type Mounter struct {
	mount.Interface
	mounterPath string
}

// NewMounter returns a MountInterface for the current system.
// It provides options to override the default mounter behavior.
// mounterPath allows using an alternative to `/bin/mount` for mounting.
func NewMounter(mounter mount.Interface, mounterPath string) MountInterface {
	return &Mounter{
		Interface:   mounter,
		mounterPath: mounterPath,
	}
}

// MountSensitiveWithFlags is the same as MountSensitive() with additional mount flags but
// because mountFlags are linux mount(8) flags this method is the same as MountSensitive() in Windows
func (mounter *Mounter) MountSensitiveWithFlags(source string, target string, fstype string, options []string, sensitiveOptions []string, mountFlags []string) error {
	return mounter.MountSensitive(source, target, fstype, options, sensitiveOptions)
}

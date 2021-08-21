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
	mountutils "k8s.io/utils/mount"
)

// FakeMounter implements MountInterface for tests.
type FakeMounter struct {
	*mountutils.FakeMounter
}

var _ MountInterface = &FakeMounter{}

// NewFakeMounter returns a FakeMounter struct that implements Interface and is
// suitable for testing purposes.
func NewFakeMounter(mps []mountutils.MountPoint) *FakeMounter {
	return &FakeMounter{
		FakeMounter: &mountutils.FakeMounter{
			MountPoints: mps,
		},
	}
}

// MountSensitiveWithFlags records the mount event and updates the in-memory mount points for FakeMounter
// sensitiveOptions to be passed in a separate parameter from the normal
// mount options and ensures the sensitiveOptions are never logged. This
// method should be used by callers that pass sensitive material (like
// passwords) as mount options.
func (f *FakeMounter) MountSensitiveWithFlags(source string, target string, fstype string, options []string, sensitiveOptions []string, mountFlags []string) error {
	return f.MountSensitive(source, target, fstype, options, sensitiveOptions)
}

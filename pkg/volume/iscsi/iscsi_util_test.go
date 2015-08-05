/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package iscsi

import (
	"testing"

	"k8s.io/kubernetes/pkg/util/mount"
)

func TestGetDevicePrefixRefCount(t *testing.T) {
	fm := &mount.FakeMounter{
		MountPoints: []mount.MountPoint{
			{Device: "/dev/disk/by-path/prefix-lun-1",
				Path: "/mnt/111"},
			{Device: "/dev/disk/by-path/prefix-lun-1",
				Path: "/mnt/222"},
			{Device: "/dev/disk/by-path/prefix-lun-0",
				Path: "/mnt/333"},
			{Device: "/dev/disk/by-path/prefix-lun-0",
				Path: "/mnt/444"},
		},
	}

	tests := []struct {
		devicePrefix string
		expectedRefs int
	}{
		{
			"/dev/disk/by-path/prefix",
			4,
		},
	}

	for i, test := range tests {
		if refs, err := getDevicePrefixRefCount(fm, test.devicePrefix); err != nil || test.expectedRefs != refs {
			t.Errorf("%d. GetDevicePrefixRefCount(%s) = %d, %v; expected %d, nil", i, test.devicePrefix, refs, err, test.expectedRefs)
		}
	}
}

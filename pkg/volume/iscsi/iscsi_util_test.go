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
			{Device: "/dev/sdb",
				Path: "/127.0.0.1:3260-iqn.2014-12.com.example:test.tgt00-lun-0"},
			{Device: "/dev/sdb",
				Path: "/127.0.0.1:3260-iqn.2014-12.com.example:test.tgt00-lun-1"},
			{Device: "/dev/sdb",
				Path: "/127.0.0.1:3260-iqn.2014-12.com.example:test.tgt00-lun-2"},
			{Device: "/dev/sdb",
				Path: "/127.0.0.1:3260-iqn.2014-12.com.example:test.tgt00-lun-3"},
		},
	}

	tests := []struct {
		devicePrefix string
		expectedRefs int
	}{
		{
			"/127.0.0.1:3260-iqn.2014-12.com.example:test.tgt00",
			4,
		},
	}

	for i, test := range tests {
		if refs, err := getDevicePrefixRefCount(fm, test.devicePrefix); err != nil || test.expectedRefs != refs {
			t.Errorf("%d. GetDevicePrefixRefCount(%s) = %d, %v; expected %d, nil", i, test.devicePrefix, refs, err, test.expectedRefs)
		}
	}
}

func TestExtractDeviceAndPrefix(t *testing.T) {
	devicePath := "127.0.0.1:3260-iqn.2014-12.com.example:test.tgt00"
	lun := "-lun-0"
	device, prefix, err := extractDeviceAndPrefix("/var/lib/kubelet/plugins/kubernetes.io/iscsi/" + devicePath + lun)
	if err != nil || device != (devicePath+lun) || prefix != devicePath {
		t.Errorf("extractDeviceAndPrefix: expected %s and %s, got %v %s and %s", devicePath+lun, devicePath, err, device, prefix)
	}
}

func TestExtractPortalAndIqn(t *testing.T) {
	devicePath := "127.0.0.1:3260-iqn.2014-12.com.example:test.tgt00-lun-0"
	portal, iqn, err := extractPortalAndIqn(devicePath)
	if err != nil || portal != "127.0.0.1:3260" || iqn != "iqn.2014-12.com.example:test.tgt00" {
		t.Errorf("extractPortalAndIqn: got %v %s %s", err, portal, iqn)
	}
}

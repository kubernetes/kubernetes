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
	"os"
	"path/filepath"
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
	devicePath = "127.0.0.1:3260-eui.02004567A425678D-lun-0"
	portal, iqn, err = extractPortalAndIqn(devicePath)
	if err != nil || portal != "127.0.0.1:3260" || iqn != "eui.02004567A425678D" {
		t.Errorf("extractPortalAndIqn: got %v %s %s", err, portal, iqn)
	}
}

func fakeOsStat(devicePath string) (fi os.FileInfo, err error) {
	var cmd os.FileInfo
	return cmd, nil
}

func fakeFilepathGlob(devicePath string) (globs []string, err error) {
	return []string{devicePath}, nil
}

func TestWaitForPathToExist(t *testing.T) {
	devicePath := []string{"/dev/disk/by-path/ip-127.0.0.1:3260-iqn.2014-12.com.example:test.tgt00-lun-0",
		"/dev/disk/by-path/pci-0000:00:00.0-ip-127.0.0.1:3260-iqn.2014-12.com.example:test.tgt00-lun-0"}

	exist := waitForPathToExistInternal(devicePath[0], 1, "default", fakeOsStat, filepath.Glob)
	if exist == false {
		t.Errorf("waitForPathToExist: could not find path %s", devicePath[0])
	}
	exist = waitForPathToExistInternal(devicePath[0], 1, "fake_iface", fakeOsStat, filepath.Glob)
	if exist != false {
		t.Errorf("waitForPathToExist: wrong code path called for %s", devicePath[0])
	}

	exist = waitForPathToExistInternal(devicePath[1], 1, "fake_iface", os.Stat, fakeFilepathGlob)
	if exist == false {
		t.Errorf("waitForPathToExist: could not find path %s", devicePath[1])
	}
	exist = waitForPathToExistInternal(devicePath[1], 1, "default", os.Stat, fakeFilepathGlob)
	if exist != false {
		t.Errorf("waitForPathToExist: wrong code path called for %s", devicePath[1])
	}
}

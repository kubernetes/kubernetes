/*
Copyright 2015 The Kubernetes Authors.

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

package nutanix_volume

import (
	"os"
	"testing"

	"k8s.io/kubernetes/pkg/util/mount"
)

func TestGetDevicePrefixRefCount(t *testing.T) {
	fm := &mount.FakeMounter{
		MountPoints: []mount.MountPoint{
			{Device: "/dev/sdb",
				Path: "/var/lib/kubelet/plugins/kubernetes.io/nutanix-volume/50f9a22f-633d-11e7-94dd-506b8deaff1c/10.5.65.156:3260-iqn.2010-06.com.nutanix:nutanix-k8-volume-4777ca23-5807-4e4d-9a28-e0378f58d31a-tgt0-lun-0"},
			{Device: "/dev/sdb",
				Path: "/var/lib/kubelet/plugins/kubernetes.io/nutanix-volume/50f9a22f-633d-11e7-94dd-506b8deaff1c/10.5.65.156:3260-iqn.2010-06.com.nutanix:nutanix-k8-volume-4777ca23-5807-4e4d-9a28-e0378f58d31a-tgt0-lun-1"},
			{Device: "/dev/sdb",
				Path: "/var/lib/kubelet/plugins/kubernetes.io/nutanix-volume/50f9a22f-633d-11e7-94dd-506b8deaff1c/10.5.65.156:3260-iqn.2010-06.com.nutanix:nutanix-k8-volume-4777ca23-5807-4e4d-9a28-e0378f58d31a-tgt0-lun-2"},
		},
	}

	tests := []struct {
		devicePrefix string
		expectedRefs int
	}{
		{
			"/var/lib/kubelet/plugins/kubernetes.io/nutanix-volume/50f9a22f-633d-11e7-94dd-506b8deaff1c/10.5.65.156:3260-iqn.2010-06.com.nutanix:nutanix-k8-volume-4777ca23-5807-4e4d-9a28-e0378f58d31a-tgt0",
			3,
		},
	}

	for i, test := range tests {
		if refs, err := getDevicePrefixRefCount(fm, test.devicePrefix); err != nil || test.expectedRefs != refs {
			t.Errorf("%d. GetDevicePrefixRefCount(%s) = %d, %v; expected %d, nil", i, test.devicePrefix, refs, err, test.expectedRefs)
		}
	}
}

func TestExtractDeviceAndPrefix(t *testing.T) {
	devicePath := "10.5.65.156:3260-iqn.2010-06.com.nutanix:c50d776301b75e568ab232af3123087f492b02fccecacc8e3aa0a9ddb78d8912:nutanix-k8-volume-plugin-tgt0"
	mountPrefix := "/var/lib/kubelet/plugins/kubernetes.io/nutanix-volume/80dbbf9d-6333-11e7-94dd-506b8deaff1c/" + devicePath
	lun := "-lun-0"
	device, prefix, err := extractDeviceAndPrefix(mountPrefix + lun)
	if err != nil || device != (devicePath+lun) || prefix != mountPrefix {
		t.Errorf("extractDeviceAndPrefix: expected %s and %s, got %v %s and %s", devicePath+lun, mountPrefix, err, device, prefix)
	}
}

func TestExtractPortalAndIqn(t *testing.T) {
	devicePath := "10.5.65.156:3260-c50d776301b75e568ab232af3123087f492b02fccecacc8e3aa0a9ddb78d8912:nutanix-k8-volume-plugin-lun-0"
	portal, iqn, err := extractPortalAndIqn(devicePath)
	if err != nil || portal != "10.5.65.156:3260" || iqn != "c50d776301b75e568ab232af3123087f492b02fccecacc8e3aa0a9ddb78d8912:nutanix-k8-volume-plugin" {
		t.Errorf("extractPortalAndIqn: got %v %s %s", err, portal, iqn)
	}
}

func fakeOsStat(devicePath string) (fi os.FileInfo, err error) {
	var cmd os.FileInfo
	return cmd, nil
}


func TestWaitForPathToExist(t *testing.T) {
	devicePath := []string{"/dev/disk/by-path/ip-10.5.65.156:3260-iscsi-iqn.2010-06.com.nutanix:c50d776301b75e568ab232af3123087f492b02fccecacc8e3aa0a9ddb78d8912:nutanix-k8-volume-plugin-tgt0-lun-0"}

	exist := waitForPathToExistInternal(devicePath[0], 1, fakeOsStat)
	if exist == false {
		t.Errorf("waitForPathToExist: could not find path %s", devicePath[0])
	}
}

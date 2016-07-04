// +build linux

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
	"strings"
	"testing"
)

func TestReadProcMountsFrom(t *testing.T) {
	successCase :=
		`/dev/0 /path/to/0 type0 flags 0 0
		/dev/1    /path/to/1   type1	flags 1 1
		/dev/2 /path/to/2 type2 flags,1,2=3 2 2
		`
	hash, err := readProcMountsFrom(strings.NewReader(successCase), nil)
	if err != nil {
		t.Errorf("expected success")
	}
	if hash != 0xa3522051 {
		t.Errorf("expected 0xa3522051, got %#x", hash)
	}
	mounts := []MountPoint{}
	hash, err = readProcMountsFrom(strings.NewReader(successCase), &mounts)
	if err != nil {
		t.Errorf("expected success")
	}
	if hash != 0xa3522051 {
		t.Errorf("expected 0xa3522051, got %#x", hash)
	}
	if len(mounts) != 3 {
		t.Fatalf("expected 3 mounts, got %d", len(mounts))
	}
	mp := MountPoint{"/dev/0", "/path/to/0", "type0", []string{"flags"}, 0, 0}
	if !mountPointsEqual(&mounts[0], &mp) {
		t.Errorf("got unexpected MountPoint[0]: %#v", mounts[0])
	}
	mp = MountPoint{"/dev/1", "/path/to/1", "type1", []string{"flags"}, 1, 1}
	if !mountPointsEqual(&mounts[1], &mp) {
		t.Errorf("got unexpected MountPoint[1]: %#v", mounts[1])
	}
	mp = MountPoint{"/dev/2", "/path/to/2", "type2", []string{"flags", "1", "2=3"}, 2, 2}
	if !mountPointsEqual(&mounts[2], &mp) {
		t.Errorf("got unexpected MountPoint[2]: %#v", mounts[2])
	}

	errorCases := []string{
		"/dev/0 /path/to/mount\n",
		"/dev/1 /path/to/mount type flags a 0\n",
		"/dev/2 /path/to/mount type flags 0 b\n",
	}
	for _, ec := range errorCases {
		_, err := readProcMountsFrom(strings.NewReader(ec), &mounts)
		if err == nil {
			t.Errorf("expected error")
		}
	}
}

func mountPointsEqual(a, b *MountPoint) bool {
	if a.Device != b.Device || a.Path != b.Path || a.Type != b.Type || !slicesEqual(a.Opts, b.Opts) || a.Pass != b.Pass || a.Freq != b.Freq {
		return false
	}
	return true
}

func slicesEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestGetMountRefs(t *testing.T) {
	fm := &FakeMounter{
		MountPoints: []MountPoint{
			{Device: "/dev/sdb", Path: "/var/lib/kubelet/plugins/kubernetes.io/gce-pd/mounts/gce-pd"},
			{Device: "/dev/sdb", Path: "/var/lib/kubelet/pods/some-pod/volumes/kubernetes.io~gce-pd/gce-pd-in-pod"},
			{Device: "/dev/sdc", Path: "/var/lib/kubelet/plugins/kubernetes.io/gce-pd/mounts/gce-pd2"},
			{Device: "/dev/sdc", Path: "/var/lib/kubelet/pods/some-pod/volumes/kubernetes.io~gce-pd/gce-pd2-in-pod"},
			{Device: "/dev/sdc", Path: "/var/lib/kubelet/pods/some-pod/volumes/kubernetes.io~gce-pd/gce-pd2-in-pod2"},
		},
	}

	tests := []struct {
		mountPath    string
		expectedRefs []string
	}{
		{
			"/var/lib/kubelet/pods/some-pod/volumes/kubernetes.io~gce-pd/gce-pd-in-pod",
			[]string{
				"/var/lib/kubelet/plugins/kubernetes.io/gce-pd/mounts/gce-pd",
			},
		},
		{
			"/var/lib/kubelet/pods/some-pod/volumes/kubernetes.io~gce-pd/gce-pd2-in-pod",
			[]string{
				"/var/lib/kubelet/pods/some-pod/volumes/kubernetes.io~gce-pd/gce-pd2-in-pod2",
				"/var/lib/kubelet/plugins/kubernetes.io/gce-pd/mounts/gce-pd2",
			},
		},
	}

	for i, test := range tests {
		if refs, err := GetMountRefs(fm, test.mountPath); err != nil || !setEquivalent(test.expectedRefs, refs) {
			t.Errorf("%d. getMountRefs(%q) = %v, %v; expected %v, nil", i, test.mountPath, refs, err, test.expectedRefs)
		}
	}
}

func setEquivalent(set1, set2 []string) bool {
	map1 := make(map[string]bool)
	map2 := make(map[string]bool)
	for _, s := range set1 {
		map1[s] = true
	}
	for _, s := range set2 {
		map2[s] = true
	}

	for s := range map1 {
		if !map2[s] {
			return false
		}
	}
	for s := range map2 {
		if !map1[s] {
			return false
		}
	}
	return true
}

func TestGetDeviceNameFromMount(t *testing.T) {
	fm := &FakeMounter{
		MountPoints: []MountPoint{
			{Device: "/dev/disk/by-path/prefix-lun-1",
				Path: "/mnt/111"},
			{Device: "/dev/disk/by-path/prefix-lun-1",
				Path: "/mnt/222"},
		},
	}

	tests := []struct {
		mountPath      string
		expectedDevice string
		expectedRefs   int
	}{
		{
			"/mnt/222",
			"/dev/disk/by-path/prefix-lun-1",
			2,
		},
	}

	for i, test := range tests {
		if device, refs, err := GetDeviceNameFromMount(fm, test.mountPath); err != nil || test.expectedRefs != refs || test.expectedDevice != device {
			t.Errorf("%d. GetDeviceNameFromMount(%s) = (%s, %d), %v; expected (%s,%d), nil", i, test.mountPath, device, refs, err, test.expectedDevice, test.expectedRefs)
		}
	}
}

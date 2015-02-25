/*
Copyright 2015 Google Inc. All rights reserved.

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

package gce_pd

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
)

type fakeMounter struct {
	mountPoints []mount.MountPoint
}

func (f *fakeMounter) Mount(source string, target string, fstype string, flags uintptr, data string) error {
	return nil
}

func (f *fakeMounter) Unmount(target string, flags int) error {
	return nil
}

func (f *fakeMounter) List() ([]mount.MountPoint, error) {
	return f.mountPoints, nil
}

func TestGetMountRefs(t *testing.T) {
	fm := &fakeMounter{
		[]mount.MountPoint{
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
		if refs, err := getMountRefs(fm, test.mountPath); err != nil || !setEquivalent(test.expectedRefs, refs) {
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

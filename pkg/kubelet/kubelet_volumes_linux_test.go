//go:build linux
// +build linux

/*
Copyright 2018 The Kubernetes Authors.

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

package kubelet

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"k8s.io/apimachinery/pkg/types"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	"k8s.io/mount-utils"
)

func validateDirExists(dir string) error {
	_, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	return nil
}

func validateDirNotExists(dir string) error {
	_, err := os.ReadDir(dir)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	return fmt.Errorf("dir %q still exists", dir)
}

func TestCleanupOrphanedPodDirs(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	testCases := map[string]struct {
		pods         []*v1.Pod
		prepareFunc  func(kubelet *Kubelet) error
		validateFunc func(kubelet *Kubelet) error
		expectErr    bool
	}{
		"nothing-to-do": {},
		"pods-dir-not-found": {
			prepareFunc: func(kubelet *Kubelet) error {
				return os.Remove(kubelet.getPodsDir())
			},
			expectErr: true,
		},
		"pod-doesnot-exist-novolume": {
			prepareFunc: func(kubelet *Kubelet) error {
				podDir := kubelet.getPodDir("pod1uid")
				return os.MkdirAll(filepath.Join(podDir, "not/a/volume"), 0750)
			},
			validateFunc: func(kubelet *Kubelet) error {
				podDir := kubelet.getPodDir("pod1uid")
				return validateDirNotExists(filepath.Join(podDir, "not"))
			},
		},
		"pod-exists-with-volume": {
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "pod1",
						UID:  "pod1uid",
					},
				},
			},
			prepareFunc: func(kubelet *Kubelet) error {
				podDir := kubelet.getPodDir("pod1uid")
				return os.MkdirAll(filepath.Join(podDir, "volumes/plugin/name"), 0750)
			},
			validateFunc: func(kubelet *Kubelet) error {
				podDir := kubelet.getPodDir("pod1uid")
				return validateDirExists(filepath.Join(podDir, "volumes/plugin/name"))
			},
		},
		"pod-doesnot-exist-with-volume": {
			prepareFunc: func(kubelet *Kubelet) error {
				podDir := kubelet.getPodDir("pod1uid")
				return os.MkdirAll(filepath.Join(podDir, "volumes/plugin/name"), 0750)
			},
			validateFunc: func(kubelet *Kubelet) error {
				podDir := kubelet.getPodDir("pod1uid")
				return validateDirNotExists(podDir)
			},
		},
		"pod-doesnot-exist-with-volume-subdir": {
			prepareFunc: func(kubelet *Kubelet) error {
				podDir := kubelet.getPodDir("pod1uid")
				return os.MkdirAll(filepath.Join(podDir, "volumes/plugin/name/subdir"), 0750)
			},
			validateFunc: func(kubelet *Kubelet) error {
				podDir := kubelet.getPodDir("pod1uid")
				return validateDirNotExists(filepath.Join(podDir, "volumes"))
			},
		},
		"pod-doesnot-exist-with-subpath": {
			prepareFunc: func(kubelet *Kubelet) error {
				podDir := kubelet.getPodDir("pod1uid")
				return os.MkdirAll(filepath.Join(podDir, "volume-subpaths/volume/container/index"), 0750)
			},
			validateFunc: func(kubelet *Kubelet) error {
				podDir := kubelet.getPodDir("pod1uid")
				return validateDirNotExists(podDir)
			},
		},
		"pod-doesnot-exist-with-subpath-top": {
			prepareFunc: func(kubelet *Kubelet) error {
				podDir := kubelet.getPodDir("pod1uid")
				return os.MkdirAll(filepath.Join(podDir, "volume-subpaths"), 0750)
			},
			validateFunc: func(kubelet *Kubelet) error {
				podDir := kubelet.getPodDir("pod1uid")
				return validateDirNotExists(podDir)
			},
		},
		"pod-doesnot-exists-with-populated-volume": {
			prepareFunc: func(kubelet *Kubelet) error {
				podDir := kubelet.getPodDir("pod1uid")
				volumePath := filepath.Join(podDir, "volumes/plugin/name")
				if err := os.MkdirAll(volumePath, 0750); err != nil {
					return err
				}
				return ioutil.WriteFile(filepath.Join(volumePath, "test.txt"), []byte("test1"), 0640)
			},
			validateFunc: func(kubelet *Kubelet) error {
				podDir := kubelet.getPodDir("pod1uid")
				return validateDirExists(filepath.Join(podDir, "volumes/plugin/name"))
			},
		},
		"pod-doesnot-exists-with-populated-subpath": {
			prepareFunc: func(kubelet *Kubelet) error {
				podDir := kubelet.getPodDir("pod1uid")
				subPath := filepath.Join(podDir, "volume-subpaths/volume/container/index")
				if err := os.MkdirAll(subPath, 0750); err != nil {
					return err
				}
				return ioutil.WriteFile(filepath.Join(subPath, "test.txt"), []byte("test1"), 0640)
			},
			validateFunc: func(kubelet *Kubelet) error {
				podDir := kubelet.getPodDir("pod1uid")
				return validateDirExists(filepath.Join(podDir, "volume-subpaths/volume/container/index"))
			},
		},
		// TODO: test volume in volume-manager
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
			defer testKubelet.Cleanup()
			kubelet := testKubelet.kubelet

			if tc.prepareFunc != nil {
				if err := tc.prepareFunc(kubelet); err != nil {
					t.Fatalf("%s failed preparation: %v", name, err)
				}
			}

			err := kubelet.cleanupOrphanedPodDirs(tc.pods, nil)
			if tc.expectErr && err == nil {
				t.Errorf("%s failed: expected error, got success", name)
			}
			if !tc.expectErr && err != nil {
				t.Errorf("%s failed: got error %v", name, err)
			}

			if tc.validateFunc != nil {
				if err := tc.validateFunc(kubelet); err != nil {
					t.Errorf("%s failed validation: %v", name, err)
				}
			}

		})
	}
}

func TestPodVolumesExistWithMount(t *testing.T) {
	poduid := types.UID("poduid")
	testCases := map[string]struct {
		prepareFunc func(kubelet *Kubelet) error
		expected    bool
	}{
		"noncsivolume-dir-not-exist": {
			prepareFunc: func(kubelet *Kubelet) error {
				return nil
			},
			expected: false,
		},
		"noncsivolume-dir-exist-noplugins": {
			prepareFunc: func(kubelet *Kubelet) error {
				podDir := kubelet.getPodDir(poduid)
				return os.MkdirAll(filepath.Join(podDir, "volumes/"), 0750)
			},
			expected: false,
		},
		"noncsivolume-dir-exist-nomount": {
			prepareFunc: func(kubelet *Kubelet) error {
				podDir := kubelet.getPodDir(poduid)
				return os.MkdirAll(filepath.Join(podDir, "volumes/plugin/name"), 0750)
			},
			expected: false,
		},
		"noncsivolume-dir-exist-with-mount": {
			prepareFunc: func(kubelet *Kubelet) error {
				podDir := kubelet.getPodDir(poduid)
				volumePath := filepath.Join(podDir, "volumes/plugin/name")
				if err := os.MkdirAll(volumePath, 0750); err != nil {
					return err
				}
				fm := mount.NewFakeMounter(
					[]mount.MountPoint{
						{Device: "/dev/sdb", Path: volumePath},
					})
				kubelet.mounter = fm
				return nil
			},
			expected: true,
		},
		"noncsivolume-dir-exist-nomount-withcsimountpath": {
			prepareFunc: func(kubelet *Kubelet) error {
				podDir := kubelet.getPodDir(poduid)
				volumePath := filepath.Join(podDir, "volumes/plugin/name/mount")
				if err := os.MkdirAll(volumePath, 0750); err != nil {
					return err
				}
				fm := mount.NewFakeMounter(
					[]mount.MountPoint{
						{Device: "/dev/sdb", Path: volumePath},
					})
				kubelet.mounter = fm
				return nil
			},
			expected: false,
		},
		"csivolume-dir-exist-nomount": {
			prepareFunc: func(kubelet *Kubelet) error {
				podDir := kubelet.getPodDir(poduid)
				volumePath := filepath.Join(podDir, "volumes/kubernetes.io~csi/name")
				return os.MkdirAll(volumePath, 0750)
			},
			expected: false,
		},
		"csivolume-dir-exist-mount-nocsimountpath": {
			prepareFunc: func(kubelet *Kubelet) error {
				podDir := kubelet.getPodDir(poduid)
				volumePath := filepath.Join(podDir, "volumes/kubernetes.io~csi/name/mount")
				return os.MkdirAll(volumePath, 0750)
			},
			expected: false,
		},
		"csivolume-dir-exist-withcsimountpath": {
			prepareFunc: func(kubelet *Kubelet) error {
				podDir := kubelet.getPodDir(poduid)
				volumePath := filepath.Join(podDir, "volumes/kubernetes.io~csi/name/mount")
				if err := os.MkdirAll(volumePath, 0750); err != nil {
					return err
				}
				fm := mount.NewFakeMounter(
					[]mount.MountPoint{
						{Device: "/dev/sdb", Path: volumePath},
					})
				kubelet.mounter = fm
				return nil
			},
			expected: true,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
			defer testKubelet.Cleanup()
			kubelet := testKubelet.kubelet

			if tc.prepareFunc != nil {
				if err := tc.prepareFunc(kubelet); err != nil {
					t.Fatalf("%s failed preparation: %v", name, err)
				}
			}

			exist := kubelet.podVolumesExist(poduid)
			if tc.expected != exist {
				t.Errorf("%s failed: expected %t, got %t", name, tc.expected, exist)
			}
		})
	}
}

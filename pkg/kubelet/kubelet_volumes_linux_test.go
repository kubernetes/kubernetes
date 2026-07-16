//go:build linux

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
	"os"
	"path/filepath"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2/ktesting"
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
				return os.WriteFile(filepath.Join(volumePath, "test.txt"), []byte("test1"), 0640)
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
				return os.WriteFile(filepath.Join(subPath, "test.txt"), []byte("test1"), 0640)
			},
			validateFunc: func(kubelet *Kubelet) error {
				podDir := kubelet.getPodDir("pod1uid")
				return validateDirExists(filepath.Join(podDir, "volume-subpaths/volume/container/index"))
			},
		},

		// Residual CSI vol_data.json after unmount/reboot must not block orphan cleanup.
		// https://github.com/kubernetes/kubernetes/issues/105536
		"pod-doesnot-exist-with-csi-vol-data-json": {
			prepareFunc: func(kubelet *Kubelet) error {
				podDir := kubelet.getPodDir("pod1uid")
				// Escaped CSI plugin name on disk is kubernetes.io~csi
				volumePath := filepath.Join(podDir, "volumes/kubernetes.io~csi/pvc-fake")
				if err := os.MkdirAll(filepath.Join(volumePath, "mount"), 0750); err != nil {
					return err
				}
				return os.WriteFile(filepath.Join(volumePath, "vol_data.json"), []byte(`{"driverName":"test.csi"}`), 0640)
			},
			validateFunc: func(kubelet *Kubelet) error {
				podDir := kubelet.getPodDir("pod1uid")
				return validateDirNotExists(podDir)
			},
		},
		// Arbitrary non-metadata content under a volume path must still be preserved.
		"pod-doesnot-exist-with-csi-userdata-preserved": {
			prepareFunc: func(kubelet *Kubelet) error {
				podDir := kubelet.getPodDir("pod1uid")
				volumePath := filepath.Join(podDir, "volumes/kubernetes.io~csi/pvc-fake")
				if err := os.MkdirAll(volumePath, 0750); err != nil {
					return err
				}
				if err := os.WriteFile(filepath.Join(volumePath, "vol_data.json"), []byte(`{"driverName":"test.csi"}`), 0640); err != nil {
					return err
				}
				return os.WriteFile(filepath.Join(volumePath, "userdata.txt"), []byte("keep-me"), 0640)
			},
			validateFunc: func(kubelet *Kubelet) error {
				podDir := kubelet.getPodDir("pod1uid")
				// volumes dir remains because of userdata; vol_data.json should be gone
				dataPath := filepath.Join(podDir, "volumes/kubernetes.io~csi/pvc-fake/vol_data.json")
				if _, err := os.Stat(dataPath); !os.IsNotExist(err) {
					return fmt.Errorf("expected vol_data.json removed, stat err=%v", err)
				}
				userPath := filepath.Join(podDir, "volumes/kubernetes.io~csi/pvc-fake/userdata.txt")
				if _, err := os.Stat(userPath); err != nil {
					return fmt.Errorf("expected userdata.txt preserved: %v", err)
				}
				return nil
			},
		},

		// CSI plugin path exists as a file: ReadDir fails and is reported.
		"pod-csi-plugin-dir-is-file": {
			prepareFunc: func(kubelet *Kubelet) error {
				podDir := kubelet.getPodDir("pod1uid")
				pluginPath := filepath.Join(podDir, "volumes/kubernetes.io~csi")
				if err := os.MkdirAll(filepath.Dir(pluginPath), 0750); err != nil {
					return err
				}
				return os.WriteFile(pluginPath, []byte("not-a-dir"), 0640)
			},
			// cleanup still runs; errors are rolled up for volume cleanup (not always returned as top-level err)
			validateFunc: func(kubelet *Kubelet) error {
				// plugin path remains as file
				pluginPath := filepath.Join(kubelet.getPodDir("pod1uid"), "volumes/kubernetes.io~csi")
				fi, err := os.Stat(pluginPath)
				if err != nil {
					return err
				}
				if fi.IsDir() {
					return fmt.Errorf("expected plugin path to remain a file")
				}
				return nil
			},
		},
		// Non-directory entry under CSI plugin dir is skipped safely.
		"pod-csi-plugin-has-file-entry-skipped": {
			prepareFunc: func(kubelet *Kubelet) error {
				podDir := kubelet.getPodDir("pod1uid")
				pluginPath := filepath.Join(podDir, "volumes/kubernetes.io~csi")
				if err := os.MkdirAll(pluginPath, 0750); err != nil {
					return err
				}
				// file entry + a cleanable volume dir
				if err := os.WriteFile(filepath.Join(pluginPath, "not-a-volume"), []byte("x"), 0640); err != nil {
					return err
				}
				volumePath := filepath.Join(pluginPath, "pvc-ok")
				if err := os.MkdirAll(volumePath, 0750); err != nil {
					return err
				}
				return os.WriteFile(filepath.Join(volumePath, "vol_data.json"), []byte(`{}`), 0640)
			},
			validateFunc: func(kubelet *Kubelet) error {
				// residual metadata cleaned; file entry may remain and block full volumes dir removal
				dataPath := filepath.Join(kubelet.getPodDir("pod1uid"), "volumes/kubernetes.io~csi/pvc-ok/vol_data.json")
				if _, err := os.Stat(dataPath); !os.IsNotExist(err) {
					return fmt.Errorf("expected vol_data cleaned, err=%v", err)
				}
				return nil
			},
		},
		// Mount path non-empty: CSI cleanup fails for that volume; userdata path keeps erroring remove.
		"pod-csi-nonempty-mount-blocks-cleanup": {
			prepareFunc: func(kubelet *Kubelet) error {
				podDir := kubelet.getPodDir("pod1uid")
				volumePath := filepath.Join(podDir, "volumes/kubernetes.io~csi/pvc-fake")
				mountPath := filepath.Join(volumePath, "mount")
				if err := os.MkdirAll(mountPath, 0750); err != nil {
					return err
				}
				if err := os.WriteFile(filepath.Join(mountPath, "stuck"), []byte("x"), 0640); err != nil {
					return err
				}
				return os.WriteFile(filepath.Join(volumePath, "vol_data.json"), []byte(`{}`), 0640)
			},
			validateFunc: func(kubelet *Kubelet) error {
				// vol_data must remain because mount removal failed first
				dataPath := filepath.Join(kubelet.getPodDir("pod1uid"), "volumes/kubernetes.io~csi/pvc-fake/vol_data.json")
				if _, err := os.Stat(dataPath); err != nil {
					return fmt.Errorf("expected vol_data to remain when mount dir non-empty: %v", err)
				}
				return nil
			},
		},
		// TODO: test volume in volume-manager
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
			defer testKubelet.Cleanup()
			kubelet := testKubelet.kubelet

			if tc.prepareFunc != nil {
				if err := tc.prepareFunc(kubelet); err != nil {
					t.Fatalf("%s failed preparation: %v", name, err)
				}
			}

			err := kubelet.cleanupOrphanedPodDirs(logger, tc.pods, nil)
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
			logger, _ := ktesting.NewTestContext(t)
			testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
			defer testKubelet.Cleanup()
			kubelet := testKubelet.kubelet

			if tc.prepareFunc != nil {
				if err := tc.prepareFunc(kubelet); err != nil {
					t.Fatalf("%s failed preparation: %v", name, err)
				}
			}

			exist := kubelet.podVolumesExist(logger, poduid)
			if tc.expected != exist {
				t.Errorf("%s failed: expected %t, got %t", name, tc.expected, exist)
			}
		})
	}
}

func TestCleanOrphanedCSIVolumeDirs(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}
	logger, _ := ktesting.NewTestContext(t)

	t.Run("plugin-dir-missing", func(t *testing.T) {
		testKubelet := newTestKubelet(t, false)
		defer testKubelet.Cleanup()
		kl := testKubelet.kubelet
		errs := kl.cleanOrphanedCSIVolumeDirs(logger, "pod-missing")
		if len(errs) != 0 {
			t.Fatalf("expected no errors, got %v", errs)
		}
	})

	t.Run("plugin-dir-is-file", func(t *testing.T) {
		testKubelet := newTestKubelet(t, false)
		defer testKubelet.Cleanup()
		kl := testKubelet.kubelet
		uid := types.UID("pod-file")
		pluginPath := filepath.Join(kl.getPodVolumesDir(uid), "kubernetes.io~csi")
		if err := os.MkdirAll(filepath.Dir(pluginPath), 0750); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(pluginPath, []byte("x"), 0640); err != nil {
			t.Fatal(err)
		}
		errs := kl.cleanOrphanedCSIVolumeDirs(logger, uid)
		if len(errs) != 1 {
			t.Fatalf("expected 1 error, got %v", errs)
		}
		if !strings.Contains(errs[0].Error(), "reading CSI volume plugin dir") {
			t.Fatalf("unexpected error: %v", errs[0])
		}
	})

	t.Run("cleanup-artifact-error-propagates", func(t *testing.T) {
		testKubelet := newTestKubelet(t, false)
		defer testKubelet.Cleanup()
		kl := testKubelet.kubelet
		uid := types.UID("pod-block")
		volumePath := filepath.Join(kl.getPodVolumesDir(uid), "kubernetes.io~csi", "pvc-x")
		mountPath := filepath.Join(volumePath, "mount")
		if err := os.MkdirAll(mountPath, 0750); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(mountPath, "stuck"), []byte("x"), 0640); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(volumePath, "vol_data.json"), []byte(`{}`), 0640); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(filepath.Dir(volumePath), "not-dir"), []byte("x"), 0640); err != nil {
			t.Fatal(err)
		}
		errs := kl.cleanOrphanedCSIVolumeDirs(logger, uid)
		if len(errs) != 1 {
			t.Fatalf("expected 1 error from nonempty mount, got %v", errs)
		}
		if !strings.Contains(errs[0].Error(), "failed to clean residual CSI volume artifacts") {
			t.Fatalf("unexpected error: %v", errs[0])
		}
	})
}

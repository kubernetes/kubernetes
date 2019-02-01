/*
Copyright 2019 The Kubernetes Authors.

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

package rbd

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/util/mount"
)

type fakeMounter struct {
	*mount.FakeMounter
	unmountFunc func() error
	remounted   bool
}

func (fm *fakeMounter) Unmount(target string) error {
	return fm.unmountFunc()
}
func (fm *fakeMounter) Mount(source string, target string, fstype string, options []string) error {
	fm.remounted = true
	return nil
}

type fakeDetachDiskManger struct {
	fakeDiskManager
	detachFunc func() error
}

func (fddm *fakeDetachDiskManger) DetachDisk(r *rbdPlugin, deviceMountPath string, device string) error {
	return fddm.detachFunc()
}
func TestDetachRBDDisk(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("rbd_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	mountPath := tmpDir + "/plugins/kubernetes.io/rbd/mounts/k8s_storage_kc1-image-kubernetes-dynamic-pvc-d95297dd-0b1a-11e9-a20e-0e392efe0609"
	os.MkdirAll(mountPath, 0700)
	mountPath, _ = filepath.EvalSymlinks(mountPath)

	mounter := &mount.FakeMounter{
		MountPoints: []mount.MountPoint{
			{
				Device: "/dev/rbd0",
				Path:   mountPath,
				Type:   "ext4",
				Opts:   []string{"rw", "relatime", "stripe=4096", "data=ordered"},
			},
		},
	}

	tests := []struct {
		comment    string
		mounter    *fakeMounter
		manager    *fakeDetachDiskManger
		expectFunc func(mounter *fakeMounter, manager *fakeDetachDiskManger, err error) error
	}{
		{
			comment: "unmount mountpoint failed",
			mounter: &fakeMounter{
				FakeMounter: mounter,
				unmountFunc: func() func() error {
					errs := []string{"unmount failed"}
					index := 0
					return func() error {
						defer func() { index++ }()
						return errors.New(errs[index])
					}
				}(),
			},
			manager: &fakeDetachDiskManger{
				detachFunc: func() func() error {
					return nil
				}(),
			},
			expectFunc: func(_ *fakeMounter, _ *fakeDetachDiskManger, err error) error {
				expect := "unmount failed"
				if err.Error() != expect {
					return fmt.Errorf("expect error: %v, got: %v", expect, err)
				}
				return nil
			},
		},
		{
			comment: "detach device failed after extra 5 retries, elapsed 15 seconds, and device got remounted",
			mounter: &fakeMounter{
				FakeMounter: mounter,
				unmountFunc: func() func() error {
					return func() error {
						return nil
					}
				}(),
			},
			manager: &fakeDetachDiskManger{
				detachFunc: func() func() error {
					errs := []string{
						"rbd: failed to unmap device /dev/rbd0",
						"rbd: failed to unmap device /dev/rbd0",
						"rbd: failed to unmap device /dev/rbd0",
						"rbd: failed to unmap device /dev/rbd0",
						"rbd: failed to unmap device /dev/rbd0",
					}
					index := 0
					return func() error {
						defer func() { index++ }()
						return errors.New(errs[index])
					}
				}(),
			},
			expectFunc: func(mounter *fakeMounter, _ *fakeDetachDiskManger, err error) error {
				expect := "rbd: failed to unmap device /dev/rbd0"
				if err.Error() != expect {
					return fmt.Errorf("expect error: %v, got: %v", expect, err)
				}
				if !mounter.remounted {
					return fmt.Errorf("the device is not remounted")
				}
				return nil
			},
		},
		{
			comment: "detach device successfully after extra 2 retries, elapsed 3 seconds",
			mounter: &fakeMounter{
				FakeMounter: mounter,
				unmountFunc: func() func() error {
					return func() error {
						return nil
					}
				}(),
			},
			manager: &fakeDetachDiskManger{
				detachFunc: func() func() error {
					errs := []string{
						"rbd: failed to unmap device /dev/rbd0",
						"rbd: failed to unmap device /dev/rbd0",
						"",
					}
					index := 0
					return func() error {
						defer func() { index++ }()
						if len(errs[index]) == 0 {
							return nil
						}
						return errors.New(errs[index])
					}
				}(),
			},
			expectFunc: func(mounter *fakeMounter, _ *fakeDetachDiskManger, err error) error {
				if err != nil {
					return fmt.Errorf("expect no error, but got: %v", err)
				}
				return nil
			},
		},
	}
	for _, test := range tests {
		detacher := &rbdDetacher{
			manager: test.manager,
			mounter: &mount.SafeFormatAndMount{Interface: test.mounter},
		}
		err := detacher.UnmountDevice(mountPath)
		if ee := test.expectFunc(test.mounter, test.manager, err); ee != nil {
			t.Fatal(ee)
		}
	}
}

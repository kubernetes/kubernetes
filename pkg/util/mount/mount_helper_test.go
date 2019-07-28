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

package mount

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"syscall"
	"testing"
)

func TestDoCleanupMountPoint(t *testing.T) {

	if runtime.GOOS == "darwin" {
		t.Skipf("not supported on GOOS=%s", runtime.GOOS)
	}

	const testMount = "test-mount"
	const defaultPerm = 0750

	tests := map[string]struct {
		corruptedMnt bool
		// Function that prepares the directory structure for the test under
		// the given base directory.
		// Returns a fake MountPoint, a fake error for the mount point,
		// and error if the prepare function encountered a fatal error.
		prepare   func(base string) (MountPoint, error, error)
		expectErr bool
	}{
		"mount-ok": {
			prepare: func(base string) (MountPoint, error, error) {
				path := filepath.Join(base, testMount)
				if err := os.MkdirAll(path, defaultPerm); err != nil {
					return MountPoint{}, nil, err
				}
				return MountPoint{Device: "/dev/sdb", Path: path}, nil, nil
			},
		},
		"mount-corrupted": {
			prepare: func(base string) (MountPoint, error, error) {
				path := filepath.Join(base, testMount)
				if err := os.MkdirAll(path, defaultPerm); err != nil {
					return MountPoint{}, nil, err
				}
				return MountPoint{Device: "/dev/sdb", Path: path}, os.NewSyscallError("fake", syscall.ESTALE), nil
			},
			corruptedMnt: true,
		},
		"mount-err-not-corrupted": {
			prepare: func(base string) (MountPoint, error, error) {
				path := filepath.Join(base, testMount)
				if err := os.MkdirAll(path, defaultPerm); err != nil {
					return MountPoint{}, nil, err
				}
				return MountPoint{Device: "/dev/sdb", Path: path}, os.NewSyscallError("fake", syscall.ETIMEDOUT), nil
			},
			expectErr: true,
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {

			tmpDir, err := ioutil.TempDir("", "unmount-mount-point-test")
			if err != nil {
				t.Fatalf("failed to create tmpdir: %v", err)
			}
			defer os.RemoveAll(tmpDir)

			if tt.prepare == nil {
				t.Fatalf("prepare function required")
			}

			mountPoint, mountError, err := tt.prepare(tmpDir)
			if err != nil {
				t.Fatalf("failed to prepare test: %v", err)
			}

			fake := &FakeMounter{
				MountPoints:      []MountPoint{mountPoint},
				MountCheckErrors: map[string]error{mountPoint.Path: mountError},
			}

			err = doCleanupMountPoint(mountPoint.Path, fake, true, tt.corruptedMnt)
			if tt.expectErr {
				if err == nil {
					t.Errorf("test %s failed, expected error, got none", name)
				}
				if err := validateDirExists(mountPoint.Path); err != nil {
					t.Errorf("test %s failed, mount path doesn't exist: %v", name, err)
				}
			}
			if !tt.expectErr {
				if err != nil {
					t.Errorf("test %s failed: %v", name, err)
				}
				if err := validateDirNotExists(mountPoint.Path); err != nil {
					t.Errorf("test %s failed, mount path still exists: %v", name, err)
				}
			}
		})
	}
}

func validateDirExists(dir string) error {
	_, err := ioutil.ReadDir(dir)
	if err != nil {
		return err
	}
	return nil
}

func validateDirNotExists(dir string) error {
	_, err := ioutil.ReadDir(dir)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	return fmt.Errorf("dir %q still exists", dir)
}

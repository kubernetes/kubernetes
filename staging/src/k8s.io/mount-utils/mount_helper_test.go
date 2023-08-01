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
	const defaultPerm = 0o750

	tests := map[string]struct {
		corruptedMnt bool
		// Function that prepares the directory structure for the test under
		// the given base directory.
		// Returns a fake MountPoint, a fake error for the mount point,
		// and error if the prepare function encountered a fatal error.
		prepareMnt func(base string) (MountPoint, error, error)
		// Function that prepares the FakeMounter for the test.
		prepareMntr func(mntr *FakeMounter)
		expectErr   bool
	}{
		"mount-ok": {
			prepareMnt: func(base string) (MountPoint, error, error) {
				path := filepath.Join(base, testMount)
				if err := os.MkdirAll(path, defaultPerm); err != nil {
					return MountPoint{}, nil, err
				}
				return MountPoint{Device: "/dev/sdb", Path: path}, nil, nil
			},
			corruptedMnt: false,
			expectErr:    false,
		},
		"path-not-exist": {
			prepareMnt: func(base string) (MountPoint, error, error) {
				path := filepath.Join(base, testMount)
				return MountPoint{Device: "/dev/sdb", Path: path}, nil, nil
			},
			corruptedMnt: false,
			expectErr:    false,
		},
		"mount-corrupted": {
			prepareMnt: func(base string) (MountPoint, error, error) {
				path := filepath.Join(base, testMount)
				if err := os.MkdirAll(path, defaultPerm); err != nil {
					return MountPoint{}, nil, err
				}
				return MountPoint{Device: "/dev/sdb", Path: path}, os.NewSyscallError("fake", syscall.ESTALE), nil
			},
			corruptedMnt: true,
			expectErr:    false,
		},
		"mount-err-not-corrupted": {
			prepareMnt: func(base string) (MountPoint, error, error) {
				path := filepath.Join(base, testMount)
				if err := os.MkdirAll(path, defaultPerm); err != nil {
					return MountPoint{}, nil, err
				}
				return MountPoint{Device: "/dev/sdb", Path: path}, os.NewSyscallError("fake", syscall.ETIMEDOUT), nil
			},
			corruptedMnt: false,
			expectErr:    true,
		},
		"skip-mount-point-check": {
			prepareMnt: func(base string) (MountPoint, error, error) {
				path := filepath.Join(base, testMount)
				if err := os.MkdirAll(path, defaultPerm); err != nil {
					return MountPoint{Device: "/dev/sdb", Path: path}, nil, err
				}
				return MountPoint{Device: "/dev/sdb", Path: path}, nil, nil
			},
			prepareMntr: func(mntr *FakeMounter) {
				mntr.WithSkipMountPointCheck()
			},
			expectErr: false,
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			tmpDir := t.TempDir()

			if tt.prepareMnt == nil {
				t.Fatalf("prepareMnt function required")
			}

			mountPoint, mountError, err := tt.prepareMnt(tmpDir)
			if err != nil {
				t.Fatalf("failed to prepareMnt for test: %v", err)
			}

			fake := NewFakeMounter(
				[]MountPoint{mountPoint},
			)
			fake.MountCheckErrors = map[string]error{mountPoint.Path: mountError}
			if tt.prepareMntr != nil {
				tt.prepareMntr(fake)
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
	_, err := os.ReadDir(dir)
	return err
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

//go:build windows
// +build windows

/*
Copyright 2017 The Kubernetes Authors.

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
		"path-not-exist": {
			prepare: func(base string) (MountPoint, error, error) {
				path := filepath.Join(base, testMount)
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

			fake := NewFakeMounter(
				[]MountPoint{mountPoint},
			)
			fake.MountCheckErrors = map[string]error{mountPoint.Path: mountError}

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

func TestNormalizeWindowsPath(t *testing.T) {
	path := `/var/lib/kubelet/pods/146f8428-83e7-11e7-8dd4-000d3a31dac4/volumes/kubernetes.io~azure-disk`
	normalizedPath := NormalizeWindowsPath(path)
	if normalizedPath != `c:\var\lib\kubelet\pods\146f8428-83e7-11e7-8dd4-000d3a31dac4\volumes\kubernetes.io~azure-disk` {
		t.Errorf("normizeWindowsPath test failed, normalizedPath : %q", normalizedPath)
	}

	path = `/var/lib/kubelet/pods/146f8428-83e7-11e7-8dd4-000d3a31dac4\volumes\kubernetes.io~azure-disk`
	normalizedPath = NormalizeWindowsPath(path)
	if normalizedPath != `c:\var\lib\kubelet\pods\146f8428-83e7-11e7-8dd4-000d3a31dac4\volumes\kubernetes.io~azure-disk` {
		t.Errorf("normizeWindowsPath test failed, normalizedPath : %q", normalizedPath)
	}

	path = `/`
	normalizedPath = NormalizeWindowsPath(path)
	if normalizedPath != `c:\` {
		t.Errorf("normizeWindowsPath test failed, normalizedPath : %q", normalizedPath)
	}
}

func TestValidateDiskNumber(t *testing.T) {
	tests := []struct {
		diskNum     string
		expectError bool
	}{
		{
			diskNum:     "0",
			expectError: false,
		},
		{
			diskNum:     "invalid",
			expectError: true,
		},
		{
			diskNum:     "99",
			expectError: false,
		},
		{
			diskNum:     "100",
			expectError: false,
		},
		{
			diskNum:     "200",
			expectError: false,
		},
	}

	for _, test := range tests {
		err := ValidateDiskNumber(test.diskNum)
		if (err != nil) != test.expectError {
			t.Errorf("TestValidateDiskNumber test failed, disk number: %s, error: %v", test.diskNum, err)
		}
	}
}

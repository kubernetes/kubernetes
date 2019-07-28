// +build linux

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

package subpath

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"golang.org/x/sys/unix"

	"k8s.io/utils/nsenter"

	nsutil "k8s.io/kubernetes/pkg/volume/util/nsenter"
)

func TestCheckDeviceInode(t *testing.T) {
	testDir, err := ioutil.TempDir("", "nsenter-mounter-device-")
	if err != nil {
		t.Fatalf("Cannot create temporary directory: %s", err)
	}
	defer os.RemoveAll(testDir)

	tests := []struct {
		name        string
		srcPath     string
		dstPath     string
		expectError string
	}{
		{
			name:        "the same file",
			srcPath:     filepath.Join(testDir, "1"),
			dstPath:     filepath.Join(testDir, "1"),
			expectError: "",
		},
		{
			name:        "different file on the same FS",
			srcPath:     filepath.Join(testDir, "2.1"),
			dstPath:     filepath.Join(testDir, "2.2"),
			expectError: "different inode",
		},
		{
			name:    "different file on different device",
			srcPath: filepath.Join(testDir, "3"),
			// /proc is always on a different "device" than /tmp (or $TEMP)
			dstPath:     "/proc/self/status",
			expectError: "different device",
		},
	}

	for _, test := range tests {
		if err := ioutil.WriteFile(test.srcPath, []byte{}, 0644); err != nil {
			t.Errorf("Test %q: cannot create srcPath %s: %s", test.name, test.srcPath, err)
			continue
		}

		// Don't create dst if it exists
		if _, err := os.Stat(test.dstPath); os.IsNotExist(err) {
			if err := ioutil.WriteFile(test.dstPath, []byte{}, 0644); err != nil {
				t.Errorf("Test %q: cannot create dstPath %s: %s", test.name, test.dstPath, err)
				continue
			}
		} else if err != nil {
			t.Errorf("Test %q: cannot check existence of dstPath %s: %s", test.name, test.dstPath, err)
			continue
		}

		fd, err := unix.Open(test.srcPath, unix.O_CREAT|unix.O_CLOEXEC, 0644)
		if err != nil {
			t.Errorf("Test %q: cannot open srcPath %s: %s", test.name, test.srcPath, err)
			continue
		}

		err = checkDeviceInode(fd, test.dstPath)

		if test.expectError == "" && err != nil {
			t.Errorf("Test %q: expected no error, got %s", test.name, err)
		}
		if test.expectError != "" {
			if err == nil {
				t.Errorf("Test %q: expected error, got none", test.name)
			} else {
				if !strings.Contains(err.Error(), test.expectError) {
					t.Errorf("Test %q: expected error %q, got %q", test.name, test.expectError, err)
				}
			}
		}
	}
}

func newFakeNsenterMounter(tmpdir string, t *testing.T) (*nsutil.Mounter, string, string, *nsenter.Nsenter, error) {
	rootfsPath := filepath.Join(tmpdir, "rootfs")
	if err := os.Mkdir(rootfsPath, 0755); err != nil {
		return nil, "", "", nil, err
	}
	ne, err := nsenter.NewFakeNsenter(rootfsPath)
	if err != nil {
		return nil, "", "", nil, err
	}

	varlibPath := filepath.Join(tmpdir, "/var/lib/kubelet")
	if err := os.MkdirAll(varlibPath, 0755); err != nil {
		return nil, "", "", nil, err
	}

	return nsutil.NewMounter(varlibPath, ne), rootfsPath, varlibPath, ne, nil
}

func TestNsenterSafeMakeDir(t *testing.T) {
	tests := []struct {
		name        string
		prepare     func(base, rootfs, varlib string) (expectedDir string, err error)
		subdir      string
		expectError bool
		// If true, "base" directory for SafeMakeDir will be /var/lib/kubelet
		baseIsVarLib bool
	}{
		{
			name: "simple directory",
			// evaluated in base
			subdir: "some/subdirectory/structure",
			prepare: func(base, rootfs, varlib string) (expectedDir string, err error) {
				// expected to be created in /roots/
				expectedDir = filepath.Join(rootfs, base, "some/subdirectory/structure")
				return expectedDir, nil
			},
		},
		{
			name: "simple existing directory",
			// evaluated in base
			subdir: "some/subdirectory/structure",
			prepare: func(base, rootfs, varlib string) (expectedDir string, err error) {
				// On the host: directory exists
				hostPath := filepath.Join(base, "some/subdirectory/structure")
				if err := os.MkdirAll(hostPath, 0755); err != nil {
					return "", err
				}
				// In rootfs: directory exists
				kubeletPath := filepath.Join(rootfs, hostPath)
				if err := os.MkdirAll(kubeletPath, 0755); err != nil {
					return "", err
				}
				// expected to be created in /roots/
				expectedDir = kubeletPath
				return expectedDir, nil
			},
		},
		{
			name: "absolute symlink into safe place",
			// evaluated in base
			subdir: "some/subdirectory/structure",
			prepare: func(base, rootfs, varlib string) (expectedDir string, err error) {
				// On the host: /base/other/subdirectory exists, /base/some is link to /base/other
				hostPath := filepath.Join(base, "other/subdirectory")
				if err := os.MkdirAll(hostPath, 0755); err != nil {
					return "", err
				}
				somePath := filepath.Join(base, "some")
				otherPath := filepath.Join(base, "other")
				if err := os.Symlink(otherPath, somePath); err != nil {
					return "", err
				}

				// In rootfs: /base/other/subdirectory exists
				kubeletPath := filepath.Join(rootfs, hostPath)
				if err := os.MkdirAll(kubeletPath, 0755); err != nil {
					return "", err
				}
				// expected 'structure' to be created
				expectedDir = filepath.Join(rootfs, hostPath, "structure")
				return expectedDir, nil
			},
		},
		{
			name: "relative symlink into safe place",
			// evaluated in base
			subdir: "some/subdirectory/structure",
			prepare: func(base, rootfs, varlib string) (expectedDir string, err error) {
				// On the host: /base/other/subdirectory exists, /base/some is link to other
				hostPath := filepath.Join(base, "other/subdirectory")
				if err := os.MkdirAll(hostPath, 0755); err != nil {
					return "", err
				}
				somePath := filepath.Join(base, "some")
				if err := os.Symlink("other", somePath); err != nil {
					return "", err
				}

				// In rootfs: /base/other/subdirectory exists
				kubeletPath := filepath.Join(rootfs, hostPath)
				if err := os.MkdirAll(kubeletPath, 0755); err != nil {
					return "", err
				}
				// expected 'structure' to be created
				expectedDir = filepath.Join(rootfs, hostPath, "structure")
				return expectedDir, nil
			},
		},
		{
			name: "symlink into unsafe place",
			// evaluated in base
			subdir: "some/subdirectory/structure",
			prepare: func(base, rootfs, varlib string) (expectedDir string, err error) {
				// On the host: /base/some is link to /bin/other
				somePath := filepath.Join(base, "some")
				if err := os.Symlink("/bin", somePath); err != nil {
					return "", err
				}
				return "", nil
			},
			expectError: true,
		},
		{
			name: "simple directory in /var/lib/kubelet",
			// evaluated in varlib
			subdir:       "some/subdirectory/structure",
			baseIsVarLib: true,
			prepare: func(base, rootfs, varlib string) (expectedDir string, err error) {
				// expected to be created in /base/var/lib/kubelet, not in /rootfs!
				expectedDir = filepath.Join(varlib, "some/subdirectory/structure")
				return expectedDir, nil
			},
		},
		{
			name: "safe symlink in /var/lib/kubelet",
			// evaluated in varlib
			subdir:       "some/subdirectory/structure",
			baseIsVarLib: true,
			prepare: func(base, rootfs, varlib string) (expectedDir string, err error) {
				// On the host: /varlib/kubelet/other/subdirectory exists, /varlib/some is link to other
				hostPath := filepath.Join(varlib, "other/subdirectory")
				if err := os.MkdirAll(hostPath, 0755); err != nil {
					return "", err
				}
				somePath := filepath.Join(varlib, "some")
				if err := os.Symlink("other", somePath); err != nil {
					return "", err
				}

				// expected to be created in /base/var/lib/kubelet, not in /rootfs!
				expectedDir = filepath.Join(varlib, "other/subdirectory/structure")
				return expectedDir, nil
			},
		},
		{
			name: "unsafe symlink in /var/lib/kubelet",
			// evaluated in varlib
			subdir:       "some/subdirectory/structure",
			baseIsVarLib: true,
			prepare: func(base, rootfs, varlib string) (expectedDir string, err error) {
				// On the host: /varlib/some is link to /bin
				somePath := filepath.Join(varlib, "some")
				if err := os.Symlink("/bin", somePath); err != nil {
					return "", err
				}

				return "", nil
			},
			expectError: true,
		},
	}
	for _, test := range tests {
		tmpdir, err := ioutil.TempDir("", "nsenter-get-safedir-")
		if err != nil {
			t.Error(err)
			continue
		}
		defer os.RemoveAll(tmpdir)

		mounter, rootfs, varlib, ne, err := newFakeNsenterMounter(tmpdir, t)
		if err != nil {
			t.Error(err)
			continue
		}

		fsp := NewNSEnter(mounter, ne, varlib)

		// Prepare base directory for the test
		testBase := filepath.Join(tmpdir, "base")
		if err := os.Mkdir(testBase, 0755); err != nil {
			t.Error(err)
			continue
		}
		// Prepare base directory also in /rootfs
		rootfsBase := filepath.Join(rootfs, testBase)
		if err := os.MkdirAll(rootfsBase, 0755); err != nil {
			t.Error(err)
			continue
		}

		expectedDir := ""
		if test.prepare != nil {
			expectedDir, err = test.prepare(testBase, rootfs, varlib)
			if err != nil {
				t.Error(err)
				continue
			}
		}

		if test.baseIsVarLib {
			// use /var/lib/kubelet as the test base so we can test creating
			// subdirs there directly in /var/lib/kubenet and not in
			// /rootfs/var/lib/kubelet
			testBase = varlib
		}

		err = fsp.SafeMakeDir(test.subdir, testBase, 0755)
		if err != nil && !test.expectError {
			t.Errorf("Test %q: unexpected error: %s", test.name, err)
		}
		if test.expectError {
			if err == nil {
				t.Errorf("Test %q: expected error, got none", test.name)
			} else {
				if !strings.Contains(err.Error(), "is outside of allowed base") {
					t.Errorf("Test %q: expected error to contain \"is outside of allowed base\", got this one instead: %s", test.name, err)
				}
			}
		}

		if expectedDir != "" {
			_, err := os.Stat(expectedDir)
			if err != nil {
				t.Errorf("Test %q: expected %q to exist, got error: %s", test.name, expectedDir, err)
			}
		}
	}
}

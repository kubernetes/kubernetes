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

package nsenter

import (
	"io/ioutil"
	"os"
	"os/user"
	"path/filepath"
	"testing"

	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/utils/nsenter"
)

func TestParseFindMnt(t *testing.T) {
	tests := []struct {
		input       string
		target      string
		expectError bool
	}{
		{
			// standard mount name, e.g. for AWS
			"/var/lib/kubelet/plugins/kubernetes.io/aws-ebs/mounts/aws/us-east-1d/vol-020f82b0759f72389 ext4\n",
			"/var/lib/kubelet/plugins/kubernetes.io/aws-ebs/mounts/aws/us-east-1d/vol-020f82b0759f72389",
			false,
		},
		{
			// mount name with space, e.g. vSphere
			"/var/lib/kubelet/plugins/kubernetes.io/vsphere-volume/mounts/[datastore1] kubevols/kubernetes-dynamic-pvc-4aacaa9b-6ba5-11e7-8f64-0050569f1b82.vmdk ext2\n",
			"/var/lib/kubelet/plugins/kubernetes.io/vsphere-volume/mounts/[datastore1] kubevols/kubernetes-dynamic-pvc-4aacaa9b-6ba5-11e7-8f64-0050569f1b82.vmdk",
			false,
		},
		{
			// hypotetic mount with several spaces
			"/var/lib/kubelet/plugins/kubernetes.io/vsphere-volume/mounts/[ d a t a s t o r e 1 ] kubevols/kubernetes-dynamic-pvc-4aacaa9b-6ba5-11e7-8f64-0050569f1b82.vmdk ext2\n",
			"/var/lib/kubelet/plugins/kubernetes.io/vsphere-volume/mounts/[ d a t a s t o r e 1 ] kubevols/kubernetes-dynamic-pvc-4aacaa9b-6ba5-11e7-8f64-0050569f1b82.vmdk",
			false,
		},
		{
			// invalid output - no filesystem type
			"/var/lib/kubelet/plugins/kubernetes.io/vsphere-volume/mounts/blabla",
			"",
			true,
		},
	}

	for i, test := range tests {
		target, err := parseFindMnt(test.input)
		if test.expectError && err == nil {
			t.Errorf("test %d expected error, got nil", i)
		}
		if !test.expectError && err != nil {
			t.Errorf("test %d returned error: %s", i, err)
		}
		if target != test.target {
			t.Errorf("test %d expected %q, got %q", i, test.target, target)
		}
	}
}

func newFakeNsenterHostUtil(tmpdir string, t *testing.T) (mount.HostUtils, string, string, error) {
	rootfsPath := filepath.Join(tmpdir, "rootfs")

	if err := os.Mkdir(rootfsPath, 0755); err != nil {
		return nil, "", "", err
	}
	ne, err := nsenter.NewFakeNsenter(rootfsPath)
	if err != nil {
		return nil, "", "", err
	}

	varlibPath := filepath.Join(tmpdir, "var/lib/kubelet")
	if err := os.MkdirAll(varlibPath, 0755); err != nil {
		return nil, "", "", err
	}

	hu := NewHostUtil(ne, varlibPath)

	return hu, rootfsPath, varlibPath, nil
}

func TestNsenterExistsFile(t *testing.T) {
	var isRoot bool
	usr, err := user.Current()
	if err == nil {
		isRoot = usr.Username == "root"
	} else {
		switch err.(type) {
		case user.UnknownUserIdError:
			// Root should be always known, this is some random UID
			isRoot = false
		default:
			t.Fatal(err)
		}
	}

	tests := []struct {
		name           string
		prepare        func(base, rootfs string) (string, error)
		expectedOutput bool
		expectError    bool
	}{
		{
			name: "simple existing file",
			prepare: func(base, rootfs string) (string, error) {
				// On the host: /base/file
				path := filepath.Join(base, "file")
				if err := ioutil.WriteFile(path, []byte{}, 0644); err != nil {
					return "", err
				}
				// In kubelet: /rootfs/base/file
				if _, err := writeRootfsFile(rootfs, path, 0644); err != nil {
					return "", err
				}
				return path, nil
			},
			expectedOutput: true,
		},
		{
			name: "simple non-existing file",
			prepare: func(base, rootfs string) (string, error) {
				path := filepath.Join(base, "file")
				return path, nil
			},
			expectedOutput: false,
		},
		{
			name: "simple non-accessible file",
			prepare: func(base, rootfs string) (string, error) {
				// On the host:
				// create /base/dir/file, then make the dir inaccessible
				dir := filepath.Join(base, "dir")
				if err := os.MkdirAll(dir, 0755); err != nil {
					return "", err
				}
				path := filepath.Join(dir, "file")
				if err := ioutil.WriteFile(path, []byte{}, 0); err != nil {
					return "", err
				}
				if err := os.Chmod(dir, 0644); err != nil {
					return "", err
				}

				// In kubelet: do the same with /rootfs/base/dir/file
				rootfsPath, err := writeRootfsFile(rootfs, path, 0777)
				if err != nil {
					return "", err
				}
				rootfsDir := filepath.Dir(rootfsPath)
				if err := os.Chmod(rootfsDir, 0644); err != nil {
					return "", err
				}

				return path, nil
			},
			expectedOutput: isRoot,  // PathExists success when running as root
			expectError:    !isRoot, // PathExists must fail when running as not-root
		},
		{
			name: "relative symlink to existing file",
			prepare: func(base, rootfs string) (string, error) {
				// On the host: /base/link -> file
				file := filepath.Join(base, "file")
				if err := ioutil.WriteFile(file, []byte{}, 0); err != nil {
					return "", err
				}
				path := filepath.Join(base, "link")
				if err := os.Symlink("file", path); err != nil {
					return "", err
				}
				// In kubelet: /rootfs/base/file
				if _, err := writeRootfsFile(rootfs, file, 0644); err != nil {
					return "", err
				}
				return path, nil
			},
			expectedOutput: true,
		},
		{
			name: "absolute symlink to existing file",
			prepare: func(base, rootfs string) (string, error) {
				// On the host: /base/link -> /base/file
				file := filepath.Join(base, "file")
				if err := ioutil.WriteFile(file, []byte{}, 0); err != nil {
					return "", err
				}
				path := filepath.Join(base, "link")
				if err := os.Symlink(file, path); err != nil {
					return "", err
				}
				// In kubelet: /rootfs/base/file
				if _, err := writeRootfsFile(rootfs, file, 0644); err != nil {
					return "", err
				}

				return path, nil
			},
			expectedOutput: true,
		},
		{
			name: "relative symlink to non-existing file",
			prepare: func(base, rootfs string) (string, error) {
				path := filepath.Join(base, "link")
				if err := os.Symlink("file", path); err != nil {
					return "", err
				}
				return path, nil
			},
			expectedOutput: false,
		},
		{
			name: "absolute symlink to non-existing file",
			prepare: func(base, rootfs string) (string, error) {
				file := filepath.Join(base, "file")
				path := filepath.Join(base, "link")
				if err := os.Symlink(file, path); err != nil {
					return "", err
				}
				return path, nil
			},
			expectedOutput: false,
		},
		{
			name: "symlink loop",
			prepare: func(base, rootfs string) (string, error) {
				path := filepath.Join(base, "link")
				if err := os.Symlink(path, path); err != nil {
					return "", err
				}
				return path, nil
			},
			expectedOutput: false,
			// TODO: realpath -m is not able to detect symlink loop. Should we care?
			expectError: false,
		},
	}

	for _, test := range tests {
		tmpdir, err := ioutil.TempDir("", "nsenter-exists-file")
		if err != nil {
			t.Error(err)
			continue
		}
		defer os.RemoveAll(tmpdir)

		testBase := filepath.Join(tmpdir, "base")
		if err := os.Mkdir(testBase, 0755); err != nil {
			t.Error(err)
			continue
		}

		hu, rootfs, _, err := newFakeNsenterHostUtil(tmpdir, t)
		if err != nil {
			t.Error(err)
			continue
		}

		path, err := test.prepare(testBase, rootfs)
		if err != nil {
			t.Error(err)
			continue
		}

		out, err := hu.PathExists(path)
		if err != nil && !test.expectError {
			t.Errorf("Test %q: unexpected error: %s", test.name, err)
		}
		if err == nil && test.expectError {
			t.Errorf("Test %q: expected error, got none", test.name)
		}

		if out != test.expectedOutput {
			t.Errorf("Test %q: expected return value %v, got %v", test.name, test.expectedOutput, out)
		}
	}
}

func TestNsenterGetMode(t *testing.T) {
	tests := []struct {
		name         string
		prepare      func(base, rootfs string) (string, error)
		expectedMode os.FileMode
		expectError  bool
	}{
		{
			name: "simple file",
			prepare: func(base, rootfs string) (string, error) {
				// On the host: /base/file
				path := filepath.Join(base, "file")
				if err := ioutil.WriteFile(path, []byte{}, 0644); err != nil {
					return "", err
				}

				// Prepare a different file as /rootfs/base/file (="the host
				// visible from container") to check that NsEnterMounter calls
				// stat on this file and not on /base/file.
				// Visible from kubelet: /rootfs/base/file
				if _, err := writeRootfsFile(rootfs, path, 0777); err != nil {
					return "", err
				}

				return path, nil
			},
			expectedMode: 0777,
		},
		{
			name: "non-existing file",
			prepare: func(base, rootfs string) (string, error) {
				path := filepath.Join(base, "file")
				return path, nil
			},
			expectedMode: 0,
			expectError:  true,
		},
		{
			name: "absolute symlink to existing file",
			prepare: func(base, rootfs string) (string, error) {
				// On the host: /base/link -> /base/file
				file := filepath.Join(base, "file")
				if err := ioutil.WriteFile(file, []byte{}, 0644); err != nil {
					return "", err
				}
				path := filepath.Join(base, "link")
				if err := os.Symlink(file, path); err != nil {
					return "", err
				}

				// Visible from kubelet:
				// /rootfs/base/file
				if _, err := writeRootfsFile(rootfs, file, 0747); err != nil {
					return "", err
				}

				return path, nil
			},
			expectedMode: 0747,
		},
		{
			name: "relative symlink to existing file",
			prepare: func(base, rootfs string) (string, error) {
				// On the host: /base/link -> file
				file := filepath.Join(base, "file")
				if err := ioutil.WriteFile(file, []byte{}, 0741); err != nil {
					return "", err
				}
				path := filepath.Join(base, "link")
				if err := os.Symlink("file", path); err != nil {
					return "", err
				}

				// Visible from kubelet:
				// /rootfs/base/file
				if _, err := writeRootfsFile(rootfs, file, 0647); err != nil {
					return "", err
				}

				return path, nil
			},
			expectedMode: 0647,
		},
	}

	for _, test := range tests {
		tmpdir, err := ioutil.TempDir("", "nsenter-get-mode-")
		if err != nil {
			t.Error(err)
			continue
		}
		defer os.RemoveAll(tmpdir)

		testBase := filepath.Join(tmpdir, "base")
		if err := os.Mkdir(testBase, 0755); err != nil {
			t.Error(err)
			continue
		}

		hu, rootfs, _, err := newFakeNsenterHostUtil(tmpdir, t)
		if err != nil {
			t.Error(err)
			continue
		}

		path, err := test.prepare(testBase, rootfs)
		if err != nil {
			t.Error(err)
			continue
		}

		mode, err := hu.GetMode(path)
		if err != nil && !test.expectError {
			t.Errorf("Test %q: unexpected error: %s", test.name, err)
		}
		if err == nil && test.expectError {
			t.Errorf("Test %q: expected error, got none", test.name)
		}

		if mode != test.expectedMode {
			t.Errorf("Test %q: expected return value %v, got %v", test.name, test.expectedMode, mode)
		}
	}
}

func writeRootfsFile(rootfs, path string, mode os.FileMode) (string, error) {
	fullPath := filepath.Join(rootfs, path)
	dir := filepath.Dir(fullPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", err
	}
	if err := ioutil.WriteFile(fullPath, []byte{}, mode); err != nil {
		return "", err
	}
	// Use chmod, io.WriteFile is affected by umask
	if err := os.Chmod(fullPath, mode); err != nil {
		return "", err
	}
	return fullPath, nil
}

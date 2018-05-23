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

package mount

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"golang.org/x/sys/unix"
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

		fd, err := unix.Open(test.srcPath, unix.O_CREAT, 0644)
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

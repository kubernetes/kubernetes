// +build linux

/*
Copyright 2014 The Kubernetes Authors.

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

package hostutil

import (
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"k8s.io/utils/exec"
)

func TestIsSharedSuccess(t *testing.T) {
	successMountInfo :=
		`62 0 253:0 / / rw,relatime shared:1 - ext4 /dev/mapper/ssd-root rw,seclabel,data=ordered
76 62 8:1 / /boot rw,relatime shared:29 - ext4 /dev/sda1 rw,seclabel,data=ordered
78 62 0:41 / /tmp rw,nosuid,nodev shared:30 - tmpfs tmpfs rw,seclabel
80 62 0:42 / /var/lib/nfs/rpc_pipefs rw,relatime shared:31 - rpc_pipefs sunrpc rw
82 62 0:43 / /var/lib/foo rw,relatime shared:32 - tmpfs tmpfs rw
83 63 0:44 / /var/lib/bar rw,relatime - tmpfs tmpfs rw
227 62 253:0 /var/lib/docker/devicemapper /var/lib/docker/devicemapper rw,relatime - ext4 /dev/mapper/ssd-root rw,seclabel,data=ordered
224 62 253:0 /var/lib/docker/devicemapper/test/shared /var/lib/docker/devicemapper/test/shared rw,relatime master:1 shared:44 - ext4 /dev/mapper/ssd-root rw,seclabel,data=ordered
`
	tempDir, filename, err := writeFile(successMountInfo)
	if err != nil {
		t.Fatalf("cannot create temporary file: %v", err)
	}
	defer os.RemoveAll(tempDir)

	tests := []struct {
		name           string
		path           string
		expectedResult bool
	}{
		{
			// /var/lib/kubelet is a directory on mount '/' that is shared
			// This is the most common case.
			"shared",
			"/var/lib/kubelet",
			true,
		},
		{
			// 8a2a... is a directory on mount /var/lib/docker/devicemapper
			// that is private.
			"private",
			"/var/lib/docker/devicemapper/mnt/8a2a5c19eefb06d6f851dfcb240f8c113427f5b49b19658b5c60168e88267693/",
			false,
		},
		{
			// 'directory' is a directory on mount
			// /var/lib/docker/devicemapper/test/shared that is shared, but one
			// of its parent is private.
			"nested-shared",
			"/var/lib/docker/devicemapper/test/shared/my/test/directory",
			true,
		},
		{
			// /var/lib/foo is a mount point and it's shared
			"shared-mount",
			"/var/lib/foo",
			true,
		},
		{
			// /var/lib/bar is a mount point and it's private
			"private-mount",
			"/var/lib/bar",
			false,
		},
	}
	for _, test := range tests {
		ret, err := isShared(test.path, filename)
		if err != nil {
			t.Errorf("test %s got unexpected error: %v", test.name, err)
		}
		if ret != test.expectedResult {
			t.Errorf("test %s expected %v, got %v", test.name, test.expectedResult, ret)
		}
	}
}

func TestIsSharedFailure(t *testing.T) {
	errorTests := []struct {
		name    string
		content string
	}{
		{
			// the first line is too short
			name: "too-short-line",
			content: `62 0 253:0 / / rw,relatime
76 62 8:1 / /boot rw,relatime shared:29 - ext4 /dev/sda1 rw,seclabel,data=ordered
78 62 0:41 / /tmp rw,nosuid,nodev shared:30 - tmpfs tmpfs rw,seclabel
80 62 0:42 / /var/lib/nfs/rpc_pipefs rw,relatime shared:31 - rpc_pipefs sunrpc rw
227 62 253:0 /var/lib/docker/devicemapper /var/lib/docker/devicemapper rw,relatime - ext4 /dev/mapper/ssd-root rw,seclabel,data=ordered
224 62 253:0 /var/lib/docker/devicemapper/test/shared /var/lib/docker/devicemapper/test/shared rw,relatime master:1 shared:44 - ext4 /dev/mapper/ssd-root rw,seclabel,data=ordered
`,
		},
		{
			// there is no root mount
			name: "no-root-mount",
			content: `76 62 8:1 / /boot rw,relatime shared:29 - ext4 /dev/sda1 rw,seclabel,data=ordered
78 62 0:41 / /tmp rw,nosuid,nodev shared:30 - tmpfs tmpfs rw,seclabel
80 62 0:42 / /var/lib/nfs/rpc_pipefs rw,relatime shared:31 - rpc_pipefs sunrpc rw
227 62 253:0 /var/lib/docker/devicemapper /var/lib/docker/devicemapper rw,relatime - ext4 /dev/mapper/ssd-root rw,seclabel,data=ordered
224 62 253:0 /var/lib/docker/devicemapper/test/shared /var/lib/docker/devicemapper/test/shared rw,relatime master:1 shared:44 - ext4 /dev/mapper/ssd-root rw,seclabel,data=ordered
`,
		},
	}
	for _, test := range errorTests {
		tempDir, filename, err := writeFile(test.content)
		if err != nil {
			t.Fatalf("cannot create temporary file: %v", err)
		}
		defer os.RemoveAll(tempDir)

		_, err = isShared("/", filename)
		if err == nil {
			t.Errorf("test %q: expected error, got none", test.name)
		}
	}
}

func TestGetSELinuxSupport(t *testing.T) {
	info :=
		`62 0 253:0 / / rw,relatime shared:1 - ext4 /dev/mapper/ssd-root rw,seclabel,data=ordered
78 62 0:41 / /tmp rw,nosuid,nodev shared:30 - tmpfs tmpfs rw,seclabel
83 63 0:44 / /var/lib/bar rw,relatime - tmpfs tmpfs rw
227 62 253:0 /var/lib/docker/devicemapper /var/lib/docker/devicemapper rw,relatime - ext4 /dev/mapper/ssd-root rw,seclabel,data=ordered
150 23 1:58 / /media/nfs_vol rw,relatime shared:89 - nfs4 172.18.4.223:/srv/nfs rw,vers=4.0,rsize=524288,wsize=524288,namlen=255,hard,proto=tcp,port=0,timeo=600,retrans=2,sec=sys,clientaddr=172.18.4.223,local_lock=none,addr=172.18.4.223
`
	tempDir, filename, err := writeFile(info)
	if err != nil {
		t.Fatalf("cannot create temporary file: %v", err)
	}
	defer os.RemoveAll(tempDir)

	tests := []struct {
		name           string
		mountPoint     string
		expectedResult bool
	}{
		{
			"ext4 on /",
			"/",
			true,
		},
		{
			"tmpfs on /var/lib/bar",
			"/var/lib/bar",
			false,
		},
		{
			"nfsv4",
			"/media/nfs_vol",
			false,
		},
	}

	for _, test := range tests {
		out, err := GetSELinux(test.mountPoint, filename)
		if err != nil {
			t.Errorf("Test %s failed with error: %s", test.name, err)
		}
		if test.expectedResult != out {
			t.Errorf("Test %s failed: expected %v, got %v", test.name, test.expectedResult, out)
		}
	}
}

func createSocketFile(socketDir string) (string, error) {
	testSocketFile := filepath.Join(socketDir, "mt.sock")

	// Switch to volume path and create the socket file
	// socket file can not have length of more than 108 character
	// and hence we must use relative path
	oldDir, _ := os.Getwd()

	err := os.Chdir(socketDir)
	if err != nil {
		return "", err
	}
	defer func() {
		os.Chdir(oldDir)
	}()
	_, socketCreateError := net.Listen("unix", "mt.sock")
	return testSocketFile, socketCreateError
}

func TestGetFileType(t *testing.T) {
	hu := NewHostUtil()

	testCase := []struct {
		name         string
		expectedType FileType
		setUp        func() (string, string, error)
	}{
		{
			"Directory Test",
			FileTypeDirectory,
			func() (string, string, error) {
				tempDir, err := ioutil.TempDir("", "test-get-filetype-")
				return tempDir, tempDir, err
			},
		},
		{
			"File Test",
			FileTypeFile,
			func() (string, string, error) {
				tempFile, err := ioutil.TempFile("", "test-get-filetype")
				if err != nil {
					return "", "", err
				}
				tempFile.Close()
				return tempFile.Name(), tempFile.Name(), nil
			},
		},
		{
			"Socket Test",
			FileTypeSocket,
			func() (string, string, error) {
				tempDir, err := ioutil.TempDir("", "test-get-filetype-")
				if err != nil {
					return "", "", err
				}
				tempSocketFile, err := createSocketFile(tempDir)
				return tempSocketFile, tempDir, err
			},
		},
		{
			"Block Device Test",
			FileTypeBlockDev,
			func() (string, string, error) {
				tempDir, err := ioutil.TempDir("", "test-get-filetype-")
				if err != nil {
					return "", "", err
				}

				tempBlockFile := filepath.Join(tempDir, "test_blk_dev")
				outputBytes, err := exec.New().Command("mknod", tempBlockFile, "b", "89", "1").CombinedOutput()
				if err != nil {
					err = fmt.Errorf("%v: %s ", err, outputBytes)
				}
				return tempBlockFile, tempDir, err
			},
		},
		{
			"Character Device Test",
			FileTypeCharDev,
			func() (string, string, error) {
				tempDir, err := ioutil.TempDir("", "test-get-filetype-")
				if err != nil {
					return "", "", err
				}

				tempCharFile := filepath.Join(tempDir, "test_char_dev")
				outputBytes, err := exec.New().Command("mknod", tempCharFile, "c", "89", "1").CombinedOutput()
				if err != nil {
					err = fmt.Errorf("%v: %s ", err, outputBytes)
				}
				return tempCharFile, tempDir, err
			},
		},
	}

	for idx, tc := range testCase {
		path, cleanUpPath, err := tc.setUp()
		defer os.RemoveAll(cleanUpPath) // RemoveAll can deal with a empty path ""
		if err != nil {
			// Locally passed, but upstream CI is not friendly to create such device files
			// Leave "Operation not permitted" out, which can be covered in an e2e test
			if isOperationNotPermittedError(err) {
				continue
			}
			t.Fatalf("[%d-%s] unexpected error : %v", idx, tc.name, err)
		}

		fileType, err := hu.GetFileType(path)
		if err != nil {
			t.Fatalf("[%d-%s] unexpected error : %v", idx, tc.name, err)
		}
		if fileType != tc.expectedType {
			t.Fatalf("[%d-%s] expected %s, but got %s", idx, tc.name, tc.expectedType, fileType)
		}
	}
}

func isOperationNotPermittedError(err error) bool {
	if strings.Contains(err.Error(), "Operation not permitted") {
		return true
	}
	return false
}

func writeFile(content string) (string, string, error) {
	tempDir, err := ioutil.TempDir("", "mounter_shared_test")
	if err != nil {
		return "", "", err
	}
	filename := filepath.Join(tempDir, "mountinfo")
	err = ioutil.WriteFile(filename, []byte(content), 0600)
	if err != nil {
		os.RemoveAll(tempDir)
		return "", "", err
	}
	return tempDir, filename, nil
}

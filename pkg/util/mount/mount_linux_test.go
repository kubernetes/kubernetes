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

package mount

import (
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"syscall"
	"testing"

	"k8s.io/utils/exec"

	"k8s.io/klog"
)

func TestReadProcMountsFrom(t *testing.T) {
	successCase :=
		`/dev/0 /path/to/0 type0 flags 0 0
/dev/1    /path/to/1   type1	flags 1 1
/dev/2 /path/to/2 type2 flags,1,2=3 2 2
`
	// NOTE: readProcMountsFrom has been updated to using fnv.New32a()
	mounts, err := parseProcMounts([]byte(successCase))
	if err != nil {
		t.Errorf("expected success, got %v", err)
	}
	if len(mounts) != 3 {
		t.Fatalf("expected 3 mounts, got %d", len(mounts))
	}
	mp := MountPoint{"/dev/0", "/path/to/0", "type0", []string{"flags"}, 0, 0}
	if !mountPointsEqual(&mounts[0], &mp) {
		t.Errorf("got unexpected MountPoint[0]: %#v", mounts[0])
	}
	mp = MountPoint{"/dev/1", "/path/to/1", "type1", []string{"flags"}, 1, 1}
	if !mountPointsEqual(&mounts[1], &mp) {
		t.Errorf("got unexpected MountPoint[1]: %#v", mounts[1])
	}
	mp = MountPoint{"/dev/2", "/path/to/2", "type2", []string{"flags", "1", "2=3"}, 2, 2}
	if !mountPointsEqual(&mounts[2], &mp) {
		t.Errorf("got unexpected MountPoint[2]: %#v", mounts[2])
	}

	errorCases := []string{
		"/dev/0 /path/to/mount\n",
		"/dev/1 /path/to/mount type flags a 0\n",
		"/dev/2 /path/to/mount type flags 0 b\n",
	}
	for _, ec := range errorCases {
		_, err := parseProcMounts([]byte(ec))
		if err == nil {
			t.Errorf("expected error")
		}
	}
}

func mountPointsEqual(a, b *MountPoint) bool {
	if a.Device != b.Device || a.Path != b.Path || a.Type != b.Type || !reflect.DeepEqual(a.Opts, b.Opts) || a.Pass != b.Pass || a.Freq != b.Freq {
		return false
	}
	return true
}

func TestGetMountRefs(t *testing.T) {
	fm := &FakeMounter{
		MountPoints: []MountPoint{
			{Device: "/dev/sdb", Path: "/var/lib/kubelet/plugins/kubernetes.io/gce-pd/mounts/gce-pd"},
			{Device: "/dev/sdb", Path: "/var/lib/kubelet/pods/some-pod/volumes/kubernetes.io~gce-pd/gce-pd-in-pod"},
			{Device: "/dev/sdc", Path: "/var/lib/kubelet/plugins/kubernetes.io/gce-pd/mounts/gce-pd2"},
			{Device: "/dev/sdc", Path: "/var/lib/kubelet/pods/some-pod/volumes/kubernetes.io~gce-pd/gce-pd2-in-pod1"},
			{Device: "/dev/sdc", Path: "/var/lib/kubelet/pods/some-pod/volumes/kubernetes.io~gce-pd/gce-pd2-in-pod2"},
		},
	}

	tests := []struct {
		mountPath    string
		expectedRefs []string
	}{
		{
			"/var/lib/kubelet/pods/some-pod/volumes/kubernetes.io~gce-pd/gce-pd-in-pod",
			[]string{
				"/var/lib/kubelet/plugins/kubernetes.io/gce-pd/mounts/gce-pd",
			},
		},
		{
			"/var/lib/kubelet/pods/some-pod/volumes/kubernetes.io~gce-pd/gce-pd2-in-pod1",
			[]string{
				"/var/lib/kubelet/pods/some-pod/volumes/kubernetes.io~gce-pd/gce-pd2-in-pod2",
				"/var/lib/kubelet/plugins/kubernetes.io/gce-pd/mounts/gce-pd2",
			},
		},
		{
			"/var/fake/directory/that/doesnt/exist",
			[]string{},
		},
	}

	for i, test := range tests {
		if refs, err := fm.GetMountRefs(test.mountPath); err != nil || !setEquivalent(test.expectedRefs, refs) {
			t.Errorf("%d. getMountRefs(%q) = %v, %v; expected %v, nil", i, test.mountPath, refs, err, test.expectedRefs)
		}
	}
}

func setEquivalent(set1, set2 []string) bool {
	map1 := make(map[string]bool)
	map2 := make(map[string]bool)
	for _, s := range set1 {
		map1[s] = true
	}
	for _, s := range set2 {
		map2[s] = true
	}

	for s := range map1 {
		if !map2[s] {
			return false
		}
	}
	for s := range map2 {
		if !map1[s] {
			return false
		}
	}
	return true
}

func TestGetDeviceNameFromMount(t *testing.T) {
	fm := &FakeMounter{
		MountPoints: []MountPoint{
			{Device: "/dev/disk/by-path/prefix-lun-1",
				Path: "/mnt/111"},
			{Device: "/dev/disk/by-path/prefix-lun-1",
				Path: "/mnt/222"},
		},
	}

	tests := []struct {
		mountPath      string
		expectedDevice string
		expectedRefs   int
	}{
		{
			"/mnt/222",
			"/dev/disk/by-path/prefix-lun-1",
			2,
		},
	}

	for i, test := range tests {
		if device, refs, err := GetDeviceNameFromMount(fm, test.mountPath); err != nil || test.expectedRefs != refs || test.expectedDevice != device {
			t.Errorf("%d. GetDeviceNameFromMount(%s) = (%s, %d), %v; expected (%s,%d), nil", i, test.mountPath, device, refs, err, test.expectedDevice, test.expectedRefs)
		}
	}
}

func TestGetMountRefsByDev(t *testing.T) {
	fm := &FakeMounter{
		MountPoints: []MountPoint{
			{Device: "/dev/sdb", Path: "/var/lib/kubelet/plugins/kubernetes.io/gce-pd/mounts/gce-pd"},
			{Device: "/dev/sdb", Path: "/var/lib/kubelet/pods/some-pod/volumes/kubernetes.io~gce-pd/gce-pd-in-pod"},
			{Device: "/dev/sdc", Path: "/var/lib/kubelet/plugins/kubernetes.io/gce-pd/mounts/gce-pd2"},
			{Device: "/dev/sdc", Path: "/var/lib/kubelet/pods/some-pod/volumes/kubernetes.io~gce-pd/gce-pd2-in-pod1"},
			{Device: "/dev/sdc", Path: "/var/lib/kubelet/pods/some-pod/volumes/kubernetes.io~gce-pd/gce-pd2-in-pod2"},
		},
	}

	tests := []struct {
		mountPath    string
		expectedRefs []string
	}{
		{
			"/var/lib/kubelet/plugins/kubernetes.io/gce-pd/mounts/gce-pd",
			[]string{
				"/var/lib/kubelet/pods/some-pod/volumes/kubernetes.io~gce-pd/gce-pd-in-pod",
			},
		},
		{
			"/var/lib/kubelet/plugins/kubernetes.io/gce-pd/mounts/gce-pd2",
			[]string{
				"/var/lib/kubelet/pods/some-pod/volumes/kubernetes.io~gce-pd/gce-pd2-in-pod1",
				"/var/lib/kubelet/pods/some-pod/volumes/kubernetes.io~gce-pd/gce-pd2-in-pod2",
			},
		},
	}

	for i, test := range tests {

		if refs, err := getMountRefsByDev(fm, test.mountPath); err != nil || !setEquivalent(test.expectedRefs, refs) {
			t.Errorf("%d. getMountRefsByDev(%q) = %v, %v; expected %v, nil", i, test.mountPath, refs, err, test.expectedRefs)
		}
	}
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

func TestPathWithinBase(t *testing.T) {
	tests := []struct {
		name     string
		fullPath string
		basePath string
		expected bool
	}{
		{
			name:     "good subpath",
			fullPath: "/a/b/c",
			basePath: "/a",
			expected: true,
		},
		{
			name:     "good subpath 2",
			fullPath: "/a/b/c",
			basePath: "/a/b",
			expected: true,
		},
		{
			name:     "good subpath end slash",
			fullPath: "/a/b/c/",
			basePath: "/a/b",
			expected: true,
		},
		{
			name:     "good subpath backticks",
			fullPath: "/a/b/../c",
			basePath: "/a",
			expected: true,
		},
		{
			name:     "good subpath equal",
			fullPath: "/a/b/c",
			basePath: "/a/b/c",
			expected: true,
		},
		{
			name:     "good subpath equal 2",
			fullPath: "/a/b/c/",
			basePath: "/a/b/c",
			expected: true,
		},
		{
			name:     "good subpath root",
			fullPath: "/a",
			basePath: "/",
			expected: true,
		},
		{
			name:     "bad subpath parent",
			fullPath: "/a/b/c",
			basePath: "/a/b/c/d",
			expected: false,
		},
		{
			name:     "bad subpath outside",
			fullPath: "/b/c",
			basePath: "/a/b/c",
			expected: false,
		},
		{
			name:     "bad subpath prefix",
			fullPath: "/a/b/cd",
			basePath: "/a/b/c",
			expected: false,
		},
		{
			name:     "bad subpath backticks",
			fullPath: "/a/../b",
			basePath: "/a",
			expected: false,
		},
		{
			name:     "configmap subpath",
			fullPath: "/var/lib/kubelet/pods/uuid/volumes/kubernetes.io~configmap/config/..timestamp/file.txt",
			basePath: "/var/lib/kubelet/pods/uuid/volumes/kubernetes.io~configmap/config",
			expected: true,
		},
	}
	for _, test := range tests {
		if PathWithinBase(test.fullPath, test.basePath) != test.expected {
			t.Errorf("test %q failed: expected %v", test.name, test.expected)
		}

	}
}

func TestSafeMakeDir(t *testing.T) {
	defaultPerm := os.FileMode(0750) + os.ModeDir
	tests := []struct {
		name string
		// Function that prepares directory structure for the test under given
		// base.
		prepare     func(base string) error
		path        string
		checkPath   string
		perm        os.FileMode
		expectError bool
	}{
		{
			"directory-does-not-exist",
			func(base string) error {
				return nil
			},
			"test/directory",
			"test/directory",
			defaultPerm,
			false,
		},
		{
			"directory-with-sgid",
			func(base string) error {
				return nil
			},
			"test/directory",
			"test/directory",
			os.FileMode(0777) + os.ModeDir + os.ModeSetgid,
			false,
		},
		{
			"directory-with-suid",
			func(base string) error {
				return nil
			},
			"test/directory",
			"test/directory",
			os.FileMode(0777) + os.ModeDir + os.ModeSetuid,
			false,
		},
		{
			"directory-with-sticky-bit",
			func(base string) error {
				return nil
			},
			"test/directory",
			"test/directory",
			os.FileMode(0777) + os.ModeDir + os.ModeSticky,
			false,
		},
		{
			"directory-exists",
			func(base string) error {
				return os.MkdirAll(filepath.Join(base, "test/directory"), 0750)
			},
			"test/directory",
			"test/directory",
			defaultPerm,
			false,
		},
		{
			"create-base",
			func(base string) error {
				return nil
			},
			"",
			"",
			defaultPerm,
			false,
		},
		{
			"escape-base-using-dots",
			func(base string) error {
				return nil
			},
			"..",
			"",
			defaultPerm,
			true,
		},
		{
			"escape-base-using-dots-2",
			func(base string) error {
				return nil
			},
			"test/../../..",
			"",
			defaultPerm,
			true,
		},
		{
			"follow-symlinks",
			func(base string) error {
				if err := os.MkdirAll(filepath.Join(base, "destination"), defaultPerm); err != nil {
					return err
				}
				return os.Symlink("destination", filepath.Join(base, "test"))
			},
			"test/directory",
			"destination/directory",
			defaultPerm,
			false,
		},
		{
			"follow-symlink-loop",
			func(base string) error {
				return os.Symlink("test", filepath.Join(base, "test"))
			},
			"test/directory",
			"",
			defaultPerm,
			true,
		},
		{
			"follow-symlink-multiple follow",
			func(base string) error {
				/* test1/dir points to test2 and test2/dir points to test1 */
				if err := os.MkdirAll(filepath.Join(base, "test1"), defaultPerm); err != nil {
					return err
				}
				if err := os.MkdirAll(filepath.Join(base, "test2"), defaultPerm); err != nil {
					return err
				}
				if err := os.Symlink(filepath.Join(base, "test2"), filepath.Join(base, "test1/dir")); err != nil {
					return err
				}
				if err := os.Symlink(filepath.Join(base, "test1"), filepath.Join(base, "test2/dir")); err != nil {
					return err
				}
				return nil
			},
			"test1/dir/dir/dir/dir/dir/dir/dir/foo",
			"test2/foo",
			defaultPerm,
			false,
		},
		{
			"danglink-symlink",
			func(base string) error {
				return os.Symlink("non-existing", filepath.Join(base, "test"))
			},
			"test/directory",
			"",
			defaultPerm,
			true,
		},
		{
			"non-directory",
			func(base string) error {
				return ioutil.WriteFile(filepath.Join(base, "test"), []byte{}, defaultPerm)
			},
			"test/directory",
			"",
			defaultPerm,
			true,
		},
		{
			"non-directory-final",
			func(base string) error {
				return ioutil.WriteFile(filepath.Join(base, "test"), []byte{}, defaultPerm)
			},
			"test",
			"",
			defaultPerm,
			true,
		},
		{
			"escape-with-relative-symlink",
			func(base string) error {
				if err := os.MkdirAll(filepath.Join(base, "dir"), defaultPerm); err != nil {
					return err
				}
				if err := os.MkdirAll(filepath.Join(base, "exists"), defaultPerm); err != nil {
					return err
				}
				return os.Symlink("../exists", filepath.Join(base, "dir/test"))
			},
			"dir/test",
			"",
			defaultPerm,
			false,
		},
		{
			"escape-with-relative-symlink-not-exists",
			func(base string) error {
				if err := os.MkdirAll(filepath.Join(base, "dir"), defaultPerm); err != nil {
					return err
				}
				return os.Symlink("../not-exists", filepath.Join(base, "dir/test"))
			},
			"dir/test",
			"",
			defaultPerm,
			true,
		},
		{
			"escape-with-symlink",
			func(base string) error {
				return os.Symlink("/", filepath.Join(base, "test"))
			},
			"test/directory",
			"",
			defaultPerm,
			true,
		},
	}

	for _, test := range tests {
		klog.V(4).Infof("test %q", test.name)
		base, err := ioutil.TempDir("", "safe-make-dir-"+test.name+"-")
		if err != nil {
			t.Fatalf(err.Error())
		}
		test.prepare(base)
		pathToCreate := filepath.Join(base, test.path)
		err = doSafeMakeDir(pathToCreate, base, test.perm)
		if err != nil && !test.expectError {
			t.Errorf("test %q: %s", test.name, err)
		}
		if err != nil {
			klog.Infof("got error: %s", err)
		}
		if err == nil && test.expectError {
			t.Errorf("test %q: expected error, got none", test.name)
		}

		if test.checkPath != "" {
			st, err := os.Stat(filepath.Join(base, test.checkPath))
			if err != nil {
				t.Errorf("test %q: cannot read path %s", test.name, test.checkPath)
			}
			if st.Mode() != test.perm {
				t.Errorf("test %q: expected permissions %o, got %o", test.name, test.perm, st.Mode())
			}
		}

		os.RemoveAll(base)
	}
}

func TestRemoveEmptyDirs(t *testing.T) {
	defaultPerm := os.FileMode(0750)
	tests := []struct {
		name string
		// Function that prepares directory structure for the test under given
		// base.
		prepare func(base string) error
		// Function that validates directory structure after the test
		validate    func(base string) error
		baseDir     string
		endDir      string
		expectError bool
	}{
		{
			name: "all-empty",
			prepare: func(base string) error {
				return os.MkdirAll(filepath.Join(base, "a/b/c"), defaultPerm)
			},
			validate: func(base string) error {
				return validateDirEmpty(filepath.Join(base, "a"))
			},
			baseDir:     "a",
			endDir:      "a/b/c",
			expectError: false,
		},
		{
			name: "dir-not-empty",
			prepare: func(base string) error {
				if err := os.MkdirAll(filepath.Join(base, "a/b/c"), defaultPerm); err != nil {
					return err
				}
				return os.Mkdir(filepath.Join(base, "a/b/d"), defaultPerm)
			},
			validate: func(base string) error {
				if err := validateDirNotExists(filepath.Join(base, "a/b/c")); err != nil {
					return err
				}
				return validateDirExists(filepath.Join(base, "a/b"))
			},
			baseDir:     "a",
			endDir:      "a/b/c",
			expectError: false,
		},
		{
			name: "path-not-within-base",
			prepare: func(base string) error {
				return os.MkdirAll(filepath.Join(base, "a/b/c"), defaultPerm)
			},
			validate: func(base string) error {
				return validateDirExists(filepath.Join(base, "a"))
			},
			baseDir:     "a",
			endDir:      "b/c",
			expectError: true,
		},
		{
			name: "path-already-deleted",
			prepare: func(base string) error {
				return nil
			},
			validate: func(base string) error {
				return nil
			},
			baseDir:     "a",
			endDir:      "a/b/c",
			expectError: false,
		},
		{
			name: "path-not-dir",
			prepare: func(base string) error {
				if err := os.MkdirAll(filepath.Join(base, "a/b"), defaultPerm); err != nil {
					return err
				}
				return ioutil.WriteFile(filepath.Join(base, "a/b", "c"), []byte{}, defaultPerm)
			},
			validate: func(base string) error {
				if err := validateDirExists(filepath.Join(base, "a/b")); err != nil {
					return err
				}
				return validateFileExists(filepath.Join(base, "a/b/c"))
			},
			baseDir:     "a",
			endDir:      "a/b/c",
			expectError: true,
		},
	}

	for _, test := range tests {
		klog.V(4).Infof("test %q", test.name)
		base, err := ioutil.TempDir("", "remove-empty-dirs-"+test.name+"-")
		if err != nil {
			t.Fatalf(err.Error())
		}
		if err = test.prepare(base); err != nil {
			os.RemoveAll(base)
			t.Fatalf("failed to prepare test %q: %v", test.name, err.Error())
		}

		err = removeEmptyDirs(filepath.Join(base, test.baseDir), filepath.Join(base, test.endDir))
		if err != nil && !test.expectError {
			t.Errorf("test %q failed: %v", test.name, err)
		}
		if err == nil && test.expectError {
			t.Errorf("test %q failed: expected error, got success", test.name)
		}

		if err = test.validate(base); err != nil {
			t.Errorf("test %q failed validation: %v", test.name, err)
		}

		os.RemoveAll(base)
	}
}

func TestCleanSubPaths(t *testing.T) {
	defaultPerm := os.FileMode(0750)
	testVol := "vol1"

	tests := []struct {
		name string
		// Function that prepares directory structure for the test under given
		// base.
		prepare func(base string) ([]MountPoint, error)
		// Function that validates directory structure after the test
		validate    func(base string) error
		expectError bool
	}{
		{
			name: "not-exists",
			prepare: func(base string) ([]MountPoint, error) {
				return nil, nil
			},
			validate: func(base string) error {
				return nil
			},
			expectError: false,
		},
		{
			name: "subpath-not-mount",
			prepare: func(base string) ([]MountPoint, error) {
				return nil, os.MkdirAll(filepath.Join(base, containerSubPathDirectoryName, testVol, "container1", "0"), defaultPerm)
			},
			validate: func(base string) error {
				return validateDirNotExists(filepath.Join(base, containerSubPathDirectoryName))
			},
			expectError: false,
		},
		{
			name: "subpath-file",
			prepare: func(base string) ([]MountPoint, error) {
				path := filepath.Join(base, containerSubPathDirectoryName, testVol, "container1")
				if err := os.MkdirAll(path, defaultPerm); err != nil {
					return nil, err
				}
				return nil, ioutil.WriteFile(filepath.Join(path, "0"), []byte{}, defaultPerm)
			},
			validate: func(base string) error {
				return validateDirNotExists(filepath.Join(base, containerSubPathDirectoryName))
			},
			expectError: false,
		},
		{
			name: "subpath-container-not-dir",
			prepare: func(base string) ([]MountPoint, error) {
				path := filepath.Join(base, containerSubPathDirectoryName, testVol)
				if err := os.MkdirAll(path, defaultPerm); err != nil {
					return nil, err
				}
				return nil, ioutil.WriteFile(filepath.Join(path, "container1"), []byte{}, defaultPerm)
			},
			validate: func(base string) error {
				return validateDirExists(filepath.Join(base, containerSubPathDirectoryName, testVol))
			},
			expectError: true,
		},
		{
			name: "subpath-multiple-container-not-dir",
			prepare: func(base string) ([]MountPoint, error) {
				path := filepath.Join(base, containerSubPathDirectoryName, testVol)
				if err := os.MkdirAll(filepath.Join(path, "container1"), defaultPerm); err != nil {
					return nil, err
				}
				return nil, ioutil.WriteFile(filepath.Join(path, "container2"), []byte{}, defaultPerm)
			},
			validate: func(base string) error {
				path := filepath.Join(base, containerSubPathDirectoryName, testVol)
				if err := validateDirNotExists(filepath.Join(path, "container1")); err != nil {
					return err
				}
				return validateFileExists(filepath.Join(path, "container2"))
			},
			expectError: true,
		},
		{
			name: "subpath-mount",
			prepare: func(base string) ([]MountPoint, error) {
				path := filepath.Join(base, containerSubPathDirectoryName, testVol, "container1", "0")
				if err := os.MkdirAll(path, defaultPerm); err != nil {
					return nil, err
				}
				mounts := []MountPoint{{Device: "/dev/sdb", Path: path}}
				return mounts, nil
			},
			validate: func(base string) error {
				return validateDirNotExists(filepath.Join(base, containerSubPathDirectoryName))
			},
		},
		{
			name: "subpath-mount-multiple",
			prepare: func(base string) ([]MountPoint, error) {
				path := filepath.Join(base, containerSubPathDirectoryName, testVol, "container1", "0")
				path2 := filepath.Join(base, containerSubPathDirectoryName, testVol, "container1", "1")
				path3 := filepath.Join(base, containerSubPathDirectoryName, testVol, "container2", "1")
				if err := os.MkdirAll(path, defaultPerm); err != nil {
					return nil, err
				}
				if err := os.MkdirAll(path2, defaultPerm); err != nil {
					return nil, err
				}
				if err := os.MkdirAll(path3, defaultPerm); err != nil {
					return nil, err
				}
				mounts := []MountPoint{
					{Device: "/dev/sdb", Path: path},
					{Device: "/dev/sdb", Path: path3},
				}
				return mounts, nil
			},
			validate: func(base string) error {
				return validateDirNotExists(filepath.Join(base, containerSubPathDirectoryName))
			},
		},
		{
			name: "subpath-mount-multiple-vols",
			prepare: func(base string) ([]MountPoint, error) {
				path := filepath.Join(base, containerSubPathDirectoryName, testVol, "container1", "0")
				path2 := filepath.Join(base, containerSubPathDirectoryName, "vol2", "container1", "1")
				if err := os.MkdirAll(path, defaultPerm); err != nil {
					return nil, err
				}
				if err := os.MkdirAll(path2, defaultPerm); err != nil {
					return nil, err
				}
				mounts := []MountPoint{
					{Device: "/dev/sdb", Path: path},
				}
				return mounts, nil
			},
			validate: func(base string) error {
				baseSubdir := filepath.Join(base, containerSubPathDirectoryName)
				if err := validateDirNotExists(filepath.Join(baseSubdir, testVol)); err != nil {
					return err
				}
				return validateDirExists(baseSubdir)
			},
		},
	}

	for _, test := range tests {
		klog.V(4).Infof("test %q", test.name)
		base, err := ioutil.TempDir("", "clean-subpaths-"+test.name+"-")
		if err != nil {
			t.Fatalf(err.Error())
		}
		mounts, err := test.prepare(base)
		if err != nil {
			os.RemoveAll(base)
			t.Fatalf("failed to prepare test %q: %v", test.name, err.Error())
		}

		fm := &FakeMounter{MountPoints: mounts}

		err = doCleanSubPaths(fm, base, testVol)
		if err != nil && !test.expectError {
			t.Errorf("test %q failed: %v", test.name, err)
		}
		if err == nil && test.expectError {
			t.Errorf("test %q failed: expected error, got success", test.name)
		}
		if err = test.validate(base); err != nil {
			t.Errorf("test %q failed validation: %v", test.name, err)
		}

		os.RemoveAll(base)
	}
}

var (
	testVol       = "vol1"
	testPod       = "pod0"
	testContainer = "container0"
	testSubpath   = 1
)

func setupFakeMounter(testMounts []string) *FakeMounter {
	mounts := []MountPoint{}
	for _, mountPoint := range testMounts {
		mounts = append(mounts, MountPoint{Device: "/foo", Path: mountPoint})
	}
	return &FakeMounter{MountPoints: mounts}
}

func getTestPaths(base string) (string, string) {
	return filepath.Join(base, testVol),
		filepath.Join(base, testPod, containerSubPathDirectoryName, testVol, testContainer, strconv.Itoa(testSubpath))
}

func TestBindSubPath(t *testing.T) {
	defaultPerm := os.FileMode(0750)

	tests := []struct {
		name string
		// Function that prepares directory structure for the test under given
		// base.
		prepare     func(base string) ([]string, string, string, error)
		expectError bool
	}{
		{
			name: "subpath-dir",
			prepare: func(base string) ([]string, string, string, error) {
				volpath, _ := getTestPaths(base)
				subpath := filepath.Join(volpath, "dir0")
				return nil, volpath, subpath, os.MkdirAll(subpath, defaultPerm)
			},
			expectError: false,
		},
		{
			name: "subpath-dir-symlink",
			prepare: func(base string) ([]string, string, string, error) {
				volpath, _ := getTestPaths(base)
				subpath := filepath.Join(volpath, "dir0")
				if err := os.MkdirAll(subpath, defaultPerm); err != nil {
					return nil, "", "", err
				}
				subpathLink := filepath.Join(volpath, "dirLink")
				return nil, volpath, subpath, os.Symlink(subpath, subpathLink)
			},
			expectError: false,
		},
		{
			name: "subpath-file",
			prepare: func(base string) ([]string, string, string, error) {
				volpath, _ := getTestPaths(base)
				subpath := filepath.Join(volpath, "file0")
				if err := os.MkdirAll(volpath, defaultPerm); err != nil {
					return nil, "", "", err
				}
				return nil, volpath, subpath, ioutil.WriteFile(subpath, []byte{}, defaultPerm)
			},
			expectError: false,
		},
		{
			name: "subpath-not-exists",
			prepare: func(base string) ([]string, string, string, error) {
				volpath, _ := getTestPaths(base)
				subpath := filepath.Join(volpath, "file0")
				return nil, volpath, subpath, nil
			},
			expectError: true,
		},
		{
			name: "subpath-outside",
			prepare: func(base string) ([]string, string, string, error) {
				volpath, _ := getTestPaths(base)
				subpath := filepath.Join(volpath, "dir0")
				if err := os.MkdirAll(volpath, defaultPerm); err != nil {
					return nil, "", "", err
				}
				return nil, volpath, subpath, os.Symlink(base, subpath)
			},
			expectError: true,
		},
		{
			name: "subpath-symlink-child-outside",
			prepare: func(base string) ([]string, string, string, error) {
				volpath, _ := getTestPaths(base)
				subpathDir := filepath.Join(volpath, "dir0")
				subpath := filepath.Join(subpathDir, "child0")
				if err := os.MkdirAll(subpathDir, defaultPerm); err != nil {
					return nil, "", "", err
				}
				return nil, volpath, subpath, os.Symlink(base, subpath)
			},
			expectError: true,
		},
		{
			name: "subpath-child-outside-exists",
			prepare: func(base string) ([]string, string, string, error) {
				volpath, _ := getTestPaths(base)
				subpathDir := filepath.Join(volpath, "dir0")
				child := filepath.Join(base, "child0")
				subpath := filepath.Join(subpathDir, "child0")
				if err := os.MkdirAll(volpath, defaultPerm); err != nil {
					return nil, "", "", err
				}
				// touch file outside
				if err := ioutil.WriteFile(child, []byte{}, defaultPerm); err != nil {
					return nil, "", "", err
				}

				// create symlink for subpath dir
				return nil, volpath, subpath, os.Symlink(base, subpathDir)
			},
			expectError: true,
		},
		{
			name: "subpath-child-outside-not-exists",
			prepare: func(base string) ([]string, string, string, error) {
				volpath, _ := getTestPaths(base)
				subpathDir := filepath.Join(volpath, "dir0")
				subpath := filepath.Join(subpathDir, "child0")
				if err := os.MkdirAll(volpath, defaultPerm); err != nil {
					return nil, "", "", err
				}
				// create symlink for subpath dir
				return nil, volpath, subpath, os.Symlink(base, subpathDir)
			},
			expectError: true,
		},
		{
			name: "subpath-child-outside-exists-middle-dir-symlink",
			prepare: func(base string) ([]string, string, string, error) {
				volpath, _ := getTestPaths(base)
				subpathDir := filepath.Join(volpath, "dir0")
				symlinkDir := filepath.Join(subpathDir, "linkDir0")
				child := filepath.Join(base, "child0")
				subpath := filepath.Join(symlinkDir, "child0")
				if err := os.MkdirAll(subpathDir, defaultPerm); err != nil {
					return nil, "", "", err
				}
				// touch file outside
				if err := ioutil.WriteFile(child, []byte{}, defaultPerm); err != nil {
					return nil, "", "", err
				}

				// create symlink for middle dir
				return nil, volpath, subpath, os.Symlink(base, symlinkDir)
			},
			expectError: true,
		},
		{
			name: "subpath-backstepping",
			prepare: func(base string) ([]string, string, string, error) {
				volpath, _ := getTestPaths(base)
				subpath := filepath.Join(volpath, "dir0")
				symlinkBase := filepath.Join(volpath, "..")
				if err := os.MkdirAll(volpath, defaultPerm); err != nil {
					return nil, "", "", err
				}

				// create symlink for subpath
				return nil, volpath, subpath, os.Symlink(symlinkBase, subpath)
			},
			expectError: true,
		},
		{
			name: "subpath-mountdir-already-exists",
			prepare: func(base string) ([]string, string, string, error) {
				volpath, subpathMount := getTestPaths(base)
				if err := os.MkdirAll(subpathMount, defaultPerm); err != nil {
					return nil, "", "", err
				}

				subpath := filepath.Join(volpath, "dir0")
				return nil, volpath, subpath, os.MkdirAll(subpath, defaultPerm)
			},
			expectError: false,
		},
		{
			name: "subpath-mount-already-exists",
			prepare: func(base string) ([]string, string, string, error) {
				volpath, subpathMount := getTestPaths(base)
				mounts := []string{subpathMount}
				if err := os.MkdirAll(subpathMount, defaultPerm); err != nil {
					return nil, "", "", err
				}

				subpath := filepath.Join(volpath, "dir0")
				return mounts, volpath, subpath, os.MkdirAll(subpath, defaultPerm)
			},
			expectError: false,
		},
		{
			name: "mount-unix-socket",
			prepare: func(base string) ([]string, string, string, error) {
				volpath, subpathMount := getTestPaths(base)
				mounts := []string{subpathMount}
				if err := os.MkdirAll(volpath, defaultPerm); err != nil {
					return nil, "", "", err
				}

				socketFile, socketCreateError := createSocketFile(volpath)

				return mounts, volpath, socketFile, socketCreateError
			},
			expectError: false,
		},
		{
			name: "subpath-mounting-fifo",
			prepare: func(base string) ([]string, string, string, error) {
				volpath, subpathMount := getTestPaths(base)
				mounts := []string{subpathMount}
				if err := os.MkdirAll(volpath, defaultPerm); err != nil {
					return nil, "", "", err
				}

				testFifo := filepath.Join(volpath, "mount_test.fifo")
				err := syscall.Mkfifo(testFifo, 0)
				return mounts, volpath, testFifo, err
			},
			expectError: false,
		},
	}

	for _, test := range tests {
		klog.V(4).Infof("test %q", test.name)
		base, err := ioutil.TempDir("", "bind-subpath-"+test.name+"-")
		if err != nil {
			t.Fatalf(err.Error())
		}

		mounts, volPath, subPath, err := test.prepare(base)
		if err != nil {
			os.RemoveAll(base)
			t.Fatalf("failed to prepare test %q: %v", test.name, err.Error())
		}

		fm := setupFakeMounter(mounts)

		subpath := Subpath{
			VolumeMountIndex: testSubpath,
			Path:             subPath,
			VolumeName:       testVol,
			VolumePath:       volPath,
			PodDir:           filepath.Join(base, "pod0"),
			ContainerName:    testContainer,
		}

		_, subpathMount := getTestPaths(base)
		bindPathTarget, err := doBindSubPath(fm, subpath)
		if test.expectError {
			if err == nil {
				t.Errorf("test %q failed: expected error, got success", test.name)
			}
			if bindPathTarget != "" {
				t.Errorf("test %q failed: expected empty bindPathTarget, got %v", test.name, bindPathTarget)
			}
			if err = validateDirNotExists(subpathMount); err != nil {
				t.Errorf("test %q failed: %v", test.name, err)
			}
		}
		if !test.expectError {
			if err != nil {
				t.Errorf("test %q failed: %v", test.name, err)
			}
			if bindPathTarget != subpathMount {
				t.Errorf("test %q failed: expected bindPathTarget %v, got %v", test.name, subpathMount, bindPathTarget)
			}
			if err = validateFileExists(subpathMount); err != nil {
				t.Errorf("test %q failed: %v", test.name, err)
			}
		}

		os.RemoveAll(base)
	}
}

func TestParseMountInfo(t *testing.T) {
	info :=
		`62 0 253:0 / / rw,relatime shared:1 - ext4 /dev/mapper/ssd-root rw,seclabel,data=ordered
78 62 0:41 / /tmp rw,nosuid,nodev shared:30 - tmpfs tmpfs rw,seclabel
80 62 0:42 / /var/lib/nfs/rpc_pipefs rw,relatime shared:31 - rpc_pipefs sunrpc rw
82 62 0:43 / /var/lib/foo rw,relatime shared:32 - tmpfs tmpfs rw
83 63 0:44 / /var/lib/bar rw,relatime - tmpfs tmpfs rw
227 62 253:0 /var/lib/docker/devicemapper /var/lib/docker/devicemapper rw,relatime - ext4 /dev/mapper/ssd-root rw,seclabel,data=ordered
224 62 253:0 /var/lib/docker/devicemapper/test/shared /var/lib/docker/devicemapper/test/shared rw,relatime master:1 shared:44 - ext4 /dev/mapper/ssd-root rw,seclabel,data=ordered
76 17 8:1 / /mnt/stateful_partition rw,nosuid,nodev,noexec,relatime - ext4 /dev/sda1 rw,commit=30,data=ordered
80 17 8:1 /var /var rw,nosuid,nodev,noexec,relatime shared:30 - ext4 /dev/sda1 rw,commit=30,data=ordered
189 80 8:1 /var/lib/kubelet /var/lib/kubelet rw,relatime shared:30 - ext4 /dev/sda1 rw,commit=30,data=ordered
818 77 8:40 / /var/lib/kubelet/pods/c25464af-e52e-11e7-ab4d-42010a800002/volumes/kubernetes.io~gce-pd/vol1 rw,relatime shared:290 - ext4 /dev/sdc rw,data=ordered
819 78 8:48 / /var/lib/kubelet/pods/c25464af-e52e-11e7-ab4d-42010a800002/volumes/kubernetes.io~gce-pd/vol1 rw,relatime shared:290 - ext4 /dev/sdd rw,data=ordered
900 100 8:48 /dir1 /var/lib/kubelet/pods/c25464af-e52e-11e7-ab4d-42010a800002/volume-subpaths/vol1/subpath1/0 rw,relatime shared:290 - ext4 /dev/sdd rw,data=ordered
901 101 8:1 /dir1 /var/lib/kubelet/pods/c25464af-e52e-11e7-ab4d-42010a800002/volume-subpaths/vol1/subpath1/1 rw,relatime shared:290 - ext4 /dev/sdd rw,data=ordered
902 102 8:1 /var/lib/kubelet/pods/d4076f24-e53a-11e7-ba15-42010a800002/volumes/kubernetes.io~empty-dir/vol1/dir1 /var/lib/kubelet/pods/d4076f24-e53a-11e7-ba15-42010a800002/volume-subpaths/vol1/subpath1/0 rw,relatime shared:30 - ext4 /dev/sda1 rw,commit=30,data=ordered
903 103 8:1 /var/lib/kubelet/pods/d4076f24-e53a-11e7-ba15-42010a800002/volumes/kubernetes.io~empty-dir/vol2/dir1 /var/lib/kubelet/pods/d4076f24-e53a-11e7-ba15-42010a800002/volume-subpaths/vol1/subpath1/1 rw,relatime shared:30 - ext4 /dev/sda1 rw,commit=30,data=ordered
178 25 253:0 /etc/bar /var/lib/kubelet/pods/12345/volume-subpaths/vol1/subpath1/0 rw,relatime shared:1 - ext4 /dev/sdb2 rw,errors=remount-ro,data=ordered
698 186 0:41 /tmp1/dir1 /var/lib/kubelet/pods/41135147-e697-11e7-9342-42010a800002/volume-subpaths/vol1/subpath1/0 rw shared:26 - tmpfs tmpfs rw
918 77 8:50 / /var/lib/kubelet/pods/2345/volumes/kubernetes.io~gce-pd/vol1 rw,relatime shared:290 - ext4 /dev/sdc rw,data=ordered
919 78 8:58 / /var/lib/kubelet/pods/2345/volumes/kubernetes.io~gce-pd/vol1 rw,relatime shared:290 - ext4 /dev/sdd rw,data=ordered
920 100 8:50 /dir1 /var/lib/kubelet/pods/2345/volume-subpaths/vol1/subpath1/0 rw,relatime shared:290 - ext4 /dev/sdc rw,data=ordered
150 23 1:58 / /media/nfs_vol rw,relatime shared:89 - nfs4 172.18.4.223:/srv/nfs rw,vers=4.0,rsize=524288,wsize=524288,namlen=255,hard,proto=tcp,port=0,timeo=600,retrans=2,sec=sys,clientaddr=172.18.4.223,local_lock=none,addr=172.18.4.223
151 24 1:58 / /media/nfs_bindmount rw,relatime shared:89 - nfs4 172.18.4.223:/srv/nfs/foo rw,vers=4.0,rsize=524288,wsize=524288,namlen=255,hard,proto=tcp,port=0,timeo=600,retrans=2,sec=sys,clientaddr=172.18.4.223,local_lock=none,addr=172.18.4.223
134 23 0:58 / /var/lib/kubelet/pods/43219158-e5e1-11e7-a392-0e858b8eaf40/volumes/kubernetes.io~nfs/nfs1 rw,relatime shared:89 - nfs4 172.18.4.223:/srv/nfs rw,vers=4.0,rsize=524288,wsize=524288,namlen=255,hard,proto=tcp,port=0,timeo=600,retrans=2,sec=sys,clientaddr=172.18.4.223,local_lock=none,addr=172.18.4.223
187 23 0:58 / /var/lib/kubelet/pods/1fc5ea21-eff4-11e7-ac80-0e858b8eaf40/volumes/kubernetes.io~nfs/nfs2 rw,relatime shared:96 - nfs4 172.18.4.223:/srv/nfs2 rw,vers=4.0,rsize=524288,wsize=524288,namlen=255,hard,proto=tcp,port=0,timeo=600,retrans=2,sec=sys,clientaddr=172.18.4.223,local_lock=none,addr=172.18.4.223
188 24 0:58 / /var/lib/kubelet/pods/43219158-e5e1-11e7-a392-0e858b8eaf40/volume-subpaths/nfs1/subpath1/0 rw,relatime shared:89 - nfs4 172.18.4.223:/srv/nfs/foo rw,vers=4.0,rsize=524288,wsize=524288,namlen=255,hard,proto=tcp,port=0,timeo=600,retrans=2,sec=sys,clientaddr=172.18.4.223,local_lock=none,addr=172.18.4.223
347 60 0:71 / /var/lib/kubelet/pods/13195d46-f9fa-11e7-bbf1-5254007a695a/volumes/kubernetes.io~nfs/vol2 rw,relatime shared:170 - nfs 172.17.0.3:/exports/2 rw,vers=3,rsize=1048576,wsize=1048576,namlen=255,hard,proto=tcp,timeo=600,retrans=2,sec=sys,mountaddr=172.17.0.3,mountvers=3,mountport=20048,mountproto=udp,local_lock=none,addr=172.17.0.3
222 24 253:0 /tmp/src /mnt/dst rw,relatime shared:1 - ext4 /dev/mapper/vagrant--vg-root rw,errors=remount-ro,data=ordered
28 18 0:24 / /sys/fs/cgroup ro,nosuid,nodev,noexec shared:9 - tmpfs tmpfs ro,mode=755
29 28 0:25 / /sys/fs/cgroup/systemd rw,nosuid,nodev,noexec,relatime shared:10 - cgroup cgroup rw,xattr,release_agent=/lib/systemd/systemd-cgroups-agent,name=systemd
31 28 0:27 / /sys/fs/cgroup/cpuset rw,nosuid,nodev,noexec,relatime shared:13 - cgroup cgroup rw,cpuset
32 28 0:28 / /sys/fs/cgroup/cpu,cpuacct rw,nosuid,nodev,noexec,relatime shared:14 - cgroup cgroup rw,cpu,cpuacct
33 28 0:29 / /sys/fs/cgroup/freezer rw,nosuid,nodev,noexec,relatime shared:15 - cgroup cgroup rw,freezer
34 28 0:30 / /sys/fs/cgroup/net_cls,net_prio rw,nosuid,nodev,noexec,relatime shared:16 - cgroup cgroup rw,net_cls,net_prio
35 28 0:31 / /sys/fs/cgroup/pids rw,nosuid,nodev,noexec,relatime shared:17 - cgroup cgroup rw,pids
36 28 0:32 / /sys/fs/cgroup/devices rw,nosuid,nodev,noexec,relatime shared:18 - cgroup cgroup rw,devices
37 28 0:33 / /sys/fs/cgroup/hugetlb rw,nosuid,nodev,noexec,relatime shared:19 - cgroup cgroup rw,hugetlb
38 28 0:34 / /sys/fs/cgroup/blkio rw,nosuid,nodev,noexec,relatime shared:20 - cgroup cgroup rw,blkio
39 28 0:35 / /sys/fs/cgroup/memory rw,nosuid,nodev,noexec,relatime shared:21 - cgroup cgroup rw,memory
40 28 0:36 / /sys/fs/cgroup/perf_event rw,nosuid,nodev,noexec,relatime shared:22 - cgroup cgroup rw,perf_event
`
	tempDir, filename, err := writeFile(info)
	if err != nil {
		t.Fatalf("cannot create temporary file: %v", err)
	}
	defer os.RemoveAll(tempDir)

	tests := []struct {
		name         string
		id           int
		expectedInfo mountInfo
	}{
		{
			"simple bind mount",
			189,
			mountInfo{
				id:             189,
				parentID:       80,
				majorMinor:     "8:1",
				root:           "/var/lib/kubelet",
				source:         "/dev/sda1",
				mountPoint:     "/var/lib/kubelet",
				optionalFields: []string{"shared:30"},
				fsType:         "ext4",
				mountOptions:   []string{"rw", "relatime"},
				superOptions:   []string{"rw", "commit=30", "data=ordered"},
			},
		},
		{
			"bind mount a directory",
			222,
			mountInfo{
				id:             222,
				parentID:       24,
				majorMinor:     "253:0",
				root:           "/tmp/src",
				source:         "/dev/mapper/vagrant--vg-root",
				mountPoint:     "/mnt/dst",
				optionalFields: []string{"shared:1"},
				fsType:         "ext4",
				mountOptions:   []string{"rw", "relatime"},
				superOptions:   []string{"rw", "errors=remount-ro", "data=ordered"},
			},
		},
		{
			"more than one optional fields",
			224,
			mountInfo{
				id:             224,
				parentID:       62,
				majorMinor:     "253:0",
				root:           "/var/lib/docker/devicemapper/test/shared",
				source:         "/dev/mapper/ssd-root",
				mountPoint:     "/var/lib/docker/devicemapper/test/shared",
				optionalFields: []string{"master:1", "shared:44"},
				fsType:         "ext4",
				mountOptions:   []string{"rw", "relatime"},
				superOptions:   []string{"rw", "seclabel", "data=ordered"},
			},
		},
		{
			"cgroup-mountpoint",
			28,
			mountInfo{
				id:             28,
				parentID:       18,
				majorMinor:     "0:24",
				root:           "/",
				source:         "tmpfs",
				mountPoint:     "/sys/fs/cgroup",
				optionalFields: []string{"shared:9"},
				fsType:         "tmpfs",
				mountOptions:   []string{"ro", "nosuid", "nodev", "noexec"},
				superOptions:   []string{"ro", "mode=755"},
			},
		},
		{
			"cgroup-subsystem-systemd-mountpoint",
			29,
			mountInfo{
				id:             29,
				parentID:       28,
				majorMinor:     "0:25",
				root:           "/",
				source:         "cgroup",
				mountPoint:     "/sys/fs/cgroup/systemd",
				optionalFields: []string{"shared:10"},
				fsType:         "cgroup",
				mountOptions:   []string{"rw", "nosuid", "nodev", "noexec", "relatime"},
				superOptions:   []string{"rw", "xattr", "release_agent=/lib/systemd/systemd-cgroups-agent", "name=systemd"},
			},
		},
		{
			"cgroup-subsystem-cpuset-mountpoint",
			31,
			mountInfo{
				id:             31,
				parentID:       28,
				majorMinor:     "0:27",
				root:           "/",
				source:         "cgroup",
				mountPoint:     "/sys/fs/cgroup/cpuset",
				optionalFields: []string{"shared:13"},
				fsType:         "cgroup",
				mountOptions:   []string{"rw", "nosuid", "nodev", "noexec", "relatime"},
				superOptions:   []string{"rw", "cpuset"},
			},
		},
	}

	infos, err := parseMountInfo(filename)
	if err != nil {
		t.Fatalf("Cannot parse %s: %s", filename, err)
	}

	for _, test := range tests {
		found := false
		for _, info := range infos {
			if info.id == test.id {
				found = true
				if !reflect.DeepEqual(info, test.expectedInfo) {
					t.Errorf("Test case %q:\n expected: %+v\n got:      %+v", test.name, test.expectedInfo, info)
				}
				break
			}
		}
		if !found {
			t.Errorf("Test case %q: mountPoint %d not found", test.name, test.id)
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
		out, err := getSELinuxSupport(test.mountPoint, filename)
		if err != nil {
			t.Errorf("Test %s failed with error: %s", test.name, err)
		}
		if test.expectedResult != out {
			t.Errorf("Test %s failed: expected %v, got %v", test.name, test.expectedResult, out)
		}
	}
}

func TestSafeOpen(t *testing.T) {
	defaultPerm := os.FileMode(0750)

	tests := []struct {
		name string
		// Function that prepares directory structure for the test under given
		// base.
		prepare     func(base string) error
		path        string
		expectError bool
	}{
		{
			"directory-does-not-exist",
			func(base string) error {
				return nil
			},
			"test/directory",
			true,
		},
		{
			"directory-exists",
			func(base string) error {
				return os.MkdirAll(filepath.Join(base, "test/directory"), 0750)
			},
			"test/directory",
			false,
		},
		{
			"escape-base-using-dots",
			func(base string) error {
				return nil
			},
			"..",
			true,
		},
		{
			"escape-base-using-dots-2",
			func(base string) error {
				return os.MkdirAll(filepath.Join(base, "test"), 0750)
			},
			"test/../../..",
			true,
		},
		{
			"symlink",
			func(base string) error {
				if err := os.MkdirAll(filepath.Join(base, "destination"), defaultPerm); err != nil {
					return err
				}
				return os.Symlink("destination", filepath.Join(base, "test"))
			},
			"test",
			true,
		},
		{
			"symlink-nested",
			func(base string) error {
				if err := os.MkdirAll(filepath.Join(base, "dir1/dir2"), defaultPerm); err != nil {
					return err
				}
				return os.Symlink("dir1", filepath.Join(base, "dir1/dir2/test"))
			},
			"test",
			true,
		},
		{
			"symlink-loop",
			func(base string) error {
				return os.Symlink("test", filepath.Join(base, "test"))
			},
			"test",
			true,
		},
		{
			"symlink-not-exists",
			func(base string) error {
				return os.Symlink("non-existing", filepath.Join(base, "test"))
			},
			"test",
			true,
		},
		{
			"non-directory",
			func(base string) error {
				return ioutil.WriteFile(filepath.Join(base, "test"), []byte{}, defaultPerm)
			},
			"test/directory",
			true,
		},
		{
			"non-directory-final",
			func(base string) error {
				return ioutil.WriteFile(filepath.Join(base, "test"), []byte{}, defaultPerm)
			},
			"test",
			false,
		},
		{
			"escape-with-relative-symlink",
			func(base string) error {
				if err := os.MkdirAll(filepath.Join(base, "dir"), defaultPerm); err != nil {
					return err
				}
				if err := os.MkdirAll(filepath.Join(base, "exists"), defaultPerm); err != nil {
					return err
				}
				return os.Symlink("../exists", filepath.Join(base, "dir/test"))
			},
			"dir/test",
			true,
		},
		{
			"escape-with-relative-symlink-not-exists",
			func(base string) error {
				if err := os.MkdirAll(filepath.Join(base, "dir"), defaultPerm); err != nil {
					return err
				}
				return os.Symlink("../not-exists", filepath.Join(base, "dir/test"))
			},
			"dir/test",
			true,
		},
		{
			"escape-with-symlink",
			func(base string) error {
				return os.Symlink("/", filepath.Join(base, "test"))
			},
			"test",
			true,
		},
		{
			"mount-unix-socket",
			func(base string) error {
				socketFile, socketError := createSocketFile(base)

				if socketError != nil {
					return fmt.Errorf("Error preparing socket file %s with %v", socketFile, socketError)
				}
				return nil
			},
			"mt.sock",
			false,
		},
		{
			"mounting-unix-socket-in-middle",
			func(base string) error {
				testSocketFile, socketError := createSocketFile(base)

				if socketError != nil {
					return fmt.Errorf("Error preparing socket file %s with %v", testSocketFile, socketError)
				}
				return nil
			},
			"mt.sock/bar",
			true,
		},
	}

	for _, test := range tests {
		klog.V(4).Infof("test %q", test.name)
		base, err := ioutil.TempDir("", "safe-open-"+test.name+"-")
		if err != nil {
			t.Fatalf(err.Error())
		}

		test.prepare(base)
		pathToCreate := filepath.Join(base, test.path)
		fd, err := doSafeOpen(pathToCreate, base)
		if err != nil && !test.expectError {
			t.Errorf("test %q: %s", test.name, err)
		}
		if err != nil {
			klog.Infof("got error: %s", err)
		}
		if err == nil && test.expectError {
			t.Errorf("test %q: expected error, got none", test.name)
		}

		syscall.Close(fd)
		os.RemoveAll(base)
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

func TestFindExistingPrefix(t *testing.T) {
	defaultPerm := os.FileMode(0750)
	tests := []struct {
		name string
		// Function that prepares directory structure for the test under given
		// base.
		prepare      func(base string) error
		path         string
		expectedPath string
		expectedDirs []string
		expectError  bool
	}{
		{
			"directory-does-not-exist",
			func(base string) error {
				return nil
			},
			"directory",
			"",
			[]string{"directory"},
			false,
		},
		{
			"directory-exists",
			func(base string) error {
				return os.MkdirAll(filepath.Join(base, "test/directory"), 0750)
			},
			"test/directory",
			"test/directory",
			[]string{},
			false,
		},
		{
			"follow-symlinks",
			func(base string) error {
				if err := os.MkdirAll(filepath.Join(base, "destination/directory"), defaultPerm); err != nil {
					return err
				}
				return os.Symlink("destination", filepath.Join(base, "test"))
			},
			"test/directory",
			"test/directory",
			[]string{},
			false,
		},
		{
			"follow-symlink-loop",
			func(base string) error {
				return os.Symlink("test", filepath.Join(base, "test"))
			},
			"test/directory",
			"",
			nil,
			true,
		},
		{
			"follow-symlink-multiple follow",
			func(base string) error {
				/* test1/dir points to test2 and test2/dir points to test1 */
				if err := os.MkdirAll(filepath.Join(base, "test1"), defaultPerm); err != nil {
					return err
				}
				if err := os.MkdirAll(filepath.Join(base, "test2"), defaultPerm); err != nil {
					return err
				}
				if err := os.Symlink(filepath.Join(base, "test2"), filepath.Join(base, "test1/dir")); err != nil {
					return err
				}
				if err := os.Symlink(filepath.Join(base, "test1"), filepath.Join(base, "test2/dir")); err != nil {
					return err
				}
				return nil
			},
			"test1/dir/dir/foo/bar",
			"test1/dir/dir",
			[]string{"foo", "bar"},
			false,
		},
		{
			"danglink-symlink",
			func(base string) error {
				return os.Symlink("non-existing", filepath.Join(base, "test"))
			},
			// OS returns IsNotExist error both for dangling symlink and for
			// non-existing directory.
			"test/directory",
			"",
			[]string{"test", "directory"},
			false,
		},
		{
			"with-fifo-in-middle",
			func(base string) error {
				testFifo := filepath.Join(base, "mount_test.fifo")
				return syscall.Mkfifo(testFifo, 0)
			},
			"mount_test.fifo/directory",
			"",
			nil,
			true,
		},
	}

	for _, test := range tests {
		klog.V(4).Infof("test %q", test.name)
		base, err := ioutil.TempDir("", "find-prefix-"+test.name+"-")
		if err != nil {
			t.Fatalf(err.Error())
		}
		test.prepare(base)
		path := filepath.Join(base, test.path)
		existingPath, dirs, err := findExistingPrefix(base, path)
		if err != nil && !test.expectError {
			t.Errorf("test %q: %s", test.name, err)
		}
		if err != nil {
			klog.Infof("got error: %s", err)
		}
		if err == nil && test.expectError {
			t.Errorf("test %q: expected error, got none", test.name)
		}

		fullExpectedPath := filepath.Join(base, test.expectedPath)
		if existingPath != fullExpectedPath {
			t.Errorf("test %q: expected path %q, got %q", test.name, fullExpectedPath, existingPath)
		}
		if !reflect.DeepEqual(dirs, test.expectedDirs) {
			t.Errorf("test %q: expected dirs %v, got %v", test.name, test.expectedDirs, dirs)
		}
		os.RemoveAll(base)
	}
}

func TestGetFileType(t *testing.T) {
	mounter := Mounter{"fake/path", false}

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
		if err != nil {
			// Locally passed, but upstream CI is not friendly to create such device files
			// Leave "Operation not permitted" out, which can be covered in an e2e test
			if isOperationNotPermittedError(err) {
				continue
			}
			t.Fatalf("[%d-%s] unexpected error : %v", idx, tc.name, err)
		}
		if len(cleanUpPath) > 0 {
			defer os.RemoveAll(cleanUpPath)
		}

		fileType, err := mounter.GetFileType(path)
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

func TestSearchMountPoints(t *testing.T) {
	base := `
19 25 0:18 / /sys rw,nosuid,nodev,noexec,relatime shared:7 - sysfs sysfs rw
20 25 0:4 / /proc rw,nosuid,nodev,noexec,relatime shared:12 - proc proc rw
21 25 0:6 / /dev rw,nosuid,relatime shared:2 - devtmpfs udev rw,size=4058156k,nr_inodes=1014539,mode=755
22 21 0:14 / /dev/pts rw,nosuid,noexec,relatime shared:3 - devpts devpts rw,gid=5,mode=620,ptmxmode=000
23 25 0:19 / /run rw,nosuid,noexec,relatime shared:5 - tmpfs tmpfs rw,size=815692k,mode=755
25 0 252:0 / / rw,relatime shared:1 - ext4 /dev/mapper/ubuntu--vg-root rw,errors=remount-ro,data=ordered
26 19 0:12 / /sys/kernel/security rw,nosuid,nodev,noexec,relatime shared:8 - securityfs securityfs rw
27 21 0:21 / /dev/shm rw,nosuid,nodev shared:4 - tmpfs tmpfs rw
28 23 0:22 / /run/lock rw,nosuid,nodev,noexec,relatime shared:6 - tmpfs tmpfs rw,size=5120k
29 19 0:23 / /sys/fs/cgroup ro,nosuid,nodev,noexec shared:9 - tmpfs tmpfs ro,mode=755
30 29 0:24 / /sys/fs/cgroup/systemd rw,nosuid,nodev,noexec,relatime shared:10 - cgroup cgroup rw,xattr,release_agent=/lib/systemd/systemd-cgroups-agent,name=systemd
31 19 0:25 / /sys/fs/pstore rw,nosuid,nodev,noexec,relatime shared:11 - pstore pstore rw
32 29 0:26 / /sys/fs/cgroup/devices rw,nosuid,nodev,noexec,relatime shared:13 - cgroup cgroup rw,devices
33 29 0:27 / /sys/fs/cgroup/freezer rw,nosuid,nodev,noexec,relatime shared:14 - cgroup cgroup rw,freezer
34 29 0:28 / /sys/fs/cgroup/pids rw,nosuid,nodev,noexec,relatime shared:15 - cgroup cgroup rw,pids
35 29 0:29 / /sys/fs/cgroup/blkio rw,nosuid,nodev,noexec,relatime shared:16 - cgroup cgroup rw,blkio
36 29 0:30 / /sys/fs/cgroup/memory rw,nosuid,nodev,noexec,relatime shared:17 - cgroup cgroup rw,memory
37 29 0:31 / /sys/fs/cgroup/perf_event rw,nosuid,nodev,noexec,relatime shared:18 - cgroup cgroup rw,perf_event
38 29 0:32 / /sys/fs/cgroup/hugetlb rw,nosuid,nodev,noexec,relatime shared:19 - cgroup cgroup rw,hugetlb
39 29 0:33 / /sys/fs/cgroup/cpu,cpuacct rw,nosuid,nodev,noexec,relatime shared:20 - cgroup cgroup rw,cpu,cpuacct
40 29 0:34 / /sys/fs/cgroup/cpuset rw,nosuid,nodev,noexec,relatime shared:21 - cgroup cgroup rw,cpuset
41 29 0:35 / /sys/fs/cgroup/net_cls,net_prio rw,nosuid,nodev,noexec,relatime shared:22 - cgroup cgroup rw,net_cls,net_prio
58 25 7:1 / /mnt/disks/blkvol1 rw,relatime shared:38 - ext4 /dev/loop1 rw,data=ordere
`

	testcases := []struct {
		name         string
		source       string
		mountInfos   string
		expectedRefs []string
		expectedErr  error
	}{
		{
			"dir",
			"/mnt/disks/vol1",
			base,
			nil,
			nil,
		},
		{
			"dir-used",
			"/mnt/disks/vol1",
			base + `
56 25 252:0 /mnt/disks/vol1 /var/lib/kubelet/pods/1890aef5-5a60-11e8-962f-000c29bb0377/volumes/kubernetes.io~local-volume/local-pv-test rw,relatime shared:1 - ext4 /dev/mapper/ubuntu--vg-root rw,errors=remount-ro,data=ordered
57 25 0:45 / /mnt/disks/vol rw,relatime shared:36 - tmpfs tmpfs rw
`,
			[]string{"/var/lib/kubelet/pods/1890aef5-5a60-11e8-962f-000c29bb0377/volumes/kubernetes.io~local-volume/local-pv-test"},
			nil,
		},
		{
			"tmpfs-vol",
			"/mnt/disks/vol1",
			base + `120 25 0:76 / /mnt/disks/vol1 rw,relatime shared:41 - tmpfs vol1 rw,size=10000k
`,
			nil,
			nil,
		},
		{
			"tmpfs-vol-used-by-two-pods",
			"/mnt/disks/vol1",
			base + `120 25 0:76 / /mnt/disks/vol1 rw,relatime shared:41 - tmpfs vol1 rw,size=10000k
196 25 0:76 / /var/lib/kubelet/pods/ade3ac21-5a5b-11e8-8559-000c29bb0377/volumes/kubernetes.io~local-volume/local-pv-8f263585 rw,relatime shared:41 - tmpfs vol1 rw,size=10000k
228 25 0:76 / /var/lib/kubelet/pods/ac60532d-5a5b-11e8-8559-000c29bb0377/volumes/kubernetes.io~local-volume/local-pv-8f263585 rw,relatime shared:41 - tmpfs vol1 rw,size=10000k
`,
			[]string{
				"/var/lib/kubelet/pods/ade3ac21-5a5b-11e8-8559-000c29bb0377/volumes/kubernetes.io~local-volume/local-pv-8f263585",
				"/var/lib/kubelet/pods/ac60532d-5a5b-11e8-8559-000c29bb0377/volumes/kubernetes.io~local-volume/local-pv-8f263585",
			},
			nil,
		},
		{
			"tmpfs-subdir-used-indirectly-via-bindmount-dir-by-one-pod",
			"/mnt/vol1/foo",
			base + `177 25 0:46 / /mnt/data rw,relatime shared:37 - tmpfs data rw
190 25 0:46 /vol1 /mnt/vol1 rw,relatime shared:37 - tmpfs data rw
191 25 0:46 /vol2 /mnt/vol2 rw,relatime shared:37 - tmpfs data rw
62 25 0:46 /vol1/foo /var/lib/kubelet/pods/e25f2f01-5b06-11e8-8694-000c29bb0377/volumes/kubernetes.io~local-volume/local-pv-test rw,relatime shared:37 - tmpfs data rw
`,
			[]string{"/var/lib/kubelet/pods/e25f2f01-5b06-11e8-8694-000c29bb0377/volumes/kubernetes.io~local-volume/local-pv-test"},
			nil,
		},
		{
			"dir-bindmounted",
			"/mnt/disks/vol2",
			base + `342 25 252:0 /mnt/disks/vol2 /mnt/disks/vol2 rw,relatime shared:1 - ext4 /dev/mapper/ubuntu--vg-root rw,errors=remount-ro,data=ordered
`,
			nil,
			nil,
		},
		{
			"dir-bindmounted-used-by-one-pod",
			"/mnt/disks/vol2",
			base + `342 25 252:0 /mnt/disks/vol2 /mnt/disks/vol2 rw,relatime shared:1 - ext4 /dev/mapper/ubuntu--vg-root rw,errors=remount-ro,data=ordered
77 25 252:0 /mnt/disks/vol2 /var/lib/kubelet/pods/f30dc360-5a5d-11e8-962f-000c29bb0377/volumes/kubernetes.io~local-volume/local-pv-1fb30a1c rw,relatime shared:1 - ext4 /dev/mapper/ubuntu--vg-root rw,errors=remount-ro,data=ordered
`,
			[]string{"/var/lib/kubelet/pods/f30dc360-5a5d-11e8-962f-000c29bb0377/volumes/kubernetes.io~local-volume/local-pv-1fb30a1c"},
			nil,
		},
		{
			"blockfs",
			"/mnt/disks/blkvol1",
			base + `58 25 7:1 / /mnt/disks/blkvol1 rw,relatime shared:38 - ext4 /dev/loop1 rw,data=ordered
`,
			nil,
			nil,
		},
		{
			"blockfs-used-by-one-pod",
			"/mnt/disks/blkvol1",
			base + `58 25 7:1 / /mnt/disks/blkvol1 rw,relatime shared:38 - ext4 /dev/loop1 rw,data=ordered
62 25 7:1 / /var/lib/kubelet/pods/f19fe4e2-5a63-11e8-962f-000c29bb0377/volumes/kubernetes.io~local-volume/local-pv-test rw,relatime shared:38 - ext4 /dev/loop1 rw,data=ordered
`,
			[]string{"/var/lib/kubelet/pods/f19fe4e2-5a63-11e8-962f-000c29bb0377/volumes/kubernetes.io~local-volume/local-pv-test"},
			nil,
		},
		{
			"blockfs-used-by-two-pods",
			"/mnt/disks/blkvol1",
			base + `58 25 7:1 / /mnt/disks/blkvol1 rw,relatime shared:38 - ext4 /dev/loop1 rw,data=ordered
62 25 7:1 / /var/lib/kubelet/pods/f19fe4e2-5a63-11e8-962f-000c29bb0377/volumes/kubernetes.io~local-volume/local-pv-test rw,relatime shared:38 - ext4 /dev/loop1 rw,data=ordered
95 25 7:1 / /var/lib/kubelet/pods/4854a48b-5a64-11e8-962f-000c29bb0377/volumes/kubernetes.io~local-volume/local-pv-test rw,relatime shared:38 - ext4 /dev/loop1 rw,data=ordered
`,
			[]string{"/var/lib/kubelet/pods/f19fe4e2-5a63-11e8-962f-000c29bb0377/volumes/kubernetes.io~local-volume/local-pv-test",
				"/var/lib/kubelet/pods/4854a48b-5a64-11e8-962f-000c29bb0377/volumes/kubernetes.io~local-volume/local-pv-test"},
			nil,
		},
	}
	tmpFile, err := ioutil.TempFile("", "test-get-filetype")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())
	defer tmpFile.Close()
	for _, v := range testcases {
		tmpFile.Truncate(0)
		tmpFile.Seek(0, 0)
		tmpFile.WriteString(v.mountInfos)
		tmpFile.Sync()
		refs, err := searchMountPoints(v.source, tmpFile.Name())
		if !reflect.DeepEqual(refs, v.expectedRefs) {
			t.Errorf("test %q: expected Refs: %#v, got %#v", v.name, v.expectedRefs, refs)
		}
		if !reflect.DeepEqual(err, v.expectedErr) {
			t.Errorf("test %q: expected err: %v, got %v", v.name, v.expectedErr, err)
		}
	}
}

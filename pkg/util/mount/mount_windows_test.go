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
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNormalizeWindowsPath(t *testing.T) {
	path := `/var/lib/kubelet/pods/146f8428-83e7-11e7-8dd4-000d3a31dac4/volumes/kubernetes.io~azure-disk`
	normalizedPath := normalizeWindowsPath(path)
	if normalizedPath != `c:\var\lib\kubelet\pods\146f8428-83e7-11e7-8dd4-000d3a31dac4\volumes\kubernetes.io~azure-disk` {
		t.Errorf("normizeWindowsPath test failed, normalizedPath : %q", normalizedPath)
	}

	path = `/var/lib/kubelet/pods/146f8428-83e7-11e7-8dd4-000d3a31dac4\volumes\kubernetes.io~azure-disk`
	normalizedPath = normalizeWindowsPath(path)
	if normalizedPath != `c:\var\lib\kubelet\pods\146f8428-83e7-11e7-8dd4-000d3a31dac4\volumes\kubernetes.io~azure-disk` {
		t.Errorf("normizeWindowsPath test failed, normalizedPath : %q", normalizedPath)
	}

	path = `/`
	normalizedPath = normalizeWindowsPath(path)
	if normalizedPath != `c:\` {
		t.Errorf("normizeWindowsPath test failed, normalizedPath : %q", normalizedPath)
	}
}

func TestValidateDiskNumber(t *testing.T) {
	diskNum := "0"
	if err := ValidateDiskNumber(diskNum); err != nil {
		t.Errorf("TestValidateDiskNumber test failed, disk number : %s", diskNum)
	}

	diskNum = "99"
	if err := ValidateDiskNumber(diskNum); err != nil {
		t.Errorf("TestValidateDiskNumber test failed, disk number : %s", diskNum)
	}

	diskNum = "ab"
	if err := ValidateDiskNumber(diskNum); err == nil {
		t.Errorf("TestValidateDiskNumber test failed, disk number : %s", diskNum)
	}

	diskNum = "100"
	if err := ValidateDiskNumber(diskNum); err == nil {
		t.Errorf("TestValidateDiskNumber test failed, disk number : %s", diskNum)
	}
}

func makeLink(link, target string) error {
	if output, err := exec.Command("cmd", "/c", "mklink", "/D", link, target).CombinedOutput(); err != nil {
		return fmt.Errorf("mklink failed: %v, link(%q) target(%q) output: %q", err, link, target, string(output))
	}
	return nil
}

func removeLink(link string) error {
	if output, err := exec.Command("cmd", "/c", "rmdir", link).CombinedOutput(); err != nil {
		return fmt.Errorf("rmdir failed: %v, output: %q", err, string(output))
	}
	return nil
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

// this func must run in admin mode, otherwise it will fail
func TestGetMountRefs(t *testing.T) {
	tests := []struct {
		mountPath    string
		expectedRefs []string
	}{
		{
			mountPath:    `c:\windows`,
			expectedRefs: []string{`c:\windows`},
		},
		{
			mountPath:    `c:\doesnotexist`,
			expectedRefs: []string{},
		},
	}

	mounter := Mounter{"fake/path"}

	for _, test := range tests {
		if refs, err := mounter.GetMountRefs(test.mountPath); err != nil || !setEquivalent(test.expectedRefs, refs) {
			t.Errorf("getMountRefs(%q) = %v, error: %v; expected %v", test.mountPath, refs, err, test.expectedRefs)
		}
	}
}

func TestDoSafeMakeDir(t *testing.T) {
	base, err := ioutil.TempDir("", "TestDoSafeMakeDir")
	if err != nil {
		t.Fatalf(err.Error())
	}

	defer os.RemoveAll(base)

	testingVolumePath := filepath.Join(base, "testingVolumePath")
	os.MkdirAll(testingVolumePath, 0755)
	defer os.RemoveAll(testingVolumePath)

	tests := []struct {
		volumePath    string
		subPath       string
		expectError   bool
		symlinkTarget string
	}{
		{
			volumePath:    testingVolumePath,
			subPath:       ``,
			expectError:   true,
			symlinkTarget: "",
		},
		{
			volumePath:    testingVolumePath,
			subPath:       filepath.Join(testingVolumePath, `x`),
			expectError:   false,
			symlinkTarget: "",
		},
		{
			volumePath:    testingVolumePath,
			subPath:       filepath.Join(testingVolumePath, `a\b\c\d`),
			expectError:   false,
			symlinkTarget: "",
		},
		{
			volumePath:    testingVolumePath,
			subPath:       filepath.Join(testingVolumePath, `symlink`),
			expectError:   false,
			symlinkTarget: base,
		},
		{
			volumePath:    testingVolumePath,
			subPath:       filepath.Join(testingVolumePath, `symlink\c\d`),
			expectError:   true,
			symlinkTarget: "",
		},
		{
			volumePath:    testingVolumePath,
			subPath:       filepath.Join(testingVolumePath, `symlink\y926`),
			expectError:   true,
			symlinkTarget: "",
		},
		{
			volumePath:    testingVolumePath,
			subPath:       filepath.Join(testingVolumePath, `a\b\symlink`),
			expectError:   false,
			symlinkTarget: base,
		},
		{
			volumePath:    testingVolumePath,
			subPath:       filepath.Join(testingVolumePath, `a\x\symlink`),
			expectError:   false,
			symlinkTarget: filepath.Join(testingVolumePath, `a`),
		},
	}

	for _, test := range tests {
		if len(test.volumePath) > 0 && len(test.subPath) > 0 && len(test.symlinkTarget) > 0 {
			// make all parent sub directories
			if parent := filepath.Dir(test.subPath); parent != "." {
				os.MkdirAll(parent, 0755)
			}

			// make last element as symlink
			linkPath := test.subPath
			if _, err := os.Stat(linkPath); err != nil && os.IsNotExist(err) {
				if err := makeLink(linkPath, test.symlinkTarget); err != nil {
					t.Fatalf("unexpected error: %v", fmt.Errorf("mklink link(%q) target(%q) error: %q", linkPath, test.symlinkTarget, err))
				}
			}
		}

		err := doSafeMakeDir(test.subPath, test.volumePath, os.FileMode(0755))
		if test.expectError {
			assert.NotNil(t, err, "Expect error during doSafeMakeDir(%s, %s)", test.subPath, test.volumePath)
			continue
		}
		assert.Nil(t, err, "Expect no error during doSafeMakeDir(%s, %s)", test.subPath, test.volumePath)
		if _, err := os.Stat(test.subPath); os.IsNotExist(err) {
			t.Errorf("subPath should exists after doSafeMakeDir(%s, %s)", test.subPath, test.volumePath)
		}
	}
}

func TestLockAndCheckSubPath(t *testing.T) {
	base, err := ioutil.TempDir("", "TestLockAndCheckSubPath")
	if err != nil {
		t.Fatalf(err.Error())
	}

	defer os.RemoveAll(base)

	testingVolumePath := filepath.Join(base, "testingVolumePath")

	tests := []struct {
		volumePath          string
		subPath             string
		expectedHandleCount int
		expectError         bool
		symlinkTarget       string
	}{
		{
			volumePath:          `c:\`,
			subPath:             ``,
			expectedHandleCount: 0,
			expectError:         false,
			symlinkTarget:       "",
		},
		{
			volumePath:          ``,
			subPath:             `a`,
			expectedHandleCount: 0,
			expectError:         false,
			symlinkTarget:       "",
		},
		{
			volumePath:          testingVolumePath,
			subPath:             filepath.Join(testingVolumePath, `a`),
			expectedHandleCount: 1,
			expectError:         false,
			symlinkTarget:       "",
		},
		{
			volumePath:          testingVolumePath,
			subPath:             filepath.Join(testingVolumePath, `a\b\c\d`),
			expectedHandleCount: 4,
			expectError:         false,
			symlinkTarget:       "",
		},
		{
			volumePath:          testingVolumePath,
			subPath:             filepath.Join(testingVolumePath, `symlink`),
			expectedHandleCount: 0,
			expectError:         true,
			symlinkTarget:       base,
		},
		{
			volumePath:          testingVolumePath,
			subPath:             filepath.Join(testingVolumePath, `a\b\c\symlink`),
			expectedHandleCount: 0,
			expectError:         true,
			symlinkTarget:       base,
		},
		{
			volumePath:          testingVolumePath,
			subPath:             filepath.Join(testingVolumePath, `a\b\c\d\symlink`),
			expectedHandleCount: 2,
			expectError:         false,
			symlinkTarget:       filepath.Join(testingVolumePath, `a\b`),
		},
	}

	for _, test := range tests {
		if len(test.volumePath) > 0 && len(test.subPath) > 0 {
			os.MkdirAll(test.volumePath, 0755)
			if len(test.symlinkTarget) == 0 {
				// make all intermediate sub directories
				os.MkdirAll(test.subPath, 0755)
			} else {
				// make all parent sub directories
				if parent := filepath.Dir(test.subPath); parent != "." {
					os.MkdirAll(parent, 0755)
				}

				// make last element as symlink
				linkPath := test.subPath
				if _, err := os.Stat(linkPath); err != nil && os.IsNotExist(err) {
					if err := makeLink(linkPath, test.symlinkTarget); err != nil {
						t.Fatalf("unexpected error: %v", fmt.Errorf("mklink link(%q) target(%q) error: %q", linkPath, test.symlinkTarget, err))
					}
				}
			}
		}

		fileHandles, err := lockAndCheckSubPath(test.volumePath, test.subPath)
		unlockPath(fileHandles)
		assert.Equal(t, test.expectedHandleCount, len(fileHandles))
		if test.expectError {
			assert.NotNil(t, err, "Expect error during LockAndCheckSubPath(%s, %s)", test.volumePath, test.subPath)
			continue
		}
		assert.Nil(t, err, "Expect no error during LockAndCheckSubPath(%s, %s)", test.volumePath, test.subPath)
	}

	// remove dir will happen after closing all file handles
	assert.Nil(t, os.RemoveAll(testingVolumePath), "Expect no error during remove dir %s", testingVolumePath)
}

func TestLockAndCheckSubPathWithoutSymlink(t *testing.T) {
	base, err := ioutil.TempDir("", "TestLockAndCheckSubPathWithoutSymlink")
	if err != nil {
		t.Fatalf(err.Error())
	}

	defer os.RemoveAll(base)

	testingVolumePath := filepath.Join(base, "testingVolumePath")

	tests := []struct {
		volumePath          string
		subPath             string
		expectedHandleCount int
		expectError         bool
		symlinkTarget       string
	}{
		{
			volumePath:          `c:\`,
			subPath:             ``,
			expectedHandleCount: 0,
			expectError:         false,
			symlinkTarget:       "",
		},
		{
			volumePath:          ``,
			subPath:             `a`,
			expectedHandleCount: 0,
			expectError:         false,
			symlinkTarget:       "",
		},
		{
			volumePath:          testingVolumePath,
			subPath:             filepath.Join(testingVolumePath, `a`),
			expectedHandleCount: 1,
			expectError:         false,
			symlinkTarget:       "",
		},
		{
			volumePath:          testingVolumePath,
			subPath:             filepath.Join(testingVolumePath, `a\b\c\d`),
			expectedHandleCount: 4,
			expectError:         false,
			symlinkTarget:       "",
		},
		{
			volumePath:          testingVolumePath,
			subPath:             filepath.Join(testingVolumePath, `symlink`),
			expectedHandleCount: 1,
			expectError:         true,
			symlinkTarget:       base,
		},
		{
			volumePath:          testingVolumePath,
			subPath:             filepath.Join(testingVolumePath, `a\b\c\symlink`),
			expectedHandleCount: 4,
			expectError:         true,
			symlinkTarget:       base,
		},
		{
			volumePath:          testingVolumePath,
			subPath:             filepath.Join(testingVolumePath, `a\b\c\d\symlink`),
			expectedHandleCount: 5,
			expectError:         true,
			symlinkTarget:       filepath.Join(testingVolumePath, `a\b`),
		},
	}

	for _, test := range tests {
		if len(test.volumePath) > 0 && len(test.subPath) > 0 {
			os.MkdirAll(test.volumePath, 0755)
			if len(test.symlinkTarget) == 0 {
				// make all intermediate sub directories
				os.MkdirAll(test.subPath, 0755)
			} else {
				// make all parent sub directories
				if parent := filepath.Dir(test.subPath); parent != "." {
					os.MkdirAll(parent, 0755)
				}

				// make last element as symlink
				linkPath := test.subPath
				if _, err := os.Stat(linkPath); err != nil && os.IsNotExist(err) {
					if err := makeLink(linkPath, test.symlinkTarget); err != nil {
						t.Fatalf("unexpected error: %v", fmt.Errorf("mklink link(%q) target(%q) error: %q", linkPath, test.symlinkTarget, err))
					}
				}
			}
		}

		fileHandles, err := lockAndCheckSubPathWithoutSymlink(test.volumePath, test.subPath)
		unlockPath(fileHandles)
		assert.Equal(t, test.expectedHandleCount, len(fileHandles))
		if test.expectError {
			assert.NotNil(t, err, "Expect error during LockAndCheckSubPath(%s, %s)", test.volumePath, test.subPath)
			continue
		}
		assert.Nil(t, err, "Expect no error during LockAndCheckSubPath(%s, %s)", test.volumePath, test.subPath)
	}

	// remove dir will happen after closing all file handles
	assert.Nil(t, os.RemoveAll(testingVolumePath), "Expect no error during remove dir %s", testingVolumePath)
}

func TestFindExistingPrefix(t *testing.T) {
	base, err := ioutil.TempDir("", "TestFindExistingPrefix")
	if err != nil {
		t.Fatalf(err.Error())
	}

	defer os.RemoveAll(base)

	testingVolumePath := filepath.Join(base, "testingVolumePath")

	tests := []struct {
		base                    string
		pathname                string
		expectError             bool
		expectedExistingPath    string
		expectedToCreateDirs    []string
		createSubPathBeforeTest bool
	}{
		{
			base:                    `c:\tmp\a`,
			pathname:                `c:\tmp\b`,
			expectError:             true,
			expectedExistingPath:    "",
			expectedToCreateDirs:    []string{},
			createSubPathBeforeTest: false,
		},
		{
			base:                    ``,
			pathname:                `c:\tmp\b`,
			expectError:             true,
			expectedExistingPath:    "",
			expectedToCreateDirs:    []string{},
			createSubPathBeforeTest: false,
		},
		{
			base:                    `c:\tmp\a`,
			pathname:                `d:\tmp\b`,
			expectError:             true,
			expectedExistingPath:    "",
			expectedToCreateDirs:    []string{},
			createSubPathBeforeTest: false,
		},
		{
			base:                    testingVolumePath,
			pathname:                testingVolumePath,
			expectError:             false,
			expectedExistingPath:    testingVolumePath,
			expectedToCreateDirs:    []string{},
			createSubPathBeforeTest: false,
		},
		{
			base:                    testingVolumePath,
			pathname:                filepath.Join(testingVolumePath, `a\b`),
			expectError:             false,
			expectedExistingPath:    filepath.Join(testingVolumePath, `a\b`),
			expectedToCreateDirs:    []string{},
			createSubPathBeforeTest: true,
		},
		{
			base:                    testingVolumePath,
			pathname:                filepath.Join(testingVolumePath, `a\b\c\`),
			expectError:             false,
			expectedExistingPath:    filepath.Join(testingVolumePath, `a\b`),
			expectedToCreateDirs:    []string{`c`},
			createSubPathBeforeTest: false,
		},
		{
			base:                    testingVolumePath,
			pathname:                filepath.Join(testingVolumePath, `a\b\c\d`),
			expectError:             false,
			expectedExistingPath:    filepath.Join(testingVolumePath, `a\b`),
			expectedToCreateDirs:    []string{`c`, `d`},
			createSubPathBeforeTest: false,
		},
	}

	for _, test := range tests {
		if test.createSubPathBeforeTest {
			os.MkdirAll(test.pathname, 0755)
		}

		existingPath, toCreate, err := findExistingPrefix(test.base, test.pathname)
		if test.expectError {
			assert.NotNil(t, err, "Expect error during findExistingPrefix(%s, %s)", test.base, test.pathname)
			continue
		}
		assert.Nil(t, err, "Expect no error during findExistingPrefix(%s, %s)", test.base, test.pathname)

		assert.Equal(t, test.expectedExistingPath, existingPath, "Expect result not equal with findExistingPrefix(%s, %s) return: %q, expected: %q",
			test.base, test.pathname, existingPath, test.expectedExistingPath)

		assert.Equal(t, test.expectedToCreateDirs, toCreate, "Expect result not equal with findExistingPrefix(%s, %s) return: %q, expected: %q",
			test.base, test.pathname, toCreate, test.expectedToCreateDirs)

	}
	// remove dir will happen after closing all file handles
	assert.Nil(t, os.RemoveAll(testingVolumePath), "Expect no error during remove dir %s", testingVolumePath)
}

func TestPathWithinBase(t *testing.T) {
	tests := []struct {
		fullPath       string
		basePath       string
		expectedResult bool
	}{
		{
			fullPath:       `c:\tmp\a\b\c`,
			basePath:       `c:\tmp`,
			expectedResult: true,
		},
		{
			fullPath:       `c:\tmp1`,
			basePath:       `c:\tmp2`,
			expectedResult: false,
		},
		{
			fullPath:       `c:\tmp`,
			basePath:       `c:\tmp`,
			expectedResult: true,
		},
		{
			fullPath:       `c:\tmp`,
			basePath:       `c:\tmp\a\b\c`,
			expectedResult: false,
		},
		{
			fullPath:       `c:\kubelet\pods\uuid\volumes\kubernetes.io~configmap\config\..timestamp\file.txt`,
			basePath:       `c:\kubelet\pods\uuid\volumes\kubernetes.io~configmap\config`,
			expectedResult: true,
		},
	}

	for _, test := range tests {
		result := PathWithinBase(test.fullPath, test.basePath)
		assert.Equal(t, result, test.expectedResult, "Expect result not equal with PathWithinBase(%s, %s) return: %q, expected: %q",
			test.fullPath, test.basePath, result, test.expectedResult)
	}
}

func TestGetFileType(t *testing.T) {
	mounter := New("fake/path")

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
	}

	for idx, tc := range testCase {
		path, cleanUpPath, err := tc.setUp()
		if err != nil {
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

func TestIsLikelyNotMountPoint(t *testing.T) {
	mounter := Mounter{"fake/path"}

	tests := []struct {
		fileName       string
		targetLinkName string
		setUp          func(base, fileName, targetLinkName string) error
		expectedResult bool
		expectError    bool
	}{
		{
			"Dir",
			"",
			func(base, fileName, targetLinkName string) error {
				return os.Mkdir(filepath.Join(base, fileName), 0750)
			},
			true,
			false,
		},
		{
			"InvalidDir",
			"",
			func(base, fileName, targetLinkName string) error {
				return nil
			},
			true,
			true,
		},
		{
			"ValidSymLink",
			"targetSymLink",
			func(base, fileName, targetLinkName string) error {
				targeLinkPath := filepath.Join(base, targetLinkName)
				if err := os.Mkdir(targeLinkPath, 0750); err != nil {
					return err
				}

				filePath := filepath.Join(base, fileName)
				if err := makeLink(filePath, targeLinkPath); err != nil {
					return err
				}
				return nil
			},
			false,
			false,
		},
		{
			"InvalidSymLink",
			"targetSymLink2",
			func(base, fileName, targetLinkName string) error {
				targeLinkPath := filepath.Join(base, targetLinkName)
				if err := os.Mkdir(targeLinkPath, 0750); err != nil {
					return err
				}

				filePath := filepath.Join(base, fileName)
				if err := makeLink(filePath, targeLinkPath); err != nil {
					return err
				}
				return removeLink(targeLinkPath)
			},
			true,
			false,
		},
	}

	for _, test := range tests {
		base, err := ioutil.TempDir("", test.fileName)
		if err != nil {
			t.Fatalf(err.Error())
		}

		defer os.RemoveAll(base)

		if err := test.setUp(base, test.fileName, test.targetLinkName); err != nil {
			t.Fatalf("unexpected error in setUp(%s, %s): %v", test.fileName, test.targetLinkName, err)
		}

		filePath := filepath.Join(base, test.fileName)
		result, err := mounter.IsLikelyNotMountPoint(filePath)
		assert.Equal(t, result, test.expectedResult, "Expect result not equal with IsLikelyNotMountPoint(%s) return: %q, expected: %q",
			filePath, result, test.expectedResult)

		if test.expectError {
			assert.NotNil(t, err, "Expect error during IsLikelyNotMountPoint(%s)", filePath)
		} else {
			assert.Nil(t, err, "Expect error is nil during IsLikelyNotMountPoint(%s)", filePath)
		}
	}
}

func TestFormatAndMount(t *testing.T) {
	fakeMounter := ErrorMounter{&FakeMounter{}, 0, nil}
	execCallback := func(cmd string, args ...string) ([]byte, error) {
		for j := range args {
			if strings.Contains(args[j], "Get-Disk -Number") {
				return []byte("0"), nil
			}

			if strings.Contains(args[j], "Get-Partition -DiskNumber") {
				return []byte("0"), nil
			}

			if strings.Contains(args[j], "mklink") {
				return nil, nil
			}
		}
		return nil, fmt.Errorf("Unexpected cmd %s, args %v", cmd, args)
	}
	fakeExec := NewFakeExec(execCallback)

	mounter := SafeFormatAndMount{
		Interface: &fakeMounter,
		Exec:      fakeExec,
	}

	tests := []struct {
		device       string
		target       string
		fstype       string
		mountOptions []string
		expectError  bool
	}{
		{
			"0",
			"disk",
			"NTFS",
			[]string{},
			false,
		},
		{
			"0",
			"disk",
			"",
			[]string{},
			false,
		},
		{
			"invalidDevice",
			"disk",
			"NTFS",
			[]string{},
			true,
		},
	}

	for _, test := range tests {
		base, err := ioutil.TempDir("", test.device)
		if err != nil {
			t.Fatalf(err.Error())
		}
		defer os.RemoveAll(base)

		target := filepath.Join(base, test.target)
		err = mounter.FormatAndMount(test.device, target, test.fstype, test.mountOptions)
		if test.expectError {
			assert.NotNil(t, err, "Expect error during FormatAndMount(%s, %s, %s, %v)", test.device, test.target, test.fstype, test.mountOptions)
		} else {
			assert.Nil(t, err, "Expect error is nil during FormatAndMount(%s, %s, %s, %v)", test.device, test.target, test.fstype, test.mountOptions)
		}
	}
}

func TestNewSMBMapping(t *testing.T) {
	tests := []struct {
		username    string
		password    string
		remotepath  string
		expectError bool
	}{
		{
			"",
			"password",
			`\\remotepath`,
			true,
		},
		{
			"username",
			"",
			`\\remotepath`,
			true,
		},
		{
			"username",
			"password",
			"",
			true,
		},
	}

	for _, test := range tests {
		_, err := newSMBMapping(test.username, test.password, test.remotepath)
		if test.expectError {
			assert.NotNil(t, err, "Expect error during newSMBMapping(%s, %s, %s, %v)", test.username, test.password, test.remotepath)
		} else {
			assert.Nil(t, err, "Expect error is nil during newSMBMapping(%s, %s, %s, %v)", test.username, test.password, test.remotepath)
		}
	}
}

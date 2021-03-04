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

package subpath

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
)

func makeLink(link, target string) error {
	if output, err := exec.Command("cmd", "/c", "mklink", "/D", link, target).CombinedOutput(); err != nil {
		return fmt.Errorf("mklink failed: %v, link(%q) target(%q) output: %q", err, link, target, string(output))
	}
	return nil
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

func TestIsDriveLetterorEmptyPath(t *testing.T) {
	tests := []struct {
		path           string
		expectedResult bool
	}{
		{
			path:           ``,
			expectedResult: true,
		},
		{
			path:           `\tmp`,
			expectedResult: false,
		},
		{
			path:           `c:\tmp`,
			expectedResult: false,
		},
		{
			path:           `c:\\`,
			expectedResult: true,
		},
		{
			path:           `c:\`,
			expectedResult: true,
		},
		{
			path:           `c:`,
			expectedResult: true,
		},
	}

	for _, test := range tests {
		result := isDriveLetterorEmptyPath(test.path)
		assert.Equal(t, test.expectedResult, result, "Expect result not equal with isDriveLetterorEmptyPath(%s) return: %t, expected: %t",
			test.path, result, test.expectedResult)
	}
}

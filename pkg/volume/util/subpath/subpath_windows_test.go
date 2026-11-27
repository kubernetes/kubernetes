//go:build windows

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
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/sys/windows"
	"k8s.io/mount-utils"
)

func makeLink(link, target, linkType string) error {
	if linkType == "" {
		linkType = "/D"
	}
	if output, err := exec.Command("cmd", "/c", "mklink", linkType, link, target).CombinedOutput(); err != nil {
		return fmt.Errorf("mklink failed: %v, link(%q) target(%q) output: %q", err, link, target, string(output))
	}
	return nil
}

func TestDoSafeMakeDir(t *testing.T) {
	base, err := os.MkdirTemp("", "TestDoSafeMakeDir")
	if err != nil {
		t.Fatalf("failed to create temporary directory: %v", err)
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
				if err := makeLink(linkPath, test.symlinkTarget, "/D"); err != nil {
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
	base, err := os.MkdirTemp("", "TestLockAndCheckSubPath")
	if err != nil {
		t.Fatalf("failed to create temporary directory: %v", err)
	}

	defer os.RemoveAll(base)

	testingVolumePath := filepath.Join(base, "testingVolumePath")

	tests := []struct {
		volumePath          string
		subPath             string
		expectedHandleCount int
		expectError         bool
		symlinkTarget       string
		linkType            string
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
		{
			volumePath:          testingVolumePath,
			subPath:             filepath.Join(testingVolumePath, `junction-outside`),
			expectedHandleCount: 0,
			expectError:         true,
			symlinkTarget:       base,
			linkType:            "/J",
		},
		{
			volumePath:          testingVolumePath,
			subPath:             filepath.Join(testingVolumePath, `junction-inside`),
			expectedHandleCount: 2,
			symlinkTarget:       filepath.Join(testingVolumePath, `a\b`),
			linkType:            "/J",
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
					if err := makeLink(linkPath, test.symlinkTarget, test.linkType); err != nil {
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
	base, err := os.MkdirTemp("", "TestLockAndCheckSubPathWithoutSymlink")
	if err != nil {
		t.Fatalf("failed to create temporary directory: %v", err)
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
					if err := makeLink(linkPath, test.symlinkTarget, "/D"); err != nil {
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
	base, err := os.MkdirTemp("", "TestFindExistingPrefix")
	if err != nil {
		t.Fatalf("failed to create temporary directory: %v", err)
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

func TestIsDeviceOrUncPath(t *testing.T) {
	tests := []struct {
		path           string
		expectedResult bool
	}{
		{
			// ordinary local path must be resolvable
			path:           `c:\tmp\foo`,
			expectedResult: false,
		},
		{
			// empty path is not a device/UNC path
			path:           ``,
			expectedResult: false,
		},
		{
			// UNC network path: must be refused so it is never followed,
			// otherwise resolving it triggers forced NTLM authentication.
			path:           `\\attacker\share`,
			expectedResult: true,
		},
		{
			// UNC network path referenced by IP
			path:           `\\127.0.0.1\share\dir`,
			expectedResult: true,
		},
		{
			// forward-slash UNC form: evalSymlink normalizes "/" to "\" via
			// mount.NormalizeWindowsPath before this check, but asserting it
			// here guards against callers that pass an unnormalized target.
			path:           `//attacker/share`,
			expectedResult: true,
		},
		{
			// device-form UNC path (extended-length prefix)
			path:           `\\?\UNC\server\share`,
			expectedResult: true,
		},
		{
			// device namespace path
			path:           `\\.\PhysicalDrive0`,
			expectedResult: true,
		},
		{
			// extended-length local path
			path:           `\\?\c:\tmp`,
			expectedResult: true,
		},
		{
			// stripped device-form UNC path
			path:           `UNC\server\share`,
			expectedResult: true,
		},
		{
			// volume GUID path
			path:           `Volume{00000000-0000-0000-0000-000000000000}\`,
			expectedResult: true,
		},
	}

	for _, test := range tests {
		result := isDeviceOrUncPath(test.path)
		assert.Equal(t, test.expectedResult, result, "Expect result not equal with isDeviceOrUncPath(%s) return: %t, expected: %t",
			test.path, result, test.expectedResult)
	}
}

func makeTestSymlink(t *testing.T, target, link string) {
	t.Helper()
	err := os.Symlink(target, link)
	if errors.Is(err, windows.ERROR_PRIVILEGE_NOT_HELD) {
		t.Skipf("symbolic link creation requires SeCreateSymbolicLinkPrivilege: %v", err)
	}
	require.NoError(t, err)
}

func TestNativeLinkType(t *testing.T) {
	root := t.TempDir()
	file, dir := filepath.Join(root, "file.cfg"), filepath.Join(root, "dir")
	require.NoError(t, os.WriteFile(file, []byte("fixture"), 0600))
	require.NoError(t, os.Mkdir(dir, 0700))
	tests := []struct {
		name, path, target string
		linkType           windowsLinkType
	}{
		{name: "file", path: file},
		{name: "directory", path: dir},
		{name: "file symlink", target: file, linkType: windowsSymbolicLink},
		{name: "directory symlink", target: dir, linkType: windowsSymbolicLink},
		{name: "junction", target: dir, linkType: windowsJunction},
		{name: "hardlink", target: file, linkType: windowsHardLink},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			path := test.path
			if path == "" {
				path = filepath.Join(root, test.name)
			}
			switch test.linkType {
			case windowsSymbolicLink:
				makeTestSymlink(t, test.target, path)
			case windowsJunction:
				require.NoError(t, makeLink(path, test.target, "/J"))
			case windowsHardLink:
				require.NoError(t, os.Link(test.target, path))
			}
			paths := []string{path}
			if test.linkType == windowsHardLink {
				paths = append(paths, test.target)
			}
			for _, path := range paths {
				kind, err := nativeLinkType(path)
				assert.NoError(t, err)
				assert.Equal(t, test.linkType, kind)
				isLink, err := isLinkPath(path)
				assert.NoError(t, err)
				assert.Equal(t, test.linkType != windowsNotLink, isLink)
			}
		})
	}
	_, err := nativeLinkType(filepath.Join(root, "missing"))
	assert.True(t, os.IsNotExist(err), "expected a not-exist error, got %v", err)
}

func TestEvalSymlink(t *testing.T) {
	root := t.TempDir()
	dir, file := filepath.Join(root, "target"), filepath.Join(root, "target", "file.cfg")
	require.NoError(t, os.MkdirAll(filepath.Join(dir, "leaf"), 0700))
	require.NoError(t, os.WriteFile(file, []byte("fixture"), 0600))
	tests := []struct {
		name, target, suffix, expected string
		junction                       bool
	}{
		{name: "absolute file", target: file, expected: file},
		{name: "relative file", target: filepath.Join("target", "file.cfg"), expected: file},
		{name: "absolute directory parent", target: dir, suffix: "leaf", expected: filepath.Join(dir, "leaf")},
		{name: "relative directory parent", target: "target", suffix: "leaf", expected: filepath.Join(dir, "leaf")},
		{name: "junction parent", target: dir, suffix: "leaf", expected: filepath.Join(dir, "leaf"), junction: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			link := filepath.Join(root, test.name)
			if test.junction {
				require.NoError(t, makeLink(link, test.target, "/J"))
			} else {
				makeTestSymlink(t, test.target, link)
			}
			// Resolution must work without a PowerShell executable in PATH.
			t.Setenv("PATH", filepath.Join(root, "no-executables"))
			actual, err := evalSymlink(filepath.Join(link, test.suffix))
			assert.NoError(t, err)
			assert.Equal(t, strings.ToLower(test.expected), strings.ToLower(actual))
		})
	}
	actual, err := evalSymlink(file)
	assert.NoError(t, err)
	assert.Equal(t, file, actual)
	_, err = evalSymlink(filepath.Join(root, "missing"))
	assert.True(t, os.IsNotExist(err), "expected a not-exist error, got %v", err)
}

func TestHardLinkTargets(t *testing.T) {
	// TempDir may inherit a short-name TEMP path from the Windows account.
	root, err := filepath.EvalSymlinks(t.TempDir())
	require.NoError(t, err)
	file := filepath.Join(root, "file.cfg")
	require.NoError(t, os.WriteFile(file, []byte("fixture"), 0600))
	names := []string{"file.cfg", "alias1.cfg", "alias2.cfg"}
	for _, name := range names[1:] {
		require.NoError(t, os.Link(file, filepath.Join(root, name)))
	}
	for _, name := range names {
		path := filepath.Join(root, name)
		targets, err := hardLinkTargets(path)
		require.NoError(t, err)
		for i := range targets {
			targets[i] = strings.ToLower(targets[i])
		}
		var expected []string
		for _, other := range names {
			if other != name {
				expected = append(expected, strings.ToLower(filepath.Join(root, other)))
			}
		}
		assert.ElementsMatch(t, expected, targets, "hardlink %q", path)
		resolved, err := evalSymlink(path)
		assert.NoError(t, err)
		assert.Equal(t, strings.ToLower(path), strings.ToLower(resolved))
	}
}

func TestHardLinkShortPaths(t *testing.T) {
	root, err := filepath.EvalSymlinks(t.TempDir())
	require.NoError(t, err)
	volume := filepath.Join(root, "long-volume-directory")
	require.NoError(t, os.Mkdir(volume, 0700))
	file, alias := filepath.Join(volume, "file.cfg"), filepath.Join(volume, "alias.cfg")
	require.NoError(t, os.WriteFile(file, []byte("fixture"), 0600))
	require.NoError(t, os.Link(file, alias))
	shortName := func(path string) string {
		t.Helper()
		p, err := windows.UTF16PtrFromString(path)
		require.NoError(t, err)
		buffer := make([]uint16, MaxPathLength+1)
		n, err := windows.GetShortPathName(p, &buffer[0], uint32(len(buffer)))
		require.NoError(t, err)
		require.Less(t, n, uint32(len(buffer)))
		return windows.UTF16ToString(buffer[:n])
	}
	shortVolume, shortFile := shortName(volume), shortName(file)
	if strings.EqualFold(shortVolume, volume) {
		t.Skip("test filesystem does not provide 8.3 names")
	}
	targets, err := hardLinkTargets(shortFile)
	require.NoError(t, err)
	require.Len(t, targets, 1)
	assert.Equal(t, strings.ToLower(alias), strings.ToLower(targets[0]))
	handles, err := lockAndCheckSubPath(shortVolume, shortFile)
	unlockPath(handles)
	assert.NoError(t, err)
	require.NoError(t, os.Link(file, filepath.Join(root, "outside.cfg")))
	handles, err = lockAndCheckSubPath(shortVolume, shortFile)
	unlockPath(handles)
	assert.Error(t, err, "expected rejection of a hardlink outside the volume")
}

func TestLockAndCheckSubPathHardLinks(t *testing.T) {
	root := t.TempDir()
	volume := filepath.Join(root, "volume")
	require.NoError(t, os.Mkdir(volume, 0700))
	file, alias := filepath.Join(volume, "file.cfg"), filepath.Join(volume, "alias.cfg")
	require.NoError(t, os.WriteFile(file, []byte("fixture"), 0600))
	require.NoError(t, os.Link(file, alias))
	handles, err := lockAndCheckSubPath(volume, alias)
	assert.NoError(t, err)
	assert.Len(t, handles, 1)
	assert.Error(t, os.Rename(alias, alias+".renamed"), "hardlink name changed while locked")
	unlockPath(handles)
	assert.NoError(t, os.Rename(alias, alias+".renamed"), "cleanup did not release the handle")
	require.NoError(t, os.Link(file, filepath.Join(root, "outside.cfg")))
	handles, err = lockAndCheckSubPath(volume, file)
	unlockPath(handles)
	assert.Error(t, err, "expected rejection of a hardlink outside the volume")
}

func TestEvalSymlinkSpecialPaths(t *testing.T) {
	for _, path := range []string{`\\127.0.0.1\not-a-share`, `//127.0.0.1/not-a-share`, `\\?\C:\not-a-path`, `\\.\PhysicalDrive999`, `UNC\127.0.0.1\not-a-share`, `Volume{00000000-0000-0000-0000-000000000000}\`, `C:`, `C:\`, ""} {
		actual, err := evalSymlink(path)
		assert.NoError(t, err, "path %q", path)
		assert.Equal(t, mount.NormalizeWindowsPath(path), actual)
	}
	link := filepath.Join(t.TempDir(), "unc-target")
	p, err := windows.UTF16PtrFromString(link)
	require.NoError(t, err)
	target, err := windows.UTF16PtrFromString(`\\127.0.0.1\not-a-share`)
	require.NoError(t, err)
	// Create the directory link without probing its remote target.
	err = windows.CreateSymbolicLink(p, target, windows.SYMBOLIC_LINK_FLAG_DIRECTORY)
	if errors.Is(err, windows.ERROR_PRIVILEGE_NOT_HELD) {
		t.Skipf("symbolic link creation requires SeCreateSymbolicLinkPrivilege: %v", err)
	}
	require.NoError(t, err)
	actual, err := evalSymlink(link)
	assert.NoError(t, err)
	assert.Equal(t, link, actual, "UNC target must not be followed")
}

func TestLockAndCheckSubPathAccessDenied(t *testing.T) {
	volume := t.TempDir()
	path := filepath.Join(volume, "denied.cfg")
	require.NoError(t, os.WriteFile(path, []byte("fixture"), 0600))
	original, err := windows.GetNamedSecurityInfo(path, windows.SE_FILE_OBJECT, windows.DACL_SECURITY_INFORMATION)
	require.NoError(t, err)
	originalDACL, _, err := original.DACL()
	require.NoError(t, err)
	denied, err := windows.SecurityDescriptorFromString("D:(D;;GR;;;WD)(A;;GA;;;BA)(A;;GA;;;SY)")
	require.NoError(t, err)
	deniedDACL, _, err := denied.DACL()
	require.NoError(t, err)
	require.NoError(t, windows.SetNamedSecurityInfo(path, windows.SE_FILE_OBJECT, windows.DACL_SECURITY_INFORMATION|windows.PROTECTED_DACL_SECURITY_INFORMATION, nil, nil, deniedDACL, nil))
	t.Cleanup(func() {
		assert.NoError(t, windows.SetNamedSecurityInfo(path, windows.SE_FILE_OBJECT, windows.DACL_SECURITY_INFORMATION|windows.UNPROTECTED_DACL_SECURITY_INFORMATION, nil, nil, originalDACL, nil))
	})
	handles, err := lockAndCheckSubPath(volume, path)
	unlockPath(handles)
	if err == nil {
		t.Skip("account bypasses the fixture's deny-read DACL")
	}
	assert.ErrorContains(t, err, "Access is denied")
}

func BenchmarkLockAndCheckSubPath(b *testing.B) {
	volume := b.TempDir()
	path := filepath.Join(volume, "a", "b", "c", "d")
	require.NoError(b, os.MkdirAll(path, 0700))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		handles, err := lockAndCheckSubPath(volume, path)
		unlockPath(handles)
		require.NoError(b, err)
	}
}

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

package path

import (
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/spf13/afero"
	"github.com/stretchr/testify/assert"
)

func TestFileUtils(t *testing.T) {
	fs := &afero.Afero{Fs: afero.NewOsFs()}
	// Create tmp dir
	tmpDir, err := fs.TempDir(os.TempDir(), "util_file_test_")
	if err != nil {
		t.Fatal("Failed to test: failed to create temp dir.")
	}

	// create tmp file
	tmpFile, err := fs.TempFile(tmpDir, "test_file_exists_")
	if err != nil {
		t.Fatal("Failed to test: failed to create temp file.")
	}

	// create tmp sym link
	tmpSymlinkName := filepath.Join(tmpDir, "test_file_exists_sym_link")
	err = os.Symlink(tmpFile.Name(), tmpSymlinkName)
	if err != nil {
		t.Fatal("Failed to test: failed to create sym link.")
	}

	// create tmp sub dir
	tmpSubDir, err := fs.TempDir(tmpDir, "sub_")
	if err != nil {
		t.Fatal("Failed to test: failed to create temp sub dir.")
	}

	// record the current dir
	currentDir, err := os.Getwd()
	if err != nil {
		t.Fatal("Failed to test: failed to get current dir.")
	}

	// change the work dir to temp dir
	err = os.Chdir(tmpDir)
	if err != nil {
		t.Fatal("Failed to test: failed to change work dir.")
	}

	// recover test environment
	defer func() {
		os.Chdir(currentDir)
		os.RemoveAll(tmpDir)
	}()

	t.Run("TestExists", func(t *testing.T) {
		tests := []struct {
			name          string
			fileName      string
			expectedError bool
			expectedValue bool
		}{
			{"file_not_exists", filepath.Join(tmpDir, "file_not_exist_case"), false, false},
			{"file_exists", tmpFile.Name(), false, true},
		}

		for _, test := range tests {
			realValued, realError := Exists(CheckFollowSymlink, test.fileName)
			if test.expectedError {
				assert.Errorf(t, realError, "Failed to test with '%s': %s", test.fileName, test.name)
			} else {
				assert.EqualValuesf(t, test.expectedValue, realValued, "Failed to test with '%s': %s", test.fileName, test.name)
			}
		}
	})

	t.Run("TestFileOrSymlinkExists", func(t *testing.T) {
		tests := []struct {
			name          string
			fileName      string
			expectedError bool
			expectedValue bool
		}{
			{"file_not_exists", filepath.Join(tmpDir, "file_not_exist_case"), false, false},
			{"file_exists", tmpFile.Name(), false, true},
			{"symlink_exists", tmpSymlinkName, false, true},
		}

		for _, test := range tests {
			realValued, realError := Exists(CheckSymlinkOnly, test.fileName)
			if test.expectedError {
				assert.Errorf(t, realError, "Failed to test with '%s': %s", test.fileName, test.name)
			} else {
				assert.EqualValuesf(t, test.expectedValue, realValued, "Failed to test with '%s': %s", test.fileName, test.name)
			}
		}
	})

	t.Run("TestReadDirNoStat", func(t *testing.T) {
		_, tmpFileSimpleName := filepath.Split(tmpFile.Name())
		_, tmpSymlinkSimpleName := filepath.Split(tmpSymlinkName)
		_, tmpSubDirSimpleName := filepath.Split(tmpSubDir)

		tests := []struct {
			name          string
			dirName       string
			expectedError bool
			expectedValue []string
		}{
			{"dir_not_exists", filepath.Join(tmpDir, "file_not_exist_case"), true, []string{}},
			{"dir_is_empty", "", false, []string{tmpFileSimpleName, tmpSymlinkSimpleName, tmpSubDirSimpleName}},
			{"dir_exists", tmpDir, false, []string{tmpFileSimpleName, tmpSymlinkSimpleName, tmpSubDirSimpleName}},
		}

		for _, test := range tests {
			realValued, realError := ReadDirNoStat(test.dirName)

			// execute sort action before compare
			sort.Strings(realValued)
			sort.Strings(test.expectedValue)

			if test.expectedError {
				assert.Errorf(t, realError, "Failed to test with '%s': %s", test.dirName, test.name)
			} else {
				assert.EqualValuesf(t, test.expectedValue, realValued, "Failed to test with '%s': %s", test.dirName, test.name)
			}
		}
	})
}

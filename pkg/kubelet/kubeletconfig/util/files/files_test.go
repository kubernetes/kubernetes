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

package files

import (
	"fmt"
	utilfs "k8s.io/kubernetes/pkg/util/filesystem"
	"os"
	"reflect"
	"testing"
)

func TestFileExists(t *testing.T) {
	fs := utilfs.NewFakeFs()

	testcases := []struct {
		name         string // name of test case
		filePath     string // file name to be created
		needsCreated bool   // if should create the file before test
		isFile       bool   // is the fileName a file or folder
		expectedBool bool   // expected result in bool
	}{
		{
			name:         "not exists",
			filePath:     "/tmp/FileExists",
			needsCreated: false,
			isFile:       false,
			expectedBool: false,
		},
		{
			name:         "exists as folder",
			filePath:     "/tmp/FileExists",
			needsCreated: true,
			isFile:       false,
			expectedBool: true,
		},
		{
			name:         "exists as file",
			filePath:     "/tmp/FileExists/file",
			needsCreated: true,
			isFile:       true,
			expectedBool: true,
		},
	}

	for _, testcase := range testcases {
		var err error
		t.Run(testcase.name, func(t *testing.T) {
			if testcase.needsCreated {
				if testcase.isFile {
					// create as file
					_, err = fs.TempFile(testcase.filePath, "")
				} else {
					// create as folder
					err = fs.MkdirAll(testcase.filePath, defaultPerm)
				}
			}

			if err != nil {
				t.Errorf("unexpected error in creating file %s, err: %v", testcase.filePath, err)
			} else {
				if result, err := FileExists(fs, testcase.filePath); err != nil {
					t.Errorf("unexpected error: %v", err)
				} else {
					if testcase.expectedBool != result {
						t.Errorf("unexpected result, expected: %v, actual: %v", testcase.expectedBool, result)
					}
				}
			}
		})

	}
}

func TestEnsureFile(t *testing.T) {
	fs := utilfs.NewFakeFs()

	testcases := []struct {
		name     string
		filePath string
		ifExist  bool
	}{
		{
			name:     "already exists",
			filePath: "/tmp/EnsureFile",
			ifExist:  true,
		},
		{
			name:     "file not exists",
			filePath: "/tmp/EnsureFile/test1",
			ifExist:  false,
		},
	}

	for _, testcase := range testcases {
		var err error
		t.Run(testcase.name, func(t *testing.T) {
			if testcase.ifExist {
				// should create the file before running the test
				if _, err = fs.TempFile(testcase.filePath, ""); err != nil {
					t.Fatalf("unexpeted error: %v", err)
				}
			}

			if err = EnsureFile(fs, testcase.filePath); err != nil {
				t.Errorf("unexpeted error in ensuring file: %v", err)
			} else {
				// verify if the file has been created
				if _, err := fs.Stat(testcase.filePath); err != nil {
					t.Errorf("unexpected error in verifying file: %s, error: %v", testcase.filePath, err)
				}
			}
		})
	}
}

func TestWriteTmpFile(t *testing.T) {
	fs := utilfs.NewFakeFs()
	path := "/tmp/WriteTmpFile/test"
	fileContent := "a test case of WriteTmpFile"

	// write content into file
	filePath, err := WriteTmpFile(fs, path, []byte(fileContent))

	if err != nil {
		t.Errorf("unexpected error in WriteTmpFile: %v", err)
	} else {
		// read the content out and check if match the fileContent
		result, err := fs.ReadFile(filePath)
		if err != nil {
			t.Errorf("unexpected error in reading written file: %v", err)
		} else {
			if fileContent != string(result) {
				t.Errorf("unexpected content in written file, expected: %s, actual: %s", fileContent, result)
			}
		}
	}
}

func TestReplaceFile(t *testing.T) {
	fileName := "/tmp/ReplaceFile/test"
	fileContent := "a test case of ReplaceFile"
	fs := utilfs.NewFakeFs()

	if _, err := WriteTmpFile(fs, fileName, []byte(fileContent)); err != nil {
		t.Fatalf("error in writing file: %s, error: %v", fileName, err)
	} else {
		// replace the file with new content, and verify if new content written
		fileContent = "new test to replace file"
		if err = ReplaceFile(fs, fileName, []byte(fileContent)); err != nil {
			t.Errorf("error in ReplaceFile: %s, error: %v", fileName, err)
		} else {
			// read the content out and verify if matches the new one
			if result, err := fs.ReadFile(fileName); err != nil {
				t.Errorf("error in reading file: %s, error: %v", fileName, err)
			} else {
				if fileContent != string(result) {
					t.Errorf("unexpected content in %s, expected: %s, actual: %s", fileName, fileContent, string(result))
				}
			}
		}
	}
}

func TestDirExists(t *testing.T) {
	folderName := "/tmp/DirExists"
	fs := utilfs.NewFakeFs()

	testcases := []struct {
		name         string // name of test case
		dirName      string // dirName to be created
		isFile       bool   // is the dirName a file or folder
		needsCreated bool   // if should create dirName before test
		expectedBool bool   // expected result in bool
		expectedErr  error  // expected error
	}{
		{
			name:         "not exists",
			dirName:      folderName,
			isFile:       false,
			needsCreated: false,
			expectedBool: false,
			expectedErr:  nil,
		},
		{
			name:         "exists as file",
			dirName:      folderName + "/file",
			isFile:       true,
			needsCreated: true,
			expectedBool: false,
			expectedErr:  fmt.Errorf("expected dir at %q, but mode is %q", folderName+"/file", os.ModeTemporary.String()),
		},
		{
			name:         "exists as dir",
			dirName:      folderName,
			isFile:       false,
			needsCreated: true,
			expectedBool: true,
			expectedErr:  nil,
		},
	}

	for _, testcase := range testcases {
		var err error
		t.Run(testcase.name, func(t *testing.T) {
			if testcase.needsCreated {
				// should create dirName in either file or dir
				if testcase.isFile {
					_, err = fs.Create(testcase.dirName)
				} else {
					err = fs.MkdirAll(testcase.dirName, defaultPerm)
				}
			}

			if err != nil {
				t.Errorf("unexpected error in creating file %s, err: %v", testcase.dirName, err)
			} else {
				if result, err := DirExists(fs, testcase.dirName); err != nil {
					// if the dirName is file, should get error
					if !reflect.DeepEqual(testcase.expectedErr, err) {
						t.Errorf("unexpected error in creating dir, expected: %v, actual: %v", testcase.expectedErr, err)
					}
				} else {
					if testcase.expectedBool != result {
						t.Errorf("unexpected result, expected: %v, actual: %v", testcase.expectedBool, result)
					}
				}
			}
		})

	}
}

func TestEnsureDir(t *testing.T) {
	folderName := "/tmp/EnsureDir/"
	fs := utilfs.NewFakeFs()

	testcases := []struct {
		name         string // name of test case
		dirName      string // dirName to be created
		needsCreated bool   // if should create dir before test
		expectedBool bool   // expected result in bool
	}{
		{
			name:         "not exists",
			dirName:      folderName,
			needsCreated: false,
			expectedBool: false,
		},
		{
			name:         "exists as dir",
			dirName:      folderName,
			needsCreated: true,
			expectedBool: true,
		},
	}

	for _, testcase := range testcases {
		var err error
		t.Run(testcase.name, func(t *testing.T) {
			if testcase.needsCreated {
				// should create dir in prior
				err = fs.MkdirAll(testcase.dirName, defaultPerm)
			}

			if err != nil {
				t.Errorf("unexpected error in creating dir: %s, error: %v", testcase.dirName, err)
			} else {
				// ensure dir exists
				if err = EnsureDir(fs, testcase.dirName); err != nil {
					t.Errorf("unexpected error in ensuring dir: %s, error: %v", testcase.dirName, err)
				} else {
					// check if dir created
					_, err = fs.Stat(testcase.dirName)
					if err != nil {
						t.Errorf("unexpected error in verifying dir: %s, error: %v", testcase.dirName, err)
					}
				}
			}
		})
	}
}

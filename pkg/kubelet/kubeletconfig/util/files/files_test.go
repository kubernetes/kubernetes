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
	"io/ioutil"
	utilfs "k8s.io/kubernetes/pkg/util/filesystem"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestFileExists(t *testing.T) {
	fs := utilfs.DefaultFs{}
	prePath := getTempDir(t, "TestFileExists")
	defer cleanup(t, fs, prePath)

	testcases := map[string]struct {
		path         string // file path to be created
		create       bool   // if should create the file before test
		isFile       bool   // if the path indicates a file
		expectedBool bool   // expected result in bool
		expectedErr  error  // expected error
	}{
		"not exists": {
			path:         prePath + "/NotExists",
			create:       false,
			isFile:       false,
			expectedBool: false,
			expectedErr:  nil,
		},
		"exists as dir": {
			path:         prePath + "/ExistsAsDir",
			create:       true,
			isFile:       false,
			expectedBool: false,
			expectedErr:  fmt.Errorf(""), // should have error, but the error info can only be fulfilled after creating the dir
		},
		"exists as file": {
			path:         prePath + "/ExistsAsFile",
			create:       true,
			isFile:       true,
			expectedBool: true,
			expectedErr:  nil,
		},
	}

	for tesetcaseName, testcase := range testcases {
		var result bool
		var err error
		t.Run(tesetcaseName, func(t *testing.T) {
			if testcase.create {
				if testcase.isFile {
					// create as file
					if err = createFile(fs, testcase.path); err != nil {
						t.Fatalf("unexpected error in creating file %s, err: %v", testcase.path, err)
					}
				} else {
					// create as dir
					if err = createDir(fs, testcase.path); err != nil {
						t.Fatalf("unexpected error in creating dir %s, err: %v", testcase.path, err)
					}
				}
			}

			// not nil indicates that we should specify the error
			if testcase.expectedErr != nil {
				testcase.expectedErr = fmt.Errorf("expected regular file at %q, but mode is %q", testcase.path, getFileMode(fs, testcase.path))
			}

			if result, err = FileExists(fs, testcase.path); !reflect.DeepEqual(testcase.expectedErr, err) {
				t.Errorf("unexpected error in checking file exists, expected: %v, actual: %v", testcase.expectedErr, err)
			}
			if testcase.expectedBool != result {
				t.Errorf("unexpected result, expected: %t, actual: %t", testcase.expectedBool, result)
			}
		})
	}
}

func TestEnsureFile(t *testing.T) {
	fs := utilfs.DefaultFs{}
	prePath := getTempDir(t, "TestEnsureFile")
	defer cleanup(t, fs, prePath)

	testcases := map[string]struct {
		path   string
		create bool
	}{
		"already exists": {
			path:   prePath + "/AlreadyExists",
			create: true,
		},
		"file not exists": {
			path:   prePath + "/FileNotExists",
			create: false,
		},
		"file not exists in subdir": {
			path:   prePath + "/FileNotExistsInSubDir/test1",
			create: false,
		},
	}

	for testcaseName, testcase := range testcases {
		var err error
		t.Run(testcaseName, func(t *testing.T) {
			if testcase.create {
				// should create the file before running the test
				if err = createFile(fs, testcase.path); err != nil {
					t.Fatalf("unexpeted error: %v", err)
				}
			}

			if err = EnsureFile(fs, testcase.path); err != nil {
				t.Fatalf("unexpeted error in ensuring file: %v", err)
			}
			// verify if the file has been created
			if _, err := fs.Stat(testcase.path); err != nil {
				if os.IsNotExist(err) {
					t.Errorf("missing file in %q, should have been created.", testcase.path)
				} else {
					t.Errorf("unexpected error in verifying file: %s, error: %v", testcase.path, err)
				}
			}
		})
	}
}

func TestWriteTmpFile(t *testing.T) {
	fs := utilfs.DefaultFs{}
	prePath := getTempDir(t, "TestWriteTmpFile")
	filePath := prePath + "/test"
	fileContent := "a test case of WriteTmpFile"
	defer cleanup(t, fs, prePath)

	if err := createDir(fs, prePath); err != nil {
		t.Fatalf("error in creating dir %q, error: %v", prePath, err)
	}
	// write content into file
	filePath, err := WriteTmpFile(fs, filePath, []byte(fileContent))

	if err != nil {
		t.Errorf("unexpected error in WriteTmpFile: %v", err)
	} else {
		// read the content out and check if match the fileContent
		result, err := fs.ReadFile(filePath)
		if err != nil {
			t.Fatalf("unexpected error in reading written file: %v", err)
		}
		if fileContent != string(result) {
			t.Errorf("unexpected content in written file, expected: %s, actual: %s", fileContent, result)
		}
	}
}

func TestReplaceFile(t *testing.T) {
	prePath := getTempDir(t, "TestReplaceFile")
	filePath := prePath + "/test"
	fileContent := "a test case of ReplaceFile"
	fs := utilfs.DefaultFs{}
	defer cleanup(t, fs, prePath)

	if err := createDir(fs, prePath); err != nil {
		t.Fatalf("error in creating dir %q, error: %v", prePath, err)
	}
	if _, err := WriteTmpFile(fs, filePath, []byte(fileContent)); err != nil {
		t.Fatalf("error in writing file: %s, error: %v", filePath, err)
	} else {
		// replace the file with new content, and verify if new content written
		fileContent = "new test to replace file"
		if err = ReplaceFile(fs, filePath, []byte(fileContent)); err != nil {
			t.Errorf("error in ReplaceFile: %s, error: %v", filePath, err)
		} else {
			// read the content out and verify if matches the new one
			if result, err := fs.ReadFile(filePath); err != nil {
				t.Errorf("error in reading file: %s, error: %v", filePath, err)
			} else {
				if fileContent != string(result) {
					t.Errorf("unexpected content in %s, expected: %s, actual: %s", filePath, fileContent, string(result))
				}
			}
		}
	}
}

func TestDirExists(t *testing.T) {
	fs := utilfs.DefaultFs{}
	prePath := getTempDir(t, "TestDirExists")
	defer cleanup(t, fs, prePath)

	testcases := map[string]struct {
		path         string // path to be created
		isFile       bool   // is the path a file or dir
		create       bool   // if should create path before test
		expectedBool bool   // expected result in bool
		expectedErr  error  // expected error
	}{
		"not exists": {
			path:         prePath + "/NotExists",
			isFile:       false,
			create:       false,
			expectedBool: false,
			expectedErr:  nil,
		},
		"exists as file": {
			path:         prePath + "/ExistsAsFile/file",
			isFile:       true,
			create:       true,
			expectedBool: false,
			expectedErr:  fmt.Errorf(""), // should have error, but the error info can only be fulfilled after creating the file
		},
		"exists as dir": {
			path:         prePath + "/ExistsAsDir",
			isFile:       false,
			create:       true,
			expectedBool: true,
			expectedErr:  nil,
		},
	}

	for testcaseName, testcase := range testcases {
		var err error
		t.Run(testcaseName, func(t *testing.T) {
			if testcase.create {
				if testcase.isFile {
					err = createFile(fs, testcase.path)
				} else {
					err = createDir(fs, testcase.path)
				}
			}

			// not nil indicates that we should specify the error
			if testcase.expectedErr != nil {
				testcase.expectedErr = fmt.Errorf("expected dir at %q, but mode is %q", testcase.path, getFileMode(fs, testcase.path))
			}

			if err != nil {
				t.Errorf("unexpected error in creating %q, err: %v", testcase.path, err)
			} else {
				if result, err := DirExists(fs, testcase.path); err != nil {
					// if the path is file, should get error
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
	prePath := getTempDir(t, "TestEnsureDir")
	fs := utilfs.DefaultFs{}
	defer cleanup(t, fs, prePath)

	testcases := map[string]struct {
		path        string // path to be created
		isFile      bool   // if the path indicates a file
		create      bool   // if should create dir before test
		expectedErr error
	}{
		"exists as file": {
			path:        prePath + "/ExistsAsFile",
			isFile:      true,
			create:      true,
			expectedErr: fmt.Errorf(""), // should have error, but the error info can only be fulfilled after creating the file
		},
		"exists as dir": {
			path:        prePath + "/NotExists",
			isFile:      false,
			create:      true,
			expectedErr: nil,
		},
		"not exists in sub dir": {
			path:        prePath + "/SubDir1/SubDir2/Exists",
			isFile:      false,
			create:      false,
			expectedErr: nil,
		},
	}

	for testcaseName, testcase := range testcases {
		var err error
		t.Run(testcaseName, func(t *testing.T) {
			if testcase.create {
				if testcase.isFile {
					err = createFile(fs, testcase.path)
				} else {
					err = createDir(fs, testcase.path)
				}
			}

			// not nil indicates we should specify the error
			if testcase.expectedErr != nil {
				testcase.expectedErr = fmt.Errorf("expected dir at %q, but mode is %q", testcase.path, getFileMode(fs, testcase.path))
			}

			if err != nil {
				t.Errorf("unexpected error in creating dir: %s, error: %v", testcase.path, err)
			} else {
				// ensure dir exists
				if err = EnsureDir(fs, testcase.path); err != nil {
					// if the path is file, should get error
					if !reflect.DeepEqual(testcase.expectedErr, err) {
						t.Errorf("unexpected error in creating dir, expected: %v, actual: %v", testcase.expectedErr, err)
					}
				} else {
					// check if dir created
					info, err := fs.Stat(testcase.path)
					if err != nil {
						t.Errorf("unexpected error in verifying dir: %s, error: %v", testcase.path, err)
					}
					if !info.IsDir() {
						t.Errorf("expected dir in path: %q, but got: %v", testcase.path, info.Mode())
					}
				}
			}
		})
	}
}

func getTempDir(t *testing.T, subPath string) string {
	dir, err := ioutil.TempDir("", subPath)
	if err != nil {
		t.Fatalf("error in getting temp dir via ioutil, error: %v", err)
	}
	return dir
}

func getFileMode(fs utilfs.Filesystem, path string) string {
	if info, err := fs.Stat(path); err == nil {
		return info.Mode().String()
	}
	return ""
}

func createFile(fs utilfs.Filesystem, path string) error {
	if err := createDir(fs, filepath.Dir(path)); err != nil {
		return err
	}

	// create the file
	if _, err := fs.Create(path); err != nil && !os.IsExist(err) {
		return err
	}

	return nil
}

func createDir(fs utilfs.Filesystem, path string) error {
	if err := fs.MkdirAll(path, 0755); err != nil && !os.IsExist(err) {
		return err
	}

	return nil
}

func cleanup(t *testing.T, fs utilfs.Filesystem, path string) {
	if err := fs.RemoveAll(path); err != nil {
		t.Fatalf("unexpected error in cleaning up path: %s", path)
	}
}

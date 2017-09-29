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
	"os"
	"path/filepath"

	utilfs "k8s.io/kubernetes/pkg/util/filesystem"
)

const defaultPerm = 0666

// FileExists returns true if a regular file exists at `path`, false if `path` does not exist, otherwise an error
func FileExists(fs utilfs.Filesystem, path string) (bool, error) {
	if info, err := fs.Stat(path); err == nil {
		if info.Mode().IsRegular() {
			return true, nil
		}
		return false, fmt.Errorf("expected regular file at %q, but mode is %q", path, info.Mode().String())
	} else if os.IsNotExist(err) {
		return false, nil
	} else {
		return false, err
	}
}

// EnsureFile ensures that a regular file exists at `path`, and if it must create the file any
// necessary parent directories will also be created and the new file will be empty.
func EnsureFile(fs utilfs.Filesystem, path string) error {
	// if file exists, don't change it, but do report any unexpected errors
	if ok, err := FileExists(fs, path); ok || err != nil {
		return err
	} // Assert: file does not exist

	// create any necessary parents
	err := fs.MkdirAll(filepath.Dir(path), defaultPerm)
	if err != nil {
		return err
	}

	// create the file
	file, err := fs.Create(path)
	if err != nil {
		return err
	}
	// close the file, since we don't intend to use it yet
	return file.Close()
}

// WriteTmpFile creates a temporary file at `path`, writes `data` into it, and fsyncs the file
func WriteTmpFile(fs utilfs.Filesystem, path string, data []byte) (tmpPath string, retErr error) {
	dir := filepath.Dir(path)
	prefix := filepath.Base(path)

	// create the tmp file
	tmpFile, err := fs.TempFile(dir, prefix)
	if err != nil {
		return "", err
	}
	defer func() {
		// close the file, return the close error only if there haven't been any other errors
		if err := tmpFile.Close(); retErr == nil {
			retErr = err
		}
		// if there was an error writing, syncing, or closing, delete the temporary file and return the error
		if retErr != nil {
			if err := fs.Remove(tmpPath); err != nil {
				retErr = fmt.Errorf("attempted to remove temporary file %q after error %v, but failed due to error: %v", path, retErr, err)
			}
			tmpPath = ""
		}
	}()

	// Name() will be an absolute path when using utilfs.DefaultFS, because ioutil.TempFile passes
	// an absolute path to os.Open, and we ensure similar behavior in utilfs.FakeFS for testing.
	tmpPath = tmpFile.Name()

	// write data
	if _, err := tmpFile.Write(data); err != nil {
		return tmpPath, err
	}
	// sync file, to ensure it's written in case a hard reset happens
	return tmpPath, tmpFile.Sync()
}

// ReplaceFile replaces the contents of the file at `path` with `data` by writing to a tmp file in the same
// dir as `path` and renaming the tmp file over `path`. The file does not have to exist to use ReplaceFile.
// Note ReplaceFile calls fsync.
func ReplaceFile(fs utilfs.Filesystem, path string, data []byte) error {
	// write data to a temporary file
	tmpPath, err := WriteTmpFile(fs, path, data)
	if err != nil {
		return err
	}
	// rename over existing file
	if err := fs.Rename(tmpPath, path); err != nil {
		return err
	}
	return nil
}

// DirExists returns true if a directory exists at `path`, false if `path` does not exist, otherwise an error
func DirExists(fs utilfs.Filesystem, path string) (bool, error) {
	if info, err := fs.Stat(path); err == nil {
		if info.IsDir() {
			return true, nil
		}
		return false, fmt.Errorf("expected dir at %q, but mode is is %q", path, info.Mode().String())
	} else if os.IsNotExist(err) {
		return false, nil
	} else {
		return false, err
	}
}

// EnsureDir ensures that a directory exists at `path`, and if it must create the directory any
// necessary parent directories will also be created and the new directory will be empty.
func EnsureDir(fs utilfs.Filesystem, path string) error {
	// if dir exists, don't change it, but do report any unexpected errors
	if ok, err := DirExists(fs, path); ok || err != nil {
		return err
	} // Assert: dir does not exist

	// create the dir
	if err := fs.MkdirAll(path, defaultPerm); err != nil {
		return err
	}
	return nil
}

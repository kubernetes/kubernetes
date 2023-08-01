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

const (
	defaultPerm = 0755
	tmpTag      = "tmp_" // additional prefix to prevent accidental collisions
)

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
// Expects the parent directory to exist.
func WriteTmpFile(fs utilfs.Filesystem, path string, data []byte) (tmpPath string, retErr error) {
	dir := filepath.Dir(path)
	prefix := tmpTag + filepath.Base(path)

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
				retErr = fmt.Errorf("attempted to remove temporary file %q after error %v, but failed due to error: %v", tmpPath, retErr, err)
			}
			tmpPath = ""
		}
	}()

	// Name() will be an absolute path when using utilfs.DefaultFS, because os.CreateTemp passes
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
// dir as `path` and renaming the tmp file over `path`. The file does not have to exist to use ReplaceFile,
// but the parent directory must exist.
// Note ReplaceFile calls fsync.
func ReplaceFile(fs utilfs.Filesystem, path string, data []byte) error {
	// write data to a temporary file
	tmpPath, err := WriteTmpFile(fs, path, data)
	if err != nil {
		return err
	}
	// rename over existing file
	return fs.Rename(tmpPath, path)
}

// DirExists returns true if a directory exists at `path`, false if `path` does not exist, otherwise an error
func DirExists(fs utilfs.Filesystem, path string) (bool, error) {
	if info, err := fs.Stat(path); err == nil {
		if info.IsDir() {
			return true, nil
		}
		return false, fmt.Errorf("expected dir at %q, but mode is %q", path, info.Mode().String())
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
	return fs.MkdirAll(path, defaultPerm)
}

// WriteTempDir creates a temporary dir at `path`, writes `files` into it, and fsyncs all the files
// The keys of `files` represent file names. These names must not:
// - be empty
// - be a path that contains more than the base name of a file (e.g. foo/bar is invalid, as is /bar)
// - match `.` or `..` exactly
// - be longer than 255 characters
// The above validation rules are based on atomic_writer.go, though in this case are more restrictive
// because we only allow a flat hierarchy.
func WriteTempDir(fs utilfs.Filesystem, path string, files map[string]string) (tmpPath string, retErr error) {
	// validate the filename keys; for now we only allow a flat keyset
	for name := range files {
		// invalidate empty names
		if name == "" {
			return "", fmt.Errorf("invalid file key: must not be empty: %q", name)
		}
		// invalidate: foo/bar and /bar
		if name != filepath.Base(name) {
			return "", fmt.Errorf("invalid file key %q, only base names are allowed", name)
		}
		// invalidate `.` and `..`
		if name == "." || name == ".." {
			return "", fmt.Errorf("invalid file key, may not be '.' or '..'")
		}
		// invalidate length > 255 characters
		if len(name) > 255 {
			return "", fmt.Errorf("invalid file key %q, must be less than 255 characters", name)
		}
	}

	// write the temp directory in parent dir and return path to the tmp directory
	dir := filepath.Dir(path)
	prefix := tmpTag + filepath.Base(path)

	// create the tmp dir
	var err error
	tmpPath, err = fs.TempDir(dir, prefix)
	if err != nil {
		return "", err
	}
	// be sure to clean up if there was an error
	defer func() {
		if retErr != nil {
			if err := fs.RemoveAll(tmpPath); err != nil {
				retErr = fmt.Errorf("attempted to remove temporary directory %q after error %v, but failed due to error: %v", tmpPath, retErr, err)
			}
		}
	}()
	// write data
	for name, data := range files {
		// create the file
		file, err := fs.Create(filepath.Join(tmpPath, name))
		if err != nil {
			return tmpPath, err
		}
		// be sure to close the file when we're done
		defer func() {
			// close the file when we're done, don't overwrite primary retErr if close fails
			if err := file.Close(); retErr == nil {
				retErr = err
			}
		}()
		// write the file
		if _, err := file.Write([]byte(data)); err != nil {
			return tmpPath, err
		}
		// sync the file, to ensure it's written in case a hard reset happens
		if err := file.Sync(); err != nil {
			return tmpPath, err
		}
	}
	return tmpPath, nil
}

// ReplaceDir replaces the contents of the dir at `path` with `files` by writing to a tmp dir in the same
// dir as `path` and renaming the tmp dir over `path`. The dir does not have to exist to use ReplaceDir.
func ReplaceDir(fs utilfs.Filesystem, path string, files map[string]string) error {
	// write data to a temporary directory
	tmpPath, err := WriteTempDir(fs, path, files)
	if err != nil {
		return err
	}
	// rename over target directory
	return fs.Rename(tmpPath, path)
}

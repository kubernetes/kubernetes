/*
Copyright 2015 The Kubernetes Authors.

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

package container

import (
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"syscall"
	"time"

	"k8s.io/kubernetes/pkg/util/mount"
)

// OSInterface collects system level operations that need to be mocked out
// during tests.
type OSInterface interface {
	MkdirAll(path string, perm os.FileMode) error
	Symlink(oldname string, newname string) error
	Stat(path string) (os.FileInfo, error)
	Remove(path string) error
	RemoveAll(path string) error
	Create(path string) (*os.File, error)
	Chmod(path string, perm os.FileMode) error
	Hostname() (name string, err error)
	Chtimes(path string, atime time.Time, mtime time.Time) error
	Pipe() (r *os.File, w *os.File, err error)
	ReadDir(dirname string) ([]os.FileInfo, error)
	Glob(pattern string) ([]string, error)
}

// RealOS is used to dispatch the real system level operations.
type RealOS struct{}

// MkdirAll will call os.MkdirAll to create a directory.
func (RealOS) MkdirAll(path string, perm os.FileMode) error {
	return os.MkdirAll(path, perm)
}

// Symlink will call os.Symlink to create a symbolic link.
func (RealOS) Symlink(oldname string, newname string) error {
	return os.Symlink(oldname, newname)
}

// Stat will call os.Stat to get the FileInfo for a given path
func (RealOS) Stat(path string) (os.FileInfo, error) {
	return os.Stat(path)
}

// Remove will call os.Remove to remove the path.
func (RealOS) Remove(path string) error {
	return os.Remove(path)
}

// RemoveAll will call os.RemoveAll to remove the path and its children.
func (RealOS) RemoveAll(path string) error {
	return os.RemoveAll(path)
}

// Create will call os.Create to create and return a file
// at path.
func (RealOS) Create(path string) (*os.File, error) {
	return os.Create(path)
}

// Chmod will change the permissions on the specified path or return
// an error.
func (RealOS) Chmod(path string, perm os.FileMode) error {
	return os.Chmod(path, perm)
}

// Hostname will call os.Hostname to return the hostname.
func (RealOS) Hostname() (name string, err error) {
	return os.Hostname()
}

// Chtimes will call os.Chtimes to change the atime and mtime of the path
func (RealOS) Chtimes(path string, atime time.Time, mtime time.Time) error {
	return os.Chtimes(path, atime, mtime)
}

// Pipe will call os.Pipe to return a connected pair of pipe.
func (RealOS) Pipe() (r *os.File, w *os.File, err error) {
	return os.Pipe()
}

// ReadDir will call ioutil.ReadDir to return the files under the directory.
func (RealOS) ReadDir(dirname string) ([]os.FileInfo, error) {
	return ioutil.ReadDir(dirname)
}

// Glob will call filepath.Glob to return the names of all files matching
// pattern.
func (RealOS) Glob(pattern string) ([]string, error) {
	return filepath.Glob(pattern)
}

// RemoveAllOneFilesystem removes path and any children it contains.
// It removes everything it can but returns the first error
// it encounters. If the path does not exist, RemoveAll
// returns nil (no error).
// It makes sure it does not cross mount boundary, i.e. it does *not* remove
// files from another filesystems. Like 'rm -rf --one-file-system'.
// It is copied from RemoveAll() sources, with IsLikelyNotMountPoint
func RemoveAllOneFilesystem(mounter mount.Interface, path string) error {
	// Simple case: if Remove works, we're done.
	err := os.Remove(path)
	if err == nil || os.IsNotExist(err) {
		return nil
	}

	// Otherwise, is this a directory we need to recurse into?
	dir, serr := os.Lstat(path)
	if serr != nil {
		if serr, ok := serr.(*os.PathError); ok && (os.IsNotExist(serr.Err) || serr.Err == syscall.ENOTDIR) {
			return nil
		}
		return serr
	}
	if !dir.IsDir() {
		// Not a directory; return the error from Remove.
		return err
	}

	// Directory.
	isNotMount, err := mounter.IsLikelyNotMountPoint(path)
	if err != nil {
		return err
	}
	if !isNotMount {
		return fmt.Errorf("cannot delete directory %s: it is a mount point", path)
	}

	fd, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			// Race. It was deleted between the Lstat and Open.
			// Return nil per RemoveAll's docs.
			return nil
		}
		return err
	}

	// Remove contents & return first error.
	err = nil
	for {
		names, err1 := fd.Readdirnames(100)
		for _, name := range names {
			err1 := RemoveAllOneFilesystem(mounter, path+string(os.PathSeparator)+name)
			if err == nil {
				err = err1
			}
		}
		if err1 == io.EOF {
			break
		}
		// If Readdirnames returned an error, use it.
		if err == nil {
			err = err1
		}
		if len(names) == 0 {
			break
		}
	}

	// Close directory, because windows won't remove opened directory.
	fd.Close()

	// Remove directory.
	err1 := os.Remove(path)
	if err1 == nil || os.IsNotExist(err1) {
		return nil
	}
	if err == nil {
		err = err1
	}
	return err
}

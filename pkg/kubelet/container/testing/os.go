/*
Copyright 2016 The Kubernetes Authors.

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

package testing

import (
	"errors"
	"os"
	"time"
)

// FakeOS mocks out certain OS calls to avoid perturbing the filesystem
// If a member of the form `*Fn` is set, that function will be called in place
// of the real call.
type FakeOS struct {
	StatFn     func(string) (os.FileInfo, error)
	ReadDirFn  func(string) ([]os.FileInfo, error)
	MkdirAllFn func(string, os.FileMode) error
	SymlinkFn  func(string, string) error
	GlobFn     func(string, string) bool
	HostName   string
	Removes    []string
	Files      map[string][]*os.FileInfo
}

func NewFakeOS() *FakeOS {
	return &FakeOS{
		Removes: []string{},
		Files:   make(map[string][]*os.FileInfo),
	}
}

// Mkdir is a fake call that just returns nil.
func (f *FakeOS) MkdirAll(path string, perm os.FileMode) error {
	if f.MkdirAllFn != nil {
		return f.MkdirAllFn(path, perm)
	}
	return nil
}

// Symlink is a fake call that just returns nil.
func (f *FakeOS) Symlink(oldname string, newname string) error {
	if f.SymlinkFn != nil {
		return f.SymlinkFn(oldname, newname)
	}
	return nil
}

// Stat is a fake that returns an error
func (f FakeOS) Stat(path string) (os.FileInfo, error) {
	if f.StatFn != nil {
		return f.StatFn(path)
	}
	return nil, errors.New("unimplemented testing mock")
}

// Remove is a fake call that returns nil.
func (f *FakeOS) Remove(path string) error {
	f.Removes = append(f.Removes, path)
	return nil
}

// RemoveAll is a fake call that just returns nil.
func (f *FakeOS) RemoveAll(path string) error {
	f.Removes = append(f.Removes, path)
	return nil
}

// Create is a fake call that creates a virtual file and returns nil.
func (f *FakeOS) Create(path string) (*os.File, error) {
	if f.Files == nil {
		f.Files = make(map[string][]*os.FileInfo)
	}
	f.Files[path] = []*os.FileInfo{}
	return nil, nil
}

// Chmod is a fake call that returns nil.
func (FakeOS) Chmod(path string, perm os.FileMode) error {
	return nil
}

// Hostname is a fake call that returns nil.
func (f *FakeOS) Hostname() (name string, err error) {
	return f.HostName, nil
}

// Chtimes is a fake call that returns nil.
func (FakeOS) Chtimes(path string, atime time.Time, mtime time.Time) error {
	return nil
}

// Pipe is a fake call that returns nil.
func (FakeOS) Pipe() (r *os.File, w *os.File, err error) {
	return nil, nil, nil
}

// ReadDir is a fake call that returns the files under the directory.
func (f *FakeOS) ReadDir(dirname string) ([]os.FileInfo, error) {
	if f.ReadDirFn != nil {
		return f.ReadDirFn(dirname)
	}
	return nil, nil
}

// Glob is a fake call that returns list of virtual files matching a pattern.
func (f *FakeOS) Glob(pattern string) ([]string, error) {
	if f.GlobFn != nil {
		var res []string
		for k := range f.Files {
			if f.GlobFn(pattern, k) {
				res = append(res, k)
			}
		}
		return res, nil
	}
	return nil, nil
}

// Open is a fake call that returns nil.
func (FakeOS) Open(name string) (*os.File, error) {
	return nil, nil
}

// OpenFile is a fake call that return nil.
func (FakeOS) OpenFile(name string, flag int, perm os.FileMode) (*os.File, error) {
	return nil, nil
}

// Rename is a fake call that return nil.
func (FakeOS) Rename(oldpath, newpath string) error {
	return nil
}

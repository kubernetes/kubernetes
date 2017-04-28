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

package filesystem

import (
	"io/ioutil"
	"os"
	"time"
)

// DefaultFs implements Filesystem using same-named functions from "os" and "io/ioutil"
type DefaultFs struct{}

// Stat via os.Stat
func (DefaultFs) Stat(name string) (os.FileInfo, error) {
	return os.Stat(name)
}

// Create via os.Create
func (DefaultFs) Create(name string) (File, error) {
	file, err := os.Create(name)
	if err != nil {
		return nil, err
	}
	return &defaultFile{file}, nil
}

// Rename via os.Rename
func (DefaultFs) Rename(oldpath, newpath string) error {
	return os.Rename(oldpath, newpath)
}

// MkdirAll via os.MkdirAll
func (DefaultFs) MkdirAll(path string, perm os.FileMode) error {
	return os.MkdirAll(path, perm)
}

// Chtimes via os.Chtimes
func (DefaultFs) Chtimes(name string, atime time.Time, mtime time.Time) error {
	return os.Chtimes(name, atime, mtime)
}

// ReadFile via os.ReadFile
func (DefaultFs) ReadFile(filename string) ([]byte, error) {
	return ioutil.ReadFile(filename)
}

// TempFile via os.TempFile
func (DefaultFs) TempFile(dir, prefix string) (File, error) {
	file, err := ioutil.TempFile(dir, prefix)
	if err != nil {
		return nil, err
	}
	return &defaultFile{file}, nil
}

// defaultFile implements File using same-named functions from "os"
type defaultFile struct {
	file *os.File
}

// Name via os.File.Name
func (file *defaultFile) Name() string {
	return file.file.Name()
}

// Write via os.File.Write
func (file *defaultFile) Write(b []byte) (n int, err error) {
	return file.file.Write(b)
}

// Close via os.File.Close
func (file *defaultFile) Close() error {
	return file.file.Close()
}

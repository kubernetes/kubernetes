/*
Copyright 2014 The Camlistore Authors

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

// Package wkfs implements the pluggable "well-known filesystem" abstraction layer.
//
// Instead of accessing files directly through the operating system
// using os.Open or os.Stat, code should use wkfs.Open or wkfs.Stat,
// which first try to intercept paths at well-known top-level
// directories representing previously-registered mount types,
// otherwise fall through to the operating system paths.
//
// Example of top-level well-known directories that might be
// registered include /gcs/bucket/object for Google Cloud Storage or
// /s3/bucket/object for AWS S3.
package wkfs // import "go4.org/wkfs"

import (
	"io"
	"io/ioutil"
	"os"
	"strings"
)

type File interface {
	io.Reader
	io.ReaderAt
	io.Closer
	io.Seeker
	Name() string
	Stat() (os.FileInfo, error)
}

type FileWriter interface {
	io.Writer
	io.Closer
}

func Open(name string) (File, error)               { return fs(name).Open(name) }
func Stat(name string) (os.FileInfo, error)        { return fs(name).Stat(name) }
func Lstat(name string) (os.FileInfo, error)       { return fs(name).Lstat(name) }
func MkdirAll(path string, perm os.FileMode) error { return fs(path).MkdirAll(path, perm) }
func OpenFile(name string, flag int, perm os.FileMode) (FileWriter, error) {
	return fs(name).OpenFile(name, flag, perm)
}
func Create(name string) (FileWriter, error) {
	// like os.Create but WRONLY instead of RDWR because we don't
	// expose a Reader here.
	return OpenFile(name, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
}

func fs(name string) FileSystem {
	for pfx, fs := range wkFS {
		if strings.HasPrefix(name, pfx) {
			return fs
		}
	}
	return osFS{}
}

type osFS struct{}

func (osFS) Open(name string) (File, error)               { return os.Open(name) }
func (osFS) Stat(name string) (os.FileInfo, error)        { return os.Stat(name) }
func (osFS) Lstat(name string) (os.FileInfo, error)       { return os.Lstat(name) }
func (osFS) MkdirAll(path string, perm os.FileMode) error { return os.MkdirAll(path, perm) }
func (osFS) OpenFile(name string, flag int, perm os.FileMode) (FileWriter, error) {
	return os.OpenFile(name, flag, perm)
}

type FileSystem interface {
	Open(name string) (File, error)
	OpenFile(name string, flag int, perm os.FileMode) (FileWriter, error)
	Stat(name string) (os.FileInfo, error)
	Lstat(name string) (os.FileInfo, error)
	MkdirAll(path string, perm os.FileMode) error
}

// well-known filesystems
var wkFS = map[string]FileSystem{}

// RegisterFS registers a well-known filesystem. It intercepts
// anything beginning with prefix (which must start and end with a
// forward slash) and forwards it to fs.
func RegisterFS(prefix string, fs FileSystem) {
	if !strings.HasPrefix(prefix, "/") || !strings.HasSuffix(prefix, "/") {
		panic("bogus prefix: " + prefix)
	}
	if _, dup := wkFS[prefix]; dup {
		panic("duplication registration of " + prefix)
	}
	wkFS[prefix] = fs
}

// WriteFile writes data to a file named by filename.
// If the file does not exist, WriteFile creates it with permissions perm;
// otherwise WriteFile truncates it before writing.
func WriteFile(filename string, data []byte, perm os.FileMode) error {
	f, err := OpenFile(filename, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, perm)
	if err != nil {
		return err
	}
	n, err := f.Write(data)
	if err == nil && n < len(data) {
		err = io.ErrShortWrite
	}
	if err1 := f.Close(); err == nil {
		err = err1
	}
	return err
}

func ReadFile(filename string) ([]byte, error) {
	f, err := Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return ioutil.ReadAll(f)
}

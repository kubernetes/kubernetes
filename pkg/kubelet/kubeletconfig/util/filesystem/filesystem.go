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
	"os"
	"time"
)

// Filesystem is an interface that we can use to mock various filesystem operations
type Filesystem interface {
	// from "os"
	Stat(name string) (os.FileInfo, error)
	Create(name string) (File, error)
	Rename(oldpath, newpath string) error
	MkdirAll(path string, perm os.FileMode) error
	Chtimes(name string, atime time.Time, mtime time.Time) error

	// from "io/ioutil"
	ReadFile(filename string) ([]byte, error)
	TempFile(dir, prefix string) (File, error)
}

// File is an interface that we can use to mock various filesystem operations typically
// accessed through the File object from the "os" package
type File interface {
	// for now, the only os.File methods used are those below, add more as necessary
	Name() string
	Write(b []byte) (n int, err error)
	Close() error
}

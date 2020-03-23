/*
Copyright 2018 The Kubernetes Authors.

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

package fs

import (
	"os"
	"time"
)

var _ os.FileInfo = &Fakefileinfo{}

// Fakefileinfo implements Fakefileinfo using a fake in-memory filesystem.
type Fakefileinfo struct {
	*FakeFile
}

// Name returns the name of the file
func (fi *Fakefileinfo) Name() string { return fi.name }

// Size returns the size of the file
func (fi *Fakefileinfo) Size() int64 { return int64(len(fi.content)) }

// Mode returns the file mode
func (fi *Fakefileinfo) Mode() os.FileMode { return 0777 }

// ModTime returns the modification time
func (fi *Fakefileinfo) ModTime() time.Time { return time.Time{} }

// IsDir returns if it is a directory
func (fi *Fakefileinfo) IsDir() bool { return fi.dir }

// Sys should return underlying data source, but it now returns nil
func (fi *Fakefileinfo) Sys() interface{} { return nil }

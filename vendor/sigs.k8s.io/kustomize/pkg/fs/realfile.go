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
)

var _ File = &realFile{}

// realFile implements File using the local filesystem.
type realFile struct {
	file *os.File
}

// Close closes a file.
func (f *realFile) Close() error { return f.file.Close() }

// Read reads a file's content.
func (f *realFile) Read(p []byte) (n int, err error) { return f.file.Read(p) }

// Write writes bytes to a file
func (f *realFile) Write(p []byte) (n int, err error) { return f.file.Write(p) }

// Stat returns an interface which has all the information regarding the file.
func (f *realFile) Stat() (os.FileInfo, error) { return f.file.Stat() }

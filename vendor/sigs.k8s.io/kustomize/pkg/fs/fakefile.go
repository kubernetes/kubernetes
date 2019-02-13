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
	"bytes"
	"os"
)

var _ File = &FakeFile{}

// FakeFile implements File in-memory for tests.
type FakeFile struct {
	name    string
	content []byte
	dir     bool
	open    bool
}

// makeDir makes a fake directory.
func makeDir(name string) *FakeFile {
	return &FakeFile{name: name, dir: true}
}

// Close marks the fake file closed.
func (f *FakeFile) Close() error {
	f.open = false
	return nil
}

// Read never fails, and doesn't mutate p.
func (f *FakeFile) Read(p []byte) (n int, err error) {
	return len(p), nil
}

// Write saves the contents of the argument to memory.
func (f *FakeFile) Write(p []byte) (n int, err error) {
	f.content = p
	return len(p), nil
}

// ContentMatches returns true if v matches fake file's content.
func (f *FakeFile) ContentMatches(v []byte) bool {
	return bytes.Equal(v, f.content)
}

// GetContent the content of a fake file.
func (f *FakeFile) GetContent() []byte {
	return f.content
}

// Stat returns nil.
func (f *FakeFile) Stat() (os.FileInfo, error) {
	return nil, nil
}

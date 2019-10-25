// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package filesys

import (
	"bytes"
	"os"
)

var _ File = &fileInMemory{}

// fileInMemory implements File in-memory for tests.
type fileInMemory struct {
	name    string
	content []byte
	dir     bool
	open    bool
}

// makeDir makes a fake directory.
func makeDir(name string) *fileInMemory {
	return &fileInMemory{name: name, dir: true}
}

// Close marks the fake file closed.
func (f *fileInMemory) Close() error {
	f.open = false
	return nil
}

// Read never fails, and doesn't mutate p.
func (f *fileInMemory) Read(p []byte) (n int, err error) {
	return len(p), nil
}

// Write saves the contents of the argument to memory.
func (f *fileInMemory) Write(p []byte) (n int, err error) {
	f.content = p
	return len(p), nil
}

// ContentMatches returns true if v matches fake file's content.
func (f *fileInMemory) ContentMatches(v []byte) bool {
	return bytes.Equal(v, f.content)
}

// GetContent the content of a fake file.
func (f *fileInMemory) GetContent() []byte {
	return f.content
}

// Stat returns nil.
func (f *fileInMemory) Stat() (os.FileInfo, error) {
	return nil, nil
}

// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package filesys

import (
	"os"
)

var _ File = &fileOnDisk{}

// fileOnDisk implements File using the local filesystem.
type fileOnDisk struct {
	file *os.File
}

// Close closes a file.
func (f *fileOnDisk) Close() error { return f.file.Close() }

// Read reads a file's content.
func (f *fileOnDisk) Read(p []byte) (n int, err error) { return f.file.Read(p) }

// Write writes bytes to a file
func (f *fileOnDisk) Write(p []byte) (n int, err error) { return f.file.Write(p) }

// Stat returns an interface which has all the information regarding the file.
func (f *fileOnDisk) Stat() (os.FileInfo, error) { return f.file.Stat() }

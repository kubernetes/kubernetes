// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package filesys

import (
	"io"
	"os"
	"time"
)

var _ os.FileInfo = &fileInfo{}

// fileInfo implements os.FileInfo for a fileInMemory instance.
type fileInfo struct {
	*fileInMemory
}

// Name returns the name of the file
func (fi *fileInfo) Name() string { return fi.name }

// Size returns the size of the file
func (fi *fileInfo) Size() int64 { return int64(len(fi.content)) }

// Mode returns the file mode
func (fi *fileInfo) Mode() os.FileMode { return 0777 }

// ModTime returns the modification time
func (fi *fileInfo) ModTime() time.Time { return time.Time{} }

// IsDir returns if it is a directory
func (fi *fileInfo) IsDir() bool { return fi.dir }

// Sys should return underlying data source, but it now returns nil
func (fi *fileInfo) Sys() interface{} { return nil }

// File groups the basic os.File methods.
type File interface {
	io.ReadWriteCloser
	Stat() (os.FileInfo, error)
}

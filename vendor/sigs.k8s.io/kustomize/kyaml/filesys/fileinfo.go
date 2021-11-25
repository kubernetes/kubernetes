// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package filesys

import (
	"os"
	"time"
)

var _ os.FileInfo = fileInfo{}

// fileInfo implements os.FileInfo for a fileInMemory instance.
type fileInfo struct {
	node *fsNode
}

// Name returns the name of the file
func (fi fileInfo) Name() string { return fi.node.Name() }

// Size returns the size of the file
func (fi fileInfo) Size() int64 { return fi.node.Size() }

// Mode returns the file mode
func (fi fileInfo) Mode() os.FileMode { return 0777 }

// ModTime returns a bogus time
func (fi fileInfo) ModTime() time.Time { return time.Time{} }

// IsDir returns true if it is a directory
func (fi fileInfo) IsDir() bool { return fi.node.isNodeADir() }

// Sys should return underlying data source, but it now returns nil
func (fi fileInfo) Sys() interface{} { return nil }

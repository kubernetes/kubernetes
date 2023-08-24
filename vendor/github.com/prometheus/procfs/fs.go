// Copyright 2018 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package procfs

import (
	"github.com/prometheus/procfs/internal/fs"
)

// FS represents the pseudo-filesystem sys, which provides an interface to
// kernel data structures.
type FS struct {
	proc fs.FS
	real bool
}

// DefaultMountPoint is the common mount point of the proc filesystem.
const DefaultMountPoint = fs.DefaultProcMountPoint

// NewDefaultFS returns a new proc FS mounted under the default proc mountPoint.
// It will error if the mount point directory can't be read or is a file.
func NewDefaultFS() (FS, error) {
	return NewFS(DefaultMountPoint)
}

// NewFS returns a new proc FS mounted under the given proc mountPoint. It will error
// if the mount point directory can't be read or is a file.
func NewFS(mountPoint string) (FS, error) {
	fs, err := fs.NewFS(mountPoint)
	if err != nil {
		return FS{}, err
	}

	real, err := isRealProc(mountPoint)
	if err != nil {
		return FS{}, err
	}

	return FS{fs, real}, nil
}

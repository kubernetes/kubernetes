// Copyright 2017 The Prometheus Authors
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

package sysfs

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/prometheus/procfs/xfs"
)

// FS represents the pseudo-filesystem sys, which provides an interface to
// kernel data structures.
type FS string

// DefaultMountPoint is the common mount point of the sys filesystem.
const DefaultMountPoint = "/sys"

// NewFS returns a new FS mounted under the given mountPoint. It will error
// if the mount point can't be read.
func NewFS(mountPoint string) (FS, error) {
	info, err := os.Stat(mountPoint)
	if err != nil {
		return "", fmt.Errorf("could not read %s: %s", mountPoint, err)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("mount point %s is not a directory", mountPoint)
	}

	return FS(mountPoint), nil
}

// Path returns the path of the given subsystem relative to the sys root.
func (fs FS) Path(p ...string) string {
	return filepath.Join(append([]string{string(fs)}, p...)...)
}

// XFSStats retrieves XFS filesystem runtime statistics for each mounted XFS
// filesystem.  Only available on kernel 4.4+.  On older kernels, an empty
// slice of *xfs.Stats will be returned.
func (fs FS) XFSStats() ([]*xfs.Stats, error) {
	matches, err := filepath.Glob(fs.Path("fs/xfs/*/stats/stats"))
	if err != nil {
		return nil, err
	}

	stats := make([]*xfs.Stats, 0, len(matches))
	for _, m := range matches {
		f, err := os.Open(m)
		if err != nil {
			return nil, err
		}

		// "*" used in glob above indicates the name of the filesystem.
		name := filepath.Base(filepath.Dir(filepath.Dir(m)))

		// File must be closed after parsing, regardless of success or
		// failure.  Defer is not used because of the loop.
		s, err := xfs.ParseStats(f)
		_ = f.Close()
		if err != nil {
			return nil, err
		}

		s.Name = name
		stats = append(stats, s)
	}

	return stats, nil
}

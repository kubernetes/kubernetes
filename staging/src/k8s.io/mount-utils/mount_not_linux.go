//go:build !linux
// +build !linux

/*
Copyright 2021 The Kubernetes Authors.

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

package mount

import (
	"os"
	"path/filepath"
)

// IsNotMountPoint determines if a directory is a mountpoint.
// It should return ErrNotExist when the directory does not exist.
// IsNotMountPoint is more expensive than IsLikelyNotMountPoint.
// IsNotMountPoint detects bind mounts in linux.
// IsNotMountPoint enumerates all the mountpoints using List() and
// the list of mountpoints may be large, then it uses
// isMountPointMatch to evaluate whether the directory is a mountpoint.
func IsNotMountPoint(mounter Interface, file string) (bool, error) {
	// IsLikelyNotMountPoint provides a quick check
	// to determine whether file IS A mountpoint.
	notMnt, notMntErr := mounter.IsLikelyNotMountPoint(file)
	if notMntErr != nil && os.IsPermission(notMntErr) {
		// We were not allowed to do the simple stat() check, e.g. on NFS with
		// root_squash. Fall back to /proc/mounts check below.
		notMnt = true
		notMntErr = nil
	}
	if notMntErr != nil {
		return notMnt, notMntErr
	}
	// identified as mountpoint, so return this fact.
	if notMnt == false {
		return notMnt, nil
	}

	// Resolve any symlinks in file, kernel would do the same and use the resolved path in /proc/mounts.
	resolvedFile, err := filepath.EvalSymlinks(file)
	if err != nil {
		return true, err
	}

	// check all mountpoints since IsLikelyNotMountPoint
	// is not reliable for some mountpoint types.
	mountPoints, mountPointsErr := mounter.List()
	if mountPointsErr != nil {
		return notMnt, mountPointsErr
	}
	for _, mp := range mountPoints {
		if isMountPointMatch(mp, resolvedFile) {
			notMnt = false
			break
		}
	}
	return notMnt, nil
}

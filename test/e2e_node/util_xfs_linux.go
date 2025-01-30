//go:build linux
// +build linux

/*
Copyright 2016 The Kubernetes Authors.

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

package e2enode

import (
	"path/filepath"
	"syscall"

	"k8s.io/mount-utils"
)

func detectMountpoint(m mount.Interface, path string) string {
	path, err := filepath.Abs(path)
	if err == nil {
		path, err = filepath.EvalSymlinks(path)
	}
	if err != nil {
		return ""
	}
	for path != "" && path != "/" {
		isNotMount, err := m.IsLikelyNotMountPoint(path)
		if err != nil {
			return ""
		}
		if !isNotMount {
			return path
		}
		path = filepath.Dir(path)
	}
	return "/"
}

const (
	xfsMagic = 0x58465342
)

// XFS over-allocates and then eventually removes that excess allocation.
// That can lead to a file growing beyond its eventual size, causing
// an unnecessary eviction:
//
// % ls -ls
// total 32704
// 32704 -rw-r--r-- 1 rkrawitz rkrawitz 20971520 Jan 15 13:16 foo.bin
//
// This issue can be hit regardless of the means used to count storage.
// It is not present in ext4fs.
func isXfs(dir string) bool {
	mountpoint := detectMountpoint(mount.New(""), dir)
	if mountpoint == "" {
		return false
	}
	var buf syscall.Statfs_t
	err := syscall.Statfs(mountpoint, &buf)
	if err != nil {
		return false
	}
	return buf.Type == xfsMagic
}

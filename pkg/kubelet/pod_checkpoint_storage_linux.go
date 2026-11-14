//go:build linux

/*
Copyright The Kubernetes Authors.

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

package kubelet

import (
	"fmt"
	"golang.org/x/sys/unix"
	"os"
	"path/filepath"
)

func inspectPodCheckpointStorage(path string) (podCheckpointStorageInfo, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return podCheckpointStorageInfo{}, err
	}
	if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return podCheckpointStorageInfo{}, fmt.Errorf("checkpoint storage %q is not a real directory", path)
	}
	var storage, parent unix.Stat_t
	if err := unix.Stat(path, &storage); err != nil {
		return podCheckpointStorageInfo{}, err
	}
	if err := unix.Stat(filepath.Dir(path), &parent); err != nil {
		return podCheckpointStorageInfo{}, err
	}
	var fs unix.Statfs_t
	if err := unix.Statfs(path, &fs); err != nil {
		return podCheckpointStorageInfo{}, err
	}
	if fs.Type != unix.EXT4_SUPER_MAGIC {
		return podCheckpointStorageInfo{}, fmt.Errorf("checkpoint storage %q requires a fixed-capacity ext4 filesystem with a fixed inode table", path)
	}
	// Requiring a different device excludes a directory or bind mount of the
	// kubelet filesystem. Dynamic inode filesystems cannot provide an inode cap.
	if fs.Bsize <= 0 || fs.Blocks > ^uint64(0)/uint64(fs.Bsize) {
		return podCheckpointStorageInfo{}, fmt.Errorf("invalid checkpoint filesystem capacity")
	}
	return podCheckpointStorageInfo{
		CapacityBytes: fs.Blocks * uint64(fs.Bsize), AvailableBytes: fs.Bavail * uint64(fs.Bsize),
		CapacityInodes: fs.Files, AvailableInodes: fs.Ffree, Isolated: storage.Dev != parent.Dev,
	}, nil
}

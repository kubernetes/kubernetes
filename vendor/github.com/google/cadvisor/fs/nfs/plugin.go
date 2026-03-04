// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build linux

package nfs

import (
	"fmt"
	"strings"

	"github.com/google/cadvisor/fs"
	"github.com/google/cadvisor/fs/vfs"

	mount "github.com/moby/sys/mountinfo"
	"k8s.io/klog/v2"
)

type nfsPlugin struct{}

// Ensure nfsPlugin implements FsCachingPlugin
var _ fs.FsCachingPlugin = &nfsPlugin{}

// NewPlugin creates a new NFS filesystem plugin.
func NewPlugin() fs.FsPlugin {
	return &nfsPlugin{}
}

func (p *nfsPlugin) Name() string {
	return "nfs"
}

// CanHandle returns true if the filesystem type is NFS (nfs, nfs3, nfs4, etc.).
func (p *nfsPlugin) CanHandle(fsType string) bool {
	return strings.HasPrefix(fsType, "nfs")
}

// Priority returns 50 - NFS has medium priority (higher than VFS but lower than specific plugins).
func (p *nfsPlugin) Priority() int {
	return 50
}

// GetStats returns filesystem statistics for NFS.
// NFS uses VFS stats.
func (p *nfsPlugin) GetStats(device string, partition fs.PartitionInfo) (*fs.FsStats, error) {
	capacity, free, avail, inodes, inodesFree, err := vfs.GetVfsStats(partition.Mountpoint)
	if err != nil {
		klog.V(4).Infof("the file system type is %s, partition mountpoint does not exist: %v, error: %v",
			partition.FsType, partition.Mountpoint, err)
		return nil, err
	}

	return &fs.FsStats{
		Capacity:   capacity,
		Free:       free,
		Available:  avail,
		Inodes:     &inodes,
		InodesFree: &inodesFree,
		Type:       fs.VFS,
	}, nil
}

// ProcessMount handles NFS mount processing.
// For NFS, no special processing is needed.
func (p *nfsPlugin) ProcessMount(mnt *mount.Info) (bool, *mount.Info, error) {
	return true, mnt, nil
}

// CacheKey returns a cache key based on device ID (major:minor).
// NFS mounts with the same device ID share the same underlying filesystem,
// so we can cache stats to avoid redundant statfs calls.
func (p *nfsPlugin) CacheKey(partition fs.PartitionInfo) string {
	return fmt.Sprintf("%d:%d", partition.Major, partition.Minor)
}

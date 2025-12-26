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

package tmpfs

import (
	"github.com/google/cadvisor/fs"
	"github.com/google/cadvisor/fs/vfs"

	mount "github.com/moby/sys/mountinfo"
)

type tmpfsPlugin struct{}

// NewPlugin creates a new tmpfs filesystem plugin.
func NewPlugin() fs.FsPlugin {
	return &tmpfsPlugin{}
}

func (p *tmpfsPlugin) Name() string {
	return "tmpfs"
}

// CanHandle returns true if the filesystem type is tmpfs.
func (p *tmpfsPlugin) CanHandle(fsType string) bool {
	return fsType == "tmpfs"
}

// Priority returns 100 - tmpfs has higher priority than VFS.
func (p *tmpfsPlugin) Priority() int {
	return 100
}

// GetStats returns filesystem statistics for tmpfs.
// tmpfs delegates to VFS for stats collection.
func (p *tmpfsPlugin) GetStats(device string, partition fs.PartitionInfo) (*fs.FsStats, error) {
	// tmpfs uses VFS stats
	capacity, free, avail, inodes, inodesFree, err := vfs.GetVfsStats(partition.Mountpoint)
	if err != nil {
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

// ProcessMount handles tmpfs mount processing.
// For tmpfs, we use the mountpoint as the source to make each mount unique.
// This allows multiple tmpfs mounts with the same "tmpfs" source to coexist.
func (p *tmpfsPlugin) ProcessMount(mnt *mount.Info) (bool, *mount.Info, error) {
	// Use mountpoint as source to make each tmpfs mount unique
	correctedMnt := *mnt
	correctedMnt.Source = mnt.Mountpoint
	return true, &correctedMnt, nil
}

// AllowDuplicateSource returns true for tmpfs since multiple tmpfs mounts
// should be tracked separately even if they appear to have the same source.
func (p *tmpfsPlugin) AllowDuplicateSource() bool {
	return true
}

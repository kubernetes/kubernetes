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

package btrfs

import (
	"strings"

	"github.com/google/cadvisor/fs"
	"github.com/google/cadvisor/fs/vfs"

	mount "github.com/moby/sys/mountinfo"
	"k8s.io/klog/v2"
)

type btrfsPlugin struct{}

// NewPlugin creates a new Btrfs filesystem plugin.
func NewPlugin() fs.FsPlugin {
	return &btrfsPlugin{}
}

func (p *btrfsPlugin) Name() string {
	return "btrfs"
}

// CanHandle returns true if the filesystem type is btrfs.
func (p *btrfsPlugin) CanHandle(fsType string) bool {
	return fsType == "btrfs"
}

// Priority returns 100 - Btrfs has higher priority than VFS.
func (p *btrfsPlugin) Priority() int {
	return 100
}

// GetStats returns filesystem statistics for Btrfs.
// Btrfs delegates to VFS for stats collection.
func (p *btrfsPlugin) GetStats(device string, partition fs.PartitionInfo) (*fs.FsStats, error) {
	// Btrfs uses VFS stats
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

// ProcessMount handles Btrfs mount processing.
// Btrfs fix: following workaround fixes wrong btrfs Major and Minor Ids reported in /proc/self/mountinfo.
// Instead of using values from /proc/self/mountinfo we use stat to get Ids from btrfs mount point.
func (p *btrfsPlugin) ProcessMount(mnt *mount.Info) (bool, *mount.Info, error) {
	// Only apply fix if Major is 0 and Source starts with /dev/
	if mnt.Major == 0 && strings.HasPrefix(mnt.Source, "/dev/") {
		major, minor, err := GetBtrfsMajorMinorIds(mnt)
		if err != nil {
			klog.Warningf("%s", err)
		} else {
			// Create a copy with corrected values
			correctedMnt := *mnt
			correctedMnt.Major = major
			correctedMnt.Minor = minor
			return true, &correctedMnt, nil
		}
	}
	return true, mnt, nil
}

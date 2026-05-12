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

package vfs

import (
	"strings"

	"github.com/google/cadvisor/fs"
	"github.com/google/cadvisor/utils"

	mount "github.com/moby/sys/mountinfo"
	"k8s.io/klog/v2"
)

type vfsPlugin struct{}

// NewPlugin creates a new VFS filesystem plugin.
func NewPlugin() fs.FsPlugin {
	return &vfsPlugin{}
}

func (p *vfsPlugin) Name() string {
	return "vfs"
}

// CanHandle returns true for standard filesystems that use VFS stats.
// This includes ext2/3/4, xfs, and similar block-based filesystems.
// Virtual/pseudo filesystems (proc, sysfs, cgroup, etc.) are excluded.
func (p *vfsPlugin) CanHandle(fsType string) bool {
	// Exclude virtual/pseudo filesystems that don't have real disk backing
	switch fsType {
	case "cgroup", "cgroup2", "cpuset", "mqueue", "proc", "sysfs",
		"devtmpfs", "devpts", "securityfs", "debugfs", "tracefs",
		"pstore", "configfs", "fusectl", "hugetlbfs", "autofs",
		"binfmt_misc", "efivarfs", "rpc_pipefs", "nsfs":
		return false
	}

	// VFS can handle most standard Linux filesystems
	if strings.HasPrefix(fsType, "ext") {
		return true
	}
	switch fsType {
	case "xfs", "squashfs", "f2fs", "jfs", "reiserfs", "hfs", "hfsplus",
		"ntfs", "vfat", "fat", "msdos", "exfat", "udf", "iso9660":
		return true
	}
	// Don't act as a general fallback - only handle known filesystem types
	return false
}

// Priority returns 0 - VFS is the lowest priority fallback plugin.
func (p *vfsPlugin) Priority() int {
	return 0
}

// GetStats returns filesystem statistics using the statfs syscall.
func (p *vfsPlugin) GetStats(device string, partition fs.PartitionInfo) (*fs.FsStats, error) {
	if !utils.FileExists(partition.Mountpoint) {
		klog.V(4).Infof("VFS: mountpoint does not exist: %v", partition.Mountpoint)
		return nil, nil
	}

	capacity, free, avail, inodes, inodesFree, err := GetVfsStats(partition.Mountpoint)
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

// ProcessMount handles standard mount processing.
// For VFS, no special processing is needed.
func (p *vfsPlugin) ProcessMount(mnt *mount.Info) (bool, *mount.Info, error) {
	return true, mnt, nil
}

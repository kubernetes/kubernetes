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

package zfs

import (
	"os"

	"github.com/google/cadvisor/fs"

	mount "github.com/moby/sys/mountinfo"
)

type zfsPlugin struct{}

// NewPlugin creates a new ZFS filesystem plugin.
func NewPlugin() fs.FsPlugin {
	return &zfsPlugin{}
}

func (p *zfsPlugin) Name() string {
	return "zfs"
}

// CanHandle returns true if the filesystem type is zfs.
func (p *zfsPlugin) CanHandle(fsType string) bool {
	return fsType == "zfs"
}

// Priority returns 100 - ZFS has higher priority than VFS.
func (p *zfsPlugin) Priority() int {
	return 100
}

// GetStats returns filesystem statistics for ZFS.
func (p *zfsPlugin) GetStats(device string, partition fs.PartitionInfo) (*fs.FsStats, error) {
	// Check if ZFS is available - if /dev/zfs doesn't exist, fall back to VFS
	if _, err := os.Stat("/dev/zfs"); os.IsNotExist(err) {
		return nil, fs.ErrFallbackToVFS
	}

	capacity, free, avail, err := GetZfsStats(device)
	if err != nil {
		return nil, err
	}

	return &fs.FsStats{
		Capacity:  capacity,
		Free:      free,
		Available: avail,
		Type:      fs.ZFS,
	}, nil
}

// ProcessMount handles ZFS mount processing.
// For ZFS, no special processing is needed.
func (p *zfsPlugin) ProcessMount(mnt *mount.Info) (bool, *mount.Info, error) {
	return true, mnt, nil
}

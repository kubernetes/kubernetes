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

package overlay

import (
	"fmt"

	"github.com/google/cadvisor/fs"
	"github.com/google/cadvisor/fs/vfs"

	mount "github.com/moby/sys/mountinfo"
)

type overlayPlugin struct{}

// NewPlugin creates a new Overlay filesystem plugin.
func NewPlugin() fs.FsPlugin {
	return &overlayPlugin{}
}

func (p *overlayPlugin) Name() string {
	return "overlay"
}

// CanHandle returns true if the filesystem type is overlay or overlay2.
func (p *overlayPlugin) CanHandle(fsType string) bool {
	return fsType == "overlay"
}

// Priority returns 100 - Overlay has higher priority than VFS.
func (p *overlayPlugin) Priority() int {
	return 100
}

// GetStats returns filesystem statistics for Overlay.
// Overlay delegates to VFS for stats collection.
func (p *overlayPlugin) GetStats(device string, partition fs.PartitionInfo) (*fs.FsStats, error) {
	// Overlay uses VFS stats
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

// ProcessMount handles Overlay mount processing.
// Overlay fix: Making mount source unique for all overlay mounts, using the mount's major and minor ids.
// This is needed because multiple overlay mounts can have the same source.
func (p *overlayPlugin) ProcessMount(mnt *mount.Info) (bool, *mount.Info, error) {
	// Create a copy with unique source
	correctedMnt := *mnt
	correctedMnt.Source = fmt.Sprintf("%s_%d-%d", mnt.Source, mnt.Major, mnt.Minor)
	return true, &correctedMnt, nil
}

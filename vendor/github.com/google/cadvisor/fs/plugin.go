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

package fs

import (
	"errors"
	"fmt"
	"sync"

	mount "github.com/moby/sys/mountinfo"
	"k8s.io/klog/v2"
)

// FsPlugin provides filesystem-specific statistics collection.
type FsPlugin interface {
	// Name returns the plugin identifier (e.g., "zfs", "devicemapper", "vfs").
	Name() string

	// CanHandle returns true if this plugin handles the given filesystem type.
	CanHandle(fsType string) bool

	// Priority returns the plugin priority (higher = checked first).
	// Allows specific plugins (zfs, btrfs) to override generic (vfs).
	Priority() int

	// GetStats returns filesystem statistics for a partition.
	GetStats(device string, partition PartitionInfo) (*FsStats, error)

	// ProcessMount optionally modifies mount info during processing.
	// Returns (shouldInclude bool, modifiedMount *mount.Info, error).
	ProcessMount(mnt *mount.Info) (bool, *mount.Info, error)
}

// FsCachingPlugin is an optional interface for plugins that want to cache
// stats by a key (e.g., device ID) to avoid redundant stat calls.
// This is useful for network filesystems like NFS where multiple mounts
// may point to the same underlying device.
type FsCachingPlugin interface {
	FsPlugin

	// CacheKey returns a cache key for the given partition.
	// Stats will be cached by this key and reused for partitions with the same key.
	// Return empty string to disable caching for a specific partition.
	CacheKey(partition PartitionInfo) string
}

// FsWatcherPlugin is an optional interface for plugins that provide
// background monitoring (e.g., ZFS watcher, ThinPool watcher).
type FsWatcherPlugin interface {
	FsPlugin

	// StartWatcher starts background monitoring.
	// Returns a Watcher that can be used to get container-level usage.
	StartWatcher() (FsWatcher, error)
}

// FsWatcher provides container-level filesystem usage from background monitoring.
type FsWatcher interface {
	// GetUsage returns filesystem usage for a specific container/path.
	GetUsage(containerID string, deviceID string) (uint64, error)

	// Stop stops the background monitoring.
	Stop()
}

// PartitionInfo contains information needed for stats collection.
type PartitionInfo struct {
	Mountpoint string
	Major      uint
	Minor      uint
	FsType     string
	BlockSize  uint
}

// FsStats contains filesystem statistics returned by plugins.
type FsStats struct {
	Capacity   uint64
	Free       uint64
	Available  uint64
	Inodes     *uint64
	InodesFree *uint64
	Type       FsType
}

// ErrFallbackToVFS signals that a specialized plugin cannot handle
// this filesystem and VFS should be used instead.
var ErrFallbackToVFS = errors.New("fallback to VFS")

// Plugin registry (init-time registration only).
var (
	pluginsLock sync.RWMutex
	plugins     = make(map[string]FsPlugin)
)

// RegisterPlugin registers a filesystem plugin.
// This should be called from init() functions.
func RegisterPlugin(name string, plugin FsPlugin) error {
	pluginsLock.Lock()
	defer pluginsLock.Unlock()
	if _, found := plugins[name]; found {
		return fmt.Errorf("FsPlugin %q was registered twice", name)
	}
	klog.V(4).Infof("Registered FsPlugin %q", name)
	plugins[name] = plugin
	return nil
}

// GetPluginForFsType returns the appropriate plugin for the filesystem type.
// Returns nil if no plugin can handle the filesystem type.
func GetPluginForFsType(fsType string) FsPlugin {
	pluginsLock.RLock()
	defer pluginsLock.RUnlock()

	var best FsPlugin
	for _, p := range plugins {
		if p.CanHandle(fsType) {
			if best == nil || p.Priority() > best.Priority() {
				best = p
			}
		}
	}
	return best
}

// GetAllPlugins returns all registered plugins.
func GetAllPlugins() []FsPlugin {
	pluginsLock.RLock()
	defer pluginsLock.RUnlock()

	result := make([]FsPlugin, 0, len(plugins))
	for _, p := range plugins {
		result = append(result, p)
	}
	return result
}

// InitializeWatchers starts all plugin watchers and returns them.
func InitializeWatchers() map[string]FsWatcher {
	pluginsLock.RLock()
	defer pluginsLock.RUnlock()

	watchers := make(map[string]FsWatcher)
	for name, plugin := range plugins {
		if wp, ok := plugin.(FsWatcherPlugin); ok {
			watcher, err := wp.StartWatcher()
			if err != nil {
				klog.V(4).Infof("Failed to start watcher for plugin %s: %v", name, err)
				continue
			}
			if watcher != nil {
				watchers[name] = watcher
				klog.V(4).Infof("Started watcher for FsPlugin %q", name)
			}
		}
	}
	return watchers
}

// StopWatchers stops all provided watchers.
func StopWatchers(watchers map[string]FsWatcher) {
	for name, watcher := range watchers {
		if watcher != nil {
			watcher.Stop()
			klog.V(4).Infof("Stopped watcher for FsPlugin %q", name)
		}
	}
}

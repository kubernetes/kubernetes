// Copyright 2016 Google Inc. All Rights Reserved.
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
	"fmt"
	"sync/atomic"
	"time"

	zfslib "github.com/mistifyio/go-zfs"
	"k8s.io/klog/v2"
)

// usageCache is a typed wrapper around atomic.Value that eliminates the need
// for type assertions at every call site. It stores filesystem name strings
// mapped to usage values (uint64).
type usageCache struct {
	v atomic.Value
}

// Load retrieves the current cache map.
func (c *usageCache) Load() map[string]uint64 {
	return c.v.Load().(map[string]uint64)
}

// Store saves a new cache map.
func (c *usageCache) Store(m map[string]uint64) {
	c.v.Store(m)
}

// ZfsWatcher maintains a cache of filesystem -> usage stats for a
// zfs filesystem
type ZfsWatcher struct {
	filesystem string
	cache      usageCache
	period     time.Duration
	stopChan   chan struct{}
}

// NewZfsWatcher returns a new ZfsWatcher for the given zfs filesystem.
func NewZfsWatcher(filesystem string) (*ZfsWatcher, error) {
	w := &ZfsWatcher{
		filesystem: filesystem,
		period:     15 * time.Second,
		stopChan:   make(chan struct{}),
	}
	w.cache.Store(map[string]uint64{})
	return w, nil
}

// Start starts the ZfsWatcher.
func (w *ZfsWatcher) Start() {
	err := w.Refresh()
	if err != nil {
		klog.Errorf("encountered error refreshing zfs watcher: %v", err)
	}

	for {
		select {
		case <-w.stopChan:
			return
		case <-time.After(w.period):
			start := time.Now()
			err = w.Refresh()
			if err != nil {
				klog.Errorf("encountered error refreshing zfs watcher: %v", err)
			}

			// print latency for refresh
			duration := time.Since(start)
			klog.V(5).Infof("zfs(%d) took %s", start.Unix(), duration)
		}
	}
}

// Stop stops the ZfsWatcher.
func (w *ZfsWatcher) Stop() {
	close(w.stopChan)
}

// GetUsage gets the cached usage value of the given filesystem.
func (w *ZfsWatcher) GetUsage(filesystem string) (uint64, error) {
	cache := w.cache.Load()
	v, ok := cache[filesystem]
	if !ok {
		return 0, fmt.Errorf("no cached value for usage of filesystem %v", filesystem)
	}
	return v, nil
}

// Refresh performs a zfs get
func (w *ZfsWatcher) Refresh() error {
	parent, err := zfslib.GetDataset(w.filesystem)
	if err != nil {
		klog.Errorf("encountered error getting zfs filesystem: %s: %v", w.filesystem, err)
		return err
	}
	children, err := parent.Children(0)
	if err != nil {
		klog.Errorf("encountered error getting children of zfs filesystem: %s: %v", w.filesystem, err)
		return err
	}

	newCache := make(map[string]uint64)
	for _, ds := range children {
		newCache[ds.Name] = ds.Used
	}

	w.cache.Store(newCache)
	return nil
}

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
	"sync"
	"time"

	zfs "github.com/mistifyio/go-zfs"
	"k8s.io/klog"
)

// zfsWatcher maintains a cache of filesystem -> usage stats for a
// zfs filesystem
type ZfsWatcher struct {
	filesystem string
	lock       *sync.RWMutex
	cache      map[string]uint64
	period     time.Duration
	stopChan   chan struct{}
}

// NewThinPoolWatcher returns a new ThinPoolWatcher for the given devicemapper
// thin pool name and metadata device or an error.
func NewZfsWatcher(filesystem string) (*ZfsWatcher, error) {

	return &ZfsWatcher{
		filesystem: filesystem,
		lock:       &sync.RWMutex{},
		cache:      make(map[string]uint64),
		period:     15 * time.Second,
		stopChan:   make(chan struct{}),
	}, nil
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
	w.lock.RLock()
	defer w.lock.RUnlock()

	v, ok := w.cache[filesystem]
	if !ok {
		return 0, fmt.Errorf("no cached value for usage of filesystem %v", filesystem)
	}

	return v, nil
}

// Refresh performs a zfs get
func (w *ZfsWatcher) Refresh() error {
	w.lock.Lock()
	defer w.lock.Unlock()
	newCache := make(map[string]uint64)
	parent, err := zfs.GetDataset(w.filesystem)
	if err != nil {
		klog.Errorf("encountered error getting zfs filesystem: %s: %v", w.filesystem, err)
		return err
	}
	children, err := parent.Children(0)
	if err != nil {
		klog.Errorf("encountered error getting children of zfs filesystem: %s: %v", w.filesystem, err)
		return err
	}

	for _, ds := range children {
		newCache[ds.Name] = ds.Used
	}

	w.cache = newCache
	return nil
}

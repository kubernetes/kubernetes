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

package noderesources

import (
	"container/list"
	"context"
	"sync"
	"time"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/klog/v2"
)

const (
	gcInterval   = 60 * time.Second
	gcBatchSize  = 512
	gcTimeBudget = 5 * time.Millisecond
)

// The index gives GC a persistent cursor. Mutations keep it consistent with the
// sync.Map; cache hits and CEL evaluation do not acquire the index lock.
type deviceMatchCache struct {
	values  sync.Map
	mu      sync.Mutex
	entries list.List
	cursor  *list.Element
}

func (c *deviceMatchCache) Load(key any) (any, bool) { return c.values.Load(key) }

func (c *deviceMatchCache) LoadOrStore(key, value any) (any, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	actual, loaded := c.values.LoadOrStore(key, value)
	if !loaded {
		value.(*sliceMatchResults).element = c.entries.PushBack(key)
	}
	return actual, loaded
}

func (c *deviceMatchCache) CompareAndSwap(key, old, value any) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.values.CompareAndSwap(key, old, value) {
		return false
	}
	element := old.(*sliceMatchResults).element
	old.(*sliceMatchResults).element = nil
	value.(*sliceMatchResults).element = element
	return true
}

func (c *deviceMatchCache) CompareAndDelete(key, value any) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.values.CompareAndDelete(key, value) {
		return false
	}
	entry := value.(*sliceMatchResults)
	if c.cursor == entry.element {
		c.cursor = entry.element.Next()
	}
	c.entries.Remove(entry.element)
	entry.element = nil
	if c.cursor == nil {
		c.cursor = c.entries.Front()
	}
	return true
}

func (c *deviceMatchCache) size() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.entries.Len()
}

func (c *deviceMatchCache) next() (any, *sliceMatchResults, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.cursor == nil {
		c.cursor = c.entries.Front()
	}
	if c.cursor == nil {
		return nil, nil, false
	}
	key := c.cursor.Value
	c.cursor = c.cursor.Next()
	if c.cursor == nil {
		c.cursor = c.entries.Front()
	}
	value, _ := c.values.Load(key)
	return key, value.(*sliceMatchResults), true
}

type deviceMatchGCStats struct {
	inspected int
	reclaimed int
	elapsed   time.Duration
}

func (c *DRACaches) gcDeviceMatchCache() deviceMatchGCStats {
	var stats deviceMatchGCStats
	if c.resourceSliceStore == nil {
		return stats
	}
	started := time.Now()
	limit := min(gcBatchSize, c.deviceMatchCache.size())
	for stats.inspected < limit && time.Since(started) < gcTimeBudget {
		key, entry, ok := c.deviceMatchCache.next()
		if !ok {
			break
		}
		obj, exists, err := c.resourceSliceStore.GetByKey(entry.name)
		current, _ := obj.(*resourceapi.ResourceSlice)
		if err == nil && (!exists || current == nil || current.UID != key || current.Generation != entry.generation) {
			if c.deviceMatchCache.CompareAndDelete(key, entry) {
				stats.reclaimed++
			}
		}
		stats.inspected++
	}
	stats.elapsed = time.Since(started)
	return stats
}

func (c *DRACaches) runDeviceMatchGC(ctx context.Context, unregister func()) {
	defer close(c.gcDone)
	defer unregister()
	ticker := time.NewTicker(gcInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			stats := c.gcDeviceMatchCache()
			if stats.elapsed > gcTimeBudget {
				klog.FromContext(ctx).Info("DRA device match cache GC exceeded time budget", "elapsed", stats.elapsed, "budget", gcTimeBudget, "inspected", stats.inspected, "reclaimed", stats.reclaimed)
			}
		}
	}
}

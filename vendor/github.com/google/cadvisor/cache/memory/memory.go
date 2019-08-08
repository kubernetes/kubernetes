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

package memory

import (
	"errors"
	"sync"
	"time"

	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/storage"
	"github.com/google/cadvisor/utils"

	"k8s.io/klog"
)

// ErrDataNotFound is the error resulting if failed to find a container in memory cache.
var ErrDataNotFound = errors.New("unable to find data in memory cache")

// TODO(vmarmol): See about refactoring this class, we have an unnecessary redirection of containerCache and InMemoryCache.
// containerCache is used to store per-container information
type containerCache struct {
	ref         info.ContainerReference
	recentStats *utils.TimedStore
	maxAge      time.Duration
	lock        sync.RWMutex
}

func (self *containerCache) AddStats(stats *info.ContainerStats) error {
	self.lock.Lock()
	defer self.lock.Unlock()

	// Add the stat to storage.
	self.recentStats.Add(stats.Timestamp, stats)
	return nil
}

func (self *containerCache) RecentStats(start, end time.Time, maxStats int) ([]*info.ContainerStats, error) {
	self.lock.RLock()
	defer self.lock.RUnlock()
	result := self.recentStats.InTimeRange(start, end, maxStats)
	converted := make([]*info.ContainerStats, len(result))
	for i, el := range result {
		converted[i] = el.(*info.ContainerStats)
	}
	return converted, nil
}

func newContainerStore(ref info.ContainerReference, maxAge time.Duration) *containerCache {
	return &containerCache{
		ref:         ref,
		recentStats: utils.NewTimedStore(maxAge, -1),
		maxAge:      maxAge,
	}
}

type InMemoryCache struct {
	lock              sync.RWMutex
	containerCacheMap map[string]*containerCache
	maxAge            time.Duration
	backend           []storage.StorageDriver
}

func (self *InMemoryCache) AddStats(cInfo *info.ContainerInfo, stats *info.ContainerStats) error {
	var cstore *containerCache
	var ok bool

	func() {
		self.lock.Lock()
		defer self.lock.Unlock()
		if cstore, ok = self.containerCacheMap[cInfo.ContainerReference.Name]; !ok {
			cstore = newContainerStore(cInfo.ContainerReference, self.maxAge)
			self.containerCacheMap[cInfo.ContainerReference.Name] = cstore
		}
	}()

	for _, backend := range self.backend {
		// TODO(monnand): To deal with long delay write operations, we
		// may want to start a pool of goroutines to do write
		// operations.
		if err := backend.AddStats(cInfo, stats); err != nil {
			klog.Error(err)
		}
	}
	return cstore.AddStats(stats)
}

func (self *InMemoryCache) RecentStats(name string, start, end time.Time, maxStats int) ([]*info.ContainerStats, error) {
	var cstore *containerCache
	var ok bool
	err := func() error {
		self.lock.RLock()
		defer self.lock.RUnlock()
		if cstore, ok = self.containerCacheMap[name]; !ok {
			return ErrDataNotFound
		}
		return nil
	}()
	if err != nil {
		return nil, err
	}

	return cstore.RecentStats(start, end, maxStats)
}

func (self *InMemoryCache) Close() error {
	self.lock.Lock()
	self.containerCacheMap = make(map[string]*containerCache, 32)
	self.lock.Unlock()
	return nil
}

func (self *InMemoryCache) RemoveContainer(containerName string) error {
	self.lock.Lock()
	delete(self.containerCacheMap, containerName)
	self.lock.Unlock()
	return nil
}

func New(
	maxAge time.Duration,
	backend []storage.StorageDriver,
) *InMemoryCache {
	ret := &InMemoryCache{
		containerCacheMap: make(map[string]*containerCache, 32),
		maxAge:            maxAge,
		backend:           backend,
	}
	return ret
}

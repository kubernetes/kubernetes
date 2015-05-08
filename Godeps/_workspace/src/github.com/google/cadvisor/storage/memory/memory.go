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
	"fmt"
	"sync"
	"time"

	"github.com/golang/glog"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/storage"
	"github.com/google/cadvisor/utils"
)

// TODO(vmarmol): See about refactoring this class, we have an unecessary redirection of containerStorage and InMemoryStorage.
// containerStorage is used to store per-container information
type containerStorage struct {
	ref         info.ContainerReference
	recentStats *utils.TimedStore
	maxAge      time.Duration
	lock        sync.RWMutex
}

func (self *containerStorage) AddStats(stats *info.ContainerStats) error {
	self.lock.Lock()
	defer self.lock.Unlock()

	// Add the stat to storage.
	self.recentStats.Add(stats.Timestamp, stats)
	return nil
}

func (self *containerStorage) RecentStats(start, end time.Time, maxStats int) ([]*info.ContainerStats, error) {
	self.lock.RLock()
	defer self.lock.RUnlock()
	result := self.recentStats.InTimeRange(start, end, maxStats)
	converted := make([]*info.ContainerStats, len(result))
	for i, el := range result {
		converted[i] = el.(*info.ContainerStats)
	}
	return converted, nil
}

func newContainerStore(ref info.ContainerReference, maxAge time.Duration) *containerStorage {
	return &containerStorage{
		ref:         ref,
		recentStats: utils.NewTimedStore(maxAge, -1),
		maxAge:      maxAge,
	}
}

type InMemoryStorage struct {
	lock                sync.RWMutex
	containerStorageMap map[string]*containerStorage
	maxAge              time.Duration
	backend             storage.StorageDriver
}

func (self *InMemoryStorage) AddStats(ref info.ContainerReference, stats *info.ContainerStats) error {
	var cstore *containerStorage
	var ok bool

	func() {
		self.lock.Lock()
		defer self.lock.Unlock()
		if cstore, ok = self.containerStorageMap[ref.Name]; !ok {
			cstore = newContainerStore(ref, self.maxAge)
			self.containerStorageMap[ref.Name] = cstore
		}
	}()

	if self.backend != nil {
		// TODO(monnand): To deal with long delay write operations, we
		// may want to start a pool of goroutines to do write
		// operations.
		if err := self.backend.AddStats(ref, stats); err != nil {
			glog.Error(err)
		}
	}
	return cstore.AddStats(stats)
}

func (self *InMemoryStorage) RecentStats(name string, start, end time.Time, maxStats int) ([]*info.ContainerStats, error) {
	var cstore *containerStorage
	var ok bool
	err := func() error {
		self.lock.RLock()
		defer self.lock.RUnlock()
		if cstore, ok = self.containerStorageMap[name]; !ok {
			return fmt.Errorf("unable to find data for container %v", name)
		}
		return nil
	}()
	if err != nil {
		return nil, err
	}

	return cstore.RecentStats(start, end, maxStats)
}

func (self *InMemoryStorage) Close() error {
	self.lock.Lock()
	self.containerStorageMap = make(map[string]*containerStorage, 32)
	self.lock.Unlock()
	return nil
}

func New(
	maxAge time.Duration,
	backend storage.StorageDriver,
) *InMemoryStorage {
	ret := &InMemoryStorage{
		containerStorageMap: make(map[string]*containerStorage, 32),
		maxAge:              maxAge,
		backend:             backend,
	}
	return ret
}

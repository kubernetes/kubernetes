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
	"container/list"
	"fmt"
	"sync"

	"github.com/google/cadvisor/info"
	"github.com/google/cadvisor/sampling"
	"github.com/google/cadvisor/storage"
)

// containerStorage is used to store per-container information
type containerStorage struct {
	ref         info.ContainerReference
	prevStats   *info.ContainerStats
	sampler     sampling.Sampler
	recentStats *list.List
	maxNumStats int
	maxMemUsage uint64
	lock        sync.RWMutex
}

func (self *containerStorage) updatePrevStats(stats *info.ContainerStats) {
	if stats == nil || stats.Cpu == nil || stats.Memory == nil {
		// discard incomplete stats
		self.prevStats = nil
		return
	}
	self.prevStats = stats.Copy(self.prevStats)
}

func (self *containerStorage) AddStats(stats *info.ContainerStats) error {
	self.lock.Lock()
	defer self.lock.Unlock()
	if self.prevStats != nil {
		sample, err := info.NewSample(self.prevStats, stats)
		if err != nil {
			return fmt.Errorf("wrong stats: %v", err)
		}
		if sample != nil {
			self.sampler.Update(sample)
		}
	}
	if stats.Memory != nil {
		if self.maxMemUsage < stats.Memory.Usage {
			self.maxMemUsage = stats.Memory.Usage
		}
	}
	if self.recentStats.Len() >= self.maxNumStats {
		self.recentStats.Remove(self.recentStats.Back())
	}
	self.recentStats.PushFront(stats)
	self.updatePrevStats(stats)
	return nil
}

func (self *containerStorage) RecentStats(numStats int) ([]*info.ContainerStats, error) {
	self.lock.RLock()
	defer self.lock.RUnlock()
	if self.recentStats.Len() < numStats || numStats < 0 {
		numStats = self.recentStats.Len()
	}

	// Stats in the recentStats list are stored in reverse chronological
	// order, i.e. most recent stats is in the front.
	// numStats will always <= recentStats.Len() so that there will be
	// always at least numStats available stats to retrieve. We traverse
	// the recentStats list from its head and fill the ret slice in
	// reverse order so that the returned slice will be in chronological
	// order. The order of the returned slice is not specified by the
	// StorageDriver interface, so it is not necessary for other storage
	// drivers to return the slice in the same order.
	ret := make([]*info.ContainerStats, numStats)
	e := self.recentStats.Front()
	for i := numStats - 1; i >= 0; i-- {
		data, ok := e.Value.(*info.ContainerStats)
		if !ok {
			return nil, fmt.Errorf("The %vth element is not a ContainerStats", i)
		}
		ret[i] = data
		e = e.Next()
	}
	return ret, nil
}

func (self *containerStorage) Samples(numSamples int) ([]*info.ContainerStatsSample, error) {
	self.lock.RLock()
	defer self.lock.RUnlock()
	if self.sampler.Len() < numSamples || numSamples < 0 {
		numSamples = self.sampler.Len()
	}
	ret := make([]*info.ContainerStatsSample, 0, numSamples)

	var err error
	self.sampler.Map(func(d interface{}) {
		if len(ret) >= numSamples || err != nil {
			return
		}
		sample, ok := d.(*info.ContainerStatsSample)
		if !ok {
			err = fmt.Errorf("An element in the sample is not a ContainerStatsSample")
		}
		ret = append(ret, sample)
	})
	if err != nil {
		return nil, err
	}
	return ret, nil
}

func (self *containerStorage) Percentiles(cpuPercentiles, memPercentiles []int) (*info.ContainerStatsPercentiles, error) {
	samples, err := self.Samples(-1)
	if err != nil {
		return nil, err
	}
	if len(samples) == 0 {
		return nil, nil
	}
	ret := info.NewPercentiles(samples, cpuPercentiles, memPercentiles)
	ret.MaxMemoryUsage = self.maxMemUsage
	return ret, nil
}

func newContainerStore(ref info.ContainerReference, maxNumSamples, maxNumStats int) *containerStorage {
	s := sampling.NewReservoirSampler(maxNumSamples)
	return &containerStorage{
		ref:         ref,
		recentStats: list.New(),
		sampler:     s,
		maxNumStats: maxNumStats,
	}
}

type InMemoryStorage struct {
	lock                sync.RWMutex
	containerStorageMap map[string]*containerStorage
	maxNumSamples       int
	maxNumStats         int
}

func (self *InMemoryStorage) AddStats(ref info.ContainerReference, stats *info.ContainerStats) error {
	var cstore *containerStorage
	var ok bool

	func() {
		self.lock.Lock()
		defer self.lock.Unlock()
		if cstore, ok = self.containerStorageMap[ref.Name]; !ok {
			cstore = newContainerStore(ref, self.maxNumSamples, self.maxNumStats)
			self.containerStorageMap[ref.Name] = cstore
		}
	}()
	return cstore.AddStats(stats)
}

func (self *InMemoryStorage) Samples(name string, numSamples int) ([]*info.ContainerStatsSample, error) {
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

	return cstore.Samples(numSamples)
}

func (self *InMemoryStorage) RecentStats(name string, numStats int) ([]*info.ContainerStats, error) {
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

	return cstore.RecentStats(numStats)
}

func (self *InMemoryStorage) Percentiles(name string, cpuPercentiles, memPercentiles []int) (*info.ContainerStatsPercentiles, error) {
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

	return cstore.Percentiles(cpuPercentiles, memPercentiles)
}

func (self *InMemoryStorage) Close() error {
	self.lock.Lock()
	self.containerStorageMap = make(map[string]*containerStorage, 32)
	self.lock.Unlock()
	return nil
}

func New(maxNumSamples, maxNumStats int) storage.StorageDriver {
	ret := &InMemoryStorage{
		containerStorageMap: make(map[string]*containerStorage, 32),
		maxNumSamples:       maxNumSamples,
		maxNumStats:         maxNumStats,
	}
	return ret
}

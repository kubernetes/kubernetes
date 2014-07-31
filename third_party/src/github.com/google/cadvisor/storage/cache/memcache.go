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

package cache

import (
	"github.com/google/cadvisor/info"
	"github.com/google/cadvisor/storage"
	"github.com/google/cadvisor/storage/memory"
)

type cachedStorageDriver struct {
	maxNumStatsInCache   int
	maxNumSamplesInCache int
	cache                storage.StorageDriver
	backend              storage.StorageDriver
}

func (self *cachedStorageDriver) AddStats(ref info.ContainerReference, stats *info.ContainerStats) error {
	err := self.cache.AddStats(ref, stats)
	if err != nil {
		return err
	}
	err = self.backend.AddStats(ref, stats)
	if err != nil {
		return err
	}
	return nil
}

func (self *cachedStorageDriver) RecentStats(containerName string, numStats int) ([]*info.ContainerStats, error) {
	if numStats < self.maxNumStatsInCache {
		return self.cache.RecentStats(containerName, numStats)
	}
	return self.backend.RecentStats(containerName, numStats)
}

func (self *cachedStorageDriver) Percentiles(containerName string, cpuUsagePercentiles []int, memUsagePercentiles []int) (*info.ContainerStatsPercentiles, error) {
	return self.backend.Percentiles(containerName, cpuUsagePercentiles, memUsagePercentiles)
}

func (self *cachedStorageDriver) Samples(containerName string, numSamples int) ([]*info.ContainerStatsSample, error) {
	if numSamples < self.maxNumSamplesInCache {
		return self.cache.Samples(containerName, numSamples)
	}
	return self.backend.Samples(containerName, numSamples)
}

func (self *cachedStorageDriver) Close() error {
	self.cache.Close()
	return self.backend.Close()
}

func MemoryCache(maxNumSamplesInCache, maxNumStatsInCache int, driver storage.StorageDriver) storage.StorageDriver {
	return &cachedStorageDriver{
		maxNumStatsInCache:   maxNumStatsInCache,
		maxNumSamplesInCache: maxNumSamplesInCache,
		cache:                memory.New(maxNumSamplesInCache, maxNumStatsInCache),
		backend:              driver,
	}
}

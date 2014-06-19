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

package storage

import "github.com/google/cadvisor/info"

type StorageDriver interface {
	AddStats(ref info.ContainerReference, stats *info.ContainerStats) error

	// Read most recent stats. numStats indicates max number of stats
	// returned. The returned stats must be consecutive observed stats. If
	// numStats < 0, then return all stats stored in the storage.
	RecentStats(containerName string, numStats int) ([]*info.ContainerStats, error)

	// Read the specified percentiles of CPU and memory usage of the container.
	// The implementation decides which time range to look at.
	Percentiles(containerName string, cpuUsagePercentiles []int, memUsagePercentiles []int) (*info.ContainerStatsPercentiles, error)

	// Returns samples of the container stats. If numSamples < 0, then
	// the number of returned samples is implementation defined. Otherwise, the driver
	// should return at most numSamples samples.
	Samples(containername string, numSamples int) ([]*info.ContainerStatsSample, error)

	// Close will clear the state of the storage driver. The elements
	// stored in the underlying storage may or may not be deleted depending
	// on the implementation of the storage driver.
	Close() error
}

// Copyright 2017 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package bcache provides access to statistics exposed by the bcache (Linux
// block cache).
package bcache

// Stats contains bcache runtime statistics, parsed from /sys/fs/bcache/.
//
// The names and meanings of each statistic were taken from bcache.txt and
// files in drivers/md/bcache in the Linux kernel source. Counters are uint64
// (in-kernel counters are mostly unsigned long).
type Stats struct {
	// The name of the bcache used to source these statistics.
	Name   string
	Bcache BcacheStats
	Bdevs  []BdevStats
	Caches []CacheStats
}

// BcacheStats contains statistics tied to a bcache ID.
type BcacheStats struct { // nolint:golint
	AverageKeySize        uint64
	BtreeCacheSize        uint64
	CacheAvailablePercent uint64
	Congested             uint64
	RootUsagePercent      uint64
	TreeDepth             uint64
	Internal              InternalStats
	FiveMin               PeriodStats
	Total                 PeriodStats
}

// BdevStats contains statistics for one backing device.
type BdevStats struct {
	Name               string
	DirtyData          uint64
	FiveMin            PeriodStats
	Total              PeriodStats
	WritebackRateDebug WritebackRateDebugStats
}

// CacheStats contains statistics for one cache device.
type CacheStats struct {
	Name            string
	IOErrors        uint64
	MetadataWritten uint64
	Written         uint64
	Priority        PriorityStats
}

// PriorityStats contains statistics from the priority_stats file.
type PriorityStats struct {
	UnusedPercent   uint64
	MetadataPercent uint64
}

// InternalStats contains internal bcache statistics.
type InternalStats struct {
	ActiveJournalEntries                uint64
	BtreeNodes                          uint64
	BtreeReadAverageDurationNanoSeconds uint64
	CacheReadRaces                      uint64
}

// PeriodStats contains statistics for a time period (5 min or total).
type PeriodStats struct {
	Bypassed            uint64
	CacheBypassHits     uint64
	CacheBypassMisses   uint64
	CacheHits           uint64
	CacheMissCollisions uint64
	CacheMisses         uint64
	CacheReadaheads     uint64
}

// WritebackRateDebugStats contains bcache writeback statistics.
type WritebackRateDebugStats struct {
	Rate         uint64
	Dirty        uint64
	Target       uint64
	Proportional int64
	Integral     int64
	Change       int64
	NextIO       int64
}

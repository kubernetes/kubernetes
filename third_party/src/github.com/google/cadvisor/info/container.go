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

package info

import (
	"fmt"
	"sort"
	"time"
)

type CpuSpecMask struct {
	Data []uint64 `json:"data,omitempty"`
}

type CpuSpec struct {
	Limit    uint64      `json:"limit"`
	MaxLimit uint64      `json:"max_limit"`
	Mask     CpuSpecMask `json:"mask,omitempty"`
}

type MemorySpec struct {
	// The amount of memory requested. Default is unlimited (-1).
	// Units: bytes.
	Limit uint64 `json:"limit,omitempty"`

	// The amount of guaranteed memory.  Default is 0.
	// Units: bytes.
	Reservation uint64 `json:"reservation,omitempty"`

	// The amount of swap space requested. Default is unlimited (-1).
	// Units: bytes.
	SwapLimit uint64 `json:"swap_limit,omitempty"`
}

type ContainerSpec struct {
	Cpu    *CpuSpec    `json:"cpu,omitempty"`
	Memory *MemorySpec `json:"memory,omitempty"`
}

// Container reference contains enough information to uniquely identify a container
type ContainerReference struct {
	// The absolute name of the container.
	Name string `json:"name"`

	Aliases []string `json:"aliases,omitempty"`
}

type ContainerInfo struct {
	ContainerReference

	// The direct subcontainers of the current container.
	Subcontainers []ContainerReference `json:"subcontainers,omitempty"`

	// The isolation used in the container.
	Spec *ContainerSpec `json:"spec,omitempty"`

	// Historical statistics gathered from the container.
	Stats []*ContainerStats `json:"stats,omitempty"`

	// Randomly sampled container states.
	Samples []*ContainerStatsSample `json:"samples,omitempty"`

	StatsPercentiles *ContainerStatsPercentiles `json:"stats_summary,omitempty"`
}

func (self *ContainerInfo) StatsAfter(ref time.Time) []*ContainerStats {
	n := len(self.Stats) + 1
	for i, s := range self.Stats {
		if s.Timestamp.After(ref) {
			n = i
			break
		}
	}
	if n > len(self.Stats) {
		return nil
	}
	return self.Stats[n:]
}

func (self *ContainerInfo) StatsStartTime() time.Time {
	var ret time.Time
	for _, s := range self.Stats {
		if s.Timestamp.Before(ret) || ret.IsZero() {
			ret = s.Timestamp
		}
	}
	return ret
}

func (self *ContainerInfo) StatsEndTime() time.Time {
	var ret time.Time
	for i := len(self.Stats) - 1; i >= 0; i-- {
		s := self.Stats[i]
		if s.Timestamp.After(ret) {
			ret = s.Timestamp
		}
	}
	return ret
}

// All CPU usage metrics are cumulative from the creation of the container
type CpuStats struct {
	Usage struct {
		// Total CPU usage.
		// Units: nanoseconds
		Total uint64 `json:"total"`

		// Per CPU/core usage of the container.
		// Unit: nanoseconds.
		PerCpu []uint64 `json:"per_cpu,omitempty"`

		// Time spent in user space.
		// Unit: nanoseconds
		User uint64 `json:"user"`

		// Time spent in kernel space.
		// Unit: nanoseconds
		System uint64 `json:"system"`
	} `json:"usage"`
	Load int32 `json:"load"`
}

type MemoryStats struct {
	// Memory limit, equivalent to "limit" in MemorySpec.
	// Units: Bytes.
	Limit uint64 `json:"limit,omitempty"`

	// Usage statistics.

	// Current memory usage, this includes all memory regardless of when it was
	// accessed.
	// Units: Bytes.
	Usage uint64 `json:"usage,omitempty"`

	// The amount of working set memory, this includes recently accessed memory,
	// dirty memory, and kernel memory. Working set is <= "usage".
	// Units: Bytes.
	WorkingSet uint64 `json:"working_set,omitempty"`

	ContainerData    MemoryStatsMemoryData `json:"container_data,omitempty"`
	HierarchicalData MemoryStatsMemoryData `json:"hierarchical_data,omitempty"`
}

type MemoryStatsMemoryData struct {
	Pgfault    uint64 `json:"pgfault,omitempty"`
	Pgmajfault uint64 `json:"pgmajfault,omitempty"`
}

type ContainerStats struct {
	// The time of this stat point.
	Timestamp time.Time    `json:"timestamp"`
	Cpu       *CpuStats    `json:"cpu,omitempty"`
	Memory    *MemoryStats `json:"memory,omitempty"`
}

type ContainerStatsSample struct {
	// Timetamp of the end of the sample period
	Timestamp time.Time `json:"timestamp"`
	// Duration of the sample period
	Duration time.Duration `json:"duration"`
	Cpu      struct {
		// number of nanoseconds of CPU time used by the container
		Usage uint64 `json:"usage"`
	} `json:"cpu"`
	Memory struct {
		// Units: Bytes.
		Usage uint64 `json:"usage"`
	} `json:"memory"`
}

type Percentile struct {
	Percentage int    `json:"percentage"`
	Value      uint64 `json:"value"`
}

type ContainerStatsPercentiles struct {
	MaxMemoryUsage         uint64       `json:"max_memory_usage,omitempty"`
	MemoryUsagePercentiles []Percentile `json:"memory_usage_percentiles,omitempty"`
	CpuUsagePercentiles    []Percentile `json:"cpu_usage_percentiles,omitempty"`
}

// Each sample needs two stats because the cpu usage in ContainerStats is
// cumulative.
// prev should be an earlier observation than current.
// This method is not thread/goroutine safe.
func NewSample(prev, current *ContainerStats) (*ContainerStatsSample, error) {
	if prev == nil || current == nil {
		return nil, fmt.Errorf("empty stats")
	}
	// Ignore this sample if it is incomplete
	if prev.Cpu == nil || prev.Memory == nil || current.Cpu == nil || current.Memory == nil {
		return nil, fmt.Errorf("incomplete stats")
	}
	// prev must be an early observation
	if !current.Timestamp.After(prev.Timestamp) {
		return nil, fmt.Errorf("wrong stats order")
	}
	// This data is invalid.
	if current.Cpu.Usage.Total < prev.Cpu.Usage.Total {
		return nil, fmt.Errorf("current CPU usage is less than prev CPU usage (cumulative).")
	}
	sample := new(ContainerStatsSample)
	// Calculate the diff to get the CPU usage within the time interval.
	sample.Cpu.Usage = current.Cpu.Usage.Total - prev.Cpu.Usage.Total
	// Memory usage is current memory usage
	sample.Memory.Usage = current.Memory.Usage
	sample.Timestamp = current.Timestamp
	sample.Duration = current.Timestamp.Sub(prev.Timestamp)

	return sample, nil
}

type uint64Slice []uint64

func (self uint64Slice) Len() int {
	return len(self)
}

func (self uint64Slice) Less(i, j int) bool {
	return self[i] < self[j]
}

func (self uint64Slice) Swap(i, j int) {
	self[i], self[j] = self[j], self[i]
}

func (self uint64Slice) Percentiles(requestedPercentiles ...int) []Percentile {
	if len(self) == 0 {
		return nil
	}
	ret := make([]Percentile, 0, len(requestedPercentiles))
	sort.Sort(self)
	for _, p := range requestedPercentiles {
		idx := (len(self) * p / 100) - 1
		if idx < 0 {
			idx = 0
		}
		ret = append(
			ret,
			Percentile{
				Percentage: p,
				Value:      self[idx],
			},
		)
	}
	return ret
}

func NewPercentiles(samples []*ContainerStatsSample, cpuPercentages, memoryPercentages []int) *ContainerStatsPercentiles {
	if len(samples) == 0 {
		return nil
	}
	cpuUsages := make([]uint64, 0, len(samples))
	memUsages := make([]uint64, 0, len(samples))

	for _, sample := range samples {
		if sample == nil {
			continue
		}
		cpuUsages = append(cpuUsages, sample.Cpu.Usage)
		memUsages = append(memUsages, sample.Memory.Usage)
	}

	ret := new(ContainerStatsPercentiles)
	ret.CpuUsagePercentiles = uint64Slice(cpuUsages).Percentiles(cpuPercentages...)
	ret.MemoryUsagePercentiles = uint64Slice(memUsages).Percentiles(memoryPercentages...)
	return ret
}

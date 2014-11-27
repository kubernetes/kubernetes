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
	"reflect"
	"time"
)

type CpuSpec struct {
	Limit    uint64 `json:"limit"`
	MaxLimit uint64 `json:"max_limit"`
	Mask     string `json:"mask,omitempty"`
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
	HasCpu bool    `json:"has_cpu"`
	Cpu    CpuSpec `json:"cpu,omitempty"`

	HasMemory bool       `json:"has_memory"`
	Memory    MemorySpec `json:"memory,omitempty"`

	HasNetwork bool `json:"has_network"`

	HasFilesystem bool `json:"has_filesystem"`
}

// Container reference contains enough information to uniquely identify a container
type ContainerReference struct {
	// The absolute name of the container.
	Name string `json:"name"`

	Aliases []string `json:"aliases,omitempty"`
}

// ContainerInfoQuery is used when users check a container info from the REST api.
// It specifies how much data users want to get about a container
type ContainerInfoRequest struct {
	// Max number of stats to return.
	NumStats int `json:"num_stats,omitempty"`
}

type ContainerInfo struct {
	ContainerReference

	// The direct subcontainers of the current container.
	Subcontainers []ContainerReference `json:"subcontainers,omitempty"`

	// The isolation used in the container.
	Spec ContainerSpec `json:"spec,omitempty"`

	// Historical statistics gathered from the container.
	Stats []*ContainerStats `json:"stats,omitempty"`
}

// ContainerInfo may be (un)marshaled by json or other en/decoder. In that
// case, the Timestamp field in each stats/sample may not be precisely
// en/decoded.  This will lead to small but acceptable differences between a
// ContainerInfo and its encode-then-decode version.  Eq() is used to compare
// two ContainerInfo accepting small difference (<10ms) of Time fields.
func (self *ContainerInfo) Eq(b *ContainerInfo) bool {

	// If both self and b are nil, then Eq() returns true
	if self == nil {
		return b == nil
	}
	if b == nil {
		return self == nil
	}

	// For fields other than time.Time, we will compare them precisely.
	// This would require that any slice should have same order.
	if !reflect.DeepEqual(self.ContainerReference, b.ContainerReference) {
		return false
	}
	if !reflect.DeepEqual(self.Subcontainers, b.Subcontainers) {
		return false
	}
	if !reflect.DeepEqual(self.Spec, b.Spec) {
		return false
	}

	for i, expectedStats := range b.Stats {
		selfStats := self.Stats[i]
		if !expectedStats.Eq(selfStats) {
			return false
		}
	}

	return true
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
		PerCpu []uint64 `json:"per_cpu_usage,omitempty"`

		// Time spent in user space.
		// Unit: nanoseconds
		User uint64 `json:"user"`

		// Time spent in kernel space.
		// Unit: nanoseconds
		System uint64 `json:"system"`
	} `json:"usage"`
	Load int32 `json:"load"`
}

type PerDiskStats struct {
	Major uint64            `json:"major"`
	Minor uint64            `json:"minor"`
	Stats map[string]uint64 `json:"stats"`
}

type DiskIoStats struct {
	IoServiceBytes []PerDiskStats `json:"io_service_bytes,omitempty"`
	IoServiced     []PerDiskStats `json:"io_serviced,omitempty"`
	IoQueued       []PerDiskStats `json:"io_queued,omitempty"`
	Sectors        []PerDiskStats `json:"sectors,omitempty"`
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

type NetworkStats struct {
	// Cumulative count of bytes received.
	RxBytes uint64 `json:"rx_bytes"`
	// Cumulative count of packets received.
	RxPackets uint64 `json:"rx_packets"`
	// Cumulative count of receive errors encountered.
	RxErrors uint64 `json:"rx_errors"`
	// Cumulative count of packets dropped while receiving.
	RxDropped uint64 `json:"rx_dropped"`
	// Cumulative count of bytes transmitted.
	TxBytes uint64 `json:"tx_bytes"`
	// Cumulative count of packets transmitted.
	TxPackets uint64 `json:"tx_packets"`
	// Cumulative count of transmit errors encountered.
	TxErrors uint64 `json:"tx_errors"`
	// Cumulative count of packets dropped while transmitting.
	TxDropped uint64 `json:"tx_dropped"`
}

type FsStats struct {
	// The block device name associated with the filesystem.
	Device string `json:"device,omitempty"`

	// Number of bytes that can be consumed by the container on this filesystem.
	Limit uint64 `json:"capacity"`

	// Number of bytes that is consumed by the container on this filesystem.
	Usage uint64 `json:"usage"`
}

type ContainerStats struct {
	// The time of this stat point.
	Timestamp time.Time     `json:"timestamp"`
	Cpu       *CpuStats     `json:"cpu,omitempty"`
	DiskIo    DiskIoStats   `json:"diskio,omitempty"`
	Memory    *MemoryStats  `json:"memory,omitempty"`
	Network   *NetworkStats `json:"network,omitempty"`
	// Filesystem statistics
	Filesystem []FsStats `json:"filesystem,omitempty"`
}

// Makes a deep copy of the ContainerStats and returns a pointer to the new
// copy. Copy() will allocate a new ContainerStats object if dst is nil.
func (self *ContainerStats) Copy(dst *ContainerStats) *ContainerStats {
	if dst == nil {
		dst = new(ContainerStats)
	}
	dst.Timestamp = self.Timestamp
	if self.Cpu != nil {
		if dst.Cpu == nil {
			dst.Cpu = new(CpuStats)
		}
		// To make a deep copy of a slice, we need to copy every value
		// in the slice. To make less memory allocation, we would like
		// to reuse the slice in dst if possible.
		percpu := dst.Cpu.Usage.PerCpu
		if len(percpu) != len(self.Cpu.Usage.PerCpu) {
			percpu = make([]uint64, len(self.Cpu.Usage.PerCpu))
		}
		dst.Cpu.Usage = self.Cpu.Usage
		dst.Cpu.Load = self.Cpu.Load
		copy(percpu, self.Cpu.Usage.PerCpu)
		dst.Cpu.Usage.PerCpu = percpu
	} else {
		dst.Cpu = nil
	}
	if self.Memory != nil {
		if dst.Memory == nil {
			dst.Memory = new(MemoryStats)
		}
		*dst.Memory = *self.Memory
	} else {
		dst.Memory = nil
	}
	return dst
}

func timeEq(t1, t2 time.Time, tolerance time.Duration) bool {
	// t1 should not be later than t2
	if t1.After(t2) {
		t1, t2 = t2, t1
	}
	diff := t2.Sub(t1)
	if diff <= tolerance {
		return true
	}
	return false
}

func durationEq(a, b time.Duration, tolerance time.Duration) bool {
	if a > b {
		a, b = b, a
	}
	diff := a - b
	if diff <= tolerance {
		return true
	}
	return false
}

const (
	// 10ms, i.e. 0.01s
	timePrecision time.Duration = 10 * time.Millisecond
)

// This function is useful because we do not require precise time
// representation.
func (a *ContainerStats) Eq(b *ContainerStats) bool {
	if !timeEq(a.Timestamp, b.Timestamp, timePrecision) {
		return false
	}
	return a.StatsEq(b)
}

// Checks equality of the stats values.
func (a *ContainerStats) StatsEq(b *ContainerStats) bool {
	if !reflect.DeepEqual(a.Cpu, b.Cpu) {
		return false
	}
	if !reflect.DeepEqual(a.Memory, b.Memory) {
		return false
	}
	return true
}

// Saturate CPU usage to 0.
func calculateCpuUsage(prev, cur uint64) uint64 {
	if prev > cur {
		return 0
	}
	return cur - prev
}

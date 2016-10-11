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

package v1

import (
	"reflect"
	"time"
)

type CpuSpec struct {
	Limit    uint64 `json:"limit"`
	MaxLimit uint64 `json:"max_limit"`
	Mask     string `json:"mask,omitempty"`
	Quota    uint64 `json:"quota,omitempty"`
	Period   uint64 `json:"period,omitempty"`
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
	// Time at which the container was created.
	CreationTime time.Time `json:"creation_time,omitempty"`

	// Metadata labels associated with this container.
	Labels map[string]string `json:"labels,omitempty"`
	// Metadata envs associated with this container. Only whitelisted envs are added.
	Envs map[string]string `json:"envs,omitempty"`

	HasCpu bool    `json:"has_cpu"`
	Cpu    CpuSpec `json:"cpu,omitempty"`

	HasMemory bool       `json:"has_memory"`
	Memory    MemorySpec `json:"memory,omitempty"`

	HasNetwork bool `json:"has_network"`

	HasFilesystem bool `json:"has_filesystem"`

	// HasDiskIo when true, indicates that DiskIo stats will be available.
	HasDiskIo bool `json:"has_diskio"`

	HasCustomMetrics bool         `json:"has_custom_metrics"`
	CustomMetrics    []MetricSpec `json:"custom_metrics,omitempty"`

	// Image name used for this container.
	Image string `json:"image,omitempty"`
}

// Container reference contains enough information to uniquely identify a container
type ContainerReference struct {
	// The container id
	Id string `json:"id,omitempty"`

	// The absolute name of the container. This is unique on the machine.
	Name string `json:"name"`

	// Other names by which the container is known within a certain namespace.
	// This is unique within that namespace.
	Aliases []string `json:"aliases,omitempty"`

	// Namespace under which the aliases of a container are unique.
	// An example of a namespace is "docker" for Docker containers.
	Namespace string `json:"namespace,omitempty"`

	Labels map[string]string `json:"labels,omitempty"`
}

// Sorts by container name.
type ContainerReferenceSlice []ContainerReference

func (self ContainerReferenceSlice) Len() int           { return len(self) }
func (self ContainerReferenceSlice) Swap(i, j int)      { self[i], self[j] = self[j], self[i] }
func (self ContainerReferenceSlice) Less(i, j int) bool { return self[i].Name < self[j].Name }

// ContainerInfoRequest is used when users check a container info from the REST API.
// It specifies how much data users want to get about a container
type ContainerInfoRequest struct {
	// Max number of stats to return. Specify -1 for all stats currently available.
	// Default: 60
	NumStats int `json:"num_stats,omitempty"`

	// Start time for which to query information.
	// If ommitted, the beginning of time is assumed.
	Start time.Time `json:"start,omitempty"`

	// End time for which to query information.
	// If ommitted, current time is assumed.
	End time.Time `json:"end,omitempty"`
}

// Returns a ContainerInfoRequest with all default values specified.
func DefaultContainerInfoRequest() ContainerInfoRequest {
	return ContainerInfoRequest{
		NumStats: 60,
	}
}

func (self *ContainerInfoRequest) Equals(other ContainerInfoRequest) bool {
	return self.NumStats == other.NumStats &&
		self.Start.Equal(other.Start) &&
		self.End.Equal(other.End)
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

// TODO(vmarmol): Refactor to not need this equality comparison.
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
	if !self.Spec.Eq(&b.Spec) {
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

func (self *ContainerSpec) Eq(b *ContainerSpec) bool {
	// Creation within 1s of each other.
	diff := self.CreationTime.Sub(b.CreationTime)
	if (diff > time.Second) || (diff < -time.Second) {
		return false
	}

	if self.HasCpu != b.HasCpu {
		return false
	}
	if !reflect.DeepEqual(self.Cpu, b.Cpu) {
		return false
	}
	if self.HasMemory != b.HasMemory {
		return false
	}
	if !reflect.DeepEqual(self.Memory, b.Memory) {
		return false
	}
	if self.HasNetwork != b.HasNetwork {
		return false
	}
	if self.HasFilesystem != b.HasFilesystem {
		return false
	}
	if self.HasDiskIo != b.HasDiskIo {
		return false
	}
	if self.HasCustomMetrics != b.HasCustomMetrics {
		return false
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

// This mirrors kernel internal structure.
type LoadStats struct {
	// Number of sleeping tasks.
	NrSleeping uint64 `json:"nr_sleeping"`

	// Number of running tasks.
	NrRunning uint64 `json:"nr_running"`

	// Number of tasks in stopped state
	NrStopped uint64 `json:"nr_stopped"`

	// Number of tasks in uninterruptible state
	NrUninterruptible uint64 `json:"nr_uninterruptible"`

	// Number of tasks waiting on IO
	NrIoWait uint64 `json:"nr_io_wait"`
}

// CPU usage time statistics.
type CpuUsage struct {
	// Total CPU usage.
	// Unit: nanoseconds.
	Total uint64 `json:"total"`

	// Per CPU/core usage of the container.
	// Unit: nanoseconds.
	PerCpu []uint64 `json:"per_cpu_usage,omitempty"`

	// Time spent in user space.
	// Unit: nanoseconds.
	User uint64 `json:"user"`

	// Time spent in kernel space.
	// Unit: nanoseconds.
	System uint64 `json:"system"`
}

// Cpu Completely Fair Scheduler statistics.
type CpuCFS struct {
	// Total number of elapsed enforcement intervals.
	Periods uint64 `json:"periods"`

	// Total number of times tasks in the cgroup have been throttled.
	ThrottledPeriods uint64 `json:"throttled_periods"`

	// Total time duration for which tasks in the cgroup have been throttled.
	// Unit: nanoseconds.
	ThrottledTime uint64 `json:"throttled_time"`
}

// All CPU usage metrics are cumulative from the creation of the container
type CpuStats struct {
	Usage CpuUsage `json:"usage"`
	CFS   CpuCFS   `json:"cfs"`
	// Smoothed average of number of runnable threads x 1000.
	// We multiply by thousand to avoid using floats, but preserving precision.
	// Load is smoothed over the last 10 seconds. Instantaneous value can be read
	// from LoadStats.NrRunning.
	LoadAverage int32 `json:"load_average"`
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
	IoServiceTime  []PerDiskStats `json:"io_service_time,omitempty"`
	IoWaitTime     []PerDiskStats `json:"io_wait_time,omitempty"`
	IoMerged       []PerDiskStats `json:"io_merged,omitempty"`
	IoTime         []PerDiskStats `json:"io_time,omitempty"`
}

type MemoryStats struct {
	// Current memory usage, this includes all memory regardless of when it was
	// accessed.
	// Units: Bytes.
	Usage uint64 `json:"usage"`

	// Number of bytes of page cache memory.
	// Units: Bytes.
	Cache uint64 `json:"cache"`

	// The amount of anonymous and swap cache memory (includes transparent
	// hugepages).
	// Units: Bytes.
	RSS uint64 `json:"rss"`

	// The amount of swap currently used by the processes in this cgroup
	// Units: Bytes.
	Swap uint64 `json:"swap"`

	// The amount of working set memory, this includes recently accessed memory,
	// dirty memory, and kernel memory. Working set is <= "usage".
	// Units: Bytes.
	WorkingSet uint64 `json:"working_set"`

	Failcnt uint64 `json:"failcnt"`

	ContainerData    MemoryStatsMemoryData `json:"container_data,omitempty"`
	HierarchicalData MemoryStatsMemoryData `json:"hierarchical_data,omitempty"`
}

type MemoryStatsMemoryData struct {
	Pgfault    uint64 `json:"pgfault"`
	Pgmajfault uint64 `json:"pgmajfault"`
}

type InterfaceStats struct {
	// The name of the interface.
	Name string `json:"name"`
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

type NetworkStats struct {
	InterfaceStats `json:",inline"`
	Interfaces     []InterfaceStats `json:"interfaces,omitempty"`
	// TCP connection stats (Established, Listen...)
	Tcp TcpStat `json:"tcp"`
	// TCP6 connection stats (Established, Listen...)
	Tcp6 TcpStat `json:"tcp6"`
}

type TcpStat struct {
	// Count of TCP connections in state "Established"
	Established uint64
	// Count of TCP connections in state "Syn_Sent"
	SynSent uint64
	// Count of TCP connections in state "Syn_Recv"
	SynRecv uint64
	// Count of TCP connections in state "Fin_Wait1"
	FinWait1 uint64
	// Count of TCP connections in state "Fin_Wait2"
	FinWait2 uint64
	// Count of TCP connections in state "Time_Wait
	TimeWait uint64
	// Count of TCP connections in state "Close"
	Close uint64
	// Count of TCP connections in state "Close_Wait"
	CloseWait uint64
	// Count of TCP connections in state "Listen_Ack"
	LastAck uint64
	// Count of TCP connections in state "Listen"
	Listen uint64
	// Count of TCP connections in state "Closing"
	Closing uint64
}

type FsStats struct {
	// The block device name associated with the filesystem.
	Device string `json:"device,omitempty"`

	// Type of the filesytem.
	Type string `json:"type"`

	// Number of bytes that can be consumed by the container on this filesystem.
	Limit uint64 `json:"capacity"`

	// Number of bytes that is consumed by the container on this filesystem.
	Usage uint64 `json:"usage"`

	// Base Usage that is consumed by the container's writable layer.
	// This field is only applicable for docker container's as of now.
	BaseUsage uint64 `json:"base_usage"`

	// Number of bytes available for non-root user.
	Available uint64 `json:"available"`

	// HasInodes when true, indicates that Inodes info will be available.
	HasInodes bool `json:"has_inodes"`

	// Number of Inodes
	Inodes uint64 `json:"inodes"`

	// Number of available Inodes
	InodesFree uint64 `json:"inodes_free"`

	// Number of reads completed
	// This is the total number of reads completed successfully.
	ReadsCompleted uint64 `json:"reads_completed"`

	// Number of reads merged
	// Reads and writes which are adjacent to each other may be merged for
	// efficiency.  Thus two 4K reads may become one 8K read before it is
	// ultimately handed to the disk, and so it will be counted (and queued)
	// as only one I/O.  This field lets you know how often this was done.
	ReadsMerged uint64 `json:"reads_merged"`

	// Number of sectors read
	// This is the total number of sectors read successfully.
	SectorsRead uint64 `json:"sectors_read"`

	// Number of milliseconds spent reading
	// This is the total number of milliseconds spent by all reads (as
	// measured from __make_request() to end_that_request_last()).
	ReadTime uint64 `json:"read_time"`

	// Number of writes completed
	// This is the total number of writes completed successfully.
	WritesCompleted uint64 `json:"writes_completed"`

	// Number of writes merged
	// See the description of reads merged.
	WritesMerged uint64 `json:"writes_merged"`

	// Number of sectors written
	// This is the total number of sectors written successfully.
	SectorsWritten uint64 `json:"sectors_written"`

	// Number of milliseconds spent writing
	// This is the total number of milliseconds spent by all writes (as
	// measured from __make_request() to end_that_request_last()).
	WriteTime uint64 `json:"write_time"`

	// Number of I/Os currently in progress
	// The only field that should go to zero. Incremented as requests are
	// given to appropriate struct request_queue and decremented as they finish.
	IoInProgress uint64 `json:"io_in_progress"`

	// Number of milliseconds spent doing I/Os
	// This field increases so long as field 9 is nonzero.
	IoTime uint64 `json:"io_time"`

	// weighted number of milliseconds spent doing I/Os
	// This field is incremented at each I/O start, I/O completion, I/O
	// merge, or read of these stats by the number of I/Os in progress
	// (field 9) times the number of milliseconds spent doing I/O since the
	// last update of this field.  This can provide an easy measure of both
	// I/O completion time and the backlog that may be accumulating.
	WeightedIoTime uint64 `json:"weighted_io_time"`
}

type ContainerStats struct {
	// The time of this stat point.
	Timestamp time.Time    `json:"timestamp"`
	Cpu       CpuStats     `json:"cpu,omitempty"`
	DiskIo    DiskIoStats  `json:"diskio,omitempty"`
	Memory    MemoryStats  `json:"memory,omitempty"`
	Network   NetworkStats `json:"network,omitempty"`

	// Filesystem statistics
	Filesystem []FsStats `json:"filesystem,omitempty"`

	// Task load stats
	TaskStats LoadStats `json:"task_stats,omitempty"`

	// Custom metrics from all collectors
	CustomMetrics map[string][]MetricVal `json:"custom_metrics,omitempty"`
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
	// TODO(vmarmol): Consider using this through reflection.
	if !reflect.DeepEqual(a.Cpu, b.Cpu) {
		return false
	}
	if !reflect.DeepEqual(a.Memory, b.Memory) {
		return false
	}
	if !reflect.DeepEqual(a.DiskIo, b.DiskIo) {
		return false
	}
	if !reflect.DeepEqual(a.Network, b.Network) {
		return false
	}
	if !reflect.DeepEqual(a.Filesystem, b.Filesystem) {
		return false
	}
	return true
}

// Event contains information general to events such as the time at which they
// occurred, their specific type, and the actual event. Event types are
// differentiated by the EventType field of Event.
type Event struct {
	// the absolute container name for which the event occurred
	ContainerName string `json:"container_name"`

	// the time at which the event occurred
	Timestamp time.Time `json:"timestamp"`

	// the type of event. EventType is an enumerated type
	EventType EventType `json:"event_type"`

	// the original event object and all of its extraneous data, ex. an
	// OomInstance
	EventData EventData `json:"event_data,omitempty"`
}

// EventType is an enumerated type which lists the categories under which
// events may fall. The Event field EventType is populated by this enum.
type EventType string

const (
	EventOom               EventType = "oom"
	EventOomKill                     = "oomKill"
	EventContainerCreation           = "containerCreation"
	EventContainerDeletion           = "containerDeletion"
)

// Extra information about an event. Only one type will be set.
type EventData struct {
	// Information about an OOM kill event.
	OomKill *OomKillEventData `json:"oom,omitempty"`
}

// Information related to an OOM kill instance
type OomKillEventData struct {
	// process id of the killed process
	Pid int `json:"pid"`

	// The name of the killed process
	ProcessName string `json:"process_name"`
}

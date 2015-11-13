/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package v1alpha1

import (
	"k8s.io/kubernetes/pkg/api/unversioned"
)

// RawNode holds node-level unprocessed sample metrics.
type RawNode struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard list metadata, applying to the lists of stats.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#types-kinds
	unversioned.ListMeta `json:"metadata,omitempty"`
	// Reference to the measured Node.
	NodeName string `json:"nodeName,omitempty"`
	// Overall machine metrics.
	Machine RawContainer `json:"machine,omitempty"`
	// Metrics of system components.
	SystemContainers []RawContainer `json:"systemContainers,omitempty"`
}

// RawPod holds pod-level unprocessed sample metrics.
type RawPod struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard list metadata, applying to the lists of stats.
	// More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#types-kinds
	unversioned.ListMeta `json:"metadata,omitempty"`
	// Reference to the measured Pod.
	PodRef unversioned.ObjectReference `json:"podRef,omitempty"`
	// Metrics of containers in the measured pod.
	Containers []RawContainer `json:"containers,omitempty"`
}

// RawContainer holds container-level unprocessed sample metrics.
type RawContainer struct {
	// Reference to the measured container.
	Name string `json:"name,omitempty"`
	// Describes the container resources.
	Spec ContainerSpec `json:"spec,omitempty"`
	// Historical metric samples gathered from the container.
	Stats []ContainerStats `json:"stats,omitempty"`
}

type ContainerSpec struct {
	// Time at which the container was created.
	CreationTime unversioned.Time `json:"creationTime,omitempty"`

	// Other names by which the container is known within a certain namespace.
	// This is unique within that namespace.
	Aliases []string `json:"aliases,omitempty"`

	// Namespace under which the aliases of a container are unique (not a Kubernetes namespace).
	// An example of a namespace is "docker" for Docker containers.
	Namespace string `json:"namespace,omitempty"`

	// Metadata labels associated with this container (not Kubernetes labels).
	Labels map[string]string `json:"labels,omitempty"`

	// Whether the container has CPU metrics.
	HasCpu bool    `json:"hasCpu,omitempty"`
	Cpu    CpuSpec `json:"cpu,omitempty"`

	// Whether the container has memory metrics.
	HasMemory bool       `json:"hasMemory,omitempty"`
	Memory    MemorySpec `json:"memory,omitempty"`

	// Whether the container has memory metrics.
	HasCustomMetrics bool         `json:"hasCustomMetrics,omitempty"`
	CustomMetrics    []MetricSpec `json:"customMetrics,omitempty"`

	// Following resources have no associated spec, but are being isolated.
	HasNetwork    bool `json:"hasNetwork,omitempty"`
	HasFilesystem bool `json:"hasFilesystem,omitempty"`
	HasDiskIo     bool `json:"hasDiskIo,omitempty"`

	// Image name used for this container.
	Image string `json:"image,omitempty"`
}

type CpuSpec struct {
	// Requested core shares. Default is 1024.
	// Units: millicore-seconds per second.
	Limit uint64 `json:"limit,omitempty"`
	// Requested cpu hard limit. Default is unlimited (0).
	// Units: millicore-seconds per second.
	MaxLimit uint64 `json:"maxLimit,omitempty"`
	// Cpu affinity mask.
	Mask string `json:"mask,omitempty"`
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
	SwapLimit uint64 `json:"swapLimit,omitempty"`
}

type ContainerStats struct {
	// The time of this stat point.
	Timestamp unversioned.Time `json:"timestamp,omitempty"`
	// CPU statistics
	HasCpu bool `json:"hasCpu,omitempty"`
	// In nanocore-seconds (aggregated)
	Cpu CpuStats `json:"cpu,omitempty"`
	// In nanocore-seconds per second (instantaneous)
	CpuInst *CpuInstStats `json:"cpuInst,omitempty"`
	// Disk IO statistics
	HasDiskIo bool        `json:"hasDiskIo,omitempty"`
	DiskIo    DiskIoStats `json:"diskIo,omitempty"`
	// Memory statistics
	HasMemory bool        `json:"hasMemory,omitempty"`
	Memory    MemoryStats `json:"memory,omitempty"`
	// Network statistics
	HasNetwork bool         `json:"hasNetwork,omitempty"`
	Network    NetworkStats `json:"network,omitempty"`
	// Filesystem statistics
	HasFilesystem bool      `json:"hasFilesystem,omitempty"`
	Filesystem    []FsStats `json:"filesystem,omitempty"`
	// Task load statistics
	HasLoad bool      `json:"hasLoad,omitempty"`
	Load    LoadStats `json:"load,omitempty"`
	// Custom Metrics
	HasCustomMetrics bool           `json:"hasCustomMetrics,omitempty"`
	CustomMetrics    []CustomMetric `json:"customMetrics,omitempty"`
}

type TcpStat struct {
	// Number of connections in the Established state.
	Established uint64 `json:"established,omitempty"`
	// Number of connections in the SynSent state.
	SynSent uint64 `json:"synSent,omitempty"`
	// Number of connections in the SynRecv state.
	SynRecv uint64 `json:"synRecv,omitempty"`
	// Number of connections in the FinWait1 state.
	FinWait1 uint64 `json:"finWait1,omitempty"`
	// Number of connections in the FinWait2 state.
	FinWait2 uint64 `json:"finWait2,omitempty"`
	// Number of connections in the TimeWait state.
	TimeWait uint64 `json:"timeWait,omitempty"`
	// Number of connections in the Close state.
	Close uint64 `json:"close,omitempty"`
	// Number of connections in the CloseWait state.
	CloseWait uint64 `json:"closeWait,omitempty"`
	// Number of connections in the LastAck state.
	LastAck uint64 `json:"lastAck,omitempty"`
	// Number of connections in the Listen state.
	Listen uint64 `json:"listen,omitempty"`
	// Number of connections in the Closing state.
	Closing uint64 `json:"closing,omitempty"`
}

type NetworkStats struct {
	// Network stats by interface.
	Interfaces []InterfaceStats `json:"interfaces,omitempty"`
	// TCP connection stats (Established, Listen...)
	Tcp TcpStat `json:"tcp,omitempty"`
	// TCP6 connection stats (Established, Listen...)
	Tcp6 TcpStat `json:"tcp6,omitempty"`
}

type InterfaceStats struct {
	// The name of the interface.
	Name string `json:"name,omitempty"`
	// Cumulative count of bytes received.
	RxBytes uint64 `json:"rxBytes,omitempty"`
	// Cumulative count of packets received.
	RxPackets uint64 `json:"rxPackets,omitempty"`
	// Cumulative count of receive errors encountered.
	RxErrors uint64 `json:"rxErrors,omitempty"`
	// Cumulative count of packets dropped while receiving.
	RxDropped uint64 `json:"rxDropped,omitempty"`
	// Cumulative count of bytes transmitted.
	TxBytes uint64 `json:"txBytes,omitempty"`
	// Cumulative count of packets transmitted.
	TxPackets uint64 `json:"txPackets,omitempty"`
	// Cumulative count of transmit errors encountered.
	TxErrors uint64 `json:"txErrors,omitempty"`
	// Cumulative count of packets dropped while transmitting.
	TxDropped uint64 `json:"txDropped,omitempty"`
}

// Instantaneous CPU stats
type CpuInstStats struct {
	Usage CpuInstUsage `json:"usage,omitempty"`
}

// CPU usage time statistics.
type CpuInstUsage struct {
	// Total CPU usage (sum of all cores).
	// Units: nanocore-seconds per second
	Total uint64 `json:"total,omitempty"`

	// Per core usage of the container, indexed by CPU index (0 to CPU MAX).
	// Unit: nanocore-seconds per second
	PerCpu []uint64 `json:"perCpu,omitempty"`

	// Usage spent in user space.
	// Unit: nanocore-seconds per second
	User uint64 `json:"user,omitempty"`

	// Usage spent in kernel space.
	// Unit: nanocore-seconds per second
	System uint64 `json:"system,omitempty"`
}

// CPU usage time statistics.
type CpuUsage struct {
	// Total CPU usage (sum of all cores).
	// Units: nanocore-seconds
	Total uint64 `json:"total,omitempty"`

	// Per core usage of the container, indexed by CPU index (0 to CPU MAX).
	// Unit: nanocore-seconds
	PerCpu []uint64 `json:"perCpu,omitempty"`

	// Usage spent in user space.
	// Unit: nanocore-seconds
	User uint64 `json:"user,omitempty"`

	// Usage spent in kernel space.
	// Unit: nanocore-seconds
	System uint64 `json:"system,omitempty"`
}

// All CPU usage metrics are cumulative from the creation of the container
type CpuStats struct {
	Usage CpuUsage `json:"usage,omitempty"`
	// CPU load that the container is experiencing, represented as a smoothed
	// average of number of runnable threads x 1000.  We multiply by thousand to
	// avoid using floats, but preserving precision.  Load is smoothed over the
	// last 10 seconds. Instantaneous value can be read from LoadStats.NrRunning.
	LoadAverage int32 `json:"loadAverage,omitempty"`
}

type PerDiskStats struct {
	// Major device number. See https://www.kernel.org/doc/Documentation/devices.txt
	Major uint64 `json:"major,omitempty"`
	// Minor device number. See https://www.kernel.org/doc/Documentation/devices.txt
	Minor uint64            `json:"minor,omitempty"`
	Stats map[string]uint64 `json:"stats,omitempty"`
}

// Disk IO stats, as reported by the cgroup block io controller.
// See https://www.kernel.org/doc/Documentation/cgroups/blkio-controller.txt
type DiskIoStats struct {
	IoServiceBytes []PerDiskStats `json:"ioServiceBytes,omitempty"`
	IoServiced     []PerDiskStats `json:"ioServiced,omitempty"`
	IoQueued       []PerDiskStats `json:"ioQueued,omitempty"`
	Sectors        []PerDiskStats `json:"sectors,omitempty"`
	IoServiceTime  []PerDiskStats `json:"ioServiceTime,omitempty"`
	IoWaitTime     []PerDiskStats `json:"ioWaitTime,omitempty"`
	IoMerged       []PerDiskStats `json:"ioMerged,omitempty"`
	IoTime         []PerDiskStats `json:"ioTime,omitempty"`
}

type MemoryStats struct {
	// Current memory usage, this includes all memory regardless of when it was
	// accessed.
	// Units: Bytes.
	Usage uint64 `json:"usage,omitempty"`

	// The amount of working set memory, this includes recently accessed memory,
	// dirty memory, and kernel memory. Working set is <= "usage".
	// Units: Bytes.
	WorkingSet uint64 `json:"workingSet,omitempty"`

	// Cumulative number of times that a usage counter hit its limit
	Failcnt uint64 `json:"failcnt,omitempty"`

	ContainerData    MemoryStatsMemoryData `json:"containerData,omitempty"`
	HierarchicalData MemoryStatsMemoryData `json:"hierarchicalData,omitempty"`
}

type MemoryStatsMemoryData struct {
	// Cumulative number of minor page faults.
	Pgfault uint64 `json:"pgfault,omitempty"`
	// Cumulative number of major page faults.
	Pgmajfault uint64 `json:"pgmajfault,omitempty"`
}

type FsStats struct {
	// The block device name associated with the filesystem.
	Device string `json:"device,omitempty"`

	// Number of bytes that can be consumed by the container on this filesystem.
	Limit uint64 `json:"limit,omitempty"`

	// Number of bytes that is consumed by the container on this filesystem.
	Usage uint64 `json:"usage,omitempty"`

	// Number of bytes available for non-root user.
	Available uint64 `json:"available,omitempty"`

	// Number of reads completed
	// This is the total number of reads completed successfully.
	ReadsCompleted uint64 `json:"readsCompleted,omitempty"`

	// Number of reads merged
	// Reads and writes which are adjacent to each other may be merged for
	// efficiency.  Thus two 4K reads may become one 8K read before it is
	// ultimately handed to the disk, and so it will be counted (and queued)
	// as only one I/O.  This field lets you know how often this was done.
	ReadsMerged uint64 `json:"readsMerged,omitempty"`

	// Number of sectors read
	// This is the total number of sectors read successfully.
	SectorsRead uint64 `json:"sectorsRead,omitempty"`

	// Number of milliseconds spent reading
	// This is the total number of milliseconds spent by all reads (as
	// measured from __make_request() to end_that_request_last()).
	ReadTime uint64 `json:"readTime,omitempty"`

	// Number of writes completed
	// This is the total number of writes completed successfully.
	WritesCompleted uint64 `json:"writesCompleted,omitempty"`

	// Number of writes merged
	// See the description of reads merged.
	WritesMerged uint64 `json:"writesMerged,omitempty"`

	// Number of sectors written
	// This is the total number of sectors written successfully.
	SectorsWritten uint64 `json:"sectorsWritten,omitempty"`

	// Number of milliseconds spent writing
	// This is the total number of milliseconds spent by all writes (as
	// measured from __make_request() to end_that_request_last()).
	WriteTime uint64 `json:"writeTime,omitempty"`

	// Number of I/Os currently in progress
	// The only field that should go to zero. Incremented as requests are
	// given to appropriate struct request_queue and decremented as they finish.
	IoInProgress uint64 `json:"ioInProgress,omitempty"`

	// Number of milliseconds spent doing I/Os
	// This field increases so long as field 9 is nonzero.
	IoTime uint64 `json:"ioTime,omitempty"`

	// weighted number of milliseconds spent doing I/Os
	// This field is incremented at each I/O start, I/O completion, I/O
	// merge, or read of these stats by the number of I/Os in progress
	// (field 9) times the number of milliseconds spent doing I/O since the
	// last update of this field.  This can provide an easy measure of both
	// I/O completion time and the backlog that may be accumulating.
	WeightedIoTime uint64 `json:"weightedIoTime,omitempty"`
}

// This mirrors kernel internal structure.
type LoadStats struct {
	// Number of sleeping tasks.
	NrSleeping uint64 `json:"nrSleeping,omitempty"`

	// Number of running tasks.
	NrRunning uint64 `json:"nrRunning,omitempty"`

	// Number of tasks in stopped state
	NrStopped uint64 `json:"nrStopped,omitempty"`

	// Number of tasks in uninterruptible state
	NrUninterruptible uint64 `json:"nrUninterruptible,omitempty"`

	// Number of tasks waiting on IO
	NrIoWait uint64 `json:"nrIoWait,omitempty"`
}

// Type of metric being exported.
type MetricType string

const (
	// Instantaneous value. May increase or decrease.
	MetricGauge MetricType = "gauge"

	// A counter-like value that is only expected to increase.
	MetricCumulative MetricType = "cumulative"

	// Rate over a time period.
	MetricDelta MetricType = "delta"
)

// DataType for metric being exported.
type DataType string

const (
	IntType   DataType = "int"
	FloatType DataType = "float"
)

// Spec for custom metric.
type MetricSpec struct {
	// The name of the metric.
	Name string `json:"name,omitempty"`

	// Type of the metric.
	Type MetricType `json:"type,omitempty"`

	// Data Type for the stats.
	Format DataType `json:"format,omitempty"`

	// Display Units for the stats.
	Units string `json:"units,omitempty"`
}

type CustomMetric struct {
	Name   string      `json:"name,omitempty"`
	Values []MetricVal `json:"values,omitempty"`
}

// An exported metric.
type MetricVal struct {
	// Label associated with a metric
	Label string `json:"label,omitempty"`

	// Time at which the metric was queried
	Timestamp unversioned.Time `json:"timestamp,omitempty"`

	// The value of the metric at this point.
	IntValue   int64   `json:"intValue,omitempty"`
	FloatValue float64 `json:"floatValue,omitempty"`
}

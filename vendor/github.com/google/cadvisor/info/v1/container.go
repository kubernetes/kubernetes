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

type ProcessSpec struct {
	Limit uint64 `json:"limit,omitempty"`
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

	HasHugetlb bool `json:"has_hugetlb"`

	HasNetwork bool `json:"has_network"`

	HasProcesses bool        `json:"has_processes"`
	Processes    ProcessSpec `json:"processes,omitempty"`

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
}

// Sorts by container name.
type ContainerReferenceSlice []ContainerReference

func (s ContainerReferenceSlice) Len() int           { return len(s) }
func (s ContainerReferenceSlice) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }
func (s ContainerReferenceSlice) Less(i, j int) bool { return s[i].Name < s[j].Name }

// ContainerInfoRequest is used when users check a container info from the REST API.
// It specifies how much data users want to get about a container
type ContainerInfoRequest struct {
	// Max number of stats to return. Specify -1 for all stats currently available.
	// Default: 60
	NumStats int `json:"num_stats,omitempty"`

	// Start time for which to query information.
	// If omitted, the beginning of time is assumed.
	Start time.Time `json:"start,omitempty"`

	// End time for which to query information.
	// If omitted, current time is assumed.
	End time.Time `json:"end,omitempty"`
}

// Returns a ContainerInfoRequest with all default values specified.
func DefaultContainerInfoRequest() ContainerInfoRequest {
	return ContainerInfoRequest{
		NumStats: 60,
	}
}

func (r *ContainerInfoRequest) Equals(other ContainerInfoRequest) bool {
	return r.NumStats == other.NumStats &&
		r.Start.Equal(other.Start) &&
		r.End.Equal(other.End)
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
func (ci *ContainerInfo) Eq(b *ContainerInfo) bool {

	// If both ci and b are nil, then Eq() returns true
	if ci == nil {
		return b == nil
	}
	if b == nil {
		return ci == nil
	}

	// For fields other than time.Time, we will compare them precisely.
	// This would require that any slice should have same order.
	if !reflect.DeepEqual(ci.ContainerReference, b.ContainerReference) {
		return false
	}
	if !reflect.DeepEqual(ci.Subcontainers, b.Subcontainers) {
		return false
	}
	if !ci.Spec.Eq(&b.Spec) {
		return false
	}

	for i, expectedStats := range b.Stats {
		selfStats := ci.Stats[i]
		if !expectedStats.Eq(selfStats) {
			return false
		}
	}

	return true
}

func (s *ContainerSpec) Eq(b *ContainerSpec) bool {
	// Creation within 1s of each other.
	diff := s.CreationTime.Sub(b.CreationTime)
	if (diff > time.Second) || (diff < -time.Second) {
		return false
	}

	if s.HasCpu != b.HasCpu {
		return false
	}
	if !reflect.DeepEqual(s.Cpu, b.Cpu) {
		return false
	}
	if s.HasMemory != b.HasMemory {
		return false
	}
	if !reflect.DeepEqual(s.Memory, b.Memory) {
		return false
	}
	if s.HasHugetlb != b.HasHugetlb {
		return false
	}
	if s.HasNetwork != b.HasNetwork {
		return false
	}
	if s.HasProcesses != b.HasProcesses {
		return false
	}
	if s.HasFilesystem != b.HasFilesystem {
		return false
	}
	if s.HasDiskIo != b.HasDiskIo {
		return false
	}
	if s.HasCustomMetrics != b.HasCustomMetrics {
		return false
	}
	if s.Image != b.Image {
		return false
	}
	return true
}

func (ci *ContainerInfo) StatsAfter(ref time.Time) []*ContainerStats {
	n := len(ci.Stats) + 1
	for i, s := range ci.Stats {
		if s.Timestamp.After(ref) {
			n = i
			break
		}
	}
	if n > len(ci.Stats) {
		return nil
	}
	return ci.Stats[n:]
}

func (ci *ContainerInfo) StatsStartTime() time.Time {
	var ret time.Time
	for _, s := range ci.Stats {
		if s.Timestamp.Before(ret) || ret.IsZero() {
			ret = s.Timestamp
		}
	}
	return ret
}

func (ci *ContainerInfo) StatsEndTime() time.Time {
	var ret time.Time
	for i := len(ci.Stats) - 1; i >= 0; i-- {
		s := ci.Stats[i]
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

// Cpu Aggregated scheduler statistics
type CpuSchedstat struct {
	// https://www.kernel.org/doc/Documentation/scheduler/sched-stats.txt

	// time spent on the cpu
	RunTime uint64 `json:"run_time"`
	// time spent waiting on a runqueue
	RunqueueTime uint64 `json:"runqueue_time"`
	// # of timeslices run on this cpu
	RunPeriods uint64 `json:"run_periods"`
}

// All CPU usage metrics are cumulative from the creation of the container
type CpuStats struct {
	Usage     CpuUsage     `json:"usage"`
	CFS       CpuCFS       `json:"cfs"`
	Schedstat CpuSchedstat `json:"schedstat"`
	// Smoothed average of number of runnable threads x 1000.
	// We multiply by thousand to avoid using floats, but preserving precision.
	// Load is smoothed over the last 10 seconds. Instantaneous value can be read
	// from LoadStats.NrRunning.
	LoadAverage int32 `json:"load_average"`
}

type PerDiskStats struct {
	Device string            `json:"device"`
	Major  uint64            `json:"major"`
	Minor  uint64            `json:"minor"`
	Stats  map[string]uint64 `json:"stats"`
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

type HugetlbStats struct {
	// current res_counter usage for hugetlb
	Usage uint64 `json:"usage,omitempty"`
	// maximum usage ever recorded.
	MaxUsage uint64 `json:"max_usage,omitempty"`
	// number of times hugetlb usage allocation failure.
	Failcnt uint64 `json:"failcnt"`
}

type MemoryStats struct {
	// Current memory usage, this includes all memory regardless of when it was
	// accessed.
	// Units: Bytes.
	Usage uint64 `json:"usage"`

	// Maximum memory usage recorded.
	// Units: Bytes.
	MaxUsage uint64 `json:"max_usage"`

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

	// The amount of memory used for mapped files (includes tmpfs/shmem)
	MappedFile uint64 `json:"mapped_file"`

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
	// UDP connection stats
	Udp UdpStat `json:"udp"`
	// UDP6 connection stats
	Udp6 UdpStat `json:"udp6"`
	// TCP advanced stats
	TcpAdvanced TcpAdvancedStat `json:"tcp_advanced"`
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

type TcpAdvancedStat struct {
	// The algorithm used to determine the timeout value used for
	// retransmitting unacknowledged octets, ref: RFC2698, default 1
	RtoAlgorithm uint64
	// The minimum value permitted by a TCP implementation for the
	// retransmission timeout, measured in milliseconds, default 200ms
	RtoMin uint64
	// The maximum value permitted by a TCP implementation for the
	// retransmission timeout, measured in milliseconds, default 120s
	RtoMax uint64
	// The limit on the total number of TCP connections the entity
	// can support., default -1, i.e. infinity
	MaxConn int64

	// The number of times TCP connections have made a direct
	// transition to the SYN-SENT state from the CLOSED state.
	ActiveOpens uint64
	// The number of times TCP connections have made a direct
	// transition to the SYN-RCVD state from the LISTEN state.
	PassiveOpens uint64
	// The number of times TCP connections have made a direct
	// transition to the CLOSED state from either the SYN-SENT
	// state or the SYN-RCVD state, plus the number of times TCP
	// connections have made a direct transition to the LISTEN
	// state from the SYN-RCVD state.
	AttemptFails uint64
	// The number of times TCP connections have made a direct
	// transition to the CLOSED state from either the ESTABLISHED
	// state or the CLOSE-WAIT state.
	EstabResets uint64
	// The number of TCP connections for which the current state
	// is either ESTABLISHED or CLOSE- WAIT.
	CurrEstab uint64

	// The total number of segments received, including those
	// received in error.
	InSegs uint64
	// The total number of segments sent, including those on
	// current connections but excluding those containing only
	// retransmitted octets.
	OutSegs uint64
	// The total number of segments retransmitted - that is, the
	// number of TCP segments transmitted containing one or more
	// previously transmitted octets.
	RetransSegs uint64
	// The total number of segments received in error (e.g., bad
	// TCP checksums).
	InErrs uint64
	// The number of TCP segments sent containing the RST flag.
	OutRsts uint64
	// The number of IP Packets with checksum errors
	InCsumErrors uint64
	// The number of resets received for embryonic SYN_RECV sockets
	EmbryonicRsts uint64

	// The number of SYN cookies sent
	SyncookiesSent uint64
	// The number of SYN cookies received
	SyncookiesRecv uint64
	// The number of invalid SYN cookies received
	SyncookiesFailed uint64

	// The number of packets pruned from receive queue because of socket buffer overrun
	PruneCalled uint64
	// The number of packets pruned from receive queue
	RcvPruned uint64
	// The number of packets dropped from out-of-order queue because of socket buffer overrun
	OfoPruned uint64
	// The number of ICMP packets dropped because they were out-of-window
	OutOfWindowIcmps uint64
	// The number of ICMP packets dropped because socket was locked
	LockDroppedIcmps uint64

	// The number of TCP sockets finished time wait in fast timer
	TW uint64
	// The number of time wait sockets recycled by time stamp
	TWRecycled uint64
	// The number of TCP sockets finished time wait in slow timer
	TWKilled uint64
	// counter, if no more mem for TIME-WAIT struct, +1
	TCPTimeWaitOverflow uint64

	// The number of RTO timer first timeout times
	TCPTimeouts uint64
	// The number of fake timeouts detected by F-RTO
	TCPSpuriousRTOs uint64
	// The number of send Tail Loss Probe (TLP) times by Probe Timeout(PTO)
	TCPLossProbes uint64
	// The number of recovery times by TLP
	TCPLossProbeRecovery uint64
	// The number of RTO failed times when in Recovery state, and remote end has no sack
	TCPRenoRecoveryFail uint64
	// The number of RTO failed times when in Recovery state, and remote end has sack
	TCPSackRecoveryFail uint64
	// The number of RTO failed times when in TCP_CA_Disorder state, and remote end has no sack
	TCPRenoFailures uint64
	// The number of RTO failed times when in TCP_CA_Disorder state, and remote end has sack
	TCPSackFailures uint64
	// The number of RTO failed times when in TCP_CA_Loss state,
	TCPLossFailures uint64

	// The number of delayed acks sent
	DelayedACKs uint64
	// The number of delayed acks further delayed because of locked socket
	DelayedACKLocked uint64
	// The number of quick ack mode was activated times
	DelayedACKLost uint64
	// The number of times the listen queue of a socket overflowed
	ListenOverflows uint64
	// The number of SYNs to LISTEN sockets dropped
	ListenDrops uint64
	// The number of packet headers predicted
	TCPHPHits uint64
	// The number of acknowledgments not containing data payload received
	TCPPureAcks uint64
	// The number of predicted acknowledgments
	TCPHPAcks uint64
	// The number of times recovered from packet loss due to fast retransmit
	TCPRenoRecovery uint64
	// The number of SACK retransmits failed
	TCPSackRecovery uint64
	// The number of bad SACK blocks received
	TCPSACKReneging uint64
	// The number of detected reordering times using FACK
	TCPFACKReorder uint64
	// The number of detected reordering times using SACK
	TCPSACKReorder uint64
	// The number of detected reordering times using Reno
	TCPRenoReorder uint64
	// The number of detected reordering times using time stamp
	TCPTSReorder uint64
	// The number of congestion windows fully recovered without slow start
	TCPFullUndo uint64
	// The number of congestion windows partially recovered using Hoe heuristic
	TCPPartialUndo uint64
	// The number of congestion windows recovered without slow start by DSACK
	TCPDSACKUndo uint64
	// The number of congestion windows recovered without slow start after partial ack
	TCPLossUndo uint64

	// The number of fast retransmits
	TCPFastRetrans uint64
	// The number of retransmits in slow start
	TCPSlowStartRetrans uint64
	// The number of retransmits lost
	TCPLostRetransmit uint64
	// The number of retransmits failed, including FastRetrans, SlowStartRetrans
	TCPRetransFail uint64

	// he number of packets collapsed in receive queue due to low socket buffer
	TCPRcvCollapsed uint64
	// The number of DSACKs sent for old packets
	TCPDSACKOldSent uint64
	// The number of DSACKs sent for out of order packets
	TCPDSACKOfoSent uint64
	// The number of DSACKs received
	TCPDSACKRecv uint64
	// The number of DSACKs for out of order packets received
	TCPDSACKOfoRecv uint64
	// The number of connections reset due to unexpected data
	TCPAbortOnData uint64
	// The number of connections reset due to early user close
	TCPAbortOnClose uint64
	// The number of connections aborted due to memory pressure
	TCPAbortOnMemory uint64
	// The number of connections aborted due to timeout
	TCPAbortOnTimeout uint64
	// The number of connections aborted after user close in linger timeout
	TCPAbortOnLinger uint64
	// The number of times unable to send RST due to no memory
	TCPAbortFailed uint64
	// The number of TCP ran low on memory times
	TCPMemoryPressures uint64
	// The number of TCP cumulative duration of
	// memory pressure events, by ms
	TCPMemoryPressuresChrono uint64
	// The number of SACKs discard
	TCPSACKDiscard uint64
	// The number of DSACKs ignore old
	TCPDSACKIgnoredOld uint64
	// The number of DSACKs ignore no undo
	TCPDSACKIgnoredNoUndo uint64

	// The number of MD5 not found
	TCPMD5NotFound uint64
	// The number of MD5 unexpected
	TCPMD5Unexpected uint64
	// The number of MD5 failed
	TCPMD5Failure uint64
	// The number of Sack shifted
	TCPSackShifted uint64
	// The number of Sack merged
	TCPSackMerged uint64
	// The number of Sack shift fall back
	TCPSackShiftFallback uint64
	// The number of Backlog drop
	TCPBacklogDrop uint64
	// The number of PFmemalloc drop
	PFMemallocDrop uint64
	// The number of memalloc drop
	TCPMinTTLDrop uint64
	// The number of DeferAccept drop
	TCPDeferAcceptDrop uint64
	// The number of IP reverse path filter
	IPReversePathFilter uint64

	// The number of request full do cookies
	TCPReqQFullDoCookies uint64
	// The number of request full drop
	TCPReqQFullDrop uint64

	// number of successful outbound TFO connections
	TCPFastOpenActive uint64
	// number of SYN-ACK packets received that did not acknowledge data
	// sent in the SYN packet and caused a retransmissions without SYN data.
	TCPFastOpenActiveFail uint64
	// number of successful inbound TFO connections
	TCPFastOpenPassive uint64
	// number of inbound SYN packets with TFO cookie that was invalid
	TCPFastOpenPassiveFail uint64
	// number of inbound SYN packets that will have TFO disabled because
	// the socket has exceeded the max queue length
	TCPFastOpenListenOverflow uint64
	// number of inbound SYN packets requesting TFO with TFO set but no cookie
	TCPFastOpenCookieReqd uint64

	// number of SYN and SYN/ACK retransmits to break down retransmissions
	// into SYN, fast-retransmits, timeout retransmits, etc.
	TCPSynRetrans uint64
	// number of outgoing packets with original data
	// (excluding retransmission but including data-in-SYN).
	TCPOrigDataSent uint64

	// The number of active connections rejected because of time stamp
	PAWSActive uint64
	// The number of packetes rejected in established connections because of timestamp
	PAWSEstab uint64
}

type UdpStat struct {
	// Count of UDP sockets in state "Listen"
	Listen uint64

	// Count of UDP packets dropped by the IP stack
	Dropped uint64

	// Count of packets Queued for Receieve
	RxQueued uint64

	// Count of packets Queued for Transmit
	TxQueued uint64
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

type AcceleratorStats struct {
	// Make of the accelerator (nvidia, amd, google etc.)
	Make string `json:"make"`

	// Model of the accelerator (tesla-p100, tesla-k80 etc.)
	Model string `json:"model"`

	// ID of the accelerator.
	ID string `json:"id"`

	// Total accelerator memory.
	// unit: bytes
	MemoryTotal uint64 `json:"memory_total"`

	// Total accelerator memory allocated.
	// unit: bytes
	MemoryUsed uint64 `json:"memory_used"`

	// Percent of time over the past sample period during which
	// the accelerator was actively processing.
	DutyCycle uint64 `json:"duty_cycle"`
}

// PerfStat represents value of a single monitored perf event.
type PerfStat struct {
	// Indicates scaling ratio for an event: time_running/time_enabled
	// (amount of time that event was being measured divided by
	// amount of time that event was enabled for).
	// value 1.0 indicates that no multiplexing occurred. Value close
	// to 0 indicates that event was measured for short time and event's
	// value might be inaccurate.
	// See: https://lwn.net/Articles/324756/
	ScalingRatio float64 `json:"scaling_ratio"`

	// Value represents value of perf event retrieved from OS. It is
	// normalized against ScalingRatio and takes multiplexing into
	// consideration.
	Value uint64 `json:"value"`

	// Name is human readable name of an event.
	Name string `json:"name"`

	// CPU that perf event was measured on.
	Cpu int `json:"cpu"`
}

// MemoryBandwidthStats corresponds to MBM (Memory Bandwidth Monitoring).
// See: https://01.org/cache-monitoring-technology
// See: https://www.kernel.org/doc/Documentation/x86/intel_rdt_ui.txt
type MemoryBandwidthStats struct {
	// The 'mbm_total_bytes'.
	TotalBytes uint64 `json:"mbm_total_bytes,omitempty"`

	// The 'mbm_local_bytes'.
	LocalBytes uint64 `json:"mbm_local_bytes,omitempty"`
}

// CacheStats corresponds to CMT (Cache Monitoring Technology).
// See: https://01.org/cache-monitoring-technology
// See: https://www.kernel.org/doc/Documentation/x86/intel_rdt_ui.txt
type CacheStats struct {
	// The 'llc_occupancy'.
	LLCOccupancy uint64 `json:"llc_occupancy,omitempty"`
}

// ResctrlStats corresponds to statistics from Resource Control.
type ResctrlStats struct {
	// Each NUMA Node statistics corresponds to one element in the array.
	MemoryBandwidth []MemoryBandwidthStats `json:"memory_bandwidth,omitempty"`
	Cache           []CacheStats           `json:"cache,omitempty"`
}

// PerfUncoreStat represents value of a single monitored perf uncore event.
type PerfUncoreStat struct {
	// Indicates scaling ratio for an event: time_running/time_enabled
	// (amount of time that event was being measured divided by
	// amount of time that event was enabled for).
	// value 1.0 indicates that no multiplexing occurred. Value close
	// to 0 indicates that event was measured for short time and event's
	// value might be inaccurate.
	// See: https://lwn.net/Articles/324756/
	ScalingRatio float64 `json:"scaling_ratio"`

	// Value represents value of perf event retrieved from OS. It is
	// normalized against ScalingRatio and takes multiplexing into
	// consideration.
	Value uint64 `json:"value"`

	// Name is human readable name of an event.
	Name string `json:"name"`

	// Socket that perf event was measured on.
	Socket int `json:"socket"`

	// PMU is Performance Monitoring Unit which collected these stats.
	PMU string `json:"pmu"`
}

type UlimitSpec struct {
	Name      string `json:"name"`
	SoftLimit int64  `json:"soft_limit"`
	HardLimit int64  `json:"hard_limit"`
}

type ProcessStats struct {
	// Number of processes
	ProcessCount uint64 `json:"process_count"`

	// Number of open file descriptors
	FdCount uint64 `json:"fd_count"`

	// Number of sockets
	SocketCount uint64 `json:"socket_count"`

	// Number of threads currently in container
	ThreadsCurrent uint64 `json:"threads_current,omitempty"`

	// Maxium number of threads allowed in container
	ThreadsMax uint64 `json:"threads_max,omitempty"`

	// Ulimits for the top-level container process
	Ulimits []UlimitSpec `json:"ulimits,omitempty"`
}

type ContainerStats struct {
	// The time of this stat point.
	Timestamp time.Time               `json:"timestamp"`
	Cpu       CpuStats                `json:"cpu,omitempty"`
	DiskIo    DiskIoStats             `json:"diskio,omitempty"`
	Memory    MemoryStats             `json:"memory,omitempty"`
	Hugetlb   map[string]HugetlbStats `json:"hugetlb,omitempty"`
	Network   NetworkStats            `json:"network,omitempty"`
	// Filesystem statistics
	Filesystem []FsStats `json:"filesystem,omitempty"`

	// Task load stats
	TaskStats LoadStats `json:"task_stats,omitempty"`

	// Metrics for Accelerators. Each Accelerator corresponds to one element in the array.
	Accelerators []AcceleratorStats `json:"accelerators,omitempty"`

	// ProcessStats for Containers
	Processes ProcessStats `json:"processes,omitempty"`

	// Custom metrics from all collectors
	CustomMetrics map[string][]MetricVal `json:"custom_metrics,omitempty"`

	// Statistics originating from perf events
	PerfStats []PerfStat `json:"perf_stats,omitempty"`

	// Statistics originating from perf uncore events.
	// Applies only for root container.
	PerfUncoreStats []PerfUncoreStat `json:"perf_uncore_stats,omitempty"`

	// Referenced memory
	ReferencedMemory uint64 `json:"referenced_memory,omitempty"`

	// Resource Control (resctrl) statistics
	Resctrl ResctrlStats `json:"resctrl,omitempty"`
}

func timeEq(t1, t2 time.Time, tolerance time.Duration) bool {
	// t1 should not be later than t2
	if t1.After(t2) {
		t1, t2 = t2, t1
	}
	diff := t2.Sub(t1)
	return diff <= tolerance
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
	if !reflect.DeepEqual(a.Hugetlb, b.Hugetlb) {
		return false
	}
	if !reflect.DeepEqual(a.DiskIo, b.DiskIo) {
		return false
	}
	if !reflect.DeepEqual(a.Network, b.Network) {
		return false
	}
	if !reflect.DeepEqual(a.Processes, b.Processes) {
		return false
	}
	if !reflect.DeepEqual(a.Filesystem, b.Filesystem) {
		return false
	}
	if !reflect.DeepEqual(a.TaskStats, b.TaskStats) {
		return false
	}
	if !reflect.DeepEqual(a.Accelerators, b.Accelerators) {
		return false
	}
	if !reflect.DeepEqual(a.CustomMetrics, b.CustomMetrics) {
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
	EventOomKill           EventType = "oomKill"
	EventContainerCreation EventType = "containerCreation"
	EventContainerDeletion EventType = "containerDeletion"
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

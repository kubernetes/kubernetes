// Copyright 2018 The Prometheus Authors
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

package procfs

// While implementing parsing of /proc/[pid]/mountstats, this blog was used
// heavily as a reference:
//   https://utcc.utoronto.ca/~cks/space/blog/linux/NFSMountstatsIndex
//
// Special thanks to Chris Siebenmann for all of his posts explaining the
// various statistics available for NFS.

import (
	"bufio"
	"fmt"
	"io"
	"strconv"
	"strings"
	"time"
)

// Constants shared between multiple functions.
const (
	deviceEntryLen = 8

	fieldBytesLen  = 8
	fieldEventsLen = 27

	statVersion10 = "1.0"
	statVersion11 = "1.1"

	fieldTransport10TCPLen = 10
	fieldTransport10UDPLen = 7

	fieldTransport11TCPLen = 13
	fieldTransport11UDPLen = 10

	// kernel version >= 4.14 MaxLen
	// See: https://elixir.bootlin.com/linux/v6.4.8/source/net/sunrpc/xprtrdma/xprt_rdma.h#L393
	fieldTransport11RDMAMaxLen = 28

	// kernel version <= 4.2 MinLen
	// See: https://elixir.bootlin.com/linux/v4.2.8/source/net/sunrpc/xprtrdma/xprt_rdma.h#L331
	fieldTransport11RDMAMinLen = 20
)

// A Mount is a device mount parsed from /proc/[pid]/mountstats.
type Mount struct {
	// Name of the device.
	Device string
	// The mount point of the device.
	Mount string
	// The filesystem type used by the device.
	Type string
	// If available additional statistics related to this Mount.
	// Use a type assertion to determine if additional statistics are available.
	Stats MountStats
}

// A MountStats is a type which contains detailed statistics for a specific
// type of Mount.
type MountStats interface {
	mountStats()
}

// A MountStatsNFS is a MountStats implementation for NFSv3 and v4 mounts.
type MountStatsNFS struct {
	// The version of statistics provided.
	StatVersion string
	// The mount options of the NFS mount.
	Opts map[string]string
	// The age of the NFS mount.
	Age time.Duration
	// Statistics related to byte counters for various operations.
	Bytes NFSBytesStats
	// Statistics related to various NFS event occurrences.
	Events NFSEventsStats
	// Statistics broken down by filesystem operation.
	Operations []NFSOperationStats
	// Statistics about the NFS RPC transport.
	Transport []NFSTransportStats
}

// mountStats implements MountStats.
func (m MountStatsNFS) mountStats() {}

// A NFSBytesStats contains statistics about the number of bytes read and written
// by an NFS client to and from an NFS server.
type NFSBytesStats struct {
	// Number of bytes read using the read() syscall.
	Read uint64
	// Number of bytes written using the write() syscall.
	Write uint64
	// Number of bytes read using the read() syscall in O_DIRECT mode.
	DirectRead uint64
	// Number of bytes written using the write() syscall in O_DIRECT mode.
	DirectWrite uint64
	// Number of bytes read from the NFS server, in total.
	ReadTotal uint64
	// Number of bytes written to the NFS server, in total.
	WriteTotal uint64
	// Number of pages read directly via mmap()'d files.
	ReadPages uint64
	// Number of pages written directly via mmap()'d files.
	WritePages uint64
}

// A NFSEventsStats contains statistics about NFS event occurrences.
type NFSEventsStats struct {
	// Number of times cached inode attributes are re-validated from the server.
	InodeRevalidate uint64
	// Number of times cached dentry nodes are re-validated from the server.
	DnodeRevalidate uint64
	// Number of times an inode cache is cleared.
	DataInvalidate uint64
	// Number of times cached inode attributes are invalidated.
	AttributeInvalidate uint64
	// Number of times files or directories have been open()'d.
	VFSOpen uint64
	// Number of times a directory lookup has occurred.
	VFSLookup uint64
	// Number of times permissions have been checked.
	VFSAccess uint64
	// Number of updates (and potential writes) to pages.
	VFSUpdatePage uint64
	// Number of pages read directly via mmap()'d files.
	VFSReadPage uint64
	// Number of times a group of pages have been read.
	VFSReadPages uint64
	// Number of pages written directly via mmap()'d files.
	VFSWritePage uint64
	// Number of times a group of pages have been written.
	VFSWritePages uint64
	// Number of times directory entries have been read with getdents().
	VFSGetdents uint64
	// Number of times attributes have been set on inodes.
	VFSSetattr uint64
	// Number of pending writes that have been forcefully flushed to the server.
	VFSFlush uint64
	// Number of times fsync() has been called on directories and files.
	VFSFsync uint64
	// Number of times locking has been attempted on a file.
	VFSLock uint64
	// Number of times files have been closed and released.
	VFSFileRelease uint64
	// Unknown.  Possibly unused.
	CongestionWait uint64
	// Number of times files have been truncated.
	Truncation uint64
	// Number of times a file has been grown due to writes beyond its existing end.
	WriteExtension uint64
	// Number of times a file was removed while still open by another process.
	SillyRename uint64
	// Number of times the NFS server gave less data than expected while reading.
	ShortRead uint64
	// Number of times the NFS server wrote less data than expected while writing.
	ShortWrite uint64
	// Number of times the NFS server indicated EJUKEBOX; retrieving data from
	// offline storage.
	JukeboxDelay uint64
	// Number of NFS v4.1+ pNFS reads.
	PNFSRead uint64
	// Number of NFS v4.1+ pNFS writes.
	PNFSWrite uint64
}

// A NFSOperationStats contains statistics for a single operation.
type NFSOperationStats struct {
	// The name of the operation.
	Operation string
	// Number of requests performed for this operation.
	Requests uint64
	// Number of times an actual RPC request has been transmitted for this operation.
	Transmissions uint64
	// Number of times a request has had a major timeout.
	MajorTimeouts uint64
	// Number of bytes sent for this operation, including RPC headers and payload.
	BytesSent uint64
	// Number of bytes received for this operation, including RPC headers and payload.
	BytesReceived uint64
	// Duration all requests spent queued for transmission before they were sent.
	CumulativeQueueMilliseconds uint64
	// Duration it took to get a reply back after the request was transmitted.
	CumulativeTotalResponseMilliseconds uint64
	// Duration from when a request was enqueued to when it was completely handled.
	CumulativeTotalRequestMilliseconds uint64
	// The count of operations that complete with tk_status < 0.  These statuses usually indicate error conditions.
	Errors uint64
}

// A NFSTransportStats contains statistics for the NFS mount RPC requests and
// responses.
type NFSTransportStats struct {
	// The transport protocol used for the NFS mount.
	Protocol string
	// The local port used for the NFS mount.
	Port uint64
	// Number of times the client has had to establish a connection from scratch
	// to the NFS server.
	Bind uint64
	// Number of times the client has made a TCP connection to the NFS server.
	Connect uint64
	// Duration (in jiffies, a kernel internal unit of time) the NFS mount has
	// spent waiting for connections to the server to be established.
	ConnectIdleTime uint64
	// Duration since the NFS mount last saw any RPC traffic.
	IdleTimeSeconds uint64
	// Number of RPC requests for this mount sent to the NFS server.
	Sends uint64
	// Number of RPC responses for this mount received from the NFS server.
	Receives uint64
	// Number of times the NFS server sent a response with a transaction ID
	// unknown to this client.
	BadTransactionIDs uint64
	// A running counter, incremented on each request as the current difference
	// ebetween sends and receives.
	CumulativeActiveRequests uint64
	// A running counter, incremented on each request by the current backlog
	// queue size.
	CumulativeBacklog uint64

	// Stats below only available with stat version 1.1.

	// Maximum number of simultaneously active RPC requests ever used.
	MaximumRPCSlotsUsed uint64
	// A running counter, incremented on each request as the current size of the
	// sending queue.
	CumulativeSendingQueue uint64
	// A running counter, incremented on each request as the current size of the
	// pending queue.
	CumulativePendingQueue uint64

	// Stats below only available with stat version 1.1.
	// Transport over RDMA

	// accessed when sending a call
	ReadChunkCount   uint64
	WriteChunkCount  uint64
	ReplyChunkCount  uint64
	TotalRdmaRequest uint64

	// rarely accessed error counters
	PullupCopyCount      uint64
	HardwayRegisterCount uint64
	FailedMarshalCount   uint64
	BadReplyCount        uint64
	MrsRecovered         uint64
	MrsOrphaned          uint64
	MrsAllocated         uint64
	EmptySendctxQ        uint64

	// accessed when receiving a reply
	TotalRdmaReply    uint64
	FixupCopyCount    uint64
	ReplyWaitsForSend uint64
	LocalInvNeeded    uint64
	NomsgCallCount    uint64
	BcallCount        uint64
}

// parseMountStats parses a /proc/[pid]/mountstats file and returns a slice
// of Mount structures containing detailed information about each mount.
// If available, statistics for each mount are parsed as well.
func parseMountStats(r io.Reader) ([]*Mount, error) {
	const (
		device            = "device"
		statVersionPrefix = "statvers="

		nfs3Type = "nfs"
		nfs4Type = "nfs4"
	)

	var mounts []*Mount

	s := bufio.NewScanner(r)
	for s.Scan() {
		// Only look for device entries in this function
		ss := strings.Fields(string(s.Bytes()))
		if len(ss) == 0 || ss[0] != device {
			continue
		}

		m, err := parseMount(ss)
		if err != nil {
			return nil, err
		}

		// Does this mount also possess statistics information?
		if len(ss) > deviceEntryLen {
			// Only NFSv3 and v4 are supported for parsing statistics
			if m.Type != nfs3Type && m.Type != nfs4Type {
				return nil, fmt.Errorf("%w: Cannot parse MountStats for %q", ErrFileParse, m.Type)
			}

			statVersion := strings.TrimPrefix(ss[8], statVersionPrefix)

			stats, err := parseMountStatsNFS(s, statVersion)
			if err != nil {
				return nil, err
			}

			m.Stats = stats
		}

		mounts = append(mounts, m)
	}

	return mounts, s.Err()
}

// parseMount parses an entry in /proc/[pid]/mountstats in the format:
//
//	device [device] mounted on [mount] with fstype [type]
func parseMount(ss []string) (*Mount, error) {
	if len(ss) < deviceEntryLen {
		return nil, fmt.Errorf("%w: Invalid device %q", ErrFileParse, ss)
	}

	// Check for specific words appearing at specific indices to ensure
	// the format is consistent with what we expect
	format := []struct {
		i int
		s string
	}{
		{i: 0, s: "device"},
		{i: 2, s: "mounted"},
		{i: 3, s: "on"},
		{i: 5, s: "with"},
		{i: 6, s: "fstype"},
	}

	for _, f := range format {
		if ss[f.i] != f.s {
			return nil, fmt.Errorf("%w: Invalid device %q", ErrFileParse, ss)
		}
	}

	return &Mount{
		Device: ss[1],
		Mount:  ss[4],
		Type:   ss[7],
	}, nil
}

// parseMountStatsNFS parses a MountStatsNFS by scanning additional information
// related to NFS statistics.
func parseMountStatsNFS(s *bufio.Scanner, statVersion string) (*MountStatsNFS, error) {
	// Field indicators for parsing specific types of data
	const (
		fieldOpts       = "opts:"
		fieldAge        = "age:"
		fieldBytes      = "bytes:"
		fieldEvents     = "events:"
		fieldPerOpStats = "per-op"
		fieldTransport  = "xprt:"
	)

	stats := &MountStatsNFS{
		StatVersion: statVersion,
	}

	for s.Scan() {
		ss := strings.Fields(string(s.Bytes()))
		if len(ss) == 0 {
			break
		}

		switch ss[0] {
		case fieldOpts:
			if len(ss) < 2 {
				return nil, fmt.Errorf("%w: Incomplete information for NFS stats: %v", ErrFileParse, ss)
			}
			if stats.Opts == nil {
				stats.Opts = map[string]string{}
			}
			for _, opt := range strings.Split(ss[1], ",") {
				split := strings.Split(opt, "=")
				if len(split) == 2 {
					stats.Opts[split[0]] = split[1]
				} else {
					stats.Opts[opt] = ""
				}
			}
		case fieldAge:
			if len(ss) < 2 {
				return nil, fmt.Errorf("%w: Incomplete information for NFS stats: %v", ErrFileParse, ss)
			}
			// Age integer is in seconds
			d, err := time.ParseDuration(ss[1] + "s")
			if err != nil {
				return nil, err
			}

			stats.Age = d
		case fieldBytes:
			if len(ss) < 2 {
				return nil, fmt.Errorf("%w: Incomplete information for NFS stats: %v", ErrFileParse, ss)
			}
			bstats, err := parseNFSBytesStats(ss[1:])
			if err != nil {
				return nil, err
			}

			stats.Bytes = *bstats
		case fieldEvents:
			if len(ss) < 2 {
				return nil, fmt.Errorf("%w: Incomplete information for NFS events: %v", ErrFileParse, ss)
			}
			estats, err := parseNFSEventsStats(ss[1:])
			if err != nil {
				return nil, err
			}

			stats.Events = *estats
		case fieldTransport:
			if len(ss) < 3 {
				return nil, fmt.Errorf("%w: Incomplete information for NFS transport stats: %v", ErrFileParse, ss)
			}

			tstats, err := parseNFSTransportStats(ss[1:], statVersion)
			if err != nil {
				return nil, err
			}

			stats.Transport = append(stats.Transport, *tstats)
		}

		// When encountering "per-operation statistics", we must break this
		// loop and parse them separately to ensure we can terminate parsing
		// before reaching another device entry; hence why this 'if' statement
		// is not just another switch case
		if ss[0] == fieldPerOpStats {
			break
		}
	}

	if err := s.Err(); err != nil {
		return nil, err
	}

	// NFS per-operation stats appear last before the next device entry
	perOpStats, err := parseNFSOperationStats(s)
	if err != nil {
		return nil, err
	}

	stats.Operations = perOpStats

	return stats, nil
}

// parseNFSBytesStats parses a NFSBytesStats line using an input set of
// integer fields.
func parseNFSBytesStats(ss []string) (*NFSBytesStats, error) {
	if len(ss) != fieldBytesLen {
		return nil, fmt.Errorf("%w: Invalid NFS bytes stats: %v", ErrFileParse, ss)
	}

	ns := make([]uint64, 0, fieldBytesLen)
	for _, s := range ss {
		n, err := strconv.ParseUint(s, 10, 64)
		if err != nil {
			return nil, err
		}

		ns = append(ns, n)
	}

	return &NFSBytesStats{
		Read:        ns[0],
		Write:       ns[1],
		DirectRead:  ns[2],
		DirectWrite: ns[3],
		ReadTotal:   ns[4],
		WriteTotal:  ns[5],
		ReadPages:   ns[6],
		WritePages:  ns[7],
	}, nil
}

// parseNFSEventsStats parses a NFSEventsStats line using an input set of
// integer fields.
func parseNFSEventsStats(ss []string) (*NFSEventsStats, error) {
	if len(ss) != fieldEventsLen {
		return nil, fmt.Errorf("%w: invalid NFS events stats: %v", ErrFileParse, ss)
	}

	ns := make([]uint64, 0, fieldEventsLen)
	for _, s := range ss {
		n, err := strconv.ParseUint(s, 10, 64)
		if err != nil {
			return nil, err
		}

		ns = append(ns, n)
	}

	return &NFSEventsStats{
		InodeRevalidate:     ns[0],
		DnodeRevalidate:     ns[1],
		DataInvalidate:      ns[2],
		AttributeInvalidate: ns[3],
		VFSOpen:             ns[4],
		VFSLookup:           ns[5],
		VFSAccess:           ns[6],
		VFSUpdatePage:       ns[7],
		VFSReadPage:         ns[8],
		VFSReadPages:        ns[9],
		VFSWritePage:        ns[10],
		VFSWritePages:       ns[11],
		VFSGetdents:         ns[12],
		VFSSetattr:          ns[13],
		VFSFlush:            ns[14],
		VFSFsync:            ns[15],
		VFSLock:             ns[16],
		VFSFileRelease:      ns[17],
		CongestionWait:      ns[18],
		Truncation:          ns[19],
		WriteExtension:      ns[20],
		SillyRename:         ns[21],
		ShortRead:           ns[22],
		ShortWrite:          ns[23],
		JukeboxDelay:        ns[24],
		PNFSRead:            ns[25],
		PNFSWrite:           ns[26],
	}, nil
}

// parseNFSOperationStats parses a slice of NFSOperationStats by scanning
// additional information about per-operation statistics until an empty
// line is reached.
func parseNFSOperationStats(s *bufio.Scanner) ([]NFSOperationStats, error) {
	const (
		// Minimum number of expected fields in each per-operation statistics set
		minFields = 9
	)

	var ops []NFSOperationStats

	for s.Scan() {
		ss := strings.Fields(string(s.Bytes()))
		if len(ss) == 0 {
			// Must break when reading a blank line after per-operation stats to
			// enable top-level function to parse the next device entry
			break
		}

		if len(ss) < minFields {
			return nil, fmt.Errorf("%w: invalid NFS per-operations stats: %v", ErrFileParse, ss)
		}

		// Skip string operation name for integers
		ns := make([]uint64, 0, minFields-1)
		for _, st := range ss[1:] {
			n, err := strconv.ParseUint(st, 10, 64)
			if err != nil {
				return nil, err
			}

			ns = append(ns, n)
		}
		opStats := NFSOperationStats{
			Operation:                           strings.TrimSuffix(ss[0], ":"),
			Requests:                            ns[0],
			Transmissions:                       ns[1],
			MajorTimeouts:                       ns[2],
			BytesSent:                           ns[3],
			BytesReceived:                       ns[4],
			CumulativeQueueMilliseconds:         ns[5],
			CumulativeTotalResponseMilliseconds: ns[6],
			CumulativeTotalRequestMilliseconds:  ns[7],
		}

		if len(ns) > 8 {
			opStats.Errors = ns[8]
		}

		ops = append(ops, opStats)
	}

	return ops, s.Err()
}

// parseNFSTransportStats parses a NFSTransportStats line using an input set of
// integer fields matched to a specific stats version.
func parseNFSTransportStats(ss []string, statVersion string) (*NFSTransportStats, error) {
	// Extract the protocol field. It is the only string value in the line
	protocol := ss[0]
	ss = ss[1:]

	switch statVersion {
	case statVersion10:
		var expectedLength int
		if protocol == "tcp" {
			expectedLength = fieldTransport10TCPLen
		} else if protocol == "udp" {
			expectedLength = fieldTransport10UDPLen
		} else {
			return nil, fmt.Errorf("%w: Invalid NFS protocol \"%s\" in stats 1.0 statement: %v", ErrFileParse, protocol, ss)
		}
		if len(ss) != expectedLength {
			return nil, fmt.Errorf("%w: Invalid NFS transport stats 1.0 statement: %v", ErrFileParse, ss)
		}
	case statVersion11:
		var expectedLength int
		if protocol == "tcp" {
			expectedLength = fieldTransport11TCPLen
		} else if protocol == "udp" {
			expectedLength = fieldTransport11UDPLen
		} else if protocol == "rdma" {
			expectedLength = fieldTransport11RDMAMinLen
		} else {
			return nil, fmt.Errorf("%w: invalid NFS protocol \"%s\" in stats 1.1 statement: %v", ErrFileParse, protocol, ss)
		}
		if (len(ss) != expectedLength && (protocol == "tcp" || protocol == "udp")) ||
			(protocol == "rdma" && len(ss) < expectedLength) {
			return nil, fmt.Errorf("%w: invalid NFS transport stats 1.1 statement: %v, protocol: %v", ErrFileParse, ss, protocol)
		}
	default:
		return nil, fmt.Errorf("%w: Unrecognized NFS transport stats version: %q, protocol: %v", ErrFileParse, statVersion, protocol)
	}

	// Allocate enough for v1.1 stats since zero value for v1.1 stats will be okay
	// in a v1.0 response. Since the stat length is bigger for TCP stats, we use
	// the TCP length here.
	//
	// Note: slice length must be set to length of v1.1 stats to avoid a panic when
	// only v1.0 stats are present.
	// See: https://github.com/prometheus/node_exporter/issues/571.
	//
	// Note: NFS Over RDMA slice length is fieldTransport11RDMAMaxLen
	ns := make([]uint64, fieldTransport11RDMAMaxLen+3)
	for i, s := range ss {
		n, err := strconv.ParseUint(s, 10, 64)
		if err != nil {
			return nil, err
		}

		ns[i] = n
	}

	// The fields differ depending on the transport protocol (TCP or UDP)
	// From https://utcc.utoronto.ca/%7Ecks/space/blog/linux/NFSMountstatsXprt
	//
	// For the udp RPC transport there is no connection count, connect idle time,
	// or idle time (fields #3, #4, and #5); all other fields are the same. So
	// we set them to 0 here.
	if protocol == "udp" {
		ns = append(ns[:2], append(make([]uint64, 3), ns[2:]...)...)
	} else if protocol == "tcp" {
		ns = append(ns[:fieldTransport11TCPLen], make([]uint64, fieldTransport11RDMAMaxLen-fieldTransport11TCPLen+3)...)
	} else if protocol == "rdma" {
		ns = append(ns[:fieldTransport10TCPLen], append(make([]uint64, 3), ns[fieldTransport10TCPLen:]...)...)
	}

	return &NFSTransportStats{
		// NFS xprt over tcp or udp
		Protocol:                 protocol,
		Port:                     ns[0],
		Bind:                     ns[1],
		Connect:                  ns[2],
		ConnectIdleTime:          ns[3],
		IdleTimeSeconds:          ns[4],
		Sends:                    ns[5],
		Receives:                 ns[6],
		BadTransactionIDs:        ns[7],
		CumulativeActiveRequests: ns[8],
		CumulativeBacklog:        ns[9],

		// NFS xprt over tcp or udp
		// And statVersion 1.1
		MaximumRPCSlotsUsed:    ns[10],
		CumulativeSendingQueue: ns[11],
		CumulativePendingQueue: ns[12],

		// NFS xprt over rdma
		// And stat Version 1.1
		ReadChunkCount:       ns[13],
		WriteChunkCount:      ns[14],
		ReplyChunkCount:      ns[15],
		TotalRdmaRequest:     ns[16],
		PullupCopyCount:      ns[17],
		HardwayRegisterCount: ns[18],
		FailedMarshalCount:   ns[19],
		BadReplyCount:        ns[20],
		MrsRecovered:         ns[21],
		MrsOrphaned:          ns[22],
		MrsAllocated:         ns[23],
		EmptySendctxQ:        ns[24],
		TotalRdmaReply:       ns[25],
		FixupCopyCount:       ns[26],
		ReplyWaitsForSend:    ns[27],
		LocalInvNeeded:       ns[28],
		NomsgCallCount:       ns[29],
		BcallCount:           ns[30],
	}, nil
}

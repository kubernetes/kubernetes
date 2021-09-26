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

// Package xfs provides access to statistics exposed by the XFS filesystem.
package xfs

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/prometheus/procfs/internal/fs"
)

// Stats contains XFS filesystem runtime statistics, parsed from
// /proc/fs/xfs/stat.
//
// The names and meanings of each statistic were taken from
// http://xfs.org/index.php/Runtime_Stats and xfs_stats.h in the Linux
// kernel source. Most counters are uint32s (same data types used in
// xfs_stats.h), but some of the "extended precision stats" are uint64s.
type Stats struct {
	// The name of the filesystem used to source these statistics.
	// If empty, this indicates aggregated statistics for all XFS
	// filesystems on the host.
	Name string

	ExtentAllocation   ExtentAllocationStats
	AllocationBTree    BTreeStats
	BlockMapping       BlockMappingStats
	BlockMapBTree      BTreeStats
	DirectoryOperation DirectoryOperationStats
	Transaction        TransactionStats
	InodeOperation     InodeOperationStats
	LogOperation       LogOperationStats
	ReadWrite          ReadWriteStats
	AttributeOperation AttributeOperationStats
	InodeClustering    InodeClusteringStats
	Vnode              VnodeStats
	Buffer             BufferStats
	ExtendedPrecision  ExtendedPrecisionStats
	Xstrat             XstratStats            // xstrat
	PushAil            PushAilStats           // push_ail
	Debug              DebugStats             // debug
	QuotaManager       QuotaManagerStats      // qm
	BtreeAllocBlocks2  BtreeAllocBlocks2Stats // abtb2
	BtreeAllocContig2  BtreeAllocContig2Stats // abtc2
	BtreeBlockMap2     BtreeBlockMap2Stats    // bmbt2
	BtreeInode2        BtreeInode2Stats       // ibt2
}

// ExtentAllocationStats contains statistics regarding XFS extent allocations.
type ExtentAllocationStats struct {
	ExtentsAllocated uint32
	BlocksAllocated  uint32
	ExtentsFreed     uint32
	BlocksFreed      uint32
}

// BTreeStats contains statistics regarding an XFS internal B-tree.
type BTreeStats struct {
	Lookups         uint32
	Compares        uint32
	RecordsInserted uint32
	RecordsDeleted  uint32
}

// BlockMappingStats contains statistics regarding XFS block maps.
type BlockMappingStats struct {
	Reads                uint32
	Writes               uint32
	Unmaps               uint32
	ExtentListInsertions uint32
	ExtentListDeletions  uint32
	ExtentListLookups    uint32
	ExtentListCompares   uint32
}

// DirectoryOperationStats contains statistics regarding XFS directory entries.
type DirectoryOperationStats struct {
	Lookups  uint32
	Creates  uint32
	Removes  uint32
	Getdents uint32
}

// TransactionStats contains statistics regarding XFS metadata transactions.
type TransactionStats struct {
	Sync  uint32
	Async uint32
	Empty uint32
}

// InodeOperationStats contains statistics regarding XFS inode operations.
type InodeOperationStats struct {
	Attempts        uint32
	Found           uint32
	Recycle         uint32
	Missed          uint32
	Duplicate       uint32
	Reclaims        uint32
	AttributeChange uint32
}

// LogOperationStats contains statistics regarding the XFS log buffer.
type LogOperationStats struct {
	Writes            uint32
	Blocks            uint32
	NoInternalBuffers uint32
	Force             uint32
	ForceSleep        uint32
}

// ReadWriteStats contains statistics regarding the number of read and write
// system calls for XFS filesystems.
type ReadWriteStats struct {
	Read  uint32
	Write uint32
}

// AttributeOperationStats contains statistics regarding manipulation of
// XFS extended file attributes.
type AttributeOperationStats struct {
	Get    uint32
	Set    uint32
	Remove uint32
	List   uint32
}

// InodeClusteringStats contains statistics regarding XFS inode clustering
// operations.
type InodeClusteringStats struct {
	Iflush     uint32
	Flush      uint32
	FlushInode uint32
}

// VnodeStats contains statistics regarding XFS vnode operations.
type VnodeStats struct {
	Active   uint32
	Allocate uint32
	Get      uint32
	Hold     uint32
	Release  uint32
	Reclaim  uint32
	Remove   uint32
	Free     uint32
}

// BufferStats contains statistics regarding XFS read/write I/O buffers.
type BufferStats struct {
	Get             uint32
	Create          uint32
	GetLocked       uint32
	GetLockedWaited uint32
	BusyLocked      uint32
	MissLocked      uint32
	PageRetries     uint32
	PageFound       uint32
	GetRead         uint32
}

// ExtendedPrecisionStats contains high precision counters used to track the
// total number of bytes read, written, or flushed, during XFS operations.
type ExtendedPrecisionStats struct {
	FlushBytes uint64
	WriteBytes uint64
	ReadBytes  uint64
}

// PushAilStats contains statistics on tail-pushing operations.
type PushAilStats struct {
	TryLogspace   uint32
	SleepLogspace uint32
	Pushes        uint32
	Success       uint32
	PushBuf       uint32
	Pinned        uint32
	Locked        uint32
	Flushing      uint32
	Restarts      uint32
	Flush         uint32
}

// QuotaManagerStats contain statistics regarding quota processing.
type QuotaManagerStats struct {
	Reclaims      uint32
	ReclaimMisses uint32
	DquoteDups    uint32
	CacheMisses   uint32
	CacheHits     uint32
	Wants         uint32
	ShakeReclaims uint32
	InactReclaims uint32
	Unused        uint32
}

// XstratStats contains statistics regarding bytes processed by the XFS daemon.
type XstratStats struct {
	Quick uint32
	Split uint32
}

// DebugStats indicate if XFS debugging is enabled.
type DebugStats struct {
	Enabled uint32
}

// BtreeAllocBlocks2Stats contains statistics on B-Tree v2 allocations.
type BtreeAllocBlocks2Stats struct {
	Lookup    uint32
	Compare   uint32
	Insrec    uint32
	Delrec    uint32
	NewRoot   uint32
	KillRoot  uint32
	Increment uint32
	Decrement uint32
	Lshift    uint32
	Rshift    uint32
	Split     uint32
	Join      uint32
	Alloc     uint32
	Free      uint32
	Moves     uint32
}

// BtreeAllocContig2Stats contain statistics on B-tree v2 free-space-by-size record operations.
type BtreeAllocContig2Stats struct {
	Lookup    uint32
	Compare   uint32
	Insrec    uint32
	Delrec    uint32
	NewRoot   uint32
	KillRoot  uint32
	Increment uint32
	Decrement uint32
	Lshift    uint32
	Rshift    uint32
	Split     uint32
	Join      uint32
	Alloc     uint32
	Free      uint32
	Moves     uint32
}

// BtreeBlockMap2Stats contain statistics on B-tree v2 block map operations.
type BtreeBlockMap2Stats struct {
	Lookup    uint32
	Compare   uint32
	Insrec    uint32
	Delrec    uint32
	NewRoot   uint32
	KillRoot  uint32
	Increment uint32
	Decrement uint32
	Lshift    uint32
	Rshift    uint32
	Split     uint32
	Join      uint32
	Alloc     uint32
	Free      uint32
	Moves     uint32
}

// BtreeInode2Stats contain statistics on B-tree v2 inode allocations.
type BtreeInode2Stats struct {
	Lookup    uint32
	Compare   uint32
	Insrec    uint32
	Delrec    uint32
	NewRoot   uint32
	KillRoot  uint32
	Increment uint32
	Decrement uint32
	Lshift    uint32
	Rshift    uint32
	Split     uint32
	Join      uint32
	Alloc     uint32
	Free      uint32
	Moves     uint32
}

// FS represents the pseudo-filesystems proc and sys, which provides an interface to
// kernel data structures.
type FS struct {
	proc *fs.FS
	sys  *fs.FS
}

// NewDefaultFS returns a new XFS handle using the default proc and sys mountPoints.
// It will error if either of the mounts point can't be read.
func NewDefaultFS() (FS, error) {
	return NewFS(fs.DefaultProcMountPoint, fs.DefaultSysMountPoint)
}

// NewFS returns a new XFS handle using the given proc and sys mountPoints. It will error
// if either of the mounts point can't be read.
func NewFS(procMountPoint string, sysMountPoint string) (FS, error) {
	if strings.TrimSpace(procMountPoint) == "" {
		procMountPoint = fs.DefaultProcMountPoint
	}
	procfs, err := fs.NewFS(procMountPoint)
	if err != nil {
		return FS{}, err
	}
	if strings.TrimSpace(sysMountPoint) == "" {
		sysMountPoint = fs.DefaultSysMountPoint
	}
	sysfs, err := fs.NewFS(sysMountPoint)
	if err != nil {
		return FS{}, err
	}
	return FS{&procfs, &sysfs}, nil
}

// ProcStat retrieves XFS filesystem runtime statistics
// from proc/fs/xfs/stat given the profs mount point.
func (fs FS) ProcStat() (*Stats, error) {
	f, err := os.Open(fs.proc.Path("fs/xfs/stat"))
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return ParseStats(f)
}

// SysStats retrieves XFS filesystem runtime statistics for each mounted XFS
// filesystem.  Only available on kernel 4.4+.  On older kernels, an empty
// slice of *xfs.Stats will be returned.
func (fs FS) SysStats() ([]*Stats, error) {
	matches, err := filepath.Glob(fs.sys.Path("fs/xfs/*/stats/stats"))
	if err != nil {
		return nil, err
	}

	stats := make([]*Stats, 0, len(matches))
	for _, m := range matches {
		f, err := os.Open(m)
		if err != nil {
			return nil, err
		}

		// "*" used in glob above indicates the name of the filesystem.
		name := filepath.Base(filepath.Dir(filepath.Dir(m)))

		// File must be closed after parsing, regardless of success or
		// failure.  Defer is not used because of the loop.
		s, err := ParseStats(f)
		_ = f.Close()
		if err != nil {
			return nil, err
		}

		s.Name = name
		stats = append(stats, s)
	}

	return stats, nil
}

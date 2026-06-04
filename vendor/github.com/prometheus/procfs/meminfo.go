// Copyright The Prometheus Authors
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

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/prometheus/procfs/internal/util"
)

// Meminfo represents memory statistics.
type Meminfo struct {
	// Total usable ram (i.e. physical ram minus a few reserved
	// bits and the kernel binary code)
	MemTotal *uint64
	// The sum of LowFree+HighFree
	MemFree *uint64
	// An estimate of how much memory is available for starting
	// new applications, without swapping. Calculated from
	// MemFree, SReclaimable, the size of the file LRU lists, and
	// the low watermarks in each zone.  The estimate takes into
	// account that the system needs some page cache to function
	// well, and that not all reclaimable slab will be
	// reclaimable, due to items being in use. The impact of those
	// factors will vary from system to system.
	MemAvailable *uint64
	// Relatively temporary storage for raw disk blocks shouldn't
	// get tremendously large (20MB or so)
	Buffers *uint64
	Cached  *uint64
	// Memory that once was swapped out, is swapped back in but
	// still also is in the swapfile (if memory is needed it
	// doesn't need to be swapped out AGAIN because it is already
	// in the swapfile. This saves I/O)
	SwapCached *uint64
	// Memory that has been used more recently and usually not
	// reclaimed unless absolutely necessary.
	Active *uint64
	// Memory which has been less recently used.  It is more
	// eligible to be reclaimed for other purposes
	Inactive     *uint64
	ActiveAnon   *uint64
	InactiveAnon *uint64
	ActiveFile   *uint64
	InactiveFile *uint64
	Unevictable  *uint64
	Mlocked      *uint64
	// total amount of swap space available
	SwapTotal *uint64
	// Memory which has been evicted from RAM, and is temporarily
	// on the disk
	SwapFree *uint64
	// Memory consumed by the zswap backend (compressed size)
	Zswap *uint64
	// Amount of anonymous memory stored in zswap (original size)
	Zswapped *uint64
	// Memory which is waiting to get written back to the disk
	Dirty *uint64
	// Memory which is actively being written back to the disk
	Writeback *uint64
	// Non-file backed pages mapped into userspace page tables
	AnonPages *uint64
	// files which have been mapped, such as libraries
	Mapped *uint64
	Shmem  *uint64
	// in-kernel data structures cache
	Slab *uint64
	// Part of Slab, that might be reclaimed, such as caches
	SReclaimable *uint64
	// Part of Slab, that cannot be reclaimed on memory pressure
	SUnreclaim  *uint64
	KernelStack *uint64
	// amount of memory dedicated to the lowest level of page
	// tables.
	PageTables *uint64
	// secondary page tables.
	SecPageTables *uint64
	// NFS pages sent to the server, but not yet committed to
	// stable storage
	NFSUnstable *uint64
	// Memory used for block device "bounce buffers"
	Bounce *uint64
	// Memory used by FUSE for temporary writeback buffers
	WritebackTmp *uint64
	// Based on the overcommit ratio ('vm.overcommit_ratio'),
	// this is the total amount of  memory currently available to
	// be allocated on the system. This limit is only adhered to
	// if strict overcommit accounting is enabled (mode 2 in
	// 'vm.overcommit_memory').
	// The CommitLimit is calculated with the following formula:
	// CommitLimit = ([total RAM pages] - [total huge TLB pages]) *
	//                overcommit_ratio / 100 + [total swap pages]
	// For example, on a system with 1G of physical RAM and 7G
	// of swap with a `vm.overcommit_ratio` of 30 it would
	// yield a CommitLimit of 7.3G.
	// For more details, see the memory overcommit documentation
	// in vm/overcommit-accounting.
	CommitLimit *uint64
	// The amount of memory presently allocated on the system.
	// The committed memory is a sum of all of the memory which
	// has been allocated by processes, even if it has not been
	// "used" by them as of yet. A process which malloc()'s 1G
	// of memory, but only touches 300M of it will show up as
	// using 1G. This 1G is memory which has been "committed" to
	// by the VM and can be used at any time by the allocating
	// application. With strict overcommit enabled on the system
	// (mode 2 in 'vm.overcommit_memory'),allocations which would
	// exceed the CommitLimit (detailed above) will not be permitted.
	// This is useful if one needs to guarantee that processes will
	// not fail due to lack of memory once that memory has been
	// successfully allocated.
	CommittedAS *uint64
	// total size of vmalloc memory area
	VmallocTotal *uint64
	// amount of vmalloc area which is used
	VmallocUsed *uint64
	// largest contiguous block of vmalloc area which is free
	VmallocChunk      *uint64
	Percpu            *uint64
	HardwareCorrupted *uint64
	AnonHugePages     *uint64
	FileHugePages     *uint64
	ShmemHugePages    *uint64
	ShmemPmdMapped    *uint64
	CmaTotal          *uint64
	CmaFree           *uint64
	Unaccepted        *uint64
	HugePagesTotal    *uint64
	HugePagesFree     *uint64
	HugePagesRsvd     *uint64
	HugePagesSurp     *uint64
	Hugepagesize      *uint64
	Hugetlb           *uint64
	DirectMap4k       *uint64
	DirectMap2M       *uint64
	DirectMap1G       *uint64

	// The struct fields below are the byte-normalized counterparts to the
	// existing struct fields. Values are normalized using the optional
	// unit field in the meminfo line.
	MemTotalBytes          *uint64
	MemFreeBytes           *uint64
	MemAvailableBytes      *uint64
	BuffersBytes           *uint64
	CachedBytes            *uint64
	SwapCachedBytes        *uint64
	ActiveBytes            *uint64
	InactiveBytes          *uint64
	ActiveAnonBytes        *uint64
	InactiveAnonBytes      *uint64
	ActiveFileBytes        *uint64
	InactiveFileBytes      *uint64
	UnevictableBytes       *uint64
	MlockedBytes           *uint64
	SwapTotalBytes         *uint64
	SwapFreeBytes          *uint64
	ZswapBytes             *uint64
	ZswappedBytes          *uint64
	DirtyBytes             *uint64
	WritebackBytes         *uint64
	AnonPagesBytes         *uint64
	MappedBytes            *uint64
	ShmemBytes             *uint64
	SlabBytes              *uint64
	SReclaimableBytes      *uint64
	SUnreclaimBytes        *uint64
	KernelStackBytes       *uint64
	PageTablesBytes        *uint64
	SecPageTablesBytes     *uint64
	NFSUnstableBytes       *uint64
	BounceBytes            *uint64
	WritebackTmpBytes      *uint64
	CommitLimitBytes       *uint64
	CommittedASBytes       *uint64
	VmallocTotalBytes      *uint64
	VmallocUsedBytes       *uint64
	VmallocChunkBytes      *uint64
	PercpuBytes            *uint64
	HardwareCorruptedBytes *uint64
	AnonHugePagesBytes     *uint64
	FileHugePagesBytes     *uint64
	ShmemHugePagesBytes    *uint64
	ShmemPmdMappedBytes    *uint64
	CmaTotalBytes          *uint64
	CmaFreeBytes           *uint64
	UnacceptedBytes        *uint64
	HugepagesizeBytes      *uint64
	HugetlbBytes           *uint64
	DirectMap4kBytes       *uint64
	DirectMap2MBytes       *uint64
	DirectMap1GBytes       *uint64
}

// Meminfo returns an information about current kernel/system memory statistics.
// See https://www.kernel.org/doc/Documentation/filesystems/proc.txt
func (fs FS) Meminfo() (Meminfo, error) {
	b, err := util.ReadFileNoStat(fs.proc.Path("meminfo"))
	if err != nil {
		return Meminfo{}, err
	}

	m, err := parseMemInfo(bytes.NewReader(b))
	if err != nil {
		return Meminfo{}, fmt.Errorf("%w: %w", ErrFileParse, err)
	}

	return *m, nil
}

func parseMemInfo(r io.Reader) (*Meminfo, error) {
	var m Meminfo
	s := bufio.NewScanner(r)
	for s.Scan() {
		fields := strings.Fields(s.Text())
		var val, valBytes uint64

		val, err := strconv.ParseUint(fields[1], 0, 64)
		if err != nil {
			return nil, err
		}

		switch len(fields) {
		case 2:
			// No unit present, use the parsed the value as bytes directly.
			valBytes = val
		case 3:
			// Unit present in optional 3rd field, convert it to
			// bytes. The only unit supported within the Linux
			// kernel is `kB`.
			if fields[2] != "kB" {
				return nil, fmt.Errorf("%w: Unsupported unit in optional 3rd field %q", ErrFileParse, fields[2])
			}

			valBytes = 1024 * val

		default:
			return nil, fmt.Errorf("%w: Malformed line %q", ErrFileParse, s.Text())
		}

		switch fields[0] {
		case "MemTotal:":
			m.MemTotal = &val
			m.MemTotalBytes = &valBytes
		case "MemFree:":
			m.MemFree = &val
			m.MemFreeBytes = &valBytes
		case "MemAvailable:":
			m.MemAvailable = &val
			m.MemAvailableBytes = &valBytes
		case "Buffers:":
			m.Buffers = &val
			m.BuffersBytes = &valBytes
		case "Cached:":
			m.Cached = &val
			m.CachedBytes = &valBytes
		case "SwapCached:":
			m.SwapCached = &val
			m.SwapCachedBytes = &valBytes
		case "Active:":
			m.Active = &val
			m.ActiveBytes = &valBytes
		case "Inactive:":
			m.Inactive = &val
			m.InactiveBytes = &valBytes
		case "Active(anon):":
			m.ActiveAnon = &val
			m.ActiveAnonBytes = &valBytes
		case "Inactive(anon):":
			m.InactiveAnon = &val
			m.InactiveAnonBytes = &valBytes
		case "Active(file):":
			m.ActiveFile = &val
			m.ActiveFileBytes = &valBytes
		case "Inactive(file):":
			m.InactiveFile = &val
			m.InactiveFileBytes = &valBytes
		case "Unevictable:":
			m.Unevictable = &val
			m.UnevictableBytes = &valBytes
		case "Mlocked:":
			m.Mlocked = &val
			m.MlockedBytes = &valBytes
		case "SwapTotal:":
			m.SwapTotal = &val
			m.SwapTotalBytes = &valBytes
		case "SwapFree:":
			m.SwapFree = &val
			m.SwapFreeBytes = &valBytes
		case "Zswap:":
			m.Zswap = &val
			m.ZswapBytes = &valBytes
		case "Zswapped:":
			m.Zswapped = &val
			m.ZswappedBytes = &valBytes
		case "Dirty:":
			m.Dirty = &val
			m.DirtyBytes = &valBytes
		case "Writeback:":
			m.Writeback = &val
			m.WritebackBytes = &valBytes
		case "AnonPages:":
			m.AnonPages = &val
			m.AnonPagesBytes = &valBytes
		case "Mapped:":
			m.Mapped = &val
			m.MappedBytes = &valBytes
		case "Shmem:":
			m.Shmem = &val
			m.ShmemBytes = &valBytes
		case "Slab:":
			m.Slab = &val
			m.SlabBytes = &valBytes
		case "SReclaimable:":
			m.SReclaimable = &val
			m.SReclaimableBytes = &valBytes
		case "SUnreclaim:":
			m.SUnreclaim = &val
			m.SUnreclaimBytes = &valBytes
		case "KernelStack:":
			m.KernelStack = &val
			m.KernelStackBytes = &valBytes
		case "PageTables:":
			m.PageTables = &val
			m.PageTablesBytes = &valBytes
		case "SecPageTables:":
			m.SecPageTables = &val
			m.SecPageTablesBytes = &valBytes
		case "NFS_Unstable:":
			m.NFSUnstable = &val
			m.NFSUnstableBytes = &valBytes
		case "Bounce:":
			m.Bounce = &val
			m.BounceBytes = &valBytes
		case "WritebackTmp:":
			m.WritebackTmp = &val
			m.WritebackTmpBytes = &valBytes
		case "CommitLimit:":
			m.CommitLimit = &val
			m.CommitLimitBytes = &valBytes
		case "Committed_AS:":
			m.CommittedAS = &val
			m.CommittedASBytes = &valBytes
		case "VmallocTotal:":
			m.VmallocTotal = &val
			m.VmallocTotalBytes = &valBytes
		case "VmallocUsed:":
			m.VmallocUsed = &val
			m.VmallocUsedBytes = &valBytes
		case "VmallocChunk:":
			m.VmallocChunk = &val
			m.VmallocChunkBytes = &valBytes
		case "Percpu:":
			m.Percpu = &val
			m.PercpuBytes = &valBytes
		case "HardwareCorrupted:":
			m.HardwareCorrupted = &val
			m.HardwareCorruptedBytes = &valBytes
		case "AnonHugePages:":
			m.AnonHugePages = &val
			m.AnonHugePagesBytes = &valBytes
		case "FileHugePages:":
			m.FileHugePages = &val
			m.FileHugePagesBytes = &valBytes
		case "ShmemHugePages:":
			m.ShmemHugePages = &val
			m.ShmemHugePagesBytes = &valBytes
		case "ShmemPmdMapped:":
			m.ShmemPmdMapped = &val
			m.ShmemPmdMappedBytes = &valBytes
		case "CmaTotal:":
			m.CmaTotal = &val
			m.CmaTotalBytes = &valBytes
		case "CmaFree:":
			m.CmaFree = &val
			m.CmaFreeBytes = &valBytes
		case "Unaccepted:":
			m.Unaccepted = &val
			m.UnacceptedBytes = &valBytes
		case "HugePages_Total:":
			m.HugePagesTotal = &val
		case "HugePages_Free:":
			m.HugePagesFree = &val
		case "HugePages_Rsvd:":
			m.HugePagesRsvd = &val
		case "HugePages_Surp:":
			m.HugePagesSurp = &val
		case "Hugepagesize:":
			m.Hugepagesize = &val
			m.HugepagesizeBytes = &valBytes
		case "Hugetlb:":
			m.Hugetlb = &val
			m.HugetlbBytes = &valBytes
		case "DirectMap4k:":
			m.DirectMap4k = &val
			m.DirectMap4kBytes = &valBytes
		case "DirectMap2M:":
			m.DirectMap2M = &val
			m.DirectMap2MBytes = &valBytes
		case "DirectMap1G:":
			m.DirectMap1G = &val
			m.DirectMap1GBytes = &valBytes
		}
	}

	return &m, nil
}

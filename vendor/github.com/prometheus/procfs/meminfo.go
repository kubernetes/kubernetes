// Copyright 2019 The Prometheus Authors
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
	HardwareCorrupted *uint64
	AnonHugePages     *uint64
	ShmemHugePages    *uint64
	ShmemPmdMapped    *uint64
	CmaTotal          *uint64
	CmaFree           *uint64
	HugePagesTotal    *uint64
	HugePagesFree     *uint64
	HugePagesRsvd     *uint64
	HugePagesSurp     *uint64
	Hugepagesize      *uint64
	DirectMap4k       *uint64
	DirectMap2M       *uint64
	DirectMap1G       *uint64
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
		return Meminfo{}, fmt.Errorf("failed to parse meminfo: %w", err)
	}

	return *m, nil
}

func parseMemInfo(r io.Reader) (*Meminfo, error) {
	var m Meminfo
	s := bufio.NewScanner(r)
	for s.Scan() {
		// Each line has at least a name and value; we ignore the unit.
		fields := strings.Fields(s.Text())
		if len(fields) < 2 {
			return nil, fmt.Errorf("malformed meminfo line: %q", s.Text())
		}

		v, err := strconv.ParseUint(fields[1], 0, 64)
		if err != nil {
			return nil, err
		}

		switch fields[0] {
		case "MemTotal:":
			m.MemTotal = &v
		case "MemFree:":
			m.MemFree = &v
		case "MemAvailable:":
			m.MemAvailable = &v
		case "Buffers:":
			m.Buffers = &v
		case "Cached:":
			m.Cached = &v
		case "SwapCached:":
			m.SwapCached = &v
		case "Active:":
			m.Active = &v
		case "Inactive:":
			m.Inactive = &v
		case "Active(anon):":
			m.ActiveAnon = &v
		case "Inactive(anon):":
			m.InactiveAnon = &v
		case "Active(file):":
			m.ActiveFile = &v
		case "Inactive(file):":
			m.InactiveFile = &v
		case "Unevictable:":
			m.Unevictable = &v
		case "Mlocked:":
			m.Mlocked = &v
		case "SwapTotal:":
			m.SwapTotal = &v
		case "SwapFree:":
			m.SwapFree = &v
		case "Dirty:":
			m.Dirty = &v
		case "Writeback:":
			m.Writeback = &v
		case "AnonPages:":
			m.AnonPages = &v
		case "Mapped:":
			m.Mapped = &v
		case "Shmem:":
			m.Shmem = &v
		case "Slab:":
			m.Slab = &v
		case "SReclaimable:":
			m.SReclaimable = &v
		case "SUnreclaim:":
			m.SUnreclaim = &v
		case "KernelStack:":
			m.KernelStack = &v
		case "PageTables:":
			m.PageTables = &v
		case "NFS_Unstable:":
			m.NFSUnstable = &v
		case "Bounce:":
			m.Bounce = &v
		case "WritebackTmp:":
			m.WritebackTmp = &v
		case "CommitLimit:":
			m.CommitLimit = &v
		case "Committed_AS:":
			m.CommittedAS = &v
		case "VmallocTotal:":
			m.VmallocTotal = &v
		case "VmallocUsed:":
			m.VmallocUsed = &v
		case "VmallocChunk:":
			m.VmallocChunk = &v
		case "HardwareCorrupted:":
			m.HardwareCorrupted = &v
		case "AnonHugePages:":
			m.AnonHugePages = &v
		case "ShmemHugePages:":
			m.ShmemHugePages = &v
		case "ShmemPmdMapped:":
			m.ShmemPmdMapped = &v
		case "CmaTotal:":
			m.CmaTotal = &v
		case "CmaFree:":
			m.CmaFree = &v
		case "HugePages_Total:":
			m.HugePagesTotal = &v
		case "HugePages_Free:":
			m.HugePagesFree = &v
		case "HugePages_Rsvd:":
			m.HugePagesRsvd = &v
		case "HugePages_Surp:":
			m.HugePagesSurp = &v
		case "Hugepagesize:":
			m.Hugepagesize = &v
		case "DirectMap4k:":
			m.DirectMap4k = &v
		case "DirectMap2M:":
			m.DirectMap2M = &v
		case "DirectMap1G:":
			m.DirectMap1G = &v
		}
	}

	return &m, nil
}

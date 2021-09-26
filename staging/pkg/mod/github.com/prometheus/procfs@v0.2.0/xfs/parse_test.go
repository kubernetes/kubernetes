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

package xfs_test

import (
	"reflect"
	"strings"
	"testing"

	"github.com/prometheus/procfs/xfs"
)

func TestParseStats(t *testing.T) {
	tests := []struct {
		name    string
		s       string
		fs      bool
		stats   *xfs.Stats
		invalid bool
	}{
		{
			name: "empty file OK",
		},
		{
			name:  "short or empty lines and unknown labels ignored",
			s:     "one\n\ntwo 1 2 3\n",
			stats: &xfs.Stats{},
		},
		{
			name:    "bad uint32",
			s:       "extent_alloc XXX",
			invalid: true,
		},
		{
			name:    "bad uint64",
			s:       "xpc XXX",
			invalid: true,
		},
		{
			name:    "extent_alloc bad",
			s:       "extent_alloc 1",
			invalid: true,
		},
		{
			name: "extent_alloc OK",
			s:    "extent_alloc 1 2 3 4",
			stats: &xfs.Stats{
				ExtentAllocation: xfs.ExtentAllocationStats{
					ExtentsAllocated: 1,
					BlocksAllocated:  2,
					ExtentsFreed:     3,
					BlocksFreed:      4,
				},
			},
		},
		{
			name:    "abt bad",
			s:       "abt 1",
			invalid: true,
		},
		{
			name: "abt OK",
			s:    "abt 1 2 3 4",
			stats: &xfs.Stats{
				AllocationBTree: xfs.BTreeStats{
					Lookups:         1,
					Compares:        2,
					RecordsInserted: 3,
					RecordsDeleted:  4,
				},
			},
		},
		{
			name:    "blk_map bad",
			s:       "blk_map 1",
			invalid: true,
		},
		{
			name: "blk_map OK",
			s:    "blk_map 1 2 3 4 5 6 7",
			stats: &xfs.Stats{
				BlockMapping: xfs.BlockMappingStats{
					Reads:                1,
					Writes:               2,
					Unmaps:               3,
					ExtentListInsertions: 4,
					ExtentListDeletions:  5,
					ExtentListLookups:    6,
					ExtentListCompares:   7,
				},
			},
		},
		{
			name:    "bmbt bad",
			s:       "bmbt 1",
			invalid: true,
		},
		{
			name: "bmbt OK",
			s:    "bmbt 1 2 3 4",
			stats: &xfs.Stats{
				BlockMapBTree: xfs.BTreeStats{
					Lookups:         1,
					Compares:        2,
					RecordsInserted: 3,
					RecordsDeleted:  4,
				},
			},
		},
		{
			name:    "dir bad",
			s:       "dir 1",
			invalid: true,
		},
		{
			name: "dir OK",
			s:    "dir 1 2 3 4",
			stats: &xfs.Stats{
				DirectoryOperation: xfs.DirectoryOperationStats{
					Lookups:  1,
					Creates:  2,
					Removes:  3,
					Getdents: 4,
				},
			},
		},
		{
			name:    "trans bad",
			s:       "trans 1",
			invalid: true,
		},
		{
			name: "trans OK",
			s:    "trans 1 2 3",
			stats: &xfs.Stats{
				Transaction: xfs.TransactionStats{
					Sync:  1,
					Async: 2,
					Empty: 3,
				},
			},
		},
		{
			name:    "ig bad",
			s:       "ig 1",
			invalid: true,
		},
		{
			name: "ig OK",
			s:    "ig 1 2 3 4 5 6 7",
			stats: &xfs.Stats{
				InodeOperation: xfs.InodeOperationStats{
					Attempts:        1,
					Found:           2,
					Recycle:         3,
					Missed:          4,
					Duplicate:       5,
					Reclaims:        6,
					AttributeChange: 7,
				},
			},
		},
		{
			name:    "log bad",
			s:       "log 1",
			invalid: true,
		},
		{
			name: "log OK",
			s:    "log 1 2 3 4 5",
			stats: &xfs.Stats{
				LogOperation: xfs.LogOperationStats{
					Writes:            1,
					Blocks:            2,
					NoInternalBuffers: 3,
					Force:             4,
					ForceSleep:        5,
				},
			},
		},
		{
			name:    "rw bad",
			s:       "rw 1",
			invalid: true,
		},
		{
			name: "rw OK",
			s:    "rw 1 2",
			stats: &xfs.Stats{
				ReadWrite: xfs.ReadWriteStats{
					Read:  1,
					Write: 2,
				},
			},
		},
		{
			name:    "attr bad",
			s:       "attr 1",
			invalid: true,
		},
		{
			name: "attr OK",
			s:    "attr 1 2 3 4",
			stats: &xfs.Stats{
				AttributeOperation: xfs.AttributeOperationStats{
					Get:    1,
					Set:    2,
					Remove: 3,
					List:   4,
				},
			},
		},
		{
			name:    "icluster bad",
			s:       "icluster 1",
			invalid: true,
		},
		{
			name: "icluster OK",
			s:    "icluster 1 2 3",
			stats: &xfs.Stats{
				InodeClustering: xfs.InodeClusteringStats{
					Iflush:     1,
					Flush:      2,
					FlushInode: 3,
				},
			},
		},
		{
			name:    "vnodes bad",
			s:       "vnodes 1",
			invalid: true,
		},
		{
			name: "vnodes (missing free) OK",
			s:    "vnodes 1 2 3 4 5 6 7",
			stats: &xfs.Stats{
				Vnode: xfs.VnodeStats{
					Active:   1,
					Allocate: 2,
					Get:      3,
					Hold:     4,
					Release:  5,
					Reclaim:  6,
					Remove:   7,
				},
			},
		},
		{
			name: "vnodes (with free) OK",
			s:    "vnodes 1 2 3 4 5 6 7 8",
			stats: &xfs.Stats{
				Vnode: xfs.VnodeStats{
					Active:   1,
					Allocate: 2,
					Get:      3,
					Hold:     4,
					Release:  5,
					Reclaim:  6,
					Remove:   7,
					Free:     8,
				},
			},
		},
		{
			name:    "buf bad",
			s:       "buf 1",
			invalid: true,
		},
		{
			name: "buf OK",
			s:    "buf 1 2 3 4 5 6 7 8 9",
			stats: &xfs.Stats{
				Buffer: xfs.BufferStats{
					Get:             1,
					Create:          2,
					GetLocked:       3,
					GetLockedWaited: 4,
					BusyLocked:      5,
					MissLocked:      6,
					PageRetries:     7,
					PageFound:       8,
					GetRead:         9,
				},
			},
		},
		{
			name:    "xpc bad",
			s:       "xpc 1",
			invalid: true,
		},
		{
			name: "xpc OK",
			s:    "xpc 1 2 3",
			stats: &xfs.Stats{
				ExtendedPrecision: xfs.ExtendedPrecisionStats{
					FlushBytes: 1,
					WriteBytes: 2,
					ReadBytes:  3,
				},
			},
		},
		{
			name:    "xstrat bad",
			s:       "xstrat 1",
			invalid: true,
		},
		{
			name: "xstrat OK",
			s:    "xstrat 1 2",
			stats: &xfs.Stats{
				Xstrat: xfs.XstratStats{
					Quick: 1,
					Split: 2,
				},
			},
		},
		{
			name:    "push_ail bad",
			s:       "push_ail 1 2 3 4 5",
			invalid: true,
		},
		{
			name: "push_ail OK",
			s:    "push_ail 1 2 3 4 5 6 7 8 9 10",
			stats: &xfs.Stats{
				PushAil: xfs.PushAilStats{
					TryLogspace:   1,
					SleepLogspace: 2,
					Pushes:        3,
					Success:       4,
					PushBuf:       5,
					Pinned:        6,
					Locked:        7,
					Flushing:      8,
					Restarts:      9,
					Flush:         10,
				},
			},
		},
		{
			name:    "debug bad",
			s:       "debug 1 2",
			invalid: true,
		},
		{
			name: "debug  OK",
			s:    "debug 1",
			stats: &xfs.Stats{
				Debug: xfs.DebugStats{
					Enabled: 1,
				},
			},
		},
		{
			name:    "qm bad",
			s:       "qm 1 2 3 4 5 6 7",
			invalid: true,
		},
		{
			name:    "qm bad",
			s:       "qm 1 2 3 4 5 6 7 8 9 10",
			invalid: true,
		},
		{
			name: "qm OK",
			s:    "qm 1 2 3 4 5 6 7 8",
			stats: &xfs.Stats{
				QuotaManager: xfs.QuotaManagerStats{
					Reclaims:      1,
					ReclaimMisses: 2,
					DquoteDups:    3,
					CacheMisses:   4,
					CacheHits:     5,
					Wants:         6,
					ShakeReclaims: 7,
					InactReclaims: 8,
				},
			},
		},
		{
			name: "qm OK",
			s:    "qm 1 2 3 4 5 6 7 8 9",
			stats: &xfs.Stats{
				QuotaManager: xfs.QuotaManagerStats{
					Reclaims:      1,
					ReclaimMisses: 2,
					DquoteDups:    3,
					CacheMisses:   4,
					CacheHits:     5,
					Wants:         6,
					ShakeReclaims: 7,
					InactReclaims: 8,
					Unused:        9,
				},
			},
		},
		{
			name:    "abtb2 bad",
			s:       "abtb2 1 2 3 4 5 6",
			invalid: true,
		},
		{
			name: "abtb2 OK",
			s:    "abtb2 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15",
			stats: &xfs.Stats{
				BtreeAllocBlocks2: xfs.BtreeAllocBlocks2Stats{
					Lookup:    1,
					Compare:   2,
					Insrec:    3,
					Delrec:    4,
					NewRoot:   5,
					KillRoot:  6,
					Increment: 7,
					Decrement: 8,
					Lshift:    9,
					Rshift:    10,
					Split:     11,
					Join:      12,
					Alloc:     13,
					Free:      14,
					Moves:     15,
				},
			},
		},
		{
			name:    "abtc2 bad",
			s:       "abtc2 1 2 3 4 5 6",
			invalid: true,
		},
		{
			name: "abtc2 OK",
			s:    "abtc2 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15",
			stats: &xfs.Stats{
				BtreeAllocContig2: xfs.BtreeAllocContig2Stats{
					Lookup:    1,
					Compare:   2,
					Insrec:    3,
					Delrec:    4,
					NewRoot:   5,
					KillRoot:  6,
					Increment: 7,
					Decrement: 8,
					Lshift:    9,
					Rshift:    10,
					Split:     11,
					Join:      12,
					Alloc:     13,
					Free:      14,
					Moves:     15,
				},
			},
		},
		{
			name:    "bmbt2 bad",
			s:       "bmbt2 1 2 3 4 5 6",
			invalid: true,
		},
		{
			name: "bmbt2 OK",
			s:    "bmbt2 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15",
			stats: &xfs.Stats{
				BtreeBlockMap2: xfs.BtreeBlockMap2Stats{
					Lookup:    1,
					Compare:   2,
					Insrec:    3,
					Delrec:    4,
					NewRoot:   5,
					KillRoot:  6,
					Increment: 7,
					Decrement: 8,
					Lshift:    9,
					Rshift:    10,
					Split:     11,
					Join:      12,
					Alloc:     13,
					Free:      14,
					Moves:     15,
				},
			},
		},
		{
			name:    "ibt2 bad",
			s:       "ibt2 1 2 3 4 5 6",
			invalid: true,
		},
		{
			name: "ibt2 OK",
			s:    "ibt2 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15",
			stats: &xfs.Stats{
				BtreeInode2: xfs.BtreeInode2Stats{
					Lookup:    1,
					Compare:   2,
					Insrec:    3,
					Delrec:    4,
					NewRoot:   5,
					KillRoot:  6,
					Increment: 7,
					Decrement: 8,
					Lshift:    9,
					Rshift:    10,
					Split:     11,
					Join:      12,
					Alloc:     13,
					Free:      14,
					Moves:     15,
				},
			},
		},
		{
			name: "fixtures OK",
			fs:   true,
			stats: &xfs.Stats{
				ExtentAllocation: xfs.ExtentAllocationStats{
					ExtentsAllocated: 92447,
					BlocksAllocated:  97589,
					ExtentsFreed:     92448,
					BlocksFreed:      93751,
				},
				AllocationBTree: xfs.BTreeStats{
					Lookups:         0,
					Compares:        0,
					RecordsInserted: 0,
					RecordsDeleted:  0,
				},
				BlockMapping: xfs.BlockMappingStats{
					Reads:                1767055,
					Writes:               188820,
					Unmaps:               184891,
					ExtentListInsertions: 92447,
					ExtentListDeletions:  92448,
					ExtentListLookups:    2140766,
					ExtentListCompares:   0,
				},
				BlockMapBTree: xfs.BTreeStats{
					Lookups:         0,
					Compares:        0,
					RecordsInserted: 0,
					RecordsDeleted:  0,
				},
				DirectoryOperation: xfs.DirectoryOperationStats{
					Lookups:  185039,
					Creates:  92447,
					Removes:  92444,
					Getdents: 136422,
				},
				Transaction: xfs.TransactionStats{
					Sync:  706,
					Async: 944304,
					Empty: 0,
				},
				InodeOperation: xfs.InodeOperationStats{
					Attempts:        185045,
					Found:           58807,
					Recycle:         0,
					Missed:          126238,
					Duplicate:       0,
					Reclaims:        33637,
					AttributeChange: 22,
				},
				LogOperation: xfs.LogOperationStats{
					Writes:            2883,
					Blocks:            113448,
					NoInternalBuffers: 9,
					Force:             17360,
					ForceSleep:        739,
				},
				ReadWrite: xfs.ReadWriteStats{
					Read:  107739,
					Write: 94045,
				},
				AttributeOperation: xfs.AttributeOperationStats{
					Get:    4,
					Set:    0,
					Remove: 0,
					List:   0,
				},
				InodeClustering: xfs.InodeClusteringStats{
					Iflush:     8677,
					Flush:      7849,
					FlushInode: 135802,
				},
				Vnode: xfs.VnodeStats{
					Active:   92601,
					Allocate: 0,
					Get:      0,
					Hold:     0,
					Release:  92444,
					Reclaim:  92444,
					Remove:   92444,
					Free:     0,
				},
				Buffer: xfs.BufferStats{
					Get:             2666287,
					Create:          7122,
					GetLocked:       2659202,
					GetLockedWaited: 3599,
					BusyLocked:      2,
					MissLocked:      7085,
					PageRetries:     0,
					PageFound:       10297,
					GetRead:         7085,
				},
				ExtendedPrecision: xfs.ExtendedPrecisionStats{
					FlushBytes: 399724544,
					WriteBytes: 92823103,
					ReadBytes:  86219234,
				},
				PushAil: xfs.PushAilStats{
					TryLogspace:   945014,
					SleepLogspace: 0,
					Pushes:        134260,
					Success:       15483,
					PushBuf:       0,
					Pinned:        3940,
					Locked:        464,
					Flushing:      159985,
					Restarts:      0,
					Flush:         40,
				},
				Xstrat: xfs.XstratStats{
					Quick: 92447,
					Split: 0,
				},
				Debug: xfs.DebugStats{
					Enabled: 0,
				},
				QuotaManager: xfs.QuotaManagerStats{
					Reclaims:      0,
					ReclaimMisses: 0,
					DquoteDups:    0,
					CacheMisses:   0,
					CacheHits:     0,
					Wants:         0,
					ShakeReclaims: 0,
					InactReclaims: 0,
				},
				BtreeAllocBlocks2: xfs.BtreeAllocBlocks2Stats{
					Lookup:    184941,
					Compare:   1277345,
					Insrec:    13257,
					Delrec:    13278,
					NewRoot:   0,
					KillRoot:  0,
					Increment: 0,
					Decrement: 0,
					Lshift:    0,
					Rshift:    0,
					Split:     0,
					Join:      0,
					Alloc:     0,
					Free:      0,
					Moves:     2746147,
				},

				BtreeAllocContig2: xfs.BtreeAllocContig2Stats{
					Lookup:    345295,
					Compare:   2416764,
					Insrec:    172637,
					Delrec:    172658,
					NewRoot:   0,
					KillRoot:  0,
					Increment: 0,
					Decrement: 0,
					Lshift:    0,
					Rshift:    0,
					Split:     0,
					Join:      0,
					Alloc:     0,
					Free:      0,
					Moves:     21406023,
				},

				BtreeBlockMap2: xfs.BtreeBlockMap2Stats{
					Lookup:    0,
					Compare:   0,
					Insrec:    0,
					Delrec:    0,
					NewRoot:   0,
					KillRoot:  0,
					Increment: 0,
					Decrement: 0,
					Lshift:    0,
					Rshift:    0,
					Split:     0,
					Join:      0,
					Alloc:     0,
					Free:      0,
					Moves:     0,
				},

				BtreeInode2: xfs.BtreeInode2Stats{
					Lookup:    343004,
					Compare:   1358467,
					Insrec:    0,
					Delrec:    0,
					NewRoot:   0,
					KillRoot:  0,
					Increment: 0,
					Decrement: 0,
					Lshift:    0,
					Rshift:    0,
					Split:     0,
					Join:      0,
					Alloc:     0,
					Free:      0,
					Moves:     0,
				},
			},
		},
	}

	for _, tt := range tests {
		var (
			stats *xfs.Stats
			err   error
		)

		if tt.s != "" {
			stats, err = xfs.ParseStats(strings.NewReader(tt.s))
		}
		if tt.fs {
			xfs, err := xfs.NewFS("../fixtures/proc", "../fixtures/sys")
			if err != nil {
				t.Fatalf("failed to access xfs fs: %v", err)
			}
			stats, err = xfs.ProcStat()
			if err != nil {
				t.Fatalf("failed to gather xfs stats: %v", err)
			}
		}

		if tt.invalid && err == nil {
			t.Error("expected an error, but none occurred")
		}
		if !tt.invalid && err != nil {
			t.Errorf("unexpected error: %v", err)
		}

		if want, have := tt.stats, stats; !reflect.DeepEqual(want, have) {
			t.Errorf("unexpected XFS stats:\nwant:\n%v\nhave:\n%v", want, have)
		}
	}
}

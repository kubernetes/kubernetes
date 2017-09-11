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

package xfs

import (
	"bufio"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// ParseStats parses a Stats from an input io.Reader, using the format
// found in /proc/fs/xfs/stat.
func ParseStats(r io.Reader) (*Stats, error) {
	const (
		// Fields parsed into stats structures.
		fieldExtentAlloc = "extent_alloc"
		fieldAbt         = "abt"
		fieldBlkMap      = "blk_map"
		fieldBmbt        = "bmbt"
		fieldDir         = "dir"
		fieldTrans       = "trans"
		fieldIg          = "ig"
		fieldLog         = "log"
		fieldRw          = "rw"
		fieldAttr        = "attr"
		fieldIcluster    = "icluster"
		fieldVnodes      = "vnodes"
		fieldBuf         = "buf"
		fieldXpc         = "xpc"

		// Unimplemented at this time due to lack of documentation.
		fieldPushAil = "push_ail"
		fieldXstrat  = "xstrat"
		fieldAbtb2   = "abtb2"
		fieldAbtc2   = "abtc2"
		fieldBmbt2   = "bmbt2"
		fieldIbt2    = "ibt2"
		fieldFibt2   = "fibt2"
		fieldQm      = "qm"
		fieldDebug   = "debug"
	)

	var xfss Stats

	s := bufio.NewScanner(r)
	for s.Scan() {
		// Expect at least a string label and a single integer value, ex:
		//   - abt 0
		//   - rw 1 2
		ss := strings.Fields(string(s.Bytes()))
		if len(ss) < 2 {
			continue
		}
		label := ss[0]

		// Extended precision counters are uint64 values.
		if label == fieldXpc {
			us, err := parseUint64s(ss[1:])
			if err != nil {
				return nil, err
			}

			xfss.ExtendedPrecision, err = extendedPrecisionStats(us)
			if err != nil {
				return nil, err
			}

			continue
		}

		// All other counters are uint32 values.
		us, err := parseUint32s(ss[1:])
		if err != nil {
			return nil, err
		}

		switch label {
		case fieldExtentAlloc:
			xfss.ExtentAllocation, err = extentAllocationStats(us)
		case fieldAbt:
			xfss.AllocationBTree, err = btreeStats(us)
		case fieldBlkMap:
			xfss.BlockMapping, err = blockMappingStats(us)
		case fieldBmbt:
			xfss.BlockMapBTree, err = btreeStats(us)
		case fieldDir:
			xfss.DirectoryOperation, err = directoryOperationStats(us)
		case fieldTrans:
			xfss.Transaction, err = transactionStats(us)
		case fieldIg:
			xfss.InodeOperation, err = inodeOperationStats(us)
		case fieldLog:
			xfss.LogOperation, err = logOperationStats(us)
		case fieldRw:
			xfss.ReadWrite, err = readWriteStats(us)
		case fieldAttr:
			xfss.AttributeOperation, err = attributeOperationStats(us)
		case fieldIcluster:
			xfss.InodeClustering, err = inodeClusteringStats(us)
		case fieldVnodes:
			xfss.Vnode, err = vnodeStats(us)
		case fieldBuf:
			xfss.Buffer, err = bufferStats(us)
		}
		if err != nil {
			return nil, err
		}
	}

	return &xfss, s.Err()
}

// extentAllocationStats builds an ExtentAllocationStats from a slice of uint32s.
func extentAllocationStats(us []uint32) (ExtentAllocationStats, error) {
	if l := len(us); l != 4 {
		return ExtentAllocationStats{}, fmt.Errorf("incorrect number of values for XFS extent allocation stats: %d", l)
	}

	return ExtentAllocationStats{
		ExtentsAllocated: us[0],
		BlocksAllocated:  us[1],
		ExtentsFreed:     us[2],
		BlocksFreed:      us[3],
	}, nil
}

// btreeStats builds a BTreeStats from a slice of uint32s.
func btreeStats(us []uint32) (BTreeStats, error) {
	if l := len(us); l != 4 {
		return BTreeStats{}, fmt.Errorf("incorrect number of values for XFS btree stats: %d", l)
	}

	return BTreeStats{
		Lookups:         us[0],
		Compares:        us[1],
		RecordsInserted: us[2],
		RecordsDeleted:  us[3],
	}, nil
}

// BlockMappingStat builds a BlockMappingStats from a slice of uint32s.
func blockMappingStats(us []uint32) (BlockMappingStats, error) {
	if l := len(us); l != 7 {
		return BlockMappingStats{}, fmt.Errorf("incorrect number of values for XFS block mapping stats: %d", l)
	}

	return BlockMappingStats{
		Reads:                us[0],
		Writes:               us[1],
		Unmaps:               us[2],
		ExtentListInsertions: us[3],
		ExtentListDeletions:  us[4],
		ExtentListLookups:    us[5],
		ExtentListCompares:   us[6],
	}, nil
}

// DirectoryOperationStats builds a DirectoryOperationStats from a slice of uint32s.
func directoryOperationStats(us []uint32) (DirectoryOperationStats, error) {
	if l := len(us); l != 4 {
		return DirectoryOperationStats{}, fmt.Errorf("incorrect number of values for XFS directory operation stats: %d", l)
	}

	return DirectoryOperationStats{
		Lookups:  us[0],
		Creates:  us[1],
		Removes:  us[2],
		Getdents: us[3],
	}, nil
}

// TransactionStats builds a TransactionStats from a slice of uint32s.
func transactionStats(us []uint32) (TransactionStats, error) {
	if l := len(us); l != 3 {
		return TransactionStats{}, fmt.Errorf("incorrect number of values for XFS transaction stats: %d", l)
	}

	return TransactionStats{
		Sync:  us[0],
		Async: us[1],
		Empty: us[2],
	}, nil
}

// InodeOperationStats builds an InodeOperationStats from a slice of uint32s.
func inodeOperationStats(us []uint32) (InodeOperationStats, error) {
	if l := len(us); l != 7 {
		return InodeOperationStats{}, fmt.Errorf("incorrect number of values for XFS inode operation stats: %d", l)
	}

	return InodeOperationStats{
		Attempts:        us[0],
		Found:           us[1],
		Recycle:         us[2],
		Missed:          us[3],
		Duplicate:       us[4],
		Reclaims:        us[5],
		AttributeChange: us[6],
	}, nil
}

// LogOperationStats builds a LogOperationStats from a slice of uint32s.
func logOperationStats(us []uint32) (LogOperationStats, error) {
	if l := len(us); l != 5 {
		return LogOperationStats{}, fmt.Errorf("incorrect number of values for XFS log operation stats: %d", l)
	}

	return LogOperationStats{
		Writes:            us[0],
		Blocks:            us[1],
		NoInternalBuffers: us[2],
		Force:             us[3],
		ForceSleep:        us[4],
	}, nil
}

// ReadWriteStats builds a ReadWriteStats from a slice of uint32s.
func readWriteStats(us []uint32) (ReadWriteStats, error) {
	if l := len(us); l != 2 {
		return ReadWriteStats{}, fmt.Errorf("incorrect number of values for XFS read write stats: %d", l)
	}

	return ReadWriteStats{
		Read:  us[0],
		Write: us[1],
	}, nil
}

// AttributeOperationStats builds an AttributeOperationStats from a slice of uint32s.
func attributeOperationStats(us []uint32) (AttributeOperationStats, error) {
	if l := len(us); l != 4 {
		return AttributeOperationStats{}, fmt.Errorf("incorrect number of values for XFS attribute operation stats: %d", l)
	}

	return AttributeOperationStats{
		Get:    us[0],
		Set:    us[1],
		Remove: us[2],
		List:   us[3],
	}, nil
}

// InodeClusteringStats builds an InodeClusteringStats from a slice of uint32s.
func inodeClusteringStats(us []uint32) (InodeClusteringStats, error) {
	if l := len(us); l != 3 {
		return InodeClusteringStats{}, fmt.Errorf("incorrect number of values for XFS inode clustering stats: %d", l)
	}

	return InodeClusteringStats{
		Iflush:     us[0],
		Flush:      us[1],
		FlushInode: us[2],
	}, nil
}

// VnodeStats builds a VnodeStats from a slice of uint32s.
func vnodeStats(us []uint32) (VnodeStats, error) {
	// The attribute "Free" appears to not be available on older XFS
	// stats versions.  Therefore, 7 or 8 elements may appear in
	// this slice.
	l := len(us)
	if l != 7 && l != 8 {
		return VnodeStats{}, fmt.Errorf("incorrect number of values for XFS vnode stats: %d", l)
	}

	s := VnodeStats{
		Active:   us[0],
		Allocate: us[1],
		Get:      us[2],
		Hold:     us[3],
		Release:  us[4],
		Reclaim:  us[5],
		Remove:   us[6],
	}

	// Skip adding free, unless it is present. The zero value will
	// be used in place of an actual count.
	if l == 7 {
		return s, nil
	}

	s.Free = us[7]
	return s, nil
}

// BufferStats builds a BufferStats from a slice of uint32s.
func bufferStats(us []uint32) (BufferStats, error) {
	if l := len(us); l != 9 {
		return BufferStats{}, fmt.Errorf("incorrect number of values for XFS buffer stats: %d", l)
	}

	return BufferStats{
		Get:             us[0],
		Create:          us[1],
		GetLocked:       us[2],
		GetLockedWaited: us[3],
		BusyLocked:      us[4],
		MissLocked:      us[5],
		PageRetries:     us[6],
		PageFound:       us[7],
		GetRead:         us[8],
	}, nil
}

// ExtendedPrecisionStats builds an ExtendedPrecisionStats from a slice of uint32s.
func extendedPrecisionStats(us []uint64) (ExtendedPrecisionStats, error) {
	if l := len(us); l != 3 {
		return ExtendedPrecisionStats{}, fmt.Errorf("incorrect number of values for XFS extended precision stats: %d", l)
	}

	return ExtendedPrecisionStats{
		FlushBytes: us[0],
		WriteBytes: us[1],
		ReadBytes:  us[2],
	}, nil
}

// parseUint32s parses a slice of strings into a slice of uint32s.
func parseUint32s(ss []string) ([]uint32, error) {
	us := make([]uint32, 0, len(ss))
	for _, s := range ss {
		u, err := strconv.ParseUint(s, 10, 32)
		if err != nil {
			return nil, err
		}

		us = append(us, uint32(u))
	}

	return us, nil
}

// parseUint64s parses a slice of strings into a slice of uint64s.
func parseUint64s(ss []string) ([]uint64, error) {
	us := make([]uint64, 0, len(ss))
	for _, s := range ss {
		u, err := strconv.ParseUint(s, 10, 64)
		if err != nil {
			return nil, err
		}

		us = append(us, u)
	}

	return us, nil
}

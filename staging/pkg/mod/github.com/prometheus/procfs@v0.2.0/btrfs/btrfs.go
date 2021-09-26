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

// Package btrfs provides access to statistics exposed by Btrfs filesystems.
package btrfs

// Stats contains statistics for a single Btrfs filesystem.
// See Linux fs/btrfs/sysfs.c for more information.
type Stats struct {
	UUID, Label    string
	Allocation     Allocation
	Devices        map[string]*Device
	Features       []string
	CloneAlignment uint64
	NodeSize       uint64
	QuotaOverride  uint64
	SectorSize     uint64
}

// Allocation contains allocation statistics for data, metadata and system data.
type Allocation struct {
	GlobalRsvReserved, GlobalRsvSize uint64
	Data, Metadata, System           *AllocationStats
}

// AllocationStats contains allocation statistics for a data type.
type AllocationStats struct {
	// Usage statistics
	DiskUsedBytes    uint64
	DiskTotalBytes   uint64
	MayUseBytes      uint64
	PinnedBytes      uint64
	TotalPinnedBytes uint64
	ReadOnlyBytes    uint64
	ReservedBytes    uint64
	UsedBytes        uint64
	TotalBytes       uint64

	// Flags marking filesystem state
	// See Linux fs/btrfs/ctree.h for more information.
	Flags uint64

	// Additional disk usage statistics depending on the disk layout.
	// At least one of these will exist and not be nil.
	Layouts map[string]*LayoutUsage
}

// LayoutUsage contains additional usage statistics for a disk layout.
type LayoutUsage struct {
	UsedBytes, TotalBytes uint64
	Ratio                 float64
}

// Device contains information about a device that is part of a Btrfs filesystem.
type Device struct {
	Size uint64
}

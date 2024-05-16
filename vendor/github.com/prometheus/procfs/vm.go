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

//go:build !windows
// +build !windows

package procfs

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/prometheus/procfs/internal/util"
)

// The VM interface is described at
//
//	https://www.kernel.org/doc/Documentation/sysctl/vm.txt
//
// Each setting is exposed as a single file.
// Each file contains one line with a single numerical value, except lowmem_reserve_ratio which holds an array
// and numa_zonelist_order (deprecated) which is a string.
type VM struct {
	AdminReserveKbytes        *int64   // /proc/sys/vm/admin_reserve_kbytes
	BlockDump                 *int64   // /proc/sys/vm/block_dump
	CompactUnevictableAllowed *int64   // /proc/sys/vm/compact_unevictable_allowed
	DirtyBackgroundBytes      *int64   // /proc/sys/vm/dirty_background_bytes
	DirtyBackgroundRatio      *int64   // /proc/sys/vm/dirty_background_ratio
	DirtyBytes                *int64   // /proc/sys/vm/dirty_bytes
	DirtyExpireCentisecs      *int64   // /proc/sys/vm/dirty_expire_centisecs
	DirtyRatio                *int64   // /proc/sys/vm/dirty_ratio
	DirtytimeExpireSeconds    *int64   // /proc/sys/vm/dirtytime_expire_seconds
	DirtyWritebackCentisecs   *int64   // /proc/sys/vm/dirty_writeback_centisecs
	DropCaches                *int64   // /proc/sys/vm/drop_caches
	ExtfragThreshold          *int64   // /proc/sys/vm/extfrag_threshold
	HugetlbShmGroup           *int64   // /proc/sys/vm/hugetlb_shm_group
	LaptopMode                *int64   // /proc/sys/vm/laptop_mode
	LegacyVaLayout            *int64   // /proc/sys/vm/legacy_va_layout
	LowmemReserveRatio        []*int64 // /proc/sys/vm/lowmem_reserve_ratio
	MaxMapCount               *int64   // /proc/sys/vm/max_map_count
	MemoryFailureEarlyKill    *int64   // /proc/sys/vm/memory_failure_early_kill
	MemoryFailureRecovery     *int64   // /proc/sys/vm/memory_failure_recovery
	MinFreeKbytes             *int64   // /proc/sys/vm/min_free_kbytes
	MinSlabRatio              *int64   // /proc/sys/vm/min_slab_ratio
	MinUnmappedRatio          *int64   // /proc/sys/vm/min_unmapped_ratio
	MmapMinAddr               *int64   // /proc/sys/vm/mmap_min_addr
	NrHugepages               *int64   // /proc/sys/vm/nr_hugepages
	NrHugepagesMempolicy      *int64   // /proc/sys/vm/nr_hugepages_mempolicy
	NrOvercommitHugepages     *int64   // /proc/sys/vm/nr_overcommit_hugepages
	NumaStat                  *int64   // /proc/sys/vm/numa_stat
	NumaZonelistOrder         string   // /proc/sys/vm/numa_zonelist_order
	OomDumpTasks              *int64   // /proc/sys/vm/oom_dump_tasks
	OomKillAllocatingTask     *int64   // /proc/sys/vm/oom_kill_allocating_task
	OvercommitKbytes          *int64   // /proc/sys/vm/overcommit_kbytes
	OvercommitMemory          *int64   // /proc/sys/vm/overcommit_memory
	OvercommitRatio           *int64   // /proc/sys/vm/overcommit_ratio
	PageCluster               *int64   // /proc/sys/vm/page-cluster
	PanicOnOom                *int64   // /proc/sys/vm/panic_on_oom
	PercpuPagelistFraction    *int64   // /proc/sys/vm/percpu_pagelist_fraction
	StatInterval              *int64   // /proc/sys/vm/stat_interval
	Swappiness                *int64   // /proc/sys/vm/swappiness
	UserReserveKbytes         *int64   // /proc/sys/vm/user_reserve_kbytes
	VfsCachePressure          *int64   // /proc/sys/vm/vfs_cache_pressure
	WatermarkBoostFactor      *int64   // /proc/sys/vm/watermark_boost_factor
	WatermarkScaleFactor      *int64   // /proc/sys/vm/watermark_scale_factor
	ZoneReclaimMode           *int64   // /proc/sys/vm/zone_reclaim_mode
}

// VM reads the VM statistics from the specified `proc` filesystem.
func (fs FS) VM() (*VM, error) {
	path := fs.proc.Path("sys/vm")
	file, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if !file.Mode().IsDir() {
		return nil, fmt.Errorf("%w: %s is not a directory", ErrFileRead, path)
	}

	files, err := os.ReadDir(path)
	if err != nil {
		return nil, err
	}

	var vm VM
	for _, f := range files {
		if f.IsDir() {
			continue
		}

		name := filepath.Join(path, f.Name())
		// ignore errors on read, as there are some write only
		// in /proc/sys/vm
		value, err := util.SysReadFile(name)
		if err != nil {
			continue
		}
		vp := util.NewValueParser(value)

		switch f.Name() {
		case "admin_reserve_kbytes":
			vm.AdminReserveKbytes = vp.PInt64()
		case "block_dump":
			vm.BlockDump = vp.PInt64()
		case "compact_unevictable_allowed":
			vm.CompactUnevictableAllowed = vp.PInt64()
		case "dirty_background_bytes":
			vm.DirtyBackgroundBytes = vp.PInt64()
		case "dirty_background_ratio":
			vm.DirtyBackgroundRatio = vp.PInt64()
		case "dirty_bytes":
			vm.DirtyBytes = vp.PInt64()
		case "dirty_expire_centisecs":
			vm.DirtyExpireCentisecs = vp.PInt64()
		case "dirty_ratio":
			vm.DirtyRatio = vp.PInt64()
		case "dirtytime_expire_seconds":
			vm.DirtytimeExpireSeconds = vp.PInt64()
		case "dirty_writeback_centisecs":
			vm.DirtyWritebackCentisecs = vp.PInt64()
		case "drop_caches":
			vm.DropCaches = vp.PInt64()
		case "extfrag_threshold":
			vm.ExtfragThreshold = vp.PInt64()
		case "hugetlb_shm_group":
			vm.HugetlbShmGroup = vp.PInt64()
		case "laptop_mode":
			vm.LaptopMode = vp.PInt64()
		case "legacy_va_layout":
			vm.LegacyVaLayout = vp.PInt64()
		case "lowmem_reserve_ratio":
			stringSlice := strings.Fields(value)
			pint64Slice := make([]*int64, 0, len(stringSlice))
			for _, value := range stringSlice {
				vp := util.NewValueParser(value)
				pint64Slice = append(pint64Slice, vp.PInt64())
			}
			vm.LowmemReserveRatio = pint64Slice
		case "max_map_count":
			vm.MaxMapCount = vp.PInt64()
		case "memory_failure_early_kill":
			vm.MemoryFailureEarlyKill = vp.PInt64()
		case "memory_failure_recovery":
			vm.MemoryFailureRecovery = vp.PInt64()
		case "min_free_kbytes":
			vm.MinFreeKbytes = vp.PInt64()
		case "min_slab_ratio":
			vm.MinSlabRatio = vp.PInt64()
		case "min_unmapped_ratio":
			vm.MinUnmappedRatio = vp.PInt64()
		case "mmap_min_addr":
			vm.MmapMinAddr = vp.PInt64()
		case "nr_hugepages":
			vm.NrHugepages = vp.PInt64()
		case "nr_hugepages_mempolicy":
			vm.NrHugepagesMempolicy = vp.PInt64()
		case "nr_overcommit_hugepages":
			vm.NrOvercommitHugepages = vp.PInt64()
		case "numa_stat":
			vm.NumaStat = vp.PInt64()
		case "numa_zonelist_order":
			vm.NumaZonelistOrder = value
		case "oom_dump_tasks":
			vm.OomDumpTasks = vp.PInt64()
		case "oom_kill_allocating_task":
			vm.OomKillAllocatingTask = vp.PInt64()
		case "overcommit_kbytes":
			vm.OvercommitKbytes = vp.PInt64()
		case "overcommit_memory":
			vm.OvercommitMemory = vp.PInt64()
		case "overcommit_ratio":
			vm.OvercommitRatio = vp.PInt64()
		case "page-cluster":
			vm.PageCluster = vp.PInt64()
		case "panic_on_oom":
			vm.PanicOnOom = vp.PInt64()
		case "percpu_pagelist_fraction":
			vm.PercpuPagelistFraction = vp.PInt64()
		case "stat_interval":
			vm.StatInterval = vp.PInt64()
		case "swappiness":
			vm.Swappiness = vp.PInt64()
		case "user_reserve_kbytes":
			vm.UserReserveKbytes = vp.PInt64()
		case "vfs_cache_pressure":
			vm.VfsCachePressure = vp.PInt64()
		case "watermark_boost_factor":
			vm.WatermarkBoostFactor = vp.PInt64()
		case "watermark_scale_factor":
			vm.WatermarkScaleFactor = vp.PInt64()
		case "zone_reclaim_mode":
			vm.ZoneReclaimMode = vp.PInt64()
		}
		if err := vp.Err(); err != nil {
			return nil, err
		}
	}

	return &vm, nil
}

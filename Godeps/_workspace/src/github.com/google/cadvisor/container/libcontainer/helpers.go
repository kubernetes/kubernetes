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

package libcontainer

import (
	"fmt"
	"time"

	"github.com/docker/libcontainer"
	"github.com/docker/libcontainer/cgroups"
	cgroupfs "github.com/docker/libcontainer/cgroups/fs"
	"github.com/docker/libcontainer/network"
	info "github.com/google/cadvisor/info/v1"
)

type CgroupSubsystems struct {
	// Cgroup subsystem mounts.
	// e.g.: "/sys/fs/cgroup/cpu" -> ["cpu", "cpuacct"]
	Mounts []cgroups.Mount

	// Cgroup subsystem to their mount location.
	// e.g.: "cpu" -> "/sys/fs/cgroup/cpu"
	MountPoints map[string]string
}

// Get information about the cgroup subsystems.
func GetCgroupSubsystems() (CgroupSubsystems, error) {
	// Get all cgroup mounts.
	allCgroups, err := cgroups.GetCgroupMounts()
	if err != nil {
		return CgroupSubsystems{}, err
	}
	if len(allCgroups) == 0 {
		return CgroupSubsystems{}, fmt.Errorf("failed to find cgroup mounts")
	}

	// Trim the mounts to only the subsystems we care about.
	supportedCgroups := make([]cgroups.Mount, 0, len(allCgroups))
	mountPoints := make(map[string]string, len(allCgroups))
	for _, mount := range allCgroups {
		for _, subsystem := range mount.Subsystems {
			if _, ok := supportedSubsystems[subsystem]; ok {
				supportedCgroups = append(supportedCgroups, mount)
				mountPoints[subsystem] = mount.Mountpoint
			}
		}
	}

	return CgroupSubsystems{
		Mounts:      supportedCgroups,
		MountPoints: mountPoints,
	}, nil
}

// Cgroup subsystems we support listing (should be the minimal set we need stats from).
var supportedSubsystems map[string]struct{} = map[string]struct{}{
	"cpu":     {},
	"cpuacct": {},
	"memory":  {},
	"cpuset":  {},
	"blkio":   {},
}

// Get stats of the specified container
func GetStats(cgroupPaths map[string]string, state *libcontainer.State) (*info.ContainerStats, error) {
	// TODO(vmarmol): Use libcontainer's Stats() in the new API when that is ready.
	stats := &libcontainer.ContainerStats{}

	var err error
	stats.CgroupStats, err = cgroupfs.GetStats(cgroupPaths)
	if err != nil {
		return &info.ContainerStats{}, err
	}

	stats.NetworkStats, err = network.GetStats(&state.NetworkState)
	if err != nil {
		return &info.ContainerStats{}, err
	}

	return toContainerStats(stats), nil
}

func DiskStatsCopy(blkio_stats []cgroups.BlkioStatEntry) (stat []info.PerDiskStats) {
	if len(blkio_stats) == 0 {
		return
	}
	type DiskKey struct {
		Major uint64
		Minor uint64
	}
	disk_stat := make(map[DiskKey]*info.PerDiskStats)
	for i := range blkio_stats {
		major := blkio_stats[i].Major
		minor := blkio_stats[i].Minor
		disk_key := DiskKey{
			Major: major,
			Minor: minor,
		}
		diskp, ok := disk_stat[disk_key]
		if !ok {
			disk := info.PerDiskStats{
				Major: major,
				Minor: minor,
			}
			disk.Stats = make(map[string]uint64)
			diskp = &disk
			disk_stat[disk_key] = diskp
		}
		op := blkio_stats[i].Op
		if op == "" {
			op = "Count"
		}
		diskp.Stats[op] = blkio_stats[i].Value
	}
	i := 0
	stat = make([]info.PerDiskStats, len(disk_stat))
	for _, disk := range disk_stat {
		stat[i] = *disk
		i++
	}
	return
}

// Convert libcontainer stats to info.ContainerStats.
func toContainerStats(libcontainerStats *libcontainer.ContainerStats) *info.ContainerStats {
	s := libcontainerStats.CgroupStats
	ret := new(info.ContainerStats)
	ret.Timestamp = time.Now()

	if s != nil {
		ret.Cpu.Usage.User = s.CpuStats.CpuUsage.UsageInUsermode
		ret.Cpu.Usage.System = s.CpuStats.CpuUsage.UsageInKernelmode
		n := len(s.CpuStats.CpuUsage.PercpuUsage)
		ret.Cpu.Usage.PerCpu = make([]uint64, n)

		ret.Cpu.Usage.Total = 0
		for i := 0; i < n; i++ {
			ret.Cpu.Usage.PerCpu[i] = s.CpuStats.CpuUsage.PercpuUsage[i]
			ret.Cpu.Usage.Total += s.CpuStats.CpuUsage.PercpuUsage[i]
		}

		ret.DiskIo.IoServiceBytes = DiskStatsCopy(s.BlkioStats.IoServiceBytesRecursive)
		ret.DiskIo.IoServiced = DiskStatsCopy(s.BlkioStats.IoServicedRecursive)
		ret.DiskIo.IoQueued = DiskStatsCopy(s.BlkioStats.IoQueuedRecursive)
		ret.DiskIo.Sectors = DiskStatsCopy(s.BlkioStats.SectorsRecursive)
		ret.DiskIo.IoServiceTime = DiskStatsCopy(s.BlkioStats.IoServiceTimeRecursive)
		ret.DiskIo.IoWaitTime = DiskStatsCopy(s.BlkioStats.IoWaitTimeRecursive)
		ret.DiskIo.IoMerged = DiskStatsCopy(s.BlkioStats.IoMergedRecursive)
		ret.DiskIo.IoTime = DiskStatsCopy(s.BlkioStats.IoTimeRecursive)

		ret.Memory.Usage = s.MemoryStats.Usage
		if v, ok := s.MemoryStats.Stats["pgfault"]; ok {
			ret.Memory.ContainerData.Pgfault = v
			ret.Memory.HierarchicalData.Pgfault = v
		}
		if v, ok := s.MemoryStats.Stats["pgmajfault"]; ok {
			ret.Memory.ContainerData.Pgmajfault = v
			ret.Memory.HierarchicalData.Pgmajfault = v
		}
		if v, ok := s.MemoryStats.Stats["total_inactive_anon"]; ok {
			ret.Memory.WorkingSet = ret.Memory.Usage - v
			if v, ok := s.MemoryStats.Stats["total_active_file"]; ok {
				ret.Memory.WorkingSet -= v
			}
		}
	}
	// TODO(vishh): Perform a deep copy or alias libcontainer network stats.
	if libcontainerStats.NetworkStats != nil {
		ret.Network = *(*info.NetworkStats)(libcontainerStats.NetworkStats)
	}

	return ret
}

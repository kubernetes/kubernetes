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
	"path"
	"time"

	"github.com/docker/libcontainer"
	"github.com/docker/libcontainer/cgroups"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/utils/sysinfo"
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

// Get cgroup and networking stats of the specified container
func GetStats(cgroupManager cgroups.Manager, networkInterfaces []string) (*info.ContainerStats, error) {
	cgroupStats, err := cgroupManager.GetStats()
	if err != nil {
		return nil, err
	}
	libcontainerStats := &libcontainer.Stats{
		CgroupStats: cgroupStats,
	}
	stats := toContainerStats(libcontainerStats)

	// TODO(rjnagal): Use networking stats directly from libcontainer.
	stats.Network.Interfaces = make([]info.InterfaceStats, len(networkInterfaces))
	for i := range networkInterfaces {
		interfaceStats, err := sysinfo.GetNetworkStats(networkInterfaces[i])
		if err != nil {
			return stats, err
		}
		stats.Network.Interfaces[i] = interfaceStats
	}
	// For backwards compatability.
	if len(networkInterfaces) > 0 {
		stats.Network.InterfaceStats = stats.Network.Interfaces[0]
	}

	return stats, nil
}

func GetProcesses(cgroupManager cgroups.Manager) ([]int, error) {
	pids, err := cgroupManager.GetPids()
	if err != nil {
		return nil, err
	}
	return pids, nil
}

func DockerStateDir(dockerRoot string) string {
	return path.Join(dockerRoot, "containers")
}

func DiskStatsCopy0(major, minor uint64) *info.PerDiskStats {
	disk := info.PerDiskStats{
		Major: major,
		Minor: minor,
	}
	disk.Stats = make(map[string]uint64)
	return &disk
}

type DiskKey struct {
	Major uint64
	Minor uint64
}

func DiskStatsCopy1(disk_stat map[DiskKey]*info.PerDiskStats) []info.PerDiskStats {
	i := 0
	stat := make([]info.PerDiskStats, len(disk_stat))
	for _, disk := range disk_stat {
		stat[i] = *disk
		i++
	}
	return stat
}

func DiskStatsCopy(blkio_stats []cgroups.BlkioStatEntry) (stat []info.PerDiskStats) {
	if len(blkio_stats) == 0 {
		return
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
			diskp = DiskStatsCopy0(major, minor)
			disk_stat[disk_key] = diskp
		}
		op := blkio_stats[i].Op
		if op == "" {
			op = "Count"
		}
		diskp.Stats[op] = blkio_stats[i].Value
	}
	return DiskStatsCopy1(disk_stat)
}

// Convert libcontainer stats to info.ContainerStats.
func toContainerStats0(s *cgroups.Stats, ret *info.ContainerStats) {
	ret.Cpu.Usage.User = s.CpuStats.CpuUsage.UsageInUsermode
	ret.Cpu.Usage.System = s.CpuStats.CpuUsage.UsageInKernelmode
	n := len(s.CpuStats.CpuUsage.PercpuUsage)
	ret.Cpu.Usage.PerCpu = make([]uint64, n)

	ret.Cpu.Usage.Total = 0
	for i := 0; i < n; i++ {
		ret.Cpu.Usage.PerCpu[i] = s.CpuStats.CpuUsage.PercpuUsage[i]
		ret.Cpu.Usage.Total += s.CpuStats.CpuUsage.PercpuUsage[i]
	}
}

func toContainerStats1(s *cgroups.Stats, ret *info.ContainerStats) {
	ret.DiskIo.IoServiceBytes = DiskStatsCopy(s.BlkioStats.IoServiceBytesRecursive)
	ret.DiskIo.IoServiced = DiskStatsCopy(s.BlkioStats.IoServicedRecursive)
	ret.DiskIo.IoQueued = DiskStatsCopy(s.BlkioStats.IoQueuedRecursive)
	ret.DiskIo.Sectors = DiskStatsCopy(s.BlkioStats.SectorsRecursive)
	ret.DiskIo.IoServiceTime = DiskStatsCopy(s.BlkioStats.IoServiceTimeRecursive)
	ret.DiskIo.IoWaitTime = DiskStatsCopy(s.BlkioStats.IoWaitTimeRecursive)
	ret.DiskIo.IoMerged = DiskStatsCopy(s.BlkioStats.IoMergedRecursive)
	ret.DiskIo.IoTime = DiskStatsCopy(s.BlkioStats.IoTimeRecursive)
}

func toContainerStats2(s *cgroups.Stats, ret *info.ContainerStats) {
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
		workingSet := ret.Memory.Usage
		if workingSet < v {
			workingSet = 0
		} else {
			workingSet -= v
		}

		if v, ok := s.MemoryStats.Stats["total_inactive_file"]; ok {
			if workingSet < v {
				workingSet = 0
			} else {
				workingSet -= v
			}
		}
		ret.Memory.WorkingSet = workingSet
	}
}

func toContainerStats3(libcontainerStats *libcontainer.Stats, ret *info.ContainerStats) {
	ret.Network.Interfaces = make([]info.InterfaceStats, len(libcontainerStats.Interfaces))
	for i := range libcontainerStats.Interfaces {
		ret.Network.Interfaces[i] = info.InterfaceStats{
			Name:      libcontainerStats.Interfaces[i].Name,
			RxBytes:   libcontainerStats.Interfaces[i].RxBytes,
			RxPackets: libcontainerStats.Interfaces[i].RxPackets,
			RxErrors:  libcontainerStats.Interfaces[i].RxErrors,
			RxDropped: libcontainerStats.Interfaces[i].RxDropped,
			TxBytes:   libcontainerStats.Interfaces[i].TxBytes,
			TxPackets: libcontainerStats.Interfaces[i].TxPackets,
			TxErrors:  libcontainerStats.Interfaces[i].TxErrors,
			TxDropped: libcontainerStats.Interfaces[i].TxDropped,
		}
	}

	// Add to base struct for backwards compatability.
	if len(ret.Network.Interfaces) > 0 {
		ret.Network.InterfaceStats = ret.Network.Interfaces[0]
	}
}

func toContainerStats(libcontainerStats *libcontainer.Stats) *info.ContainerStats {
	s := libcontainerStats.CgroupStats
	ret := new(info.ContainerStats)
	ret.Timestamp = time.Now()

	if s != nil {
		toContainerStats0(s, ret)
		toContainerStats1(s, ret)
		toContainerStats2(s, ret)
	}
	if len(libcontainerStats.Interfaces) > 0 {
		toContainerStats3(libcontainerStats, ret)
	}
	return ret
}

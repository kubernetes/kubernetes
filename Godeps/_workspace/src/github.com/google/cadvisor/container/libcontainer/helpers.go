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
	"bufio"
	"fmt"
	"net"
	"os/exec"
	"path"
	"regexp"
	"runtime"
	"strings"
	"time"

	"github.com/docker/libcontainer"
	"github.com/docker/libcontainer/cgroups"
	"github.com/golang/glog"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/utils/sysinfo"
	"github.com/vishvananda/netns"
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
func GetStats(cgroupManager cgroups.Manager, networkInterfaces []string, pid int) (*info.ContainerStats, error) {
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

	// If we know the pid & we haven't discovered any network interfaces yet
	// try the network namespace.
	if pid > 0 && len(stats.Network.Interfaces) == 0 {
		nsStats, err := networkStatsFromNs(pid)
		if err != nil {
			glog.V(2).Infof("Unable to get network stats from pid %d: %v", pid, err)
		} else {
			stats.Network.Interfaces = append(stats.Network.Interfaces, nsStats...)
		}
	}

	// For backwards compatibility.
	if len(stats.Network.Interfaces) > 0 {
		stats.Network.InterfaceStats = stats.Network.Interfaces[0]
	}

	return stats, nil
}

func networkStatsFromNs(pid int) ([]info.InterfaceStats, error) {
	// Lock the OS Thread so we only change the ns for this thread exclusively
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	stats := []info.InterfaceStats{}

	// Save the current network namespace
	origns, _ := netns.Get()
	defer origns.Close()

	// Switch to the pid netns
	pidns, err := netns.GetFromPid(pid)
	defer pidns.Close()
	if err != nil {
		return stats, nil
	}
	netns.Set(pidns)

	// Defer setting back to original ns
	defer netns.Set(origns)

	ifaceStats, err := scanInterfaceStats()
	if err != nil {
		return stats, fmt.Errorf("couldn't read network stats: %v", err)
	}

	ifaces, err := net.Interfaces()
	if err != nil {
		return stats, fmt.Errorf("cannot find interfaces: %v", err)
	}

	for _, iface := range ifaces {
		if iface.Flags&net.FlagUp != 0 && iface.Flags&net.FlagLoopback == 0 {
			if s, ok := ifaceStats[iface.Name]; ok {
				stats = append(stats, s)
			}
		}
	}

	return stats, nil
}

// Borrowed from libnetwork Stats - https://github.com/docker/libnetwork/blob/master/sandbox/interface_linux.go
// In older kernels (like the one in Centos 6.6 distro) sysctl does not have netns support. Therefore
// we cannot gather the statistics from /sys/class/net/<dev>/statistics/<counter> files. Per-netns stats
// are naturally found in /proc/net/dev in kernels which support netns (ifconfig relyes on that).
const (
	netStatsFile = "/proc/net/dev"
)

func scanInterfaceStats() (map[string]info.InterfaceStats, error) {
	re := regexp.MustCompile("[  ]*(.+):([  ]+[0-9]+){16}")

	var (
		bkt uint64
	)

	stats := map[string]info.InterfaceStats{}

	// For some reason ioutil.ReadFile(netStatsFile) reads the file in
	// the default netns when this code is invoked from docker.
	// Executing "cat <netStatsFile>" works as expected.
	data, err := exec.Command("cat", netStatsFile).Output()
	if err != nil {
		return stats, fmt.Errorf("failure opening %s: %v", netStatsFile, err)
	}

	reader := strings.NewReader(string(data))
	scanner := bufio.NewScanner(reader)

	scanner.Split(bufio.ScanLines)

	for scanner.Scan() {
		line := scanner.Text()
		if re.MatchString(line) {
			line = strings.Replace(line, ":", "", -1)

			i := info.InterfaceStats{}

			_, err := fmt.Sscanf(line, "%s %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d",
				&i.Name, &i.RxBytes, &i.RxPackets, &i.RxErrors, &i.RxDropped, &bkt, &bkt, &bkt,
				&bkt, &i.TxBytes, &i.TxPackets, &i.TxErrors, &i.TxDropped, &bkt, &bkt, &bkt, &bkt)

			if err != nil {
				return stats, fmt.Errorf("failure opening %s: %v", netStatsFile, err)
			}

			stats[i.Name] = i
		}
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

	// Add to base struct for backwards compatibility.
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

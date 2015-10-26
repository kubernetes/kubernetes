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
	"io/ioutil"
	"path"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/docker/libcontainer"
	"github.com/docker/libcontainer/cgroups"
	"github.com/golang/glog"
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

// Get cgroup and networking stats of the specified container
func GetStats(cgroupManager cgroups.Manager, rootFs string, pid int) (*info.ContainerStats, error) {
	cgroupStats, err := cgroupManager.GetStats()
	if err != nil {
		return nil, err
	}
	libcontainerStats := &libcontainer.Stats{
		CgroupStats: cgroupStats,
	}
	stats := toContainerStats(libcontainerStats)

	// If we know the pid then get network stats from /proc/<pid>/net/dev
	if pid > 0 {
		netStats, err := networkStatsFromProc(rootFs, pid)
		if err != nil {
			glog.V(2).Infof("Unable to get network stats from pid %d: %v", pid, err)
		} else {
			stats.Network.Interfaces = append(stats.Network.Interfaces, netStats...)
		}

		t, err := tcpStatsFromProc(rootFs, pid, "net/tcp")
		if err != nil {
			glog.V(2).Infof("Unable to get tcp stats from pid %d: %v", pid, err)
		} else {
			stats.Network.Tcp = t
		}

		t6, err := tcpStatsFromProc(rootFs, pid, "net/tcp6")
		if err != nil {
			glog.V(2).Infof("Unable to get tcp6 stats from pid %d: %v", pid, err)
		} else {
			stats.Network.Tcp6 = t6
		}
	}

	// For backwards compatibility.
	if len(stats.Network.Interfaces) > 0 {
		stats.Network.InterfaceStats = stats.Network.Interfaces[0]
	}

	return stats, nil
}

func networkStatsFromProc(rootFs string, pid int) ([]info.InterfaceStats, error) {
	netStatsFile := path.Join(rootFs, "proc", strconv.Itoa(pid), "/net/dev")

	ifaceStats, err := scanInterfaceStats(netStatsFile)
	if err != nil {
		return []info.InterfaceStats{}, fmt.Errorf("couldn't read network stats: %v", err)
	}

	return ifaceStats, nil
}

var (
	ignoredDevicePrefixes = []string{"lo", "veth", "docker"}
	netStatLineRE         = regexp.MustCompile("[  ]*(.+):([  ]+[0-9]+){16}")
)

func isIgnoredDevice(ifName string) bool {
	for _, prefix := range ignoredDevicePrefixes {
		if strings.HasPrefix(strings.ToLower(ifName), prefix) {
			return true
		}
	}
	return false
}

func scanInterfaceStats(netStatsFile string) ([]info.InterfaceStats, error) {
	var (
		bkt uint64
	)

	stats := []info.InterfaceStats{}

	data, err := ioutil.ReadFile(netStatsFile)
	if err != nil {
		return stats, fmt.Errorf("failure opening %s: %v", netStatsFile, err)
	}

	reader := strings.NewReader(string(data))
	scanner := bufio.NewScanner(reader)

	scanner.Split(bufio.ScanLines)

	for scanner.Scan() {
		line := scanner.Text()
		if netStatLineRE.MatchString(line) {
			line = strings.Replace(line, ":", "", -1)

			i := info.InterfaceStats{}

			_, err := fmt.Sscanf(line, "%s %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d",
				&i.Name, &i.RxBytes, &i.RxPackets, &i.RxErrors, &i.RxDropped, &bkt, &bkt, &bkt,
				&bkt, &i.TxBytes, &i.TxPackets, &i.TxErrors, &i.TxDropped, &bkt, &bkt, &bkt, &bkt)

			if err != nil {
				return stats, fmt.Errorf("failure opening %s: %v", netStatsFile, err)
			}

			if !isIgnoredDevice(i.Name) {
				stats = append(stats, i)
			}
		}
	}

	return stats, nil
}

func tcpStatsFromProc(rootFs string, pid int, file string) (info.TcpStat, error) {
	tcpStatsFile := path.Join(rootFs, "proc", strconv.Itoa(pid), file)

	tcpStats, err := scanTcpStats(tcpStatsFile)
	if err != nil {
		return tcpStats, fmt.Errorf("couldn't read tcp stats: %v", err)
	}

	return tcpStats, nil
}

func scanTcpStats(tcpStatsFile string) (info.TcpStat, error) {

	var stats info.TcpStat

	data, err := ioutil.ReadFile(tcpStatsFile)
	if err != nil {
		return stats, fmt.Errorf("failure opening %s: %v", tcpStatsFile, err)
	}

	tcpStatLineRE, _ := regexp.Compile("[0-9:].*")

	tcpStateMap := map[string]uint64{
		"01": 0, //ESTABLISHED
		"02": 0, //SYN_SENT
		"03": 0, //SYN_RECV
		"04": 0, //FIN_WAIT1
		"05": 0, //FIN_WAIT2
		"06": 0, //TIME_WAIT
		"07": 0, //CLOSE
		"08": 0, //CLOSE_WAIT
		"09": 0, //LAST_ACK
		"0A": 0, //LISTEN
		"0B": 0, //CLOSING
	}

	reader := strings.NewReader(string(data))
	scanner := bufio.NewScanner(reader)

	scanner.Split(bufio.ScanLines)

	for scanner.Scan() {

		line := scanner.Text()
		//skip header
		matched := tcpStatLineRE.MatchString(line)

		if matched {
			state := strings.Fields(line)
			//#file header tcp state is the 4 filed:
			//sl local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt  uid timeout inode
			tcpStateMap[state[3]]++
		}
	}

	stats = info.TcpStat{
		Established: tcpStateMap["01"],
		SynSent:     tcpStateMap["02"],
		SynRecv:     tcpStateMap["03"],
		FinWait1:    tcpStateMap["04"],
		FinWait2:    tcpStateMap["05"],
		TimeWait:    tcpStateMap["06"],
		Close:       tcpStateMap["07"],
		CloseWait:   tcpStateMap["08"],
		LastAck:     tcpStateMap["09"],
		Listen:      tcpStateMap["0A"],
		Closing:     tcpStateMap["0B"],
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
	ret.Memory.Failcnt = s.MemoryStats.Failcnt
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

// Copyright 2018 Google Inc. All Rights Reserved.
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

//go:build linux

package libcontainer

import (
	"bufio"
	"fmt"
	"os"
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/opencontainers/cgroups"
	"github.com/opencontainers/cgroups/fs2"
	"github.com/opencontainers/cgroups/fscommon"
	"k8s.io/klog/v2"

	"github.com/dims/libcadvisor/container"
	"github.com/dims/libcadvisor/container/common"
	info "github.com/dims/libcadvisor/model"
)

type Handler struct {
	cgroupManager   cgroups.Manager
	rootFs          string
	pid             int
	includedMetrics container.MetricSet
}

func NewHandler(cgroupManager cgroups.Manager, rootFs string, pid int, includedMetrics container.MetricSet) *Handler {
	return &Handler{
		cgroupManager:   cgroupManager,
		rootFs:          rootFs,
		pid:             pid,
		includedMetrics: includedMetrics,
	}
}

// Get cgroup and networking stats of the specified container
func (h *Handler) GetStats() (*info.ContainerStats, error) {
	ignoreStatsError := false
	if cgroups.IsCgroup2UnifiedMode() {
		// On cgroup v2 the root cgroup stats have been introduced in recent kernel versions,
		// so not all kernel versions have all the data. This means that stat fetching can fail
		// due to lacking cgroup stat files, but that some data is provided.
		if h.cgroupManager.Path("") == fs2.UnifiedMountpoint {
			ignoreStatsError = true
		}
	}

	cgroupStats, err := h.cgroupManager.GetStats()
	if err != nil {
		if !ignoreStatsError {
			return nil, err
		}
		klog.V(4).Infof("Ignoring errors when gathering stats for root cgroup since some controllers don't have stats on the root cgroup: %v", err)
	}
	stats := newContainerStats(cgroupStats, h.includedMetrics)

	if cgroups.IsCgroup2UnifiedMode() {
		setMemoryEvents(h.cgroupManager.Path(""), stats)
	}

	// If we know the pid then get network stats from /proc/<pid>/net/dev
	if h.pid > 0 {
		if h.includedMetrics.Has(container.NetworkUsageMetrics) {
			netStats, err := networkStatsFromProc(h.rootFs, h.pid)
			if err != nil {
				klog.V(4).Infof("Unable to get network stats from pid %d: %v", h.pid, err)
			} else {
				stats.Network.Interfaces = append(stats.Network.Interfaces, netStats...)
			}
		}
	}
	// some process metrics are per container ( number of processes, number of
	// file descriptors etc.) and not required a proper container's
	// root PID (systemd services don't have the root PID atm)
	if h.includedMetrics.Has(container.ProcessMetrics) {
		path, ok := common.GetControllerPath(h.cgroupManager.GetPaths(), "cpu", cgroups.IsCgroup2UnifiedMode())
		if !ok {
			klog.V(4).Infof("Could not find cgroups CPU for container %d", h.pid)
		} else {
			stats.Processes, err = processStatsFromProcs(h.rootFs, path, h.pid)
			if err != nil {
				klog.V(4).Infof("Unable to get Process Stats: %v", err)
			}
		}

		// if include processes metrics, just set threads metrics if exist, and has no relationship with cpu path
		setThreadsStats(cgroupStats, stats)
	}

	// For backwards compatibility.
	if len(stats.Network.Interfaces) > 0 {
		stats.Network.InterfaceStats = stats.Network.Interfaces[0]
	}

	return stats, nil
}

func parseUlimit(value string) (int64, error) {
	num, err := strconv.ParseInt(value, 10, 64)
	if err != nil {
		if strings.EqualFold(value, "unlimited") {
			// -1 implies unlimited except for priority and nice; man limits.conf
			num = -1
		} else {
			// Value is not a number or "unlimited"; return an error
			return 0, fmt.Errorf("unable to parse limit: %s", value)
		}
	}
	return num, nil
}

func processLimitsFile(fileData string) []info.UlimitSpec {
	const maxOpenFilesLinePrefix = "Max open files"

	limits := strings.Split(fileData, "\n")
	ulimits := make([]info.UlimitSpec, 0, len(limits))
	for _, lim := range limits {
		// Skip any headers/footers
		if strings.HasPrefix(lim, "Max open files") {
			// Remove line prefix
			ulimit, err := processMaxOpenFileLimitLine(
				"max_open_files",
				lim[len(maxOpenFilesLinePrefix):],
			)
			if err == nil {
				ulimits = append(ulimits, ulimit)
			}
		}
	}
	return ulimits
}

// Any caller of processMaxOpenFileLimitLine must ensure that the name prefix is already removed from the limit line.
// with the "Max open files" prefix.
func processMaxOpenFileLimitLine(name, line string) (info.UlimitSpec, error) {
	// Remove any leading whitespace
	line = strings.TrimSpace(line)
	// Split on whitespace
	fields := strings.Fields(line)
	if len(fields) != 3 {
		return info.UlimitSpec{}, fmt.Errorf("unable to parse max open files line: %s", line)
	}
	// The first field is the soft limit, the second is the hard limit
	soft, err := parseUlimit(fields[0])
	if err != nil {
		return info.UlimitSpec{}, err
	}
	hard, err := parseUlimit(fields[1])
	if err != nil {
		return info.UlimitSpec{}, err
	}
	return info.UlimitSpec{
		Name:      name,
		SoftLimit: soft,
		HardLimit: hard,
	}, nil
}

func processRootProcUlimits(rootFs string, rootPid int) []info.UlimitSpec {
	filePath := path.Join(rootFs, "/proc", strconv.Itoa(rootPid), "limits")
	out, err := os.ReadFile(filePath)
	if err != nil {
		klog.V(4).Infof("error while listing directory %q to read ulimits: %v", filePath, err)
		return []info.UlimitSpec{}
	}
	return processLimitsFile(string(out))
}

func processStatsFromProcs(rootFs string, cgroupPath string, rootPid int) (info.ProcessStats, error) {
	var fdCount, socketCount uint64
	filePath := path.Join(cgroupPath, "cgroup.procs")
	out, err := os.ReadFile(filePath)
	if err != nil {
		return info.ProcessStats{}, fmt.Errorf("couldn't open cpu cgroup procs file %v : %v", filePath, err)
	}

	pids := strings.Split(string(out), "\n")

	// EOL is also treated as a new line while reading "cgroup.procs" file with os.ReadFile.
	// The last value is an empty string "". Ex: pids = ["22", "1223", ""]
	// Trim the last value
	if len(pids) != 0 && pids[len(pids)-1] == "" {
		pids = pids[:len(pids)-1]
	}

	for _, pid := range pids {
		dirPath := path.Join(rootFs, "/proc", pid, "fd")
		fds, err := os.ReadDir(dirPath)
		if err != nil {
			klog.V(4).Infof("error while listing directory %q to measure fd count: %v", dirPath, err)
			continue
		}
		fdCount += uint64(len(fds))
		for _, fd := range fds {
			fdPath := path.Join(dirPath, fd.Name())
			linkName, err := os.Readlink(fdPath)
			if err != nil {
				klog.V(4).Infof("error while reading %q link: %v", fdPath, err)
				continue
			}
			if strings.HasPrefix(linkName, "socket") {
				socketCount++
			}
		}
	}

	processStats := info.ProcessStats{
		ProcessCount: uint64(len(pids)),
		FdCount:      fdCount,
		SocketCount:  socketCount,
	}

	if rootPid > 0 {
		processStats.Ulimits = processRootProcUlimits(rootFs, rootPid)
	}

	return processStats, nil
}

func networkStatsFromProc(rootFs string, pid int) ([]info.InterfaceStats, error) {
	netStatsFile := path.Join(rootFs, "proc", strconv.Itoa(pid), "/net/dev")

	ifaceStats, err := scanInterfaceStats(netStatsFile)
	if err != nil {
		return []info.InterfaceStats{}, fmt.Errorf("couldn't read network stats: %v", err)
	}

	return ifaceStats, nil
}

var ignoredDevicePrefixes = []string{"lo", "veth", "docker", "nerdctl"}

func isIgnoredDevice(ifName string) bool {
	for _, prefix := range ignoredDevicePrefixes {
		if strings.HasPrefix(strings.ToLower(ifName), prefix) {
			return true
		}
	}
	return false
}

func scanInterfaceStats(netStatsFile string) ([]info.InterfaceStats, error) {
	file, err := os.Open(netStatsFile)
	if err != nil {
		return nil, fmt.Errorf("failure opening %s: %v", netStatsFile, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	// Discard header lines
	for i := 0; i < 2; i++ {
		if b := scanner.Scan(); !b {
			return nil, scanner.Err()
		}
	}

	stats := []info.InterfaceStats{}
	for scanner.Scan() {
		line := scanner.Text()
		line = strings.Replace(line, ":", "", -1)

		fields := strings.Fields(line)
		// If the format of the  line is invalid then don't trust any of the stats
		// in this file.
		if len(fields) != 17 {
			return nil, fmt.Errorf("invalid interface stats line: %v", line)
		}

		devName := fields[0]
		if isIgnoredDevice(devName) {
			continue
		}

		i := info.InterfaceStats{
			Name: devName,
		}

		statFields := append(fields[1:5], fields[9:13]...)
		statPointers := []*uint64{
			&i.RxBytes, &i.RxPackets, &i.RxErrors, &i.RxDropped,
			&i.TxBytes, &i.TxPackets, &i.TxErrors, &i.TxDropped,
		}

		err := setInterfaceStatValues(statFields, statPointers)
		if err != nil {
			return nil, fmt.Errorf("cannot parse interface stats (%v): %v", err, line)
		}

		stats = append(stats, i)
	}

	return stats, nil
}

func setInterfaceStatValues(fields []string, pointers []*uint64) error {
	for i, v := range fields {
		val, err := strconv.ParseUint(v, 10, 64)
		if err != nil {
			return err
		}
		*pointers[i] = val
	}
	return nil
}

func (h *Handler) GetProcesses() ([]int, error) {
	pids, err := h.cgroupManager.GetPids()
	if err != nil {
		return nil, err
	}
	return pids, nil
}

// Convert libcontainer stats to info.ContainerStats.
func setCPUStats(s *cgroups.Stats, ret *info.ContainerStats, withPerCPU bool) {
	ret.Cpu.Usage.User = s.CpuStats.CpuUsage.UsageInUsermode
	ret.Cpu.Usage.System = s.CpuStats.CpuUsage.UsageInKernelmode
	ret.Cpu.Usage.Total = s.CpuStats.CpuUsage.TotalUsage
	ret.Cpu.CFS.Periods = s.CpuStats.ThrottlingData.Periods
	ret.Cpu.CFS.ThrottledPeriods = s.CpuStats.ThrottlingData.ThrottledPeriods
	ret.Cpu.CFS.ThrottledTime = s.CpuStats.ThrottlingData.ThrottledTime
	ret.Cpu.CFS.BurstsPeriods = s.CpuStats.BurstData.BurstsPeriods
	ret.Cpu.CFS.BurstTime = s.CpuStats.BurstData.BurstTime
	setPSIStats(s.CpuStats.PSI, &ret.Cpu.PSI)

	if !withPerCPU {
		return
	}
	if len(s.CpuStats.CpuUsage.PercpuUsage) == 0 {
		// libcontainer's 'GetStats' can leave 'PercpuUsage' nil if it skipped the
		// cpuacct subsystem.
		return
	}
	ret.Cpu.Usage.PerCpu = s.CpuStats.CpuUsage.PercpuUsage
}

func setDiskIoStats(s *cgroups.Stats, ret *info.ContainerStats) {
	ret.DiskIo.IoServiceBytes = diskStatsCopy(s.BlkioStats.IoServiceBytesRecursive)
	ret.DiskIo.IoServiced = diskStatsCopy(s.BlkioStats.IoServicedRecursive)
	ret.DiskIo.IoQueued = diskStatsCopy(s.BlkioStats.IoQueuedRecursive)
	ret.DiskIo.Sectors = diskStatsCopy(s.BlkioStats.SectorsRecursive)
	ret.DiskIo.IoServiceTime = diskStatsCopy(s.BlkioStats.IoServiceTimeRecursive)
	ret.DiskIo.IoWaitTime = diskStatsCopy(s.BlkioStats.IoWaitTimeRecursive)
	ret.DiskIo.IoMerged = diskStatsCopy(s.BlkioStats.IoMergedRecursive)
	ret.DiskIo.IoTime = diskStatsCopy(s.BlkioStats.IoTimeRecursive)
	ret.DiskIo.IoCostUsage = diskStatsCopy(s.BlkioStats.IoCostUsage)
	ret.DiskIo.IoCostWait = diskStatsCopy(s.BlkioStats.IoCostWait)
	ret.DiskIo.IoCostIndebt = diskStatsCopy(s.BlkioStats.IoCostIndebt)
	ret.DiskIo.IoCostIndelay = diskStatsCopy(s.BlkioStats.IoCostIndelay)
	setPSIStats(s.BlkioStats.PSI, &ret.DiskIo.PSI)
}

func setMemoryStats(s *cgroups.Stats, ret *info.ContainerStats) {
	ret.Memory.Usage = s.MemoryStats.Usage.Usage
	ret.Memory.MaxUsage = s.MemoryStats.Usage.MaxUsage
	ret.Memory.Failcnt = s.MemoryStats.Usage.Failcnt
	ret.Memory.KernelUsage = s.MemoryStats.KernelUsage.Usage
	setPSIStats(s.MemoryStats.PSI, &ret.Memory.PSI)

	if cgroups.IsCgroup2UnifiedMode() {
		ret.Memory.Cache = s.MemoryStats.Stats["file"]
		ret.Memory.RSS = s.MemoryStats.Stats["anon"]
		ret.Memory.Swap = s.MemoryStats.SwapUsage.Usage - s.MemoryStats.Usage.Usage
		ret.Memory.MappedFile = s.MemoryStats.Stats["file_mapped"]
	} else if s.MemoryStats.UseHierarchy {
		ret.Memory.Cache = s.MemoryStats.Stats["total_cache"]
		ret.Memory.RSS = s.MemoryStats.Stats["total_rss"]
		ret.Memory.Swap = s.MemoryStats.Stats["total_swap"]
		ret.Memory.MappedFile = s.MemoryStats.Stats["total_mapped_file"]
	} else {
		ret.Memory.Cache = s.MemoryStats.Stats["cache"]
		ret.Memory.RSS = s.MemoryStats.Stats["rss"]
		ret.Memory.Swap = s.MemoryStats.Stats["swap"]
		ret.Memory.MappedFile = s.MemoryStats.Stats["mapped_file"]
	}
	if v, ok := s.MemoryStats.Stats["pgfault"]; ok {
		ret.Memory.ContainerData.Pgfault = v
		ret.Memory.HierarchicalData.Pgfault = v
	}
	if v, ok := s.MemoryStats.Stats["pgmajfault"]; ok {
		ret.Memory.ContainerData.Pgmajfault = v
		ret.Memory.HierarchicalData.Pgmajfault = v
	}

	// Additional memory.stat fields (cgroup v2 only).
	//  On cgroup v1 these keys are absent, so the values are left at zero.
	if v, ok := s.MemoryStats.Stats["file_dirty"]; ok {
		ret.Memory.FileDirty = v
	}
	if v, ok := s.MemoryStats.Stats["file_writeback"]; ok {
		ret.Memory.FileWriteback = v
	}
	if v, ok := s.MemoryStats.Stats["pgscan"]; ok {
		ret.Memory.Pgscan = v
	}
	if v, ok := s.MemoryStats.Stats["pgsteal"]; ok {
		ret.Memory.Pgsteal = v
	}
	if v, ok := s.MemoryStats.Stats["workingset_refault_file"]; ok {
		ret.Memory.WorkingsetRefaultFile = v
	}
	if v, ok := s.MemoryStats.Stats["workingset_refault_anon"]; ok {
		ret.Memory.WorkingsetRefaultAnon = v
	}

	inactiveFileKeyName := "total_inactive_file"
	if cgroups.IsCgroup2UnifiedMode() {
		inactiveFileKeyName = "inactive_file"
	}

	activeFileKeyName := "total_active_file"
	if cgroups.IsCgroup2UnifiedMode() {
		activeFileKeyName = "active_file"
	}

	if v, ok := s.MemoryStats.Stats[activeFileKeyName]; ok {
		ret.Memory.TotalActiveFile = v
	}

	workingSet := ret.Memory.Usage
	if v, ok := s.MemoryStats.Stats[inactiveFileKeyName]; ok {
		ret.Memory.TotalInactiveFile = v
		if workingSet < v {
			workingSet = 0
		} else {
			workingSet -= v
		}
	}
	ret.Memory.WorkingSet = workingSet
}

func setMemoryEvents(cgroupPath string, ret *info.ContainerStats) {
	if val, err := fscommon.GetValueByKey(cgroupPath, "memory.events", "high"); err == nil {
		ret.Memory.Events.High = val
	}
	if val, err := fscommon.GetValueByKey(cgroupPath, "memory.events", "max"); err == nil {
		ret.Memory.Events.Max = val
	}
}

func setPSIData(d *cgroups.PSIData, ret *info.PSIData) {
	if d != nil {
		ret.Total = d.Total
		ret.Avg10 = d.Avg10
		ret.Avg60 = d.Avg60
		ret.Avg300 = d.Avg300
	}
}

func setPSIStats(s *cgroups.PSIStats, ret *info.PSIStats) {
	if s != nil {
		setPSIData(&s.Full, &ret.Full)
		setPSIData(&s.Some, &ret.Some)
	}
}

// read from pids path not cpu
func setThreadsStats(s *cgroups.Stats, ret *info.ContainerStats) {
	if s != nil {
		ret.Processes.ThreadsCurrent = s.PidsStats.Current
		ret.Processes.ThreadsMax = s.PidsStats.Limit
	}
}

func newContainerStats(cgroupStats *cgroups.Stats, includedMetrics container.MetricSet) *info.ContainerStats {
	ret := &info.ContainerStats{
		Timestamp: time.Now(),
	}

	if s := cgroupStats; s != nil {
		setCPUStats(s, ret, includedMetrics.Has(container.PerCpuUsageMetrics))
		if includedMetrics.Has(container.DiskIOMetrics) {
			setDiskIoStats(s, ret)
		}
		setMemoryStats(s, ret)
	}
	return ret
}

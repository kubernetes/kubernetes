// Copyright 2016 Google Inc. All Rights Reserved.
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

package common

import (
	"errors"
	"fmt"
	"io/fs"
	"math"
	"os"
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/opencontainers/cgroups"
	"golang.org/x/sys/unix"

	"github.com/google/cadvisor/container"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/utils"

	"k8s.io/klog/v2"
)

func DebugInfo(watches map[string][]string) map[string][]string {
	out := make(map[string][]string)

	lines := make([]string, 0, len(watches))
	for containerName, cgroupWatches := range watches {
		lines = append(lines, fmt.Sprintf("%s:", containerName))
		for _, cg := range cgroupWatches {
			lines = append(lines, fmt.Sprintf("\t%s", cg))
		}
	}
	out["Inotify watches"] = lines

	return out
}

var bootTime = func() time.Time {
	now := time.Now()
	var sysinfo unix.Sysinfo_t
	if err := unix.Sysinfo(&sysinfo); err != nil {
		return now
	}
	sinceBoot := time.Duration(sysinfo.Uptime) * time.Second
	return now.Add(-1 * sinceBoot).Truncate(time.Minute)
}()

func GetSpec(cgroupPaths map[string]string, machineInfoFactory info.MachineInfoFactory, hasNetwork, hasFilesystem bool) (info.ContainerSpec, error) {
	return getSpecInternal(cgroupPaths, machineInfoFactory, hasNetwork, hasFilesystem, cgroups.IsCgroup2UnifiedMode())
}

func getSpecInternal(cgroupPaths map[string]string, machineInfoFactory info.MachineInfoFactory, hasNetwork, hasFilesystem, cgroup2UnifiedMode bool) (info.ContainerSpec, error) {
	var spec info.ContainerSpec

	// Assume unified hierarchy containers.
	// Get the lowest creation time from all hierarchies as the container creation time.
	now := time.Now()
	lowestTime := now
	for _, cgroupPathDir := range cgroupPaths {
		dir, err := os.Stat(cgroupPathDir)
		if err == nil && dir.ModTime().Before(lowestTime) {
			lowestTime = dir.ModTime()
		} else if os.IsNotExist(err) {
			// Directory does not exist, skip checking for files within.
			continue
		}

		// The modified time of the cgroup directory sometimes changes whenever a subcontainer is created.
		// eg. /docker will have creation time matching the creation of latest docker container.
		// Use clone_children/events as a workaround as it isn't usually modified. It is only likely changed
		// immediately after creating a container. If the directory modified time is lower, we use that.
		cgroupPathFile := path.Join(cgroupPathDir, "cgroup.clone_children")
		if cgroup2UnifiedMode {
			cgroupPathFile = path.Join(cgroupPathDir, "cgroup.events")
		}
		fi, err := os.Stat(cgroupPathFile)
		if err == nil && fi.ModTime().Before(lowestTime) {
			lowestTime = fi.ModTime()
		}
	}
	if lowestTime.Before(bootTime) {
		lowestTime = bootTime
	}

	if lowestTime != now {
		spec.CreationTime = lowestTime
	}

	// Get machine info.
	mi, err := machineInfoFactory.GetMachineInfo()
	if err != nil {
		return spec, err
	}

	// CPU.
	cpuRoot, ok := GetControllerPath(cgroupPaths, "cpu", cgroup2UnifiedMode)
	if ok {
		if utils.FileExists(cpuRoot) {
			if cgroup2UnifiedMode {
				spec.HasCpu = true

				weight := readUInt64(cpuRoot, "cpu.weight")
				if weight > 0 {
					limit, err := convertCPUWeightToCPULimit(weight)
					if err != nil {
						klog.Errorf("GetSpec: Failed to read CPULimit from %q: %s", path.Join(cpuRoot, "cpu.weight"), err)
					} else {
						spec.Cpu.Limit = limit
					}
				}
				max := readString(cpuRoot, "cpu.max")
				if max != "" {
					splits := strings.SplitN(max, " ", 2)
					if len(splits) != 2 {
						klog.Errorf("GetSpec: Failed to parse CPUmax from %q", path.Join(cpuRoot, "cpu.max"))
					} else {
						if splits[0] != "max" {
							spec.Cpu.Quota = parseUint64String(splits[0])
						}
						spec.Cpu.Period = parseUint64String(splits[1])
					}
				}
			} else {
				spec.HasCpu = true
				spec.Cpu.Limit = readUInt64(cpuRoot, "cpu.shares")
				spec.Cpu.Period = readUInt64(cpuRoot, "cpu.cfs_period_us")
				quota := readString(cpuRoot, "cpu.cfs_quota_us")

				if quota != "" && quota != "-1" {
					val, err := strconv.ParseUint(quota, 10, 64)
					if err != nil {
						klog.Errorf("GetSpec: Failed to parse CPUQuota from %q: %s", path.Join(cpuRoot, "cpu.cfs_quota_us"), err)
					} else {
						spec.Cpu.Quota = val
					}
				}
			}
		}
	}

	// Cpu Mask.
	// This will fail for non-unified hierarchies. We'll return the whole machine mask in that case.
	cpusetRoot, ok := GetControllerPath(cgroupPaths, "cpuset", cgroup2UnifiedMode)
	if ok {
		if utils.FileExists(cpusetRoot) {
			spec.HasCpu = true
			mask := ""
			if cgroup2UnifiedMode {
				mask = readString(cpusetRoot, "cpuset.cpus.effective")
			} else {
				mask = readString(cpusetRoot, "cpuset.cpus")
			}
			spec.Cpu.Mask = utils.FixCpuMask(mask, mi.NumCores)
		}
	}

	// Memory
	memoryRoot, ok := GetControllerPath(cgroupPaths, "memory", cgroup2UnifiedMode)
	if ok {
		if cgroup2UnifiedMode {
			if utils.FileExists(path.Join(memoryRoot, "memory.max")) {
				spec.HasMemory = true
				spec.Memory.Reservation = readUInt64(memoryRoot, "memory.min")
				spec.Memory.Limit = readUInt64(memoryRoot, "memory.max")
				spec.Memory.SwapLimit = readUInt64(memoryRoot, "memory.swap.max")
			}
		} else {
			if utils.FileExists(memoryRoot) {
				spec.HasMemory = true
				spec.Memory.Limit = readUInt64(memoryRoot, "memory.limit_in_bytes")
				spec.Memory.SwapLimit = readUInt64(memoryRoot, "memory.memsw.limit_in_bytes")
				spec.Memory.Reservation = readUInt64(memoryRoot, "memory.soft_limit_in_bytes")
			}
		}
	}

	// Hugepage
	hugepageRoot, ok := cgroupPaths["hugetlb"]
	if ok {
		if utils.FileExists(hugepageRoot) {
			spec.HasHugetlb = true
		}
	}

	// Processes, read it's value from pids path directly
	pidsRoot, ok := GetControllerPath(cgroupPaths, "pids", cgroup2UnifiedMode)
	if ok {
		if utils.FileExists(pidsRoot) {
			spec.HasProcesses = true
			spec.Processes.Limit = readUInt64(pidsRoot, "pids.max")
		}
	}

	spec.HasNetwork = hasNetwork
	spec.HasFilesystem = hasFilesystem

	ioControllerName := "blkio"
	if cgroup2UnifiedMode {
		ioControllerName = "io"
	}

	if blkioRoot, ok := GetControllerPath(cgroupPaths, ioControllerName, cgroup2UnifiedMode); ok && utils.FileExists(blkioRoot) {
		spec.HasDiskIo = true
	}

	return spec, nil
}

func GetControllerPath(cgroupPaths map[string]string, controllerName string, cgroup2UnifiedMode bool) (string, bool) {

	ok := false
	path := ""

	if cgroup2UnifiedMode {
		path, ok = cgroupPaths[""]
	} else {
		path, ok = cgroupPaths[controllerName]
	}
	return path, ok
}

func readString(dirpath string, file string) string {
	cgroupFile := path.Join(dirpath, file)

	// Read
	out, err := os.ReadFile(cgroupFile)
	if err != nil {
		// Ignore non-existent files
		if !os.IsNotExist(err) {
			klog.Warningf("readString: Failed to read %q: %s", cgroupFile, err)
		}
		return ""
	}
	return strings.TrimSpace(string(out))
}

// Convert from [1-10000] to [2-262144]
func convertCPUWeightToCPULimit(weight uint64) (uint64, error) {
	const (
		// minWeight is the lowest value possible for cpu.weight
		minWeight = 1
		// maxWeight is the highest value possible for cpu.weight
		maxWeight = 10000
	)
	if weight < minWeight || weight > maxWeight {
		return 0, fmt.Errorf("convertCPUWeightToCPULimit: invalid cpu weight: %v", weight)
	}
	return 2 + ((weight-1)*262142)/9999, nil
}

func parseUint64String(strValue string) uint64 {
	if strValue == "max" {
		return math.MaxUint64
	}
	if strValue == "" {
		return 0
	}

	val, err := strconv.ParseUint(strValue, 10, 64)
	if err != nil {
		klog.Errorf("parseUint64String: Failed to parse int %q: %s", strValue, err)
		return 0
	}

	return val
}

func readUInt64(dirpath string, file string) uint64 {
	out := readString(dirpath, file)
	if out == "max" {
		return math.MaxUint64
	}
	if out == "" {
		return 0
	}

	val, err := strconv.ParseUint(out, 10, 64)
	if err != nil {
		klog.Errorf("readUInt64: Failed to parse int %q from file %q: %s", out, path.Join(dirpath, file), err)
		return 0
	}

	return val
}

// Lists all directories under "path" and outputs the results as children of "parent".
func ListDirectories(dirpath string, parent string, recursive bool, output map[string]struct{}) error {
	dirents, err := os.ReadDir(dirpath)
	if err != nil {
		// Ignore if this hierarchy does not exist.
		if errors.Is(err, fs.ErrNotExist) {
			return nil
		}
		return err
	}
	for _, dirent := range dirents {
		// We only grab directories.
		if !dirent.IsDir() {
			continue
		}
		dirname := dirent.Name()

		name := path.Join(parent, dirname)
		output[name] = struct{}{}

		// List subcontainers if asked to.
		if recursive {
			if err := ListDirectories(path.Join(dirpath, dirname), name, true, output); err != nil {
				return err
			}
		}
	}
	return nil
}

func MakeCgroupPaths(mountPoints map[string]string, name string) map[string]string {
	cgroupPaths := make(map[string]string, len(mountPoints))
	for key, val := range mountPoints {
		cgroupPaths[key] = path.Join(val, name)
	}

	return cgroupPaths
}

func CgroupExists(cgroupPaths map[string]string) bool {
	// If any cgroup exists, the container is still alive.
	for _, cgroupPath := range cgroupPaths {
		if utils.FileExists(cgroupPath) {
			return true
		}
	}
	return false
}

func ListContainers(name string, cgroupPaths map[string]string, listType container.ListType) ([]info.ContainerReference, error) {
	containers := make(map[string]struct{})
	for _, cgroupPath := range cgroupPaths {
		err := ListDirectories(cgroupPath, name, listType == container.ListRecursive, containers)
		if err != nil {
			return nil, err
		}
	}

	// Make into container references.
	ret := make([]info.ContainerReference, 0, len(containers))
	for cont := range containers {
		ret = append(ret, info.ContainerReference{
			Name: cont,
		})
	}

	return ret, nil
}

// AssignDeviceNamesToDiskStats assigns the Device field on the provided DiskIoStats by looking up
// the device major and minor identifiers in the provided device namer.
func AssignDeviceNamesToDiskStats(namer DeviceNamer, stats *info.DiskIoStats) {
	assignDeviceNamesToPerDiskStats(
		namer,
		stats.IoMerged,
		stats.IoQueued,
		stats.IoServiceBytes,
		stats.IoServiceTime,
		stats.IoServiced,
		stats.IoTime,
		stats.IoWaitTime,
		stats.Sectors,
		stats.IoCostUsage,
		stats.IoCostWait,
		stats.IoCostIndebt,
		stats.IoCostIndelay,
	)
}

// assignDeviceNamesToPerDiskStats looks up device names for the provided stats, caching names
// if necessary.
func assignDeviceNamesToPerDiskStats(namer DeviceNamer, diskStats ...[]info.PerDiskStats) {
	devices := make(deviceIdentifierMap)
	for _, stats := range diskStats {
		for i, stat := range stats {
			stats[i].Device = devices.Find(stat.Major, stat.Minor, namer)
		}
	}
}

// DeviceNamer returns string names for devices by their major and minor id.
type DeviceNamer interface {
	// DeviceName returns the name of the device by its major and minor ids, or false if no
	// such device is recognized.
	DeviceName(major, minor uint64) (string, bool)
}

type MachineInfoNamer info.MachineInfo

func (n *MachineInfoNamer) DeviceName(major, minor uint64) (string, bool) {
	for _, info := range n.DiskMap {
		if info.Major == major && info.Minor == minor {
			return "/dev/" + info.Name, true
		}
	}
	for _, info := range n.Filesystems {
		if info.DeviceMajor == major && info.DeviceMinor == minor {
			return info.Device, true
		}
	}
	return "", false
}

type deviceIdentifier struct {
	major uint64
	minor uint64
}

type deviceIdentifierMap map[deviceIdentifier]string

// Find locates the device name by device identifier out of from, caching the result as necessary.
func (m deviceIdentifierMap) Find(major, minor uint64, namer DeviceNamer) string {
	d := deviceIdentifier{major, minor}
	if s, ok := m[d]; ok {
		return s
	}
	s, _ := namer.DeviceName(major, minor)
	m[d] = s
	return s
}

// RemoveNetMetrics is used to remove any network metrics from the given MetricSet.
// It returns the original set as is if remove is false, or if there are no metrics
// to remove.
func RemoveNetMetrics(metrics container.MetricSet, remove bool) container.MetricSet {
	if !remove {
		return metrics
	}

	// Check if there is anything we can remove, to avoid useless copying.
	if !metrics.HasAny(container.AllNetworkMetrics) {
		return metrics
	}

	// A copy of all metrics except for network ones.
	return metrics.Difference(container.AllNetworkMetrics)
}

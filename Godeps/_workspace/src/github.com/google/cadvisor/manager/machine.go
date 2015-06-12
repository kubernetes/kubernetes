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

package manager

import (
	"bytes"
	"flag"
	"fmt"
	"io/ioutil"
	"regexp"
	"strconv"
	"strings"
	"syscall"

	dclient "github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"
	"github.com/google/cadvisor/container/docker"
	"github.com/google/cadvisor/fs"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/utils"
	"github.com/google/cadvisor/utils/sysfs"
	"github.com/google/cadvisor/utils/sysinfo"
	version "github.com/google/cadvisor/version"
)

var cpuRegExp = regexp.MustCompile("processor\\t*: +([0-9]+)")
var coreRegExp = regexp.MustCompile("core id\\t*: +([0-9]+)")
var nodeRegExp = regexp.MustCompile("physical id\\t*: +([0-9]+)")
var CpuClockSpeedMHz = regexp.MustCompile("cpu MHz\\t*: +([0-9]+.[0-9]+)")
var memoryCapacityRegexp = regexp.MustCompile("MemTotal: *([0-9]+) kB")

var machineIdFilePath = flag.String("machine_id_file", "/etc/machine-id,/var/lib/dbus/machine-id", "Comma-separated list of files to check for machine-id. Use the first one that exists.")
var bootIdFilePath = flag.String("boot_id_file", "/proc/sys/kernel/random/boot_id", "Comma-separated list of files to check for boot-id. Use the first one that exists.")

func getClockSpeed(procInfo []byte) (uint64, error) {
	// First look through sys to find a max supported cpu frequency.
	const maxFreqFile = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"
	if utils.FileExists(maxFreqFile) {
		val, err := ioutil.ReadFile(maxFreqFile)
		if err != nil {
			return 0, err
		}
		var maxFreq uint64
		n, err := fmt.Sscanf(string(val), "%d", &maxFreq)
		if err != nil || n != 1 {
			return 0, fmt.Errorf("could not parse frequency %q", val)
		}
		return maxFreq, nil
	}
	// Fall back to /proc/cpuinfo
	matches := CpuClockSpeedMHz.FindSubmatch(procInfo)
	if len(matches) != 2 {
		//Check if we are running on Power systems which have a different format
		CpuClockSpeedMHz, _ = regexp.Compile("clock\\t*: +([0-9]+.[0-9]+)MHz")
		matches = CpuClockSpeedMHz.FindSubmatch(procInfo)
		if len(matches) != 2 {
			return 0, fmt.Errorf("could not detect clock speed from output: %q", string(procInfo))
		}
	}
	speed, err := strconv.ParseFloat(string(matches[1]), 64)
	if err != nil {
		return 0, err
	}
	// Convert to kHz
	return uint64(speed * 1000), nil
}

func getMemoryCapacity(b []byte) (int64, error) {
	matches := memoryCapacityRegexp.FindSubmatch(b)
	if len(matches) != 2 {
		return -1, fmt.Errorf("failed to find memory capacity in output: %q", string(b))
	}
	m, err := strconv.ParseInt(string(matches[1]), 10, 64)
	if err != nil {
		return -1, err
	}

	// Convert to bytes.
	return m * 1024, err
}

func extractValue(s string, r *regexp.Regexp) (bool, int, error) {
	matches := r.FindSubmatch([]byte(s))
	if len(matches) == 2 {
		val, err := strconv.ParseInt(string(matches[1]), 10, 32)
		if err != nil {
			return true, -1, err
		}
		return true, int(val), nil
	}
	return false, -1, nil
}

func findNode(nodes []info.Node, id int) (bool, int) {
	for i, n := range nodes {
		if n.Id == id {
			return true, i
		}
	}
	return false, -1
}

func addNode(nodes *[]info.Node, id int) (int, error) {
	var idx int
	if id == -1 {
		// Some VMs don't fill topology data. Export single package.
		id = 0
	}

	ok, idx := findNode(*nodes, id)
	if !ok {
		// New node
		node := info.Node{Id: id}
		// Add per-node memory information.
		meminfo := fmt.Sprintf("/sys/devices/system/node/node%d/meminfo", id)
		out, err := ioutil.ReadFile(meminfo)
		// Ignore if per-node info is not available.
		if err == nil {
			m, err := getMemoryCapacity(out)
			if err != nil {
				return -1, err
			}
			node.Memory = uint64(m)
		}
		*nodes = append(*nodes, node)
		idx = len(*nodes) - 1
	}
	return idx, nil
}

func getTopology(sysFs sysfs.SysFs, cpuinfo string) ([]info.Node, int, error) {
	nodes := []info.Node{}
	numCores := 0
	lastThread := -1
	lastCore := -1
	lastNode := -1
	for _, line := range strings.Split(cpuinfo, "\n") {
		ok, val, err := extractValue(line, cpuRegExp)
		if err != nil {
			return nil, -1, fmt.Errorf("could not parse cpu info from %q: %v", line, err)
		}
		if ok {
			thread := val
			numCores++
			if lastThread != -1 {
				// New cpu section. Save last one.
				nodeIdx, err := addNode(&nodes, lastNode)
				if err != nil {
					return nil, -1, fmt.Errorf("failed to add node %d: %v", lastNode, err)
				}
				nodes[nodeIdx].AddThread(lastThread, lastCore)
				lastCore = -1
				lastNode = -1
			}
			lastThread = thread
		}
		ok, val, err = extractValue(line, coreRegExp)
		if err != nil {
			return nil, -1, fmt.Errorf("could not parse core info from %q: %v", line, err)
		}
		if ok {
			lastCore = val
		}
		ok, val, err = extractValue(line, nodeRegExp)
		if err != nil {
			return nil, -1, fmt.Errorf("could not parse node info from %q: %v", line, err)
		}
		if ok {
			lastNode = val
		}
	}
	nodeIdx, err := addNode(&nodes, lastNode)
	if err != nil {
		return nil, -1, fmt.Errorf("failed to add node %d: %v", lastNode, err)
	}
	nodes[nodeIdx].AddThread(lastThread, lastCore)
	if numCores < 1 {
		return nil, numCores, fmt.Errorf("could not detect any cores")
	}
	for idx, node := range nodes {
		caches, err := sysinfo.GetCacheInfo(sysFs, node.Cores[0].Threads[0])
		if err != nil {
			glog.Errorf("failed to get cache information for node %d: %v", node.Id, err)
			continue
		}
		numThreadsPerCore := len(node.Cores[0].Threads)
		numThreadsPerNode := len(node.Cores) * numThreadsPerCore
		for _, cache := range caches {
			c := info.Cache{
				Size:  cache.Size,
				Level: cache.Level,
				Type:  cache.Type,
			}
			if cache.Cpus == numThreadsPerNode && cache.Level > 2 {
				// Add a node-level cache.
				nodes[idx].AddNodeCache(c)
			} else if cache.Cpus == numThreadsPerCore {
				// Add to each core.
				nodes[idx].AddPerCoreCache(c)
			}
			// Ignore unknown caches.
		}
	}
	return nodes, numCores, nil
}

func getInfoFromFiles(filePaths string) string {
	if len(filePaths) == 0 {
		return ""
	}
	for _, file := range strings.Split(filePaths, ",") {
		id, err := ioutil.ReadFile(file)
		if err == nil {
			return strings.TrimSpace(string(id))
		}
	}
	glog.Infof("Couldn't collect info from any of the files in %q", filePaths)
	return ""
}

func getMachineInfo(sysFs sysfs.SysFs, fsInfo fs.FsInfo) (*info.MachineInfo, error) {
	cpuinfo, err := ioutil.ReadFile("/proc/cpuinfo")
	clockSpeed, err := getClockSpeed(cpuinfo)
	if err != nil {
		return nil, err
	}

	// Get the amount of usable memory from /proc/meminfo.
	out, err := ioutil.ReadFile("/proc/meminfo")
	if err != nil {
		return nil, err
	}

	memoryCapacity, err := getMemoryCapacity(out)
	if err != nil {
		return nil, err
	}

	filesystems, err := fsInfo.GetGlobalFsInfo()
	if err != nil {
		glog.Errorf("Failed to get global filesystem information: %v", err)
	}

	diskMap, err := sysinfo.GetBlockDeviceInfo(sysFs)
	if err != nil {
		glog.Errorf("Failed to get disk map: %v", err)
	}

	netDevices, err := sysinfo.GetNetworkDevices(sysFs)
	if err != nil {
		glog.Errorf("Failed to get network devices: %v", err)
	}

	topology, numCores, err := getTopology(sysFs, string(cpuinfo))
	if err != nil {
		glog.Errorf("Failed to get topology information: %v", err)
	}

	systemUUID, err := sysinfo.GetSystemUUID(sysFs)
	if err != nil {
		glog.Errorf("Failed to get system UUID: %v", err)
	}

	machineInfo := &info.MachineInfo{
		NumCores:       numCores,
		CpuFrequency:   clockSpeed,
		MemoryCapacity: memoryCapacity,
		DiskMap:        diskMap,
		NetworkDevices: netDevices,
		Topology:       topology,
		MachineID:      getInfoFromFiles(*machineIdFilePath),
		SystemUUID:     systemUUID,
		BootID:         getInfoFromFiles(*bootIdFilePath),
	}

	for _, fs := range filesystems {
		machineInfo.Filesystems = append(machineInfo.Filesystems, info.FsInfo{Device: fs.Device, Capacity: fs.Capacity})
	}

	return machineInfo, nil
}

func getVersionInfo() (*info.VersionInfo, error) {

	kernel_version := getKernelVersion()
	container_os := getContainerOsVersion()
	docker_version := getDockerVersion()

	return &info.VersionInfo{
		KernelVersion:      kernel_version,
		ContainerOsVersion: container_os,
		DockerVersion:      docker_version,
		CadvisorVersion:    version.VERSION,
	}, nil
}

func getContainerOsVersion() string {
	container_os := "Unknown"
	os_release, err := ioutil.ReadFile("/etc/os-release")
	if err == nil {
		// We might be running in a busybox or some hand-crafted image.
		// It's useful to know why cadvisor didn't come up.
		for _, line := range strings.Split(string(os_release), "\n") {
			parsed := strings.Split(line, "\"")
			if len(parsed) == 3 && parsed[0] == "PRETTY_NAME=" {
				container_os = parsed[1]
				break
			}
		}
	}
	return container_os
}

func getDockerVersion() string {
	docker_version := "Unknown"
	client, err := dclient.NewClient(*docker.ArgDockerEndpoint)
	if err == nil {
		version, err := client.Version()
		if err == nil {
			docker_version = version.Get("Version")
		}
	}
	return docker_version
}

func getKernelVersion() string {
	uname := &syscall.Utsname{}

	if err := syscall.Uname(uname); err != nil {
		return "Unknown"
	}

	release := make([]byte, len(uname.Release))
	i := 0
	for _, c := range uname.Release {
		release[i] = byte(c)
		i++
	}
	release = release[:bytes.IndexByte(release, 0)]

	return string(release)
}

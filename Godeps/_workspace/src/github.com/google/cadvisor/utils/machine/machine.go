// Copyright 2015 Google Inc. All Rights Reserved.
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

package machine

import (
	"fmt"
	"io/ioutil"
	"regexp"
	"strconv"
	"strings"

	"github.com/golang/glog"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/utils"
	"github.com/google/cadvisor/utils/sysfs"
	"github.com/google/cadvisor/utils/sysinfo"
)

// The utils/machine package contains functions that extract machine-level specs.

var cpuRegExp = regexp.MustCompile("processor\\t*: +([0-9]+)")
var coreRegExp = regexp.MustCompile("core id\\t*: +([0-9]+)")
var nodeRegExp = regexp.MustCompile("physical id\\t*: +([0-9]+)")
var CpuClockSpeedMHz = regexp.MustCompile("cpu MHz\\t*: +([0-9]+.[0-9]+)")
var memoryCapacityRegexp = regexp.MustCompile("MemTotal: *([0-9]+) kB")
var swapCapacityRegexp = regexp.MustCompile("SwapTotal: *([0-9]+) kB")

// GetClockSpeed returns the CPU clock speed, given a []byte formatted as the /proc/cpuinfo file.
func GetClockSpeed(procInfo []byte) (uint64, error) {
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

// GetMachineMemoryCapacity returns the machine's total memory from /proc/meminfo.
// Returns the total memory capacity as an int64 (number of bytes).
func GetMachineMemoryCapacity() (int64, error) {
	out, err := ioutil.ReadFile("/proc/meminfo")
	if err != nil {
		return 0, err
	}

	memoryCapacity, err := parseCapacity(out, memoryCapacityRegexp)
	if err != nil {
		return 0, err
	}
	return memoryCapacity, err
}

// GetMachineSwapCapacity returns the machine's total swap from /proc/meminfo.
// Returns the total swap capacity as an int64 (number of bytes).
func GetMachineSwapCapacity() (int64, error) {
	out, err := ioutil.ReadFile("/proc/meminfo")
	if err != nil {
		return 0, err
	}

	swapCapacity, err := parseCapacity(out, swapCapacityRegexp)
	if err != nil {
		return 0, err
	}
	return swapCapacity, err
}

// parseCapacity matches a Regexp in a []byte, returning the resulting value in bytes.
// Assumes that the value matched by the Regexp is in KB.
func parseCapacity(b []byte, r *regexp.Regexp) (int64, error) {
	matches := r.FindSubmatch(b)
	if len(matches) != 2 {
		return -1, fmt.Errorf("failed to match regexp in output: %q", string(b))
	}
	m, err := strconv.ParseInt(string(matches[1]), 10, 64)
	if err != nil {
		return -1, err
	}

	// Convert to bytes.
	return m * 1024, err
}

func GetTopology(sysFs sysfs.SysFs, cpuinfo string) ([]info.Node, int, error) {
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
			m, err := parseCapacity(out, memoryCapacityRegexp)
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

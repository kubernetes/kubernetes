//go:build linux

/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package e2enode

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	libcontainercgroups "github.com/opencontainers/cgroups"
	"k8s.io/utils/cpuset"

	"k8s.io/kubernetes/test/e2e/framework"
)

// IsCgroup2UnifiedMode returns whether we are running in cgroup v2 unified mode.
func IsCgroup2UnifiedMode() bool {
	return libcontainercgroups.IsCgroup2UnifiedMode()
}

func isHTEnabled() bool {
	outData, err := exec.Command("/bin/sh", "-c", "lscpu | grep \"Thread(s) per core:\" | cut -d \":\" -f 2").Output()
	framework.ExpectNoError(err)

	threadsPerCore, err := strconv.Atoi(strings.TrimSpace(string(outData)))
	framework.ExpectNoError(err)

	return threadsPerCore > 1
}

func isMultiNUMA() bool {
	outData, err := exec.Command("/bin/sh", "-c", "lscpu | grep \"NUMA node(s):\" | cut -d \":\" -f 2").Output()
	framework.ExpectNoError(err)

	numaNodes, err := strconv.Atoi(strings.TrimSpace(string(outData)))
	framework.ExpectNoError(err)

	return numaNodes > 1
}

func getUncoreCPUGroupSize() int {
	cpuID := 0 // this is just the most likely cpu to be present in a random system. No special meaning besides this.
	out, err := os.ReadFile(fmt.Sprintf("/sys/devices/system/cpu/cpu%d/cache/index3/shared_cpu_list", cpuID))
	if errors.Is(err, fs.ErrNotExist) {
		return 0 // no Uncore/LLC cache detected, nothing to do
	}
	framework.ExpectNoError(err)
	// how many cores share a same Uncore/LLC block?
	cpus, err := cpuset.Parse(strings.TrimSpace(string(out)))
	framework.ExpectNoError(err)
	return cpus.Size()
}

func getCPUSiblingList(cpuRes int64) string {
	out, err := exec.Command("/bin/sh", "-c", fmt.Sprintf("cat /sys/devices/system/cpu/cpu%d/topology/thread_siblings_list | tr -d \"\n\r\"", cpuRes)).Output()
	framework.ExpectNoError(err)
	return string(out)
}

func getCoreSiblingList(cpuRes int64) string {
	out, err := exec.Command("/bin/sh", "-c", fmt.Sprintf("cat /sys/devices/system/cpu/cpu%d/topology/core_siblings_list | tr -d \"\n\r\"", cpuRes)).Output()
	framework.ExpectNoError(err)
	return string(out)
}

// getNumaNodeCPUs retrieves CPUs for each NUMA node.
func getNumaNodeCPUs() (map[int]cpuset.CPUSet, error) {
	numaNodes := make(map[int]cpuset.CPUSet)
	nodePaths, err := filepath.Glob("/sys/devices/system/node/node*/cpulist")
	if err != nil {
		return nil, err
	}

	for _, nodePath := range nodePaths {
		data, err := os.ReadFile(nodePath)
		framework.ExpectNoError(err, "Error obtaning CPU information from the node")
		cpuSet := strings.TrimSpace(string(data))
		cpus, err := cpuset.Parse(cpuSet)
		framework.ExpectNoError(err, "Error parsing CPUset")

		// Extract node ID from path (e.g., "node0" -> 0)
		base := filepath.Base(filepath.Dir(nodePath))
		nodeID, err := strconv.Atoi(strings.TrimPrefix(base, "node"))
		if err != nil {
			continue
		}
		numaNodes[nodeID] = cpus
	}

	return numaNodes, nil
}

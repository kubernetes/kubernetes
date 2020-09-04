/*
Copyright 2020 The Kubernetes Authors.

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

package alignment

import (
	"fmt"
	"strconv"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"

	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	// NUMAAlignmentCommand emits the output expected by CheckNUMAAlignment
	NUMAAlignmentCommand = `export CPULIST_ALLOWED=$( awk -F":\t*" '/Cpus_allowed_list/ { print $2 }' /proc/self/status); env;`
	// NUMAAlignmentSleepCommand emits the output expected by CheckNUMAAlignment, and sleeps for a long enough time
	NUMAAlignmentSleepCommand = NUMAAlignmentCommand + `sleep 1d;`
)

// ContainerFileReader must return the full content of a file in a given container
type ContainerFileReader func(podName, cntName, path string) string

// CheckNUMAAlignment parses the output form NUMAAlignment*Command run in the given container, and returns if the container has all the resources NUMA-aligned
func CheckNUMAAlignment(pod *v1.Pod, cnt *v1.Container, envLogs string, numaNodes int, sriovResourceName string, cntFileRead ContainerFileReader) (bool, error) {
	var err error
	podEnv, err := makeEnvMap(envLogs)
	if err != nil {
		return false, err
	}

	CPUToNUMANode, err := getCPUToNUMANodeMapFromEnv(cntFileRead, pod, cnt, podEnv, numaNodes)
	if err != nil {
		return false, err
	}

	PCIDevsToNUMANode, err := getPCIDeviceToNumaNodeMapFromEnv(cntFileRead, pod, cnt, podEnv)
	if err != nil {
		return false, err
	}

	if containerWantsDevices(cnt, sriovResourceName) && len(PCIDevsToNUMANode) == 0 {
		return false, fmt.Errorf("no PCI devices found in environ")
	}

	nodeNum := -1 // not set
	for _, cpuNode := range CPUToNUMANode {
		if nodeNum == -1 {
			nodeNum = cpuNode
		} else if nodeNum != cpuNode {
			framework.Logf("misalignment: nodeNum=%d cpuNode=%d", nodeNum, cpuNode)
			return false, nil
		}
	}
	for _, devNode := range PCIDevsToNUMANode {
		if nodeNum != devNode {
			framework.Logf("misalignment: nodeNum=%d devNode=%d", nodeNum, devNode)
			return false, nil
		}
	}
	return true, nil
}

func getCPUToNUMANodeMapFromEnv(cntFileRead ContainerFileReader, pod *v1.Pod, cnt *v1.Container, environ map[string]string, numaNodes int) (map[int]int, error) {
	var cpuIDs []int
	cpuListAllowedEnvVar := "CPULIST_ALLOWED"

	for name, value := range environ {
		if name == cpuListAllowedEnvVar {
			cpus, err := cpuset.Parse(value)
			if err != nil {
				return nil, err
			}
			cpuIDs = cpus.ToSlice()
		}
	}
	if len(cpuIDs) == 0 {
		return nil, fmt.Errorf("variable %q not found in environ", cpuListAllowedEnvVar)
	}

	cpusPerNUMA := make(map[int][]int)
	for numaNode := 0; numaNode < numaNodes; numaNode++ {
		nodeCPUList := cntFileRead(pod.Name, cnt.Name, fmt.Sprintf("/sys/devices/system/node/node%d/cpulist", numaNode))

		cpus, err := cpuset.Parse(nodeCPUList)
		if err != nil {
			return nil, err
		}
		cpusPerNUMA[numaNode] = cpus.ToSlice()
	}

	// CPU IDs -> NUMA Node ID
	CPUToNUMANode := make(map[int]int)
	for nodeNum, cpus := range cpusPerNUMA {
		for _, cpu := range cpus {
			CPUToNUMANode[cpu] = nodeNum
		}
	}

	// filter out only the allowed CPUs
	CPUMap := make(map[int]int)
	for _, cpuID := range cpuIDs {
		_, ok := CPUToNUMANode[cpuID]
		if !ok {
			return nil, fmt.Errorf("CPU %d not found on NUMA map: %v", cpuID, CPUToNUMANode)
		}
		CPUMap[cpuID] = CPUToNUMANode[cpuID]
	}
	return CPUMap, nil
}

func getPCIDeviceToNumaNodeMapFromEnv(cntFileRead ContainerFileReader, pod *v1.Pod, cnt *v1.Container, environ map[string]string) (map[string]int, error) {
	pciDevPrefix := "PCIDEVICE_"
	// at this point we don't care which plugin selected the device,
	// we only need to know which devices were assigned to the POD.
	// Hence, do prefix search for the variable and fetch the device(s).

	NUMAPerDev := make(map[string]int)
	for name, value := range environ {
		if !strings.HasPrefix(name, pciDevPrefix) {
			continue
		}

		// a single plugin can allocate more than a single device
		pciDevs := strings.Split(value, ",")
		for _, pciDev := range pciDevs {
			pciDevNUMANode := cntFileRead(pod.Name, cnt.Name, fmt.Sprintf("/sys/bus/pci/devices/%s/numa_node", pciDev))
			nodeNum, err := strconv.Atoi(strings.TrimSpace(pciDevNUMANode))
			if err != nil {
				return nil, err
			}
			NUMAPerDev[pciDev] = nodeNum
		}
	}
	return NUMAPerDev, nil
}

func makeEnvMap(logs string) (map[string]string, error) {
	podEnv := strings.Split(logs, "\n")
	envMap := make(map[string]string)
	for _, envVar := range podEnv {
		if len(envVar) == 0 {
			continue
		}
		pair := strings.SplitN(envVar, "=", 2)
		if len(pair) != 2 {
			return nil, fmt.Errorf("unable to split %q", envVar)
		}
		envMap[pair[0]] = pair[1]
	}
	return envMap, nil
}

func containerWantsDevices(cnt *v1.Container, sriovResourceName string) bool {
	if sriovResourceName == "" {
		return false
	}
	_, found := cnt.Resources.Requests[v1.ResourceName(sriovResourceName)]
	return found
}

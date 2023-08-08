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

package e2enode

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/utils/cpuset"

	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
)

type numaPodResources struct {
	CPUToNUMANode     map[int]int
	PCIDevsToNUMANode map[string]int
}

func (R *numaPodResources) CheckAlignment() bool {
	nodeNum := -1 // not set
	for _, cpuNode := range R.CPUToNUMANode {
		if nodeNum == -1 {
			nodeNum = cpuNode
		} else if nodeNum != cpuNode {
			return false
		}
	}
	for _, devNode := range R.PCIDevsToNUMANode {
		if nodeNum != devNode {
			return false
		}
	}
	return true
}

func (R *numaPodResources) String() string {
	var b strings.Builder
	// To store the keys in slice in sorted order
	var cpuKeys []int
	for ck := range R.CPUToNUMANode {
		cpuKeys = append(cpuKeys, ck)
	}
	sort.Ints(cpuKeys)
	for _, k := range cpuKeys {
		nodeNum := R.CPUToNUMANode[k]
		b.WriteString(fmt.Sprintf("CPU cpu#%03d=%02d\n", k, nodeNum))
	}
	var pciKeys []string
	for pk := range R.PCIDevsToNUMANode {
		pciKeys = append(pciKeys, pk)
	}
	sort.Strings(pciKeys)
	for _, k := range pciKeys {
		nodeNum := R.PCIDevsToNUMANode[k]
		b.WriteString(fmt.Sprintf("PCI %s=%02d\n", k, nodeNum))
	}
	return b.String()
}

func getCPUsPerNUMANode(nodeNum int) ([]int, error) {
	nodeCPUList, err := os.ReadFile(fmt.Sprintf("/sys/devices/system/node/node%d/cpulist", nodeNum))
	if err != nil {
		return nil, err
	}
	cpus, err := cpuset.Parse(strings.TrimSpace(string(nodeCPUList)))
	if err != nil {
		return nil, err
	}
	return cpus.List(), nil
}

func getCPUToNUMANodeMapFromEnv(f *framework.Framework, pod *v1.Pod, cnt *v1.Container, environ map[string]string, numaNodes int) (map[int]int, error) {
	var cpuIDs []int
	cpuListAllowedEnvVar := "CPULIST_ALLOWED"

	for name, value := range environ {
		if name == cpuListAllowedEnvVar {
			cpus, err := cpuset.Parse(value)
			if err != nil {
				return nil, err
			}
			cpuIDs = cpus.List()
		}
	}
	if len(cpuIDs) == 0 {
		return nil, fmt.Errorf("variable %q not found in environ", cpuListAllowedEnvVar)
	}

	cpusPerNUMA := make(map[int][]int)
	for numaNode := 0; numaNode < numaNodes; numaNode++ {
		nodeCPUList := e2epod.ExecCommandInContainer(f, pod.Name, cnt.Name,
			"/bin/cat", fmt.Sprintf("/sys/devices/system/node/node%d/cpulist", numaNode))

		cpus, err := cpuset.Parse(nodeCPUList)
		if err != nil {
			return nil, err
		}
		cpusPerNUMA[numaNode] = cpus.List()
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

func getPCIDeviceToNumaNodeMapFromEnv(f *framework.Framework, pod *v1.Pod, cnt *v1.Container, environ map[string]string) (map[string]int, error) {
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
			pciDevNUMANode := e2epod.ExecCommandInContainer(f, pod.Name, cnt.Name,
				"/bin/cat", fmt.Sprintf("/sys/bus/pci/devices/%s/numa_node", pciDev))
			NUMAPerDev[pciDev] = numaNodeFromSysFsEntry(pciDevNUMANode)
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

type testEnvInfo struct {
	numaNodes         int
	sriovResourceName string
	policy            string
	scope             string
}

func containerWantsDevices(cnt *v1.Container, envInfo *testEnvInfo) bool {
	_, found := cnt.Resources.Requests[v1.ResourceName(envInfo.sriovResourceName)]
	return found
}

func checkNUMAAlignment(f *framework.Framework, pod *v1.Pod, cnt *v1.Container, logs string, envInfo *testEnvInfo) (*numaPodResources, error) {
	var err error
	podEnv, err := makeEnvMap(logs)
	if err != nil {
		return nil, err
	}

	CPUToNUMANode, err := getCPUToNUMANodeMapFromEnv(f, pod, cnt, podEnv, envInfo.numaNodes)
	if err != nil {
		return nil, err
	}

	PCIDevsToNUMANode, err := getPCIDeviceToNumaNodeMapFromEnv(f, pod, cnt, podEnv)
	if err != nil {
		return nil, err
	}

	if containerWantsDevices(cnt, envInfo) && len(PCIDevsToNUMANode) == 0 {
		return nil, fmt.Errorf("no PCI devices found in environ")
	}
	numaRes := numaPodResources{
		CPUToNUMANode:     CPUToNUMANode,
		PCIDevsToNUMANode: PCIDevsToNUMANode,
	}
	aligned := numaRes.CheckAlignment()
	if !aligned {
		err = fmt.Errorf("NUMA resources not aligned")
	}
	return &numaRes, err
}

type pciDeviceInfo struct {
	Address  string
	NUMANode int
	IsPhysFn bool
	IsVFn    bool
}

func getPCIDeviceInfo(sysPCIDir string) ([]pciDeviceInfo, error) {
	var pciDevs []pciDeviceInfo

	entries, err := os.ReadDir(sysPCIDir)
	if err != nil {
		return nil, err
	}

	for _, entry := range entries {
		isPhysFn := false
		isVFn := false
		if _, err := os.Stat(filepath.Join(sysPCIDir, entry.Name(), "sriov_numvfs")); err == nil {
			isPhysFn = true
		} else if !os.IsNotExist(err) {
			// unexpected error. Bail out
			return nil, err
		}
		if _, err := os.Stat(filepath.Join(sysPCIDir, entry.Name(), "physfn")); err == nil {
			isVFn = true
		} else if !os.IsNotExist(err) {
			// unexpected error. Bail out
			return nil, err
		}

		content, err := os.ReadFile(filepath.Join(sysPCIDir, entry.Name(), "numa_node"))
		if err != nil {
			return nil, err
		}

		pciDevs = append(pciDevs, pciDeviceInfo{
			Address:  entry.Name(),
			NUMANode: numaNodeFromSysFsEntry(string(content)),
			IsPhysFn: isPhysFn,
			IsVFn:    isVFn,
		})
	}

	return pciDevs, nil
}

func numaNodeFromSysFsEntry(content string) int {
	nodeNum, err := strconv.Atoi(strings.TrimSpace(content))
	framework.ExpectNoError(err, "error detecting the device numa_node from sysfs: %v", err)
	return nodeNum
}

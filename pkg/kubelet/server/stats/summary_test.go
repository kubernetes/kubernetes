/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package stats

import (
	"fmt"
	"testing"

	"github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/info/v2"
	"github.com/stretchr/testify/assert"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/leaky"
)

const (
	// Offsets from seed value in generated container stats.
	offsetCPUUsageCores = iota
	offsetCPUUsageCoreSeconds
	offsetMemPageFaults
	offsetMemMajorPageFaults
	offsetMemUsageBytes
	offsetMemWorkingSetBytes
	offsetNetRxBytes
	offsetNetRxErrors
	offsetNetTxBytes
	offsetNetTxErrors

	namespace = "test"
)

func TestBuildSummary(t *testing.T) {
	node := api.Node{}
	node.Name = "FooNode"
	nodeConfig := cm.NodeConfig{
		DockerDaemonContainerName: "/docker-daemon",
		SystemContainerName:       "/system",
		KubeletContainerName:      "/kubelet",
	}
	pods := []*api.Pod{
		summaryTestPod("pod"),
	}

	const (
		seedRoot         = 0
		seedRuntime      = 100
		seedKubelet      = 200
		seedMisc         = 300
		seedPodInfra     = 1000
		seedPodContainer = 2000
	)
	infos := map[string]v2.ContainerInfo{
		"/":              summaryTestContainerInfo(seedRoot),
		"/docker-daemon": summaryTestContainerInfo(seedRuntime),
		"/kubelet":       summaryTestContainerInfo(seedKubelet),
		"/system":        summaryTestContainerInfo(seedMisc),
		"/pod-0":         summaryTestContainerInfo(seedPodInfra),
		"/pod-1":         summaryTestContainerInfo(seedPodContainer),
	}

	summary, err := buildSummary(&node, nodeConfig, pods, infos)
	assert.NoError(t, err)
	nodeStats := summary.Node
	assert.Equal(t, "FooNode", nodeStats.NodeName)
	checkCPUStats(t, "Node", seedRoot, nodeStats.CPU)
	checkMemoryStats(t, "Node", seedRoot, nodeStats.Memory)
	checkNetworkStats(t, "Node", seedRoot, nodeStats.Network)

	systemSeeds := map[string]int{
		SystemContainerRuntime: seedRuntime,
		SystemContainerKubelet: seedKubelet,
		SystemContainerMisc:    seedMisc,
	}
	for _, sys := range nodeStats.SystemContainers {
		name := sys.Name
		seed, found := systemSeeds[name]
		if !found {
			t.Errorf("Unknown SystemContainer: %q", name)
		}
		checkCPUStats(t, name, seed, sys.CPU)
		checkMemoryStats(t, name, seed, sys.Memory)
	}

	assert.Equal(t, 1, len(summary.Pods))
	pod := summary.Pods[0]
	assert.Equal(t, "pod", pod.PodRef.Name)
	assert.Equal(t, namespace, pod.PodRef.Namespace)
	checkNetworkStats(t, "Pod", seedPodInfra, pod.Network)

	assert.Equal(t, 1, len(pod.Containers))
	container := pod.Containers[0]
	assert.Equal(t, "pod-container-1", container.Name)
	checkCPUStats(t, "container", seedPodContainer, container.CPU)
	checkMemoryStats(t, "container", seedPodContainer, container.Memory)
}

func summaryTestPod(podName string) *api.Pod {
	pod := api.Pod{}
	pod.Name = podName
	pod.Namespace = namespace

	const numContainers = 2
	containerStatuses := make([]api.ContainerStatus, numContainers)
	for i := 0; i < numContainers; i++ {
		cID := kubecontainer.ContainerID{"test", fmt.Sprintf("%s-%d", podName, i)}
		containerStatuses[i] = api.ContainerStatus{
			Name:        fmt.Sprintf("%s-container-%d", podName, i),
			ContainerID: cID.String(),
		}
	}
	containerStatuses[0].Name = leaky.PodInfraContainerName
	pod.Status.ContainerStatuses = containerStatuses
	return &pod
}

func summaryTestContainerInfo(seed int) v2.ContainerInfo {
	stats := v2.ContainerStats{
		Cpu:     &v1.CpuStats{},
		CpuInst: &v2.CpuInstStats{},
		Memory: &v1.MemoryStats{
			Usage:      uint64(seed + offsetMemUsageBytes),
			WorkingSet: uint64(seed + offsetMemWorkingSetBytes),
			ContainerData: v1.MemoryStatsMemoryData{
				Pgfault:    uint64(seed + offsetMemPageFaults),
				Pgmajfault: uint64(seed + offsetMemMajorPageFaults),
			},
		},
		Network: &v2.NetworkStats{
			Interfaces: []v1.InterfaceStats{{
				RxBytes:  uint64(seed + offsetNetRxBytes),
				RxErrors: uint64(seed + offsetNetRxErrors),
				TxBytes:  uint64(seed + offsetNetTxBytes),
				TxErrors: uint64(seed + offsetNetTxErrors),
			}},
		},
	}
	stats.Cpu.Usage.Total = uint64(seed + offsetCPUUsageCoreSeconds)
	stats.CpuInst.Usage.Total = uint64(seed + offsetCPUUsageCores)

	return v2.ContainerInfo{
		Spec: v2.ContainerSpec{
			HasCpu:     true,
			HasMemory:  true,
			HasNetwork: true,
		},
		Stats: []*v2.ContainerStats{&stats},
	}
}

func checkNetworkStats(t *testing.T, label string, seed int, stats *NetworkStats) {
	assert.EqualValues(t, seed+offsetNetRxBytes, stats.RxBytes.Value(), label+".Net.RxBytes")
	assert.EqualValues(t, seed+offsetNetRxErrors, *stats.RxErrors, label+".Net.RxErrors")
	assert.EqualValues(t, seed+offsetNetTxBytes, stats.TxBytes.Value(), label+".Net.TxBytes")
	assert.EqualValues(t, seed+offsetNetTxErrors, *stats.TxErrors, label+".Net.TxErrors")
}

func checkCPUStats(t *testing.T, label string, seed int, stats *CPUStats) {
	assert.EqualValues(t, seed+offsetCPUUsageCores, stats.UsageCores.ScaledValue(resource.Nano), label+".CPU.UsageCores")
	assert.EqualValues(t, seed+offsetCPUUsageCoreSeconds, stats.UsageCoreSeconds.ScaledValue(resource.Nano), label+".CPU.UsageCoreSeconds")
}

func checkMemoryStats(t *testing.T, label string, seed int, stats *MemoryStats) {
	assert.EqualValues(t, seed+offsetMemUsageBytes, stats.UsageBytes.Value(), label+".Mem.UsageBytes")
	assert.EqualValues(t, seed+offsetMemWorkingSetBytes, stats.WorkingSetBytes.Value(), label+".Mem.WorkingSetBytes")
	assert.EqualValues(t, seed+offsetMemPageFaults, *stats.PageFaults, label+".Mem.PageFaults")
	assert.EqualValues(t, seed+offsetMemMajorPageFaults, *stats.MajorPageFaults, label+".Mem.MajorPageFaults")
}

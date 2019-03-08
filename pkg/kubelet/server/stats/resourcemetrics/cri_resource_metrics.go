/*
Copyright 2019 The Kubernetes Authors.

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
package resourcemetrics

import (
	"fmt"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	internalapi "k8s.io/kubernetes/pkg/kubelet/apis/cri"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/runtime/v1alpha2"
	summary "k8s.io/kubernetes/pkg/kubelet/apis/stats/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/pkg/kubelet/leaky"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

type criCoreMetricsProvider struct {
	cm             cm.ContainerManager
	runtimeService internalapi.RuntimeService
}

func NewCRIMetricsProvider(cm cm.ContainerManager, rs internalapi.RuntimeService) ResourceMetricsProvider {
	return &criCoreMetricsProvider{cm, rs}
}

func (cp *criCoreMetricsProvider) GetMetrics() (*summary.Summary, error) {
	resp, err := cp.runtimeService.ListContainerStats(&runtimeapi.ContainerStatsFilter{})
	if err != nil {
		return nil, fmt.Errorf("failed to list all container stats: %v", err)
	}

	podToStats := map[summary.PodReference]*summary.PodStats{}
	for _, stats := range resp {
		containerName := kubetypes.GetContainerName(stats.Attributes.Labels)
		if containerName == leaky.PodInfraContainerName || containerName == "" {
			//jump infrastructure container
			continue
		}
		cpuStats := &summary.CPUStats{
			UsageCoreNanoSeconds: uint64Ptr(stats.Cpu.UsageCoreNanoSeconds.Value),
			Time:                 metav1.NewTime(time.Unix(0, stats.Cpu.Timestamp)),
		}
		memoryStats := &summary.MemoryStats{
			WorkingSetBytes: uint64Ptr(stats.Memory.WorkingSetBytes.Value),
			Time:            metav1.NewTime(time.Unix(0, stats.Memory.Timestamp)),
		}

		ref := buildPodRef(stats.Attributes.Labels)
		podStats, found := podToStats[ref]
		if !found {
			podStats = &summary.PodStats{PodRef: ref}
			podToStats[ref] = podStats
		}

		podStats.Containers = append(podStats.Containers, summary.ContainerStats{
			// TODO: which timestamp should we use?
			StartTime: metav1.NewTime(time.Unix(0, stats.Cpu.Timestamp)),
			Name:      containerName,
			CPU:       cpuStats,
			Memory:    memoryStats,
		})
	}

	// Extract node info from root cgroup
	cpu, memory, time, err := extractContainerInfo(cm.CgroupName([]string{}), cp.cm)
	if err != nil {
		return nil, err
	}
	nodeStats := summary.NodeStats{
		CPU:       cpu,
		Memory:    memory,
		StartTime: metav1.NewTime(time),
	}

	p := []summary.PodStats{}
	for _, ps := range podToStats {
		p = append(p, *ps)
	}
	return &summary.Summary{Pods: p, Node: nodeStats}, nil
}

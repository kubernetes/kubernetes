// +build windows

/*
Copyright 2018 The Kubernetes Authors.

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
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	statsapi "k8s.io/kubernetes/pkg/kubelet/apis/stats/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/cm"
)

func (sp *summaryProviderImpl) GetSystemContainersStats(nodeConfig cm.NodeConfig, podStats []statsapi.PodStats, updateStats bool) (stats []statsapi.ContainerStats) {
	stats = append(stats, sp.getSystemPodsStats(nodeConfig, podStats, updateStats))
	return stats
}

func (sp *summaryProviderImpl) getSystemPodsStats(nodeConfig cm.NodeConfig, podStats []statsapi.PodStats, updateStats bool) statsapi.ContainerStats {
	now := metav1.NewTime(time.Now())
	podsSummary := statsapi.ContainerStats{
		StartTime: now,
		CPU:       &statsapi.CPUStats{},
		Memory:    &statsapi.MemoryStats{},
		Name:      statsapi.SystemContainerPods,
	}

	// Sum up all pod's stats.
	var usageCoreNanoSeconds uint64
	var usageNanoCores uint64
	var availableBytes uint64
	var usageBytes uint64
	var workingSetBytes uint64
	for _, pod := range podStats {
		if pod.CPU != nil {
			podsSummary.CPU.Time = now
			if pod.CPU.UsageCoreNanoSeconds != nil {
				usageCoreNanoSeconds = usageCoreNanoSeconds + *pod.CPU.UsageCoreNanoSeconds
			}
			if pod.CPU.UsageNanoCores != nil {
				usageNanoCores = usageNanoCores + *pod.CPU.UsageNanoCores
			}
		}

		if pod.Memory != nil {
			podsSummary.Memory.Time = now
			if pod.Memory.AvailableBytes != nil {
				availableBytes = availableBytes + *pod.Memory.AvailableBytes
			}
			if pod.Memory.UsageBytes != nil {
				usageBytes = usageBytes + *pod.Memory.UsageBytes
			}
			if pod.Memory.WorkingSetBytes != nil {
				workingSetBytes = workingSetBytes + *pod.Memory.WorkingSetBytes
			}
		}
	}

	// Set results only if they are not zero.
	if usageCoreNanoSeconds != 0 {
		podsSummary.CPU.UsageCoreNanoSeconds = &usageCoreNanoSeconds
	}
	if usageNanoCores != 0 {
		podsSummary.CPU.UsageNanoCores = &usageNanoCores
	}
	if availableBytes != 0 {
		podsSummary.Memory.AvailableBytes = &availableBytes
	}
	if usageBytes != 0 {
		podsSummary.Memory.UsageBytes = &usageBytes
	}
	if workingSetBytes != 0 {
		podsSummary.Memory.WorkingSetBytes = &workingSetBytes
	}

	return podsSummary
}

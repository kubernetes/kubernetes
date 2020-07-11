// +build !windows

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
	"k8s.io/klog/v2"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/cm"
)

func (sp *summaryProviderImpl) GetSystemContainersStats(nodeConfig cm.NodeConfig, podStats []statsapi.PodStats, updateStats bool) (stats []statsapi.ContainerStats) {
	systemContainers := map[string]struct {
		name             string
		forceStatsUpdate bool
		startTime        metav1.Time
	}{
		statsapi.SystemContainerKubelet: {name: nodeConfig.KubeletCgroupsName, forceStatsUpdate: false, startTime: sp.kubeletCreationTime},
		statsapi.SystemContainerRuntime: {name: nodeConfig.RuntimeCgroupsName, forceStatsUpdate: false},
		statsapi.SystemContainerMisc:    {name: nodeConfig.SystemCgroupsName, forceStatsUpdate: false},
		statsapi.SystemContainerPods:    {name: sp.provider.GetPodCgroupRoot(), forceStatsUpdate: updateStats},
	}
	for sys, cont := range systemContainers {
		// skip if cgroup name is undefined (not all system containers are required)
		if cont.name == "" {
			continue
		}
		s, _, err := sp.provider.GetCgroupStats(cont.name, cont.forceStatsUpdate)
		if err != nil {
			klog.Errorf("Failed to get system container stats for %q: %v", cont.name, err)
			continue
		}
		// System containers don't have a filesystem associated with them.
		s.Logs, s.Rootfs = nil, nil
		s.Name = sys

		// if we know the start time of a system container, use that instead of the start time provided by cAdvisor
		if !cont.startTime.IsZero() {
			s.StartTime = cont.startTime
		}
		stats = append(stats, *s)
	}

	return stats
}

func (sp *summaryProviderImpl) GetSystemContainersCPUAndMemoryStats(nodeConfig cm.NodeConfig, podStats []statsapi.PodStats, updateStats bool) (stats []statsapi.ContainerStats) {
	systemContainers := map[string]struct {
		name             string
		forceStatsUpdate bool
		startTime        metav1.Time
	}{
		statsapi.SystemContainerKubelet: {name: nodeConfig.KubeletCgroupsName, forceStatsUpdate: false, startTime: sp.kubeletCreationTime},
		statsapi.SystemContainerRuntime: {name: nodeConfig.RuntimeCgroupsName, forceStatsUpdate: false},
		statsapi.SystemContainerMisc:    {name: nodeConfig.SystemCgroupsName, forceStatsUpdate: false},
		statsapi.SystemContainerPods:    {name: sp.provider.GetPodCgroupRoot(), forceStatsUpdate: updateStats},
	}
	for sys, cont := range systemContainers {
		// skip if cgroup name is undefined (not all system containers are required)
		if cont.name == "" {
			continue
		}
		s, err := sp.provider.GetCgroupCPUAndMemoryStats(cont.name, cont.forceStatsUpdate)
		if err != nil {
			klog.Errorf("Failed to get system container stats for %q: %v", cont.name, err)
			continue
		}
		s.Name = sys

		// if we know the start time of a system container, use that instead of the start time provided by cAdvisor
		if !cont.startTime.IsZero() {
			s.StartTime = cont.startTime
		}
		stats = append(stats, *s)
	}

	return stats
}

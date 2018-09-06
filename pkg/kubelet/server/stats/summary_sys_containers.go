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
	"github.com/golang/glog"

	statsapi "k8s.io/kubernetes/pkg/kubelet/apis/stats/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/cm"
)

func (sp *summaryProviderImpl) GetSystemContainersStats(nodeConfig cm.NodeConfig, podStats []statsapi.PodStats, updateStats bool) (stats []statsapi.ContainerStats) {
	systemContainers := map[string]struct {
		name             string
		forceStatsUpdate bool
	}{
		statsapi.SystemContainerKubelet: {nodeConfig.KubeletCgroupsName, false},
		statsapi.SystemContainerRuntime: {nodeConfig.RuntimeCgroupsName, false},
		statsapi.SystemContainerMisc:    {nodeConfig.SystemCgroupsName, false},
		statsapi.SystemContainerPods:    {sp.provider.GetPodCgroupRoot(), updateStats},
	}
	for sys, cont := range systemContainers {
		// skip if cgroup name is undefined (not all system containers are required)
		if cont.name == "" {
			continue
		}
		s, _, err := sp.provider.GetCgroupStats(cont.name, cont.forceStatsUpdate)
		if err != nil {
			glog.Errorf("Failed to get system container stats for %q: %v", cont.name, err)
			continue
		}
		// System containers don't have a filesystem associated with them.
		s.Logs, s.Rootfs = nil, nil
		s.Name = sys
		stats = append(stats, *s)
	}

	return stats
}

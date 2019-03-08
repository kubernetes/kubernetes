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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	internalapi "k8s.io/kubernetes/pkg/kubelet/apis/cri"
	summary "k8s.io/kubernetes/pkg/kubelet/apis/stats/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/pkg/kubelet/leaky"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

type ResourceMetricsProvider interface {
	GetMetrics() (*summary.Summary, error)
}

type coreMetricsProvider struct {
	cm             cm.ContainerManager
	runtimeService internalapi.RuntimeService
}

func NewCoreMetrics(cm cm.ContainerManager, rs internalapi.RuntimeService) ResourceMetricsProvider {
	return &coreMetricsProvider{cm, rs}
}

func (self *coreMetricsProvider) GetMetrics() (*summary.Summary, error) {
	containerCgroups, err := self.cm.NewPodContainerManager().GetAllPodsContainersFromCgroups()
	if err != nil {
		return nil, err
	}

	podToStats := map[summary.PodReference]*summary.PodStats{}
	for _, cg := range containerCgroups {
		id := containerNameToDockerId(cg)
		status, err := self.runtimeService.ContainerStatus(id)
		if err != nil {
			continue
		}
		if !isPodManagedContainer(status.Labels) {
			continue
		}
		containerName := kubetypes.GetContainerName(status.Labels)
		if containerName == leaky.PodInfraContainerName || containerName == "" {
			//jump infrastructure container
			continue
		}
		cpu, memory, time, err := extractContainerInfo(cg, self.cm)
		if err != nil {
			return nil, err
		}
		ref := buildPodRef(status.Labels)
		podStats, found := podToStats[ref]
		if !found {
			podStats = &summary.PodStats{PodRef: ref}
			podToStats[ref] = podStats
		}

		podStats.Containers = append(podStats.Containers, summary.ContainerStats{
			StartTime: metav1.NewTime(time),
			Name:      containerName,
			CPU:       cpu,
			Memory:    memory,
		})
	}

	// Extract node info from root cgroup
	cpu, memory, time, err := extractContainerInfo(cm.CgroupName([]string{}), self.cm)
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

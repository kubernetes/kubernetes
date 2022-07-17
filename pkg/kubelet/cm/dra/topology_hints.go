/*
Copyright 2022 The Kubernetes Authors.

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

package dra

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
)

// GetTopologyHints implements the TopologyManager HintProvider Interface which
// ensures the DRA Manager is consulted when Topology Aware Hints for each
// container are created.
func (m *ManagerImpl) GetTopologyHints(pod *v1.Pod, container *v1.Container) map[string][]topologymanager.TopologyHint {
	// The pod is during the admission phase. We need to save the pod to avoid it
	// being cleaned before the admission ended
	m.setPodPendingAdmission(pod)

	// Loop through all device resources and generate TopologyHints for them..
	deviceHints := make(map[string][]topologymanager.TopologyHint)
	for resourceObj, _ := range container.Resources.Limits {
		deviceHints[string(resourceObj)] = nil // resource doesn't have a topology preference
	}

	return deviceHints
}

// GetPodTopologyHints implements the topologymanager.HintProvider Interface which
// ensures the Device Manager is consulted when Topology Aware Hints for Pod are created.
func (m *ManagerImpl) GetPodTopologyHints(pod *v1.Pod) map[string][]topologymanager.TopologyHint {
	// The pod is during the admission phase. We need to save the pod to avoid it
	// being cleaned before the admission ended
	m.setPodPendingAdmission(pod)

	deviceHints := make(map[string][]topologymanager.TopologyHint)
	for _, container := range append(pod.Spec.InitContainers, pod.Spec.Containers...) {
		for resourceObj, _ := range container.Resources.Limits {
			deviceHints[string(resourceObj)] = nil // resource doesn't have a topology preference
		}
	}

	return deviceHints
}

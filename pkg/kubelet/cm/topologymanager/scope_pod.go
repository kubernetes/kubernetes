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

package topologymanager

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/cm/admission"
	"k8s.io/kubernetes/pkg/kubelet/cm/containermap"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
)

type podScope struct {
	scope
}

// Ensure podScope implements Scope interface
var _ Scope = &podScope{}

// NewPodScope returns a pod scope.
func NewPodScope(policy Policy, recorder record.EventRecorder) Scope {
	return &podScope{
		scope{
			name:             podTopologyScope,
			recorder:         recorder,
			podTopologyHints: podTopologyHints{},
			policy:           policy,
			podMap:           containermap.NewContainerMap(),
		},
	}
}

func (s *podScope) Admit(pod *v1.Pod) lifecycle.PodAdmitResult {
	bestHint, admit := s.calculateAffinity(pod)
	klog.InfoS("Best TopologyHint", "bestHint", bestHint, "pod", klog.KObj(pod))
	if !admit {
		metrics.TopologyManagerAdmissionErrorsTotal.Inc()
		return admission.GetPodAdmitResult(&TopologyAffinityError{
			Hint: bestHint.String(),
		})
	}

	allocs := make(allocationMap)
	for _, container := range append(pod.Spec.InitContainers, pod.Spec.Containers...) {
		klog.InfoS("Topology Affinity", "bestHint", bestHint, "pod", klog.KObj(pod), "containerName", container.Name)
		s.setTopologyHints(string(pod.UID), container.Name, bestHint)

		resources, err := s.allocateAlignedResources(pod, &container)
		if err != nil {
			metrics.TopologyManagerAdmissionErrorsTotal.Inc()
			return admission.GetPodAdmitResult(err)
		}
		allocs.Add(container.Name, resources)
	}
	s.resourceAllocationSuccessEvent(pod, allocs)
	return admission.GetPodAdmitResult(nil)
}

func (s *podScope) accumulateProvidersHints(pod *v1.Pod) []map[string][]TopologyHint {
	var providersHints []map[string][]TopologyHint

	for _, provider := range s.providers {
		// Get the TopologyHints for a Pod from a provider.
		hints := provider.GetPodTopologyHints(pod)
		providersHints = append(providersHints, hints)
		klog.InfoS("TopologyHints", "hints", hints, "pod", klog.KObj(pod))
	}
	return providersHints
}

func (s *podScope) calculateAffinity(pod *v1.Pod) (TopologyHint, bool) {
	providersHints := s.accumulateProvidersHints(pod)
	bestHint, admit := s.policy.Merge(providersHints)
	klog.InfoS("PodTopologyHint", "bestHint", bestHint)
	return bestHint, admit
}

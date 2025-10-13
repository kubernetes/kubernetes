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
	"context"

	"k8s.io/api/core/v1"
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
func NewPodScope(policy Policy) Scope {
	return &podScope{
		scope{
			name:             podTopologyScope,
			podTopologyHints: podTopologyHints{},
			policy:           policy,
			podMap:           containermap.NewContainerMap(),
		},
	}
}

func (s *podScope) Admit(ctx context.Context, pod *v1.Pod) lifecycle.PodAdmitResult {
	logger := klog.FromContext(ctx)

	bestHint, admit := s.calculateAffinity(ctx, pod)

	logger.Info("Best TopologyHint", "bestHint", bestHint, "pod", klog.KObj(pod))
	if !admit {
		if IsAlignmentGuaranteed(s.policy) {
			// increment only if we know we allocate aligned resources.
			metrics.ContainerAlignedComputeResourcesFailure.WithLabelValues(metrics.AlignScopePod, metrics.AlignedNUMANode).Inc()
		}
		metrics.TopologyManagerAdmissionErrorsTotal.Inc()
		return admission.GetPodAdmitResult(&TopologyAffinityError{})
	}

	for _, container := range append(pod.Spec.InitContainers, pod.Spec.Containers...) {
		logger.Info("Topology Affinity", "bestHint", bestHint, "pod", klog.KObj(pod), "containerName", container.Name)
		s.setTopologyHints(string(pod.UID), container.Name, bestHint)

		err := s.allocateAlignedResources(pod, &container)
		if err != nil {
			metrics.TopologyManagerAdmissionErrorsTotal.Inc()
			return admission.GetPodAdmitResult(err)
		}
	}
	if IsAlignmentGuaranteed(s.policy) {
		// increment only if we know we allocate aligned resources.
		logger.V(4).Info("Resource alignment at pod scope guaranteed", "pod", klog.KObj(pod))
		metrics.ContainerAlignedComputeResources.WithLabelValues(metrics.AlignScopePod, metrics.AlignedNUMANode).Inc()
	}
	return admission.GetPodAdmitResult(nil)
}

func (s *podScope) accumulateProvidersHints(ctx context.Context, pod *v1.Pod) []map[string][]TopologyHint {
	logger := klog.FromContext(ctx)

	var providersHints []map[string][]TopologyHint

	for _, provider := range s.hintProviders {
		// Get the TopologyHints for a Pod from a provider.
		hints := provider.GetPodTopologyHints(pod)
		providersHints = append(providersHints, hints)
		logger.Info("TopologyHints", "hints", hints, "pod", klog.KObj(pod))
	}
	return providersHints
}

func (s *podScope) calculateAffinity(ctx context.Context, pod *v1.Pod) (TopologyHint, bool) {
	logger := klog.FromContext(ctx)
	providersHints := s.accumulateProvidersHints(ctx, pod)
	bestHint, admit := s.policy.Merge(ctx, providersHints)
	logger.Info("PodTopologyHint", "bestHint", bestHint, "pod", klog.KObj(pod))
	return bestHint, admit
}

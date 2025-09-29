/*
Copyright 2016 The Kubernetes Authors.

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

package kubelet

import (
	"fmt"

	"k8s.io/klog/v2"

	corev1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	resourcehelper "k8s.io/component-helpers/resource"
	"k8s.io/kubernetes/pkg/api/v1/resource"
	kubefeatures "k8s.io/kubernetes/pkg/features"
)

// defaultPodLimitsForDownwardAPI copies the input pod, and optional container,
// and applies default resource limits. it returns a copy of the input pod,
// and a copy of the input container (if specified) with default limits
// applied.
// If a container has no limits specified, it defaults to the pod-level resources.
// If neither container-level nor pod-level resources limits are specified, it defaults
// to the node's allocatable resources.
func (kl *Kubelet) defaultPodLimitsForDownwardAPI(pod *corev1.Pod, container *corev1.Container) (*corev1.Pod, *corev1.Container, error) {
	if pod == nil {
		return nil, nil, fmt.Errorf("invalid input, pod cannot be nil")
	}

	node, err := kl.getNodeAnyWay()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to find node object, expected a node")
	}
	allocatable := node.Status.Allocatable
	if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.PodLevelResources) && resourcehelper.IsPodLevelLimitsSet(pod) {
		allocatable = allocatable.DeepCopy()
		// Resources supported by the Downward API
		for _, resource := range []corev1.ResourceName{corev1.ResourceCPU, corev1.ResourceMemory, corev1.ResourceEphemeralStorage} {
			// Skip resources not supported by Pod Level Resources
			if !resourcehelper.IsSupportedPodLevelResource(resource) {
				continue
			}
			if val, exists := pod.Spec.Resources.Limits[resource]; exists && !val.IsZero() {
				if _, exists := allocatable[resource]; exists {
					allocatable[resource] = val.DeepCopy()
				}
			}
		}
	}

	klog.InfoS("Allocatable", "allocatable", allocatable)
	outputPod := pod.DeepCopy()
	for idx := range outputPod.Spec.Containers {
		resource.MergeContainerResourceLimits(&outputPod.Spec.Containers[idx], allocatable)
	}

	var outputContainer *corev1.Container
	if container != nil {
		outputContainer = container.DeepCopy()
		resource.MergeContainerResourceLimits(outputContainer, allocatable)
	}
	return outputPod, outputContainer, nil
}

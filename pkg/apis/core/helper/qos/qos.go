/*
Copyright 2017 The Kubernetes Authors.

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

// NOTE: DO NOT use those helper functions through client-go, the
// package path will be changed in the future.
package qos

import (
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

var supportedQoSComputeResources = sets.New(core.ResourceCPU, core.ResourceMemory)

// GetPodQOS returns the QoS class of a pod persisted in the PodStatus.QOSClass field.
// If PodStatus.QOSClass is empty, it returns value of ComputePodQOS() which evaluates pod's QoS class.
func GetPodQOS(pod *core.Pod) core.PodQOSClass {
	if pod.Status.QOSClass != "" {
		return pod.Status.QOSClass
	}
	return ComputePodQOS(pod)
}

// ComputePodQOS evaluates the list of containers to determine a pod's QoS class. This function is more
// expensive than GetPodQOS which should be used for pods having a non-empty .Status.QOSClass.
// A pod is BestEffort if none of its containers have specified any cpu or memory requests or limits.
// A pod is Guaranteed only when cpu & memory requests and limits are specified for all the containers and they are equal.
// A pod is Burstable if cpu & memory limits and requests do not match across all containers.
// NOTE: This is tested in pkg/apis/core/v1/helper/qos/qos_test.go
func ComputePodQOS(pod *core.Pod) core.PodQOSClass {
	// When pod-level resources are specified, we use them to determine QoS class.
	if utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResources) &&
		pod.Spec.Resources != nil {
		return requirementsQOS(pod.Spec.Resources)
	}

	// Iterator for Init & main Containers.
	// Cannot use podutil.ContainerIter due to import cycle.
	containerIter := func(yield func(*core.Container) bool) {
		for _, c := range pod.Spec.InitContainers {
			if !yield(&c) {
				return
			}
		}
		for _, c := range pod.Spec.Containers {
			if !yield(&c) {
				return
			}
		}
	}

	var podQOS core.PodQOSClass
	for container := range containerIter {
		containerQOS := requirementsQOS(&container.Resources)
		if containerQOS == core.PodQOSBurstable {
			return containerQOS // If any container is Burstable, we know the pod isn't BestEffort or Guaranteed
		} else if podQOS == "" {
			podQOS = containerQOS
		} else if podQOS != containerQOS {
			return core.PodQOSBurstable // If one container is BestEffort and another is Guaranteed, the pod is Burstable
		}
	}
	if podQOS == "" { // This can only happen if there aren't any containers (not possible in production).
		podQOS = core.PodQOSBestEffort
	}
	return podQOS
}

// requirementsQOS gets the QOSClass based on a single set of resource requirements. This may need
// to be aggregated to determine pod QOS.
func requirementsQOS(resources *core.ResourceRequirements) core.PodQOSClass {
	if len(resources.Requests) == 0 && len(resources.Limits) == 0 {
		return core.PodQOSBestEffort
	}

	var qos core.PodQOSClass
	for res := range supportedQoSComputeResources {
		resQOS := resourceQOS(resources, res)
		if resQOS == core.PodQOSBurstable {
			return resQOS // If any resource is Burstable, we know the pod isn't BestEffort or Guaranteed
		} else if qos == "" {
			qos = resQOS
		} else if qos != resQOS {
			// A mismatch indicates some but not all QOS resources are specified, so the pod must be
			// burstable.
			return core.PodQOSBurstable
		}
	}
	return qos
}

// resourceQOS determines the QOS "shape" of the given resource in the requirements:
// - BestEffort: Request and Limit are both zero
// - Burstable: Request != Limit
// - Guaranteed: Request and Limit are equal and non-zero
func resourceQOS(resources *core.ResourceRequirements, res core.ResourceName) core.PodQOSClass {
	req := resources.Requests[res]
	lim := resources.Limits[res]

	if !req.Equal(lim) {
		// If they're not equal we know at least one is non-zero, so we know it's neither guaranteed nor best effort.
		return core.PodQOSBurstable
	} else if req.IsZero() {
		// req == lim, so no need to check lim.IsZero()
		return core.PodQOSBestEffort
	} else {
		// req == lim != 0
		return core.PodQOSGuaranteed
	}
}

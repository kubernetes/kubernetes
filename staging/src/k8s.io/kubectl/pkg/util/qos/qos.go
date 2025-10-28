/*
Copyright 2015 The Kubernetes Authors.

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

package qos

import (
	core "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
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
// A pod is besteffort if none of its containers have specified any cpu or memory requests or limits.
// A pod is guaranteed only when cpu & memory requests and limits are specified for all the containers and they are equal.
// A pod is burstable if cpu & memory limits and requests do not match across all containers.
func ComputePodQOS(pod *core.Pod) core.PodQOSClass {
	// When pod-level resources are specified, we use them to determine QoS class.
	if pod.Spec.Resources != nil {
		return resourceQOS(pod.Spec.Resources)
	}

	// Iterator for Init & main Containers.
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
		qos := resourceQOS(&container.Resources)
		if qos == core.PodQOSBurstable {
			return qos // If any container is Burstable, we know the pod isn't BestEffort or Guaranteed
		} else if podQOS == "" {
			podQOS = qos
		} else {
			if podQOS != qos {
				return core.PodQOSBurstable // If one container is BestEffort and another is Guaranteed, the pod is Burstable
			}
		}
	}
	if podQOS == "" { // This should only happen in tests
		podQOS = core.PodQOSBestEffort
	}
	return podQOS
}

// resourceQOS gets the QOSClass based on a single set of resource requirements. This may need to be aggregated to determine pod QOS.
func resourceQOS(resources *core.ResourceRequirements) core.PodQOSClass {
	if len(resources.Requests) == 0 && len(resources.Limits) == 0 {
		return core.PodQOSBestEffort
	}

	var qos core.PodQOSClass
	for res := range supportedQoSComputeResources {
		req := resources.Requests[res]
		lim := resources.Limits[res]
		if !req.Equal(lim) {
			// If they're not equal we know at least one is non-zero, so we know it's neither guaranteed nor best effort.
			return core.PodQOSBurstable
		}
		bestEffort := req.IsZero() && lim.IsZero()
		if qos == "" {
			if bestEffort {
				qos = core.PodQOSBestEffort
			} else {
				qos = core.PodQOSGuaranteed
			}
		} else {
			if bestEffort != (qos == core.PodQOSBestEffort) {
				return core.PodQOSBurstable
			}
		}
	}
	return qos
}

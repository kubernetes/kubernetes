/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package util

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
)

const (
	Guaranteed = "Guaranteed"
	Burstable  = "Burstable"
	BestEffort = "BestEffort"
)

// isResourceGuaranteed returns true if the container's resource requirements are Guaranteed.
func isResourceGuaranteed(container *api.Container, resource api.ResourceName) bool {
	// A container resource is guaranteed if its request == limit.
	// If request == limit, the user is very confident of resource consumption.
	req, hasReq := container.Resources.Requests[resource]
	limit, hasLimit := container.Resources.Limits[resource]
	if !hasReq || !hasLimit {
		return false
	}
	return req.Cmp(limit) == 0 && req.Value() != 0
}

// isResourceBestEffort returns true if the container's resource requirements are best-effort.
func isResourceBestEffort(container *api.Container, resource api.ResourceName) bool {
	// A container resource is best-effort if its request is unspecified or 0.
	// If a request is specified, then the user expects some kind of resource guarantee.
	req, hasReq := container.Resources.Requests[resource]
	return !hasReq || req.Value() == 0
}

// GetPodQos returns the QoS class of a pod.
// A pod is besteffort if none of its containers have specified any requests or limits.
// A pod is guaranteed only when requests and limits are specified for all the containers and they are equal.
// A pod is burstable if limits and requests do not match across all containers.
func GetPodQos(pod *api.Pod) string {
	requests := api.ResourceList{}
	limits := api.ResourceList{}
	zeroQuantity := resource.MustParse("0")
	isGuaranteed := true
	for _, container := range pod.Spec.Containers {
		// process requests
		for name, quantity := range container.Resources.Requests {
			if quantity.Cmp(zeroQuantity) == 1 {
				delta := quantity.Copy()
				if _, exists := requests[name]; !exists {
					requests[name] = *delta
				} else {
					delta.Add(requests[name])
					requests[name] = *delta
				}
			}
		}
		// process limits
		for name, quantity := range container.Resources.Limits {
			if quantity.Cmp(zeroQuantity) == 1 {
				delta := quantity.Copy()
				if _, exists := limits[name]; !exists {
					limits[name] = *delta
				} else {
					delta.Add(limits[name])
					limits[name] = *delta
				}
			}
		}
		if len(container.Resources.Limits) != len(supportedComputeResources) {
			isGuaranteed = false
		}
	}
	if len(requests) == 0 && len(limits) == 0 {
		return BestEffort
	}
	// Check is requests match limits for all resources.
	if isGuaranteed {
		for name, req := range requests {
			if lim, exists := limits[name]; !exists || lim.Cmp(req) != 0 {
				isGuaranteed = false
				break
			}
		}
	}
	if isGuaranteed &&
		len(requests) == len(limits) &&
		len(limits) == len(supportedComputeResources) {
		return Guaranteed
	}
	return Burstable
}

// QoSList is a set of (resource name, QoS class) pairs.
type QoSList map[api.ResourceName]string

// GetQoS returns a mapping of resource name to QoS class of a container
func GetQoS(container *api.Container) QoSList {
	resourceToQoS := QoSList{}
	for resource := range allResources(container) {
		switch {
		case isResourceGuaranteed(container, resource):
			resourceToQoS[resource] = Guaranteed
		case isResourceBestEffort(container, resource):
			resourceToQoS[resource] = BestEffort
		default:
			resourceToQoS[resource] = Burstable
		}
	}
	return resourceToQoS
}

// supportedComputeResources is the list of supported compute resources
var supportedComputeResources = []api.ResourceName{
	api.ResourceCPU,
	api.ResourceMemory,
}

// allResources returns a set of all possible resources whose mapped key value is true if present on the container
func allResources(container *api.Container) map[api.ResourceName]bool {
	resources := map[api.ResourceName]bool{}
	for _, resource := range supportedComputeResources {
		resources[resource] = false
	}
	for resource := range container.Resources.Requests {
		resources[resource] = true
	}
	for resource := range container.Resources.Limits {
		resources[resource] = true
	}
	return resources
}

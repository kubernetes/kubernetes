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
	"k8s.io/kubernetes/pkg/util/sets"
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
// The QoS class of a pod is the lowest QoS class for each resource in each container.
func GetPodQos(pod *api.Pod) string {
	qosValues := sets.NewString()
	for _, container := range pod.Spec.Containers {
		qosPerResource := GetQoS(&container)
		for _, qosValue := range qosPerResource {
			qosValues.Insert(qosValue)
		}
	}
	if qosValues.Has(BestEffort) {
		return BestEffort
	}
	if qosValues.Has(Burstable) {
		return Burstable
	}
	return Guaranteed
}

// GetQos returns a mapping of resource name to QoS class of a container
func GetQoS(container *api.Container) map[api.ResourceName]string {
	resourceToQoS := map[api.ResourceName]string{}
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

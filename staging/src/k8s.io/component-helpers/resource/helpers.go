/*
Copyright 2024 The Kubernetes Authors.

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

package resource

import (
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/sets"
)

// ContainerType signifies container type
type ContainerType int

const (
	// Containers is for normal containers
	Containers ContainerType = 1 << iota
	// InitContainers is for init containers
	InitContainers
)

// PodResourcesOptions controls the behavior of PodRequests and PodLimits.
type PodResourcesOptions struct {
	// Reuse, if provided will be reused to accumulate resources and returned by the PodRequests or PodLimits
	// functions. All existing values in Reuse will be lost.
	Reuse v1.ResourceList
	// UseStatusResources indicates whether resources reported by the PodStatus should be considered
	// when evaluating the pod resources. This MUST be false if the InPlacePodVerticalScaling
	// feature is not enabled.
	UseStatusResources bool
	// InPlacePodLevelResourcesVerticalScalingEnabled indicates whether resources reported by the
	// PodStatus should be considered when evaluating the pod resources.
	// This MUST be false if the InPlacePodLevelResourcesVerticalScaling
	// feature is not enabled.
	InPlacePodLevelResourcesVerticalScalingEnabled bool
	// ExcludeOverhead controls if pod overhead is excluded from the calculation.
	ExcludeOverhead bool
	// NonMissingContainerRequests if provided will replace any missing container level requests for the specified resources
	// with the given values.  If the requests for those resources are explicitly set, even if zero, they will not be modified.
	NonMissingContainerRequests v1.ResourceList
	// SkipPodLevelResources controls whether pod-level resources should be skipped
	// from the calculation. If pod-level resources are not set in PodSpec,
	// pod-level resources will always be skipped.
	SkipPodLevelResources bool
	// SkipContainerLevelResources
	SkipContainerLevelResources bool
	// Use node allocatable resource claim information from pod status to compute the effective pod resource request.
	UseDRANodeAllocatableResourceClaimStatus bool
}

var supportedPodLevelResources = sets.New(v1.ResourceCPU, v1.ResourceMemory)

func SupportedPodLevelResources() sets.Set[v1.ResourceName] {
	return supportedPodLevelResources.Clone().Insert(v1.ResourceHugePagesPrefix)
}

// IsSupportedPodLevelResources checks if a given resource is supported by pod-level
// resource management through the PodLevelResources feature. Returns true if
// the resource is supported.
func IsSupportedPodLevelResource(name v1.ResourceName) bool {
	return supportedPodLevelResources.Has(name) || strings.HasPrefix(string(name), v1.ResourceHugePagesPrefix)
}

// IsPodLevelResourcesSet check if PodLevelResources pod-level resources are set.
// It returns true if either the Requests or Limits maps are non-empty.
// Note: keep this in sync with k8s.io/kubernetes/pkg/apis/core/helper.IsPodLevelResourcesSet
func IsPodLevelResourcesSet(pod *v1.Pod) bool {
	if pod.Spec.Resources == nil {
		return false
	}

	if (len(pod.Spec.Resources.Requests) + len(pod.Spec.Resources.Limits)) == 0 {
		return false
	}

	for resourceName := range pod.Spec.Resources.Requests {
		if IsSupportedPodLevelResource(resourceName) {
			return true
		}
	}

	for resourceName := range pod.Spec.Resources.Limits {
		if IsSupportedPodLevelResource(resourceName) {
			return true
		}
	}

	return false
}

// IsPodLevelRequestsSet checks if pod-level requests are set. It returns true if
// Requests map is non-empty.
func IsPodLevelRequestsSet(pod *v1.Pod) bool {
	if pod.Spec.Resources == nil {
		return false
	}

	if len(pod.Spec.Resources.Requests) == 0 {
		return false
	}

	for resourceName := range pod.Spec.Resources.Requests {
		if IsSupportedPodLevelResource(resourceName) {
			return true
		}
	}

	return false
}

// IsPodLevelLimitsSet checks if pod-level limits are set. It returns true if
// Limits map is non-empty and contains at least one supported pod-level resource.
func IsPodLevelLimitsSet(pod *v1.Pod) bool {
	if pod.Spec.Resources == nil {
		return false
	}

	if len(pod.Spec.Resources.Limits) == 0 {
		return false
	}

	for resourceName := range pod.Spec.Resources.Limits {
		if IsSupportedPodLevelResource(resourceName) {
			return true
		}
	}

	return false
}

// PodRequests computes the total pod requests per the PodResourcesOptions supplied.
// If PodResourcesOptions is nil, then the requests are returned including pod overhead.
// If the PodLevelResources feature is enabled AND the pod-level resources are set,
// those pod-level values are used in calculating Pod Requests.
// The computation is part of the API and must be reviewed as an API change.
func PodRequests(pod *v1.Pod, opts PodResourcesOptions) v1.ResourceList {
	reqs := v1.ResourceList{}
	if !opts.SkipContainerLevelResources {
		reqs = AggregateContainerRequests(pod, opts)
	}

	if !opts.SkipPodLevelResources && IsPodLevelRequestsSet(pod) {
		effectiveReqs := pod.Spec.Resources.Requests
		if opts.InPlacePodLevelResourcesVerticalScalingEnabled && opts.UseStatusResources && pod.Status.Resources != nil {
			effectiveReqs = effectivePodLevelResources(pod, pod.Spec.Resources.Requests, pod.Status.Resources.Requests, pod.Status.AllocatedResources)
		}

		applyPodLevelResources(reqs, effectiveReqs)
	}

	// Add overhead for running a pod to the sum of requests if requested:
	if !opts.ExcludeOverhead && pod.Spec.Overhead != nil {
		addResourceList(reqs, pod.Spec.Overhead)
	}

	return reqs
}

func applyPodLevelResources(result, effectiveResources v1.ResourceList) {
	for resourceName, quantity := range effectiveResources {
		if IsSupportedPodLevelResource(resourceName) {
			result[resourceName] = quantity
		}
	}
}

func effectivePodLevelResources(pod *v1.Pod, spec v1.ResourceList, statuses ...v1.ResourceList) v1.ResourceList {
	if IsPodResizeInfeasible(pod) {
		spec = nil
	}
	return max(spec, statuses...)
}

func containerSpecRequests(container *v1.Container, _ *v1.ContainerStatus, _ bool) v1.ResourceList {
	return container.Resources.Requests
}

func containerAllocatedRequests(container *v1.Container, containerStatus *v1.ContainerStatus, isResizeInfeasible bool) v1.ResourceList {
	if containerStatus != nil && containerStatus.AllocatedResources != nil {
		return containerStatus.AllocatedResources
	}
	if isResizeInfeasible {
		return nil
	}
	return container.Resources.Requests
}

func containerActuatedRequests(container *v1.Container, containerStatus *v1.ContainerStatus, isResizeInfeasible bool) v1.ResourceList {
	if containerStatus != nil && containerStatus.Resources != nil && containerStatus.Resources.Requests != nil {
		return containerStatus.Resources.Requests
	}
	if containerStatus != nil && containerStatus.AllocatedResources != nil {
		return containerStatus.AllocatedResources
	}
	if isResizeInfeasible {
		return nil
	}
	return container.Resources.Requests
}

func findContainerStatus(pod *v1.Pod, name string) *v1.ContainerStatus {
	for i := range pod.Status.ContainerStatuses {
		if pod.Status.ContainerStatuses[i].Name == name {
			return &pod.Status.ContainerStatuses[i]
		}
	}
	for i := range pod.Status.InitContainerStatuses {
		if pod.Status.InitContainerStatuses[i].Name == name {
			return &pod.Status.InitContainerStatuses[i]
		}
	}
	return nil
}

func isRestartableInitContainer(container *v1.Container) bool {
	return container.RestartPolicy != nil && *container.RestartPolicy == v1.ContainerRestartPolicyAlways
}

func aggregateContainerResourcesByFn(pod *v1.Pod, opts PodResourcesOptions, getResourceList func(container *v1.Container, containerStatus *v1.ContainerStatus, isResizeInfeasible bool) v1.ResourceList) v1.ResourceList {
	var isResizeInfeasible bool
	if opts.UseStatusResources {
		isResizeInfeasible = IsPodResizeInfeasible(pod)
	}
	result := v1.ResourceList{}
	for _, container := range pod.Spec.Containers {
		var cs *v1.ContainerStatus
		if opts.UseStatusResources {
			cs = findContainerStatus(pod, container.Name)
		}
		containerResources := getResourceList(&container, cs, isResizeInfeasible)
		if len(opts.NonMissingContainerRequests) > 0 {
			containerResources = applyNonMissing(containerResources, opts.NonMissingContainerRequests)
		}
		addResourceList(result, containerResources)
	}

	restartableInitContainerResources := v1.ResourceList{}
	initContainerResources := v1.ResourceList{}
	// init containers define the minimum of any resource
	//
	// Let's say `InitContainerUse(i)` is the resource requirements when the i-th
	// init container is initializing, then
	// `InitContainerUse(i) = sum(Resources of restartable init containers with index < i) + Resources of i-th init container`.
	//
	// See https://github.com/kubernetes/enhancements/tree/master/keps/sig-node/753-sidecar-containers#exposing-pod-resource-requirements for the detail.
	for _, container := range pod.Spec.InitContainers {
		var cs *v1.ContainerStatus
		if opts.UseStatusResources {
			cs = findContainerStatus(pod, container.Name)
		}
		containerResources := getResourceList(&container, cs, isResizeInfeasible)
		if len(opts.NonMissingContainerRequests) > 0 {
			containerResources = applyNonMissing(containerResources, opts.NonMissingContainerRequests)
		}
		// Is the init container marked as a restartable init container?
		if isRestartableInitContainer(&container) {
			// and add them to the resulting cumulative container requests
			addResourceList(result, containerResources)

			// track our cumulative restartable init container resources
			addResourceList(restartableInitContainerResources, containerResources)
			containerResources = restartableInitContainerResources
		} else {
			combinedResources := v1.ResourceList{}
			addResourceList(combinedResources, containerResources)
			addResourceList(combinedResources, restartableInitContainerResources)
			containerResources = combinedResources
		}
		maxResourceList(initContainerResources, containerResources)
	}
	maxResourceList(result, initContainerResources)
	return result
}

// AggregateContainerRequests computes the total resource requests of all the containers
// in a pod. This computation folows the formula defined in the KEP for sidecar
// containers. See https://github.com/kubernetes/enhancements/tree/master/keps/sig-node/753-sidecar-containers#resources-calculation-for-scheduling-and-pod-admission
// for more details.
func AggregateContainerRequests(pod *v1.Pod, opts PodResourcesOptions) v1.ResourceList {
	// attempt to reuse the maps if passed, or allocate otherwise
	reqs := reuseOrClearResourceList(opts.Reuse)
	if !opts.UseStatusResources {
		addResourceList(reqs, aggregateContainerResourcesByFn(pod, opts, containerSpecRequests))
	} else {
		isResizeInfeasible := IsPodResizeInfeasible(pod)
		var specReqs, allocatedReqs, actuatedReqs v1.ResourceList
		// When pod-level status maps are populated, they already contain the aggregate values across all containers.
		// When unpopulated (e.g., at creation time or when feature gates are disabled), we fall back to container status aggregation.
		// Once InPlacePodLevelResourcesVerticalScaling and InPlacePodVerticalScaling are GA and feature gates are removed,
		// container-level fallback becomes redundant because max(spec, actuated, allocated) naturally evaluates to spec at creation time.
		if opts.InPlacePodLevelResourcesVerticalScalingEnabled && pod.Status.AllocatedResources != nil && pod.Status.Resources != nil && pod.Status.Resources.Requests != nil {
			specReqs = aggregateContainerResourcesByFn(pod, opts, containerSpecRequests)
			allocatedReqs = pod.Status.AllocatedResources
			actuatedReqs = pod.Status.Resources.Requests
		} else {
			specReqs = aggregateContainerResourcesByFn(pod, opts, containerSpecRequests)
			allocatedReqs = aggregateContainerResourcesByFn(pod, opts, containerAllocatedRequests)
			actuatedReqs = aggregateContainerResourcesByFn(pod, opts, containerActuatedRequests)
		}

		if isResizeInfeasible {
			addResourceList(reqs, max(actuatedReqs, allocatedReqs))
		} else {
			addResourceList(reqs, max(specReqs, actuatedReqs, allocatedReqs))
		}
	}

	// Add resources from node allocatable ResourceClaims
	if opts.UseDRANodeAllocatableResourceClaimStatus && len(pod.Status.NodeAllocatableResourceClaimStatuses) > 0 {
		for _, claimStatus := range pod.Status.NodeAllocatableResourceClaimStatuses {
			for _, mapping := range claimStatus.Mapping {
				if mapping.Quantity != nil {
					addResourceList(reqs, v1.ResourceList{mapping.Name: *mapping.Quantity})
				}
			}
			// TODO(pravk03): Handle claim references by init containers and peak resource calculations.
			for _, overhead := range claimStatus.Overhead {
				var quantity resource.Quantity
				if overhead.PerPod != nil {
					quantity.Add(*overhead.PerPod)
				}
				if overhead.PerContainer != nil && len(claimStatus.Containers) > 0 {
					varOverhead := overhead.PerContainer.DeepCopy()
					varOverhead.Mul(int64(len(claimStatus.Containers)))
					quantity.Add(varOverhead)
				}
				addResourceList(reqs, v1.ResourceList{overhead.Name: quantity})
			}
		}
	}

	return reqs
}

// IsPodResizeInfeasible returns true if the pod condition PodResizePending is set to infeasible.
func IsPodResizeInfeasible(pod *v1.Pod) bool {
	for _, condition := range pod.Status.Conditions {
		if condition.Type == v1.PodResizePending {
			return condition.Reason == v1.PodReasonInfeasible
		}
	}
	return false
}

// IsPodResizeDeferred returns true if the pod condition PodResizePending is set to deferred.
func IsPodResizeDeferred(pod *v1.Pod) bool {
	for _, condition := range pod.Status.Conditions {
		if condition.Type == v1.PodResizePending {
			return condition.Reason == v1.PodReasonDeferred
		}
	}
	return false
}

// applyNonMissing will return a copy of the given resource list with any missing values replaced by the nonMissing values
func applyNonMissing(reqs v1.ResourceList, nonMissing v1.ResourceList) v1.ResourceList {
	cp := v1.ResourceList{}
	for k, v := range reqs {
		cp[k] = v.DeepCopy()
	}

	for k, v := range nonMissing {
		if _, found := reqs[k]; !found {
			rk := cp[k]
			rk.Add(v)
			cp[k] = rk
		}
	}
	return cp
}

func containerSpecLimits(c *v1.Container, _ *v1.ContainerStatus, _ bool) v1.ResourceList {
	return c.Resources.Limits
}

func containerActuatedLimits(c *v1.Container, cs *v1.ContainerStatus, isResizeInfeasible bool) v1.ResourceList {
	if cs != nil && cs.Resources != nil && cs.Resources.Limits != nil {
		return cs.Resources.Limits
	}
	if isResizeInfeasible {
		return nil
	}
	return c.Resources.Limits
}

// PodLimits computes the pod limits per the PodResourcesOptions supplied. If PodResourcesOptions is nil, then
// the limits are returned including pod overhead for any non-zero limits. The computation is part of the API and must be reviewed
// as an API change.
func PodLimits(pod *v1.Pod, opts PodResourcesOptions) v1.ResourceList {
	// attempt to reuse the maps if passed, or allocate otherwise
	limits := AggregateContainerLimits(pod, opts)
	if !opts.SkipPodLevelResources && IsPodLevelResourcesSet(pod) {
		effectiveLims := pod.Spec.Resources.Limits
		if opts.InPlacePodLevelResourcesVerticalScalingEnabled && opts.UseStatusResources && pod.Status.Resources != nil {
			effectiveLims = effectivePodLevelResources(pod, pod.Spec.Resources.Limits, pod.Status.Resources.Limits)
		}
		applyPodLevelResources(limits, effectiveLims)
	}

	// Add overhead to non-zero limits if requested:
	if !opts.ExcludeOverhead && pod.Spec.Overhead != nil {
		for name, quantity := range pod.Spec.Overhead {
			if value, ok := limits[name]; ok && !value.IsZero() {
				value.Add(quantity)
				limits[name] = value
			}
		}
	}

	return limits
}

// AggregateContainerLimits computes the aggregated resource limits of all the containers
// in a pod. This computation follows the formula defined in the KEP for sidecar
// containers. See https://github.com/kubernetes/enhancements/tree/master/keps/sig-node/753-sidecar-containers#resources-calculation-for-scheduling-and-pod-admission
// for more details.
func AggregateContainerLimits(pod *v1.Pod, opts PodResourcesOptions) v1.ResourceList {
	opts.NonMissingContainerRequests = nil
	// attempt to reuse the maps if passed, or allocate otherwise
	limits := reuseOrClearResourceList(opts.Reuse)
	if !opts.UseStatusResources {
		addResourceList(limits, aggregateContainerResourcesByFn(pod, opts, containerSpecLimits))
	} else {
		isResizeInfeasible := IsPodResizeInfeasible(pod)
		var specLimits, actuatedLimits v1.ResourceList
		// When pod-level status maps are populated, they already contain the aggregate values across all containers.
		// When unpopulated (e.g., at creation time or when feature gates are disabled), we fall back to container status aggregation.
		// Once InPlacePodLevelResourcesVerticalScaling and InPlacePodVerticalScaling are GA and feature gates are removed,
		// container-level fallback becomes redundant because max(spec, actuated) naturally evaluates to spec at creation time.
		if opts.InPlacePodLevelResourcesVerticalScalingEnabled && pod.Status.Resources != nil && pod.Status.Resources.Limits != nil {
			specLimits = aggregateContainerResourcesByFn(pod, opts, containerSpecLimits)
			actuatedLimits = pod.Status.Resources.Limits
		} else {
			specLimits = aggregateContainerResourcesByFn(pod, opts, containerSpecLimits)
			actuatedLimits = aggregateContainerResourcesByFn(pod, opts, containerActuatedLimits)
		}

		if isResizeInfeasible {
			addResourceList(limits, actuatedLimits)
		} else {
			addResourceList(limits, max(specLimits, actuatedLimits))
		}
	}
	return limits
}

// addResourceList adds the resources in newList to list.
func addResourceList(list, newList v1.ResourceList) {
	for name, quantity := range newList {
		if value, ok := list[name]; !ok {
			list[name] = quantity.DeepCopy()
		} else {
			value.Add(quantity)
			list[name] = value
		}
	}
}

// maxResourceList sets list to the greater of list/newList for every resource in newList
func maxResourceList(list, newList v1.ResourceList) {
	for name, quantity := range newList {
		if value, ok := list[name]; !ok || quantity.Cmp(value) > 0 {
			list[name] = quantity.DeepCopy()
		}
	}
}

// max returns the result of max(a, b...) for each named resource and is only used if we can't
// accumulate into an existing resource list
func max(a v1.ResourceList, b ...v1.ResourceList) v1.ResourceList {
	var result v1.ResourceList
	if a != nil {
		result = a.DeepCopy()
	} else {
		result = v1.ResourceList{}
	}
	for _, other := range b {
		maxResourceList(result, other)
	}
	return result
}

// reuseOrClearResourceList is a helper for avoiding excessive allocations of
// resource lists within the inner loop of resource calculations.
func reuseOrClearResourceList(reuse v1.ResourceList) v1.ResourceList {
	if reuse == nil {
		return make(v1.ResourceList, 4)
	}
	for k := range reuse {
		delete(reuse, k)
	}
	return reuse
}

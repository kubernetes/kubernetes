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
	"slices"
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

func aggregateContainerResourcesByFn(pod *v1.Pod, opts PodResourcesOptions, getResourceList func(container *v1.Container, containerStatus *v1.ContainerStatus, isResizeInfeasible bool) v1.ResourceList, dra draNodeAllocatableResources) v1.ResourceList {
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
		addResourceList(result, dra.perContainer[container.Name])
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
		if draResources := dra.perContainer[container.Name]; len(draResources) > 0 {
			combinedResources := v1.ResourceList{}
			addResourceList(combinedResources, containerResources)
			addResourceList(combinedResources, draResources)
			containerResources = combinedResources
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
	// Pod fixed DRA claim resource quantities apply to the entire pod duration.
	addResourceList(result, dra.podFixed)
	return result
}

// AggregateContainerRequests computes the total resource requests of all the containers
// in a pod. This computation folows the formula defined in the KEP for sidecar
// containers. See https://github.com/kubernetes/enhancements/tree/master/keps/sig-node/753-sidecar-containers#resources-calculation-for-scheduling-and-pod-admission
// for more details.
func AggregateContainerRequests(pod *v1.Pod, opts PodResourcesOptions) v1.ResourceList {
	// attempt to reuse the maps if passed, or allocate otherwise
	reqs := reuseOrClearResourceList(opts.Reuse)
	draRes := getDRANodeAllocatableResources(pod, opts)
	if !opts.UseStatusResources {
		addResourceList(reqs, aggregateContainerResourcesByFn(pod, opts, containerSpecRequests, draRes))
	} else {
		isResizeInfeasible := IsPodResizeInfeasible(pod)
		specReqs := aggregateContainerResourcesByFn(pod, opts, containerSpecRequests, draRes)
		var allocatedReqs, actuatedReqs v1.ResourceList
		// When pod-level status maps are populated, they already contain the aggregate values across all containers.
		// When unpopulated (e.g., at creation time or when feature gates are disabled), we fall back to container status aggregation.
		// Once InPlacePodLevelResourcesVerticalScaling and InPlacePodVerticalScaling are GA and feature gates are removed,
		// container-level fallback becomes redundant because max(spec, actuated, allocated) naturally evaluates to spec at creation time.
		if opts.InPlacePodLevelResourcesVerticalScalingEnabled && pod.Status.AllocatedResources != nil && pod.Status.Resources != nil && pod.Status.Resources.Requests != nil {
			allocatedReqs = pod.Status.AllocatedResources
			actuatedReqs = pod.Status.Resources.Requests
			// DRA values are not added to allocatedReqs and actuatedReqs because
			// 1. Kubelet adds DRA allocations when setting pod.Status.AllocatedResources.
			// 2. Kubelet adds DRA allocations when configuring pod-level cgroups. Since pod.Status.Resources.Requests is read from cgroup settings, it already contains DRA values.
		} else {
			// DRA allocations are added to allocatedReqs here because Kubelet does not include DRA in pod.Status.ContainerStatuses[].AllocatedResources. Adding them prevents under-reporting.
			// This is a temporary fallback until InPlacePodLevelResourcesVerticalScaling is Beta/GA on all nodes and pod-level status fields which natively include DRA are always available.
			allocatedReqs = aggregateContainerResourcesByFn(pod, opts, containerAllocatedRequests, draRes)
			actuatedReqs = aggregateContainerResourcesByFn(pod, opts, containerActuatedRequests, draAlreadyIncluded())
		}

		if isResizeInfeasible {
			addResourceList(reqs, max(actuatedReqs, allocatedReqs))
		} else {
			addResourceList(reqs, max(specReqs, actuatedReqs, allocatedReqs))
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
	draRes := getDRANodeAllocatableResources(pod, opts)
	dropDRAValuesForUndeclaredLimits(pod, &draRes)
	if !opts.UseStatusResources {
		addResourceList(limits, aggregateContainerResourcesByFn(pod, opts, containerSpecLimits, draRes))
	} else {
		isResizeInfeasible := IsPodResizeInfeasible(pod)
		specLimits := aggregateContainerResourcesByFn(pod, opts, containerSpecLimits, draRes)
		var actuatedLimits v1.ResourceList
		// When pod-level status maps are populated, they already contain the aggregate values across all containers.
		// When unpopulated (e.g., at creation time or when feature gates are disabled), we fall back to container status aggregation.
		// Once InPlacePodLevelResourcesVerticalScaling and InPlacePodVerticalScaling are GA and feature gates are removed,
		// container-level fallback becomes redundant because max(spec, actuated) naturally evaluates to spec at creation time.
		if opts.InPlacePodLevelResourcesVerticalScalingEnabled && pod.Status.Resources != nil && pod.Status.Resources.Limits != nil {
			actuatedLimits = pod.Status.Resources.Limits
			// Kubelet includes DRA values when populating pod limits. Since pod.Status.Resources.Limits is updated based on cgroup settings, it already contains DRA values, so we should not add them again here.
		} else {
			// Kubelet considers DRA values while updating container limits. We should not be adding it here again.
			actuatedLimits = aggregateContainerResourcesByFn(pod, opts, containerActuatedLimits, draAlreadyIncluded())
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

// addResource adds quantity to list[name].
func addResource(list v1.ResourceList, name v1.ResourceName, quantity resource.Quantity) {
	q := list[name]
	q.Add(quantity)
	list[name] = q
}

// addDRAMappedResources adds each mapping's quantity to list.
func addDRAMappedResources(list v1.ResourceList, mappings []v1.NodeAllocatableMappedResources) {
	for _, mapping := range mappings {
		if mapping.Quantity != nil {
			addResource(list, mapping.Name, *mapping.Quantity)
		}
	}
}

// addDRAPerPodOverhead adds each overhead's PerPod quantity to list.
func addDRAPerPodOverhead(list v1.ResourceList, overheads []v1.NodeAllocatableOverheadResources) {
	for _, overhead := range overheads {
		if overhead.PerPod != nil {
			addResource(list, overhead.Name, *overhead.PerPod)
		}
	}
}

// addDRAPerContainerOverhead adds each overhead's PerContainer quantity multiplied by refs to list.
func addDRAPerContainerOverhead(list v1.ResourceList, overheads []v1.NodeAllocatableOverheadResources, numRefs int64) {
	if numRefs <= 0 {
		return
	}
	for _, overhead := range overheads {
		if overhead.PerContainer != nil {
			quantity := overhead.PerContainer.DeepCopy()
			quantity.Mul(numRefs)
			addResource(list, overhead.Name, quantity)
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

// GetContainerDRAAllocations returns the sum of all DRA resource allocations assigned to a container.
func GetContainerDRAAllocations(pod *v1.Pod, containerName string) v1.ResourceList {
	draAllocations := make(v1.ResourceList)
	for _, claimStatus := range pod.Status.NodeAllocatableResourceClaimStatuses {
		if !slices.Contains(claimStatus.Containers, containerName) {
			continue
		}
		// Add Mapping resources
		addDRAMappedResources(draAllocations, claimStatus.Mapping)

		// Add Overhead resources
		addDRAPerPodOverhead(draAllocations, claimStatus.Overhead)
		addDRAPerContainerOverhead(draAllocations, claimStatus.Overhead, 1)
	}
	return draAllocations
}

// dropDRAValuesForUndeclaredLimits drops DRA resource quantities (in-place) for resources without limits
// explicitly declared in the spec: kubelet skips setting limits (unlimited) if not defined in spec.
// Hugepages is an exception as they are strictly non-overcommitable so DRA values are always kept.
// Note: DRA node allocatable resources apply only to cpu, memory and hugepages.
// TODO(pravk03): Update this as part of https://github.com/kubernetes/kubernetes/issues/140810.
func dropDRAValuesForUndeclaredLimits(pod *v1.Pod, draRes *draNodeAllocatableResources) {
	if len(draRes.podFixed) == 0 && len(draRes.perContainer) == 0 {
		return
	}
	var declared sets.Set[v1.ResourceName]
	filterFunc := func(draResources v1.ResourceList) {
		for name := range draResources {
			if strings.HasPrefix(string(name), v1.ResourceHugePagesPrefix) {
				continue
			}
			if declared == nil {
				// Build a map the resource names with limits explicitly
				// declared in any container. Built lazily on first use.
				declared = sets.New[v1.ResourceName]()
				for _, containers := range [][]v1.Container{pod.Spec.InitContainers, pod.Spec.Containers} {
					for i := range containers {
						for resName := range containers[i].Resources.Limits {
							declared.Insert(resName)
						}
					}
				}
			}
			if !declared.Has(name) {
				delete(draResources, name)
			}
		}
	}
	filterFunc(draRes.podFixed)
	for _, draResContainer := range draRes.perContainer {
		filterFunc(draResContainer)
	}
}

// draNodeAllocatableResources holds the pod's node allocatable DRA claim resources.
type draNodeAllocatableResources struct {
	// podFixed is charged to this pod for its entire lifetime
	// and must always be included for resource aggregation.
	// From each NodeAllocatableResourceClaimStatus it includes:
	//   - Mapping[].Quantity
	//   - Overhead[].PerPod
	podFixed v1.ResourceList
	// perContainer holds Overhead[].PerContainer quantity for each referencing container,
	// keyed by container name. It is charged only while the referencing container runs,
	// like the container's own requests.
	// For non-restartable init containers, it counts only towards that init container's
	// peak candidate. References by container names not found in the pod spec are
	// conservatively charged to podFixed.
	perContainer map[string]v1.ResourceList
}

func getDRANodeAllocatableResources(pod *v1.Pod, opts PodResourcesOptions) draNodeAllocatableResources {
	if !opts.UseDRANodeAllocatableResourceClaimStatus || len(pod.Status.NodeAllocatableResourceClaimStatuses) == 0 {
		return draNodeAllocatableResources{}
	}
	podFixedDRARes := v1.ResourceList{}
	var perContainerDRARes map[string]v1.ResourceList
	for _, claimStatus := range pod.Status.NodeAllocatableResourceClaimStatuses {
		// Mapping quantities and PerPod overhead are charged for the pod's entire lifetime,
		// regardless of which containers reference the claim.
		addDRAMappedResources(podFixedDRARes, claimStatus.Mapping)
		addDRAPerPodOverhead(podFixedDRARes, claimStatus.Overhead)
		perContainerOverhead := slices.ContainsFunc(claimStatus.Overhead, func(overhead v1.NodeAllocatableOverheadResources) bool {
			return overhead.PerContainer != nil
		})
		if !perContainerOverhead {
			continue
		}

		// Only PerContainer overhead needs the claim's references attributed to the
		// referencing containers.
		knownRefs := 0
		for _, containers := range [][]v1.Container{pod.Spec.InitContainers, pod.Spec.Containers} {
			for i := range containers {
				name := containers[i].Name
				if !slices.Contains(claimStatus.Containers, name) {
					continue
				}
				knownRefs++
				if perContainerDRARes == nil {
					perContainerDRARes = map[string]v1.ResourceList{}
				}
				if perContainerDRARes[name] == nil {
					perContainerDRARes[name] = v1.ResourceList{}
				}
				addDRAPerContainerOverhead(perContainerDRARes[name], claimStatus.Overhead, 1)
			}
		}
		// References by container names not found in the pod spec are conservatively
		// charged for the pod's entire lifetime.
		addDRAPerContainerOverhead(podFixedDRARes, claimStatus.Overhead, int64(len(claimStatus.Containers)-knownRefs))
	}
	return draNodeAllocatableResources{podFixed: podFixedDRARes, perContainer: perContainerDRARes}
}

// draAlreadyIncluded returns an empty value for draNodeAllocatableResources.
// It exists for semantic meaning: the aggregated source already includes DRA quantities,
// like actuated values read from cgroup settings, so there is nothing to add on top.
func draAlreadyIncluded() draNodeAllocatableResources {
	return draNodeAllocatableResources{}
}

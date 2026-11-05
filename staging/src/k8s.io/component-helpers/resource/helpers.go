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
	// Use ExtraNodeAllocatableResources from pod status to compute the effective pod resource request.
	UseExtraNodeAllocatableResources bool
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

func aggregateContainerResourcesByFn(pod *v1.Pod, opts PodResourcesOptions, getResourceList func(container *v1.Container, containerStatus *v1.ContainerStatus, isResizeInfeasible bool) v1.ResourceList, extraRes extraNodeAllocatableResources) v1.ResourceList {
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
		addResourceList(result, extraRes.perContainer[container.Name])
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
		if containerExtraRes := extraRes.perContainer[container.Name]; len(containerExtraRes) > 0 {
			combinedResources := v1.ResourceList{}
			addResourceList(combinedResources, containerResources)
			addResourceList(combinedResources, containerExtraRes)
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
	addResourceList(result, extraRes.podFixed)
	return result
}

// AggregateContainerRequests computes the total resource requests of all the containers
// in a pod. This computation folows the formula defined in the KEP for sidecar
// containers. See https://github.com/kubernetes/enhancements/tree/master/keps/sig-node/753-sidecar-containers#resources-calculation-for-scheduling-and-pod-admission
// for more details.
func AggregateContainerRequests(pod *v1.Pod, opts PodResourcesOptions) v1.ResourceList {
	// attempt to reuse the maps if passed, or allocate otherwise
	reqs := reuseOrClearResourceList(opts.Reuse)
	extraRes := getExtraNodeAllocatableResources(pod, opts)
	if !opts.UseStatusResources {
		addResourceList(reqs, aggregateContainerResourcesByFn(pod, opts, containerSpecRequests, extraRes))
	} else {
		isResizeInfeasible := IsPodResizeInfeasible(pod)
		specReqs := aggregateContainerResourcesByFn(pod, opts, containerSpecRequests, extraRes)
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
			allocatedReqs = aggregateContainerResourcesByFn(pod, opts, containerAllocatedRequests, extraRes)
			actuatedReqs = aggregateContainerResourcesByFn(pod, opts, containerActuatedRequests, extraNodeAllocatableResources{})
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

// omitPartiallySetCPUMemLimits deletes the cpu and memory limits from limits unless every
// init and regular container sets them to a non-zero value.
func omitPartiallySetCPUMemLimits(pod *v1.Pod, limits v1.ResourceList) {
	_, cpuPresent := limits[v1.ResourceCPU]
	_, memoryPresent := limits[v1.ResourceMemory]
	for _, containers := range [][]v1.Container{pod.Spec.InitContainers, pod.Spec.Containers} {
		for i := range containers {
			if !cpuPresent && !memoryPresent {
				return
			}
			ctrLimits := containers[i].Resources.Limits
			if limit, found := ctrLimits[v1.ResourceCPU]; cpuPresent && (!found || limit.IsZero()) {
				cpuPresent = false
				delete(limits, v1.ResourceCPU)
			}
			if limit, found := ctrLimits[v1.ResourceMemory]; memoryPresent && (!found || limit.IsZero()) {
				memoryPresent = false
				delete(limits, v1.ResourceMemory)
			}
		}
	}
}

// omitPartiallySetEphemeralStorageLimit deletes the ephemeral-storage limit if no
// container sets one. Extra node allocatable resources (like DRA allocations) raise a declared
// limit but must not create one, so a pod without ephemeral-storage limits stays unbounded.
//
// Unlike omitPartiallySetCPUMemLimits, which omits the cpu and memory
// limits when ANY container does not set them, it keeps the partial sum when only some containers
// set an ephemeral-storage limit.
// TODO(pravk03): Omit partially set ephemeral-storage limits like cpu and memory, and remove this
// function. That changes pod-level eviction and the limits.ephemeral-storage quota for all pods.
func omitPartiallySetEphemeralStorageLimit(pod *v1.Pod, limits v1.ResourceList, extraRes extraNodeAllocatableResources) {
	if len(extraRes.podFixed) == 0 && len(extraRes.perContainer) == 0 {
		return
	}
	for _, containers := range [][]v1.Container{pod.Spec.InitContainers, pod.Spec.Containers} {
		for i := range containers {
			if _, found := containers[i].Resources.Limits[v1.ResourceEphemeralStorage]; found {
				return
			}
		}
	}
	delete(limits, v1.ResourceEphemeralStorage)
}

// AggregateContainerLimits computes the aggregated resource limits of all the containers
// in a pod. This computation follows the formula defined in the KEP for sidecar
// containers. See https://github.com/kubernetes/enhancements/tree/master/keps/sig-node/753-sidecar-containers#resources-calculation-for-scheduling-and-pod-admission
// for more details.
//
// If any container does not set a cpu or memory limit, or sets it to zero, that limit is
// omitted in the aggregate instead of returned as a partial sum, because the kubelet treats
// such a container as unlimited. Other resources, like hugepages or swap, are always summed and an unset limit
// for them means zero.
func AggregateContainerLimits(pod *v1.Pod, opts PodResourcesOptions) v1.ResourceList {
	opts.NonMissingContainerRequests = nil
	// attempt to reuse the maps if passed, or allocate otherwise
	limits := reuseOrClearResourceList(opts.Reuse)
	extraRes := getExtraNodeAllocatableResources(pod, opts)
	if !opts.UseStatusResources {
		addResourceList(limits, aggregateContainerResourcesByFn(pod, opts, containerSpecLimits, extraRes))
	} else {
		isResizeInfeasible := IsPodResizeInfeasible(pod)
		specLimits := aggregateContainerResourcesByFn(pod, opts, containerSpecLimits, extraRes)
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
			actuatedLimits = aggregateContainerResourcesByFn(pod, opts, containerActuatedLimits, extraNodeAllocatableResources{})
		}

		if isResizeInfeasible {
			addResourceList(limits, actuatedLimits)
		} else {
			addResourceList(limits, max(specLimits, actuatedLimits))
		}
	}

	omitPartiallySetCPUMemLimits(pod, limits)
	omitPartiallySetEphemeralStorageLimit(pod, limits, extraRes)
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

// addExtraNodeAllocatablePodFixedResources adds the claim's pod-lifetime quantities to list: the mapped
// quantities and the PerPod overhead.
func addExtraNodeAllocatablePodFixedResources(list v1.ResourceList, res v1.ExtraNodeAllocatableResource) {
	for _, mapping := range res.Mapping {
		if mapping.Quantity != nil {
			addResource(list, mapping.Name, *mapping.Quantity)
		}
	}
	for _, overhead := range res.Overhead {
		if overhead.PerPod != nil {
			addResource(list, overhead.Name, *overhead.PerPod)
		}
	}
}

// addExtraNodeAllocatablePerContainerOverhead adds each overhead's PerContainer quantity to list.
func addExtraNodeAllocatablePerContainerOverhead(list v1.ResourceList, overheads []v1.NodeAllocatableOverheadResources) {
	for _, overhead := range overheads {
		if overhead.PerContainer != nil {
			addResource(list, overhead.Name, *overhead.PerContainer)
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

// GetContainerExtraNodeAllocatableResources returns the sum of all extra node allocatable resources (like DRA allocations) assigned to a container.
func GetContainerExtraNodeAllocatableResources(pod *v1.Pod, containerName string) v1.ResourceList {
	extraNodeAllocatableRes := make(v1.ResourceList)
	for _, res := range pod.Status.ExtraNodeAllocatableResources {
		if !slices.Contains(res.Containers, containerName) {
			continue
		}
		addExtraNodeAllocatablePodFixedResources(extraNodeAllocatableRes, res)
		addExtraNodeAllocatablePerContainerOverhead(extraNodeAllocatableRes, res.Overhead)
	}
	return extraNodeAllocatableRes
}

// GetPodExtraNodeAllocatableResources returns the sum of all extra node allocatable
// resources (like DRA allocations) of a pod.
func GetPodExtraNodeAllocatableResources(pod *v1.Pod) v1.ResourceList {
	extraRes := getExtraNodeAllocatableResources(pod, PodResourcesOptions{UseExtraNodeAllocatableResources: true})
	extraNodeAllocatableRes := v1.ResourceList{}
	addResourceList(extraNodeAllocatableRes, extraRes.podFixed)
	for _, containerExtraRes := range extraRes.perContainer {
		addResourceList(extraNodeAllocatableRes, containerExtraRes)
	}
	return extraNodeAllocatableRes
}

// extraNodeAllocatableResources holds the pod's extra node allocatable resources (like DRA allocations).
type extraNodeAllocatableResources struct {
	// podFixed is charged to this pod for its entire lifetime
	// and must always be included for resource aggregation.
	// From each ExtraNodeAllocatableResource it includes:
	//   - Mapping[].Quantity
	//   - Overhead[].PerPod
	podFixed v1.ResourceList
	// perContainer holds Overhead[].PerContainer quantity for each referencing container,
	// keyed by container name. It is charged only while the referencing container runs,
	// like the container's own requests.
	// For non-restartable init containers, it counts only towards that init container's
	// peak candidate.
	perContainer map[string]v1.ResourceList
}

func getExtraNodeAllocatableResources(pod *v1.Pod, opts PodResourcesOptions) extraNodeAllocatableResources {
	if !opts.UseExtraNodeAllocatableResources || len(pod.Status.ExtraNodeAllocatableResources) == 0 {
		return extraNodeAllocatableResources{}
	}
	podFixed := v1.ResourceList{}
	perContainer := map[string]v1.ResourceList{}
	for _, res := range pod.Status.ExtraNodeAllocatableResources {
		// Charged regardless of which containers reference the claim.
		addExtraNodeAllocatablePodFixedResources(podFixed, res)

		// Every container referencing the claim is charged the same PerContainer overhead,
		// so compute it once. The scheduler only writes names taken from the pod spec.
		containerOverhead := v1.ResourceList{}
		addExtraNodeAllocatablePerContainerOverhead(containerOverhead, res.Overhead)
		if len(containerOverhead) == 0 {
			continue
		}
		for _, containerName := range res.Containers {
			if perContainer[containerName] == nil {
				perContainer[containerName] = v1.ResourceList{}
			}
			addResourceList(perContainer[containerName], containerOverhead)
		}
	}
	return extraNodeAllocatableResources{podFixed: podFixed, perContainer: perContainer}
}

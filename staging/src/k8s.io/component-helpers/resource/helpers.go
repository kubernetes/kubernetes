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
	"k8s.io/apimachinery/pkg/util/sets"
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
func IsPodLevelResourcesSet(pod *v1.Pod) bool {
	return IsPodLevelRequestsSet(pod) || IsPodLevelLimitsSet(pod)
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
	return Requests.PodTotal(pod, opts)
}

// AggregateContainerRequests computes the total resource requests of all the containers
// in a pod. This computation follows the formula defined in the KEP for sidecar
// containers. See https://github.com/kubernetes/enhancements/tree/master/keps/sig-node/753-sidecar-containers#resources-calculation-for-scheduling-and-pod-admission
// for more details.
func AggregateContainerRequests(pod *v1.Pod, opts PodResourcesOptions) v1.ResourceList {
	opts.SkipPodLevelResources = true
	opts.ExcludeOverhead = true
	return Requests.PodTotal(pod, opts)
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

// PodLimits computes the pod limits per the PodResourcesOptions supplied. If PodResourcesOptions is nil, then
// the limits are returned including pod overhead for any non-zero limits. The computation is part of the API and must be reviewed
// as an API change.
func PodLimits(pod *v1.Pod, opts PodResourcesOptions) v1.ResourceList {
	return Limits.PodTotal(pod, opts)
}

// AggregateContainerLimits computes the aggregated resource limits of all the containers
// in a pod. This computation follows the formula defined in the KEP for sidecar
// containers. See https://github.com/kubernetes/enhancements/tree/master/keps/sig-node/753-sidecar-containers#resources-calculation-for-scheduling-and-pod-admission
// for more details.
func AggregateContainerLimits(pod *v1.Pod, opts PodResourcesOptions) v1.ResourceList {
	opts.SkipPodLevelResources = true
	opts.ExcludeOverhead = true
	return Limits.PodTotal(pod, opts)
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

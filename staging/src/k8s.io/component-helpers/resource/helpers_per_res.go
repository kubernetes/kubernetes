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
	"iter"
	"slices"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

type Requirement int

const (
	Requests Requirement = iota
	Limits
)

func (t Requirement) PodTotal(pod *v1.Pod, opts PodResourcesOptions) v1.ResourceList {
	// attempt to reuse the maps if passed, or allocate otherwise
	reqs := reuseOrClearResourceList(opts.Reuse)

	for res := range t.PodResourceIter(pod, opts) {
		reqs[res] = t.PodResourceTotal(pod, res, opts)
	}

	return reqs
}

// PodResourceIter iterates over all resources requested across pod-level requests, container
// requests, and pod overhead.
func (t Requirement) PodResourceIter(pod *v1.Pod, opts PodResourcesOptions) iter.Seq[v1.ResourceName] {
	return func(yield func(v1.ResourceName) bool) {
		var seenBuf [8]v1.ResourceName // Pre-allocate for 8 resources on the stack.
		seen := seenBuf[:0]
		scanResources := func(rl v1.ResourceList) bool {
			for res := range rl {
				if !slices.Contains(seen, res) {
					if !yield(res) {
						return false
					}
					seen = append(seen, res)
				}
			}
			return true
		}

		if !opts.SkipPodLevelResources && pod.Spec.Resources != nil {
			for res := range t.get(pod.Spec.Resources) {
				if IsSupportedPodLevelResource(res) {
					if !slices.Contains(seen, res) {
						if !yield(res) {
							return
						}
						seen = append(seen, res)
					}
				}
			}
			if opts.UseStatusResources && opts.InPlacePodLevelResourcesVerticalScalingEnabled {
				if pod.Status.Resources != nil && !scanResources(t.get(pod.Status.Resources)) {
					return
				}
				if t == Requests && !scanResources(pod.Status.AllocatedResources) {
					return
				}
			}
		}
		if !opts.ExcludeOverhead && !scanResources(pod.Spec.Overhead) {
			return
		}
		if !opts.SkipContainerLevelResources {
			for c := range joinIter(pod.Spec.InitContainers, pod.Spec.Containers) {
				if !scanResources(t.get(&c.Resources)) {
					return
				}
			}
			if opts.UseStatusResources {
				for cs := range joinIter(pod.Status.ContainerStatuses, pod.Status.InitContainerStatuses) {
					if cs.Resources != nil && !scanResources(t.get(cs.Resources)) {
						return
					}
					if t == Requests && !scanResources(cs.AllocatedResources) {
						return
					}
				}
			}
		}
	}
}

func (t Requirement) PodResourceTotal(pod *v1.Pod, res v1.ResourceName, opts PodResourcesOptions) resource.Quantity {
	var request resource.Quantity
	if !opts.SkipPodLevelResources && IsSupportedPodLevelResource(res) && pod.Spec.Resources != nil {
		request = t.get(pod.Spec.Resources)[res]
		if opts.InPlacePodLevelResourcesVerticalScalingEnabled && opts.UseStatusResources && pod.Status.Resources != nil {
			request = t.determineEffectiveResource(pod,
				request,
				pod.Status.Resources.Requests[res],
				pod.Status.AllocatedResources[res])
		}
		request = request.DeepCopy()
	}

	if !opts.SkipContainerLevelResources && request.IsZero() {
		request = t.AggregateContainerResource(pod, res, opts)
	}

	// Add overhead for running a pod to the sum of requests if requested:
	if !opts.ExcludeOverhead && pod.Spec.Overhead != nil {
		overhead := pod.Spec.Overhead[res]
		if !overhead.IsZero() {
			if request.IsZero() {
				request = overhead.DeepCopy()
			} else {
				request.Add(overhead)
			}
		}
	}

	return request
}

func (t Requirement) AggregateContainerResource(pod *v1.Pod, res v1.ResourceName, opts PodResourcesOptions) resource.Quantity {
	var longRunningReq, maxInitReq resource.Quantity

	// init containers define the minimum of any resource
	//
	// Let's say `InitContainerUse(i)` is the resource requirements when the i-th
	// init container is initializing, then
	// `InitContainerUse(i) = sum(Resources of restartable init containers with index < i) + Resources of i-th init container`.
	//
	// See https://github.com/kubernetes/enhancements/tree/master/keps/sig-node/753-sidecar-containers#exposing-pod-resource-requirements for the detail.
	for _, container := range pod.Spec.InitContainers {
		containerReq := t.get(&container.Resources)[res]
		if opts.UseStatusResources {
			if container.RestartPolicy != nil && *container.RestartPolicy == v1.ContainerRestartPolicyAlways {
				cs := findContainerStatus(pod.Status.InitContainerStatuses, container.Name)
				if cs != nil && cs.Resources != nil {
					containerReq = t.determineEffectiveResource(pod,
						containerReq,
						t.get(cs.Resources)[res],
						cs.AllocatedResources[res])
				}
			}
		}

		if t == Requests && containerReq.IsZero() {
			if nonMissing, ok := opts.NonMissingContainerRequests[res]; ok {
				containerReq = nonMissing
			}
		}

		if container.RestartPolicy != nil && *container.RestartPolicy == v1.ContainerRestartPolicyAlways {
			// track our cumulative restartable init container resources
			longRunningReq.Add(containerReq)
			containerReq = longRunningReq
		} else {
			containerReq = containerReq.DeepCopy()
			containerReq.Add(longRunningReq)
		}

		maxInitReq = maxQuantity(maxInitReq, containerReq)
	}

	for _, container := range pod.Spec.Containers {
		containerReq := t.get(&container.Resources)[res]
		if opts.UseStatusResources {
			cs := findContainerStatus(pod.Status.ContainerStatuses, container.Name)
			if cs != nil && cs.Resources != nil {
				containerReq = t.determineEffectiveResource(pod,
					containerReq,
					t.get(cs.Resources)[res],
					cs.AllocatedResources[res])
			}
		}

		if t == Requests && containerReq.IsZero() {
			if nonMissing, ok := opts.NonMissingContainerRequests[res]; ok {
				containerReq = nonMissing
			}
		}

		longRunningReq.Add(containerReq)
	}

	return maxQuantity(longRunningReq, maxInitReq)
}

func (t Requirement) get(reqs *v1.ResourceRequirements) v1.ResourceList {
	switch t {
	case Requests:
		return reqs.Requests
	case Limits:
		return reqs.Limits
	default:
		panic("invalid requirement type")
	}
}

func (t Requirement) determineEffectiveResource(pod *v1.Pod, desired, actuated, allocated resource.Quantity) resource.Quantity {
	if t == Limits {
		allocated = resource.Quantity{}
	}
	if IsPodResizeInfeasible(pod) {
		return maxQuantity(actuated, allocated)
	}
	return maxQuantity(desired, maxQuantity(actuated, allocated))
}

// maxQuantity returns the largest quantity from two provided values.
// This assumes that q1 and q2 cannot be negative.
func maxQuantity(q1, q2 resource.Quantity) resource.Quantity {
	if q1.IsZero() {
		return q2
	}
	if q2.IsZero() {
		return q1
	}
	if q1.Cmp(q2) > 0 {
		return q1
	}
	return q2
}

func findContainerStatus(statuses []v1.ContainerStatus, name string) *v1.ContainerStatus {
	for i, status := range statuses {
		if status.Name == name {
			return &statuses[i]
		}
	}
	return nil
}

func joinIter[T any](ss ...[]T) iter.Seq[*T] {
	return func(yield func(*T) bool) {
		for _, s := range ss {
			for i := range s {
				if !yield(&s[i]) {
					return
				}
			}
		}
	}
}

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

func PodRequestsV2(pod *v1.Pod, opts PodResourcesOptions) v1.ResourceList {
	// attempt to reuse the maps if passed, or allocate otherwise
	reqs := reuseOrClearResourceList(opts.Reuse)

	for res := range allRequestResources(pod, opts) {
		reqs[res] = PodResourceRequest(pod, res, opts)
	}

	return reqs
}

// allRequestResources iterates over all resources requested across pod-level requests, container
// requests, and pod overhead.
func allRequestResources(pod *v1.Pod, opts PodResourcesOptions) iter.Seq[v1.ResourceName] {
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
			for res := range pod.Spec.Resources.Requests {
				if IsSupportedPodLevelResource(res) {
					if !slices.Contains(seen, res) {
						if !yield(res) {
							return
						}
						seen = append(seen, res)
					}
				}
			}
			if opts.UseStatusResources && opts.InPlacePodLevelResourcesVerticalScalingEnabled && pod.Status.Resources != nil {
				for res := range pod.Status.Resources.Requests {
					if !slices.Contains(seen, res) {
						if !yield(res) {
							return
						}
						seen = append(seen, res)
					}
				}
				for res := range pod.Status.AllocatedResources {
					if !slices.Contains(seen, res) {
						if !yield(res) {
							return
						}
						seen = append(seen, res)
					}
				}
			}
		}
		if !opts.ExcludeOverhead {
			for res := range pod.Spec.Overhead {
				if !slices.Contains(seen, res) {
					if !yield(res) {
						return
					}
					seen = append(seen, res)
				}
			}
		}
		if !opts.SkipContainerLevelResources {
			for c := range containerIter(pod) {
				for res := range c.Resources.Requests {
					if !slices.Contains(seen, res) {
						if !yield(res) {
							return
						}
						seen = append(seen, res)
					}
				}
			}
			if opts.UseStatusResources {
				for _, cs := range pod.Status.ContainerStatuses {
					if cs.Resources != nil {
						for res := range cs.Resources.Requests {
							if !slices.Contains(seen, res) {
								if !yield(res) {
									return
								}
								seen = append(seen, res)
							}
						}
					}
					for res := range cs.AllocatedResources {
						if !slices.Contains(seen, res) {
							if !yield(res) {
								return
							}
							seen = append(seen, res)
						}
					}
				}
				for _, cs := range pod.Status.InitContainerStatuses {
					if cs.Resources != nil {
						for res := range cs.Resources.Requests {
							if !slices.Contains(seen, res) {
								if !yield(res) {
									return
								}
								seen = append(seen, res)
							}
						}
					}
					for res := range cs.AllocatedResources {
						if !slices.Contains(seen, res) {
							if !yield(res) {
								return
							}
							seen = append(seen, res)
						}
					}
				}
			}
		}
	}
}

func PodResourceRequest(pod *v1.Pod, res v1.ResourceName, opts PodResourcesOptions) resource.Quantity {
	var request resource.Quantity
	if !opts.SkipPodLevelResources && IsSupportedPodLevelResource(res) && pod.Spec.Resources != nil {
		request = pod.Spec.Resources.Requests[res]
		if opts.InPlacePodLevelResourcesVerticalScalingEnabled && opts.UseStatusResources && pod.Status.Resources != nil {
			request = determineEffectiveRequest(pod,
				request,
				pod.Status.Resources.Requests[res],
				pod.Status.AllocatedResources[res])
		}
	}

	if !opts.SkipContainerLevelResources && request.IsZero() {
		request = AggregateContainerResourceRequest(pod, res, opts)
	}

	// Add overhead for running a pod to the sum of requests if requested:
	if !opts.ExcludeOverhead && pod.Spec.Overhead != nil {
		overhead := pod.Spec.Overhead[res]
		if !overhead.IsZero() {
			if request.IsZero() {
				request = overhead
			} else {
				request.Add(overhead)
			}
		}
	}

	return request
}

func AggregateContainerResourceRequest(pod *v1.Pod, res v1.ResourceName, opts PodResourcesOptions) resource.Quantity {

	var longRunningReq, maxInitReq resource.Quantity

	// init containers define the minimum of any resource
	//
	// Let's say `InitContainerUse(i)` is the resource requirements when the i-th
	// init container is initializing, then
	// `InitContainerUse(i) = sum(Resources of restartable init containers with index < i) + Resources of i-th init container`.
	//
	// See https://github.com/kubernetes/enhancements/tree/master/keps/sig-node/753-sidecar-containers#exposing-pod-resource-requirements for the detail.
	for _, container := range pod.Spec.InitContainers {
		containerReq := container.Resources.Requests[res]
		if opts.UseStatusResources {
			if container.RestartPolicy != nil && *container.RestartPolicy == v1.ContainerRestartPolicyAlways {
				cs := findContainerStatus(pod.Status.InitContainerStatuses, container.Name)
				if cs != nil && cs.Resources != nil {
					containerReq = determineEffectiveRequest(pod,
						containerReq,
						cs.Resources.Requests[res],
						cs.AllocatedResources[res])
				}
			}
		}

		if containerReq.IsZero() {
			if nonMissing, ok := opts.NonMissingContainerRequests[res]; ok {
				containerReq = nonMissing
			}
		}

		if container.RestartPolicy != nil && *container.RestartPolicy == v1.ContainerRestartPolicyAlways {
			// track our cumulative restartable init container resources
			longRunningReq.Add(containerReq)
			containerReq = longRunningReq
		} else {
			// We need a sum of longRunningReq + containerReq.
			// To avoid allocation if they are both int64-based:
			if longRunningReq.IsZero() {
				// use containerReq as is
			} else if containerReq.IsZero() {
				containerReq = longRunningReq
			} else {
				// This might still allocate if it's not int64.
				containerReq.Add(longRunningReq)
			}
		}

		// TODO: this doesn't support opts.ContainerFn

		maxInitReq = maxQuantity(maxInitReq, containerReq)
	}

	for _, container := range pod.Spec.Containers {
		containerReq := container.Resources.Requests[res]
		if opts.UseStatusResources {
			cs := findContainerStatus(pod.Status.ContainerStatuses, container.Name)
			if cs != nil && cs.Resources != nil {
				containerReq = determineEffectiveRequest(pod,
					containerReq,
					cs.Resources.Requests[res],
					cs.AllocatedResources[res])
			}
		}

		if containerReq.IsZero() {
			if nonMissing, ok := opts.NonMissingContainerRequests[res]; ok {
				containerReq = nonMissing
			}
		}

		// TODO: this doesn't support opts.ContainerFn

		longRunningReq.Add(containerReq)
	}

	return maxQuantity(longRunningReq, maxInitReq)
}

func determineEffectiveRequest(pod *v1.Pod, desired, actuated, allocated resource.Quantity) resource.Quantity {
	if IsPodResizeInfeasible(pod) {
		return maxQuantity(actuated, allocated)
	}
	return maxQuantity(desired, maxQuantity(actuated, allocated))
}

// maxQuantity returns the largest quantity from two provided values.
func maxQuantity(q1, q2 resource.Quantity) resource.Quantity {
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

func containerIter(pod *v1.Pod) iter.Seq2[*v1.Container, ContainerType] {
	return func(yield func(*v1.Container, ContainerType) bool) {
		for i := range pod.Spec.InitContainers {
			if !yield(&pod.Spec.InitContainers[i], InitContainers) {
				return
			}
		}
		for i := range pod.Spec.Containers {
			if !yield(&pod.Spec.Containers[i], Containers) {
				return
			}
		}
	}
}

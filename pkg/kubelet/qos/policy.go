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
	"math/bits"
	"slices"

	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	resourcehelper "k8s.io/component-helpers/resource"
	v1qos "k8s.io/kubernetes/pkg/apis/core/v1/helper/qos"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/types"
)

const (
	// KubeletOOMScoreAdj is the OOM score adjustment for Kubelet
	KubeletOOMScoreAdj int = -999
	// KubeProxyOOMScoreAdj is the OOM score adjustment for kube-proxy
	KubeProxyOOMScoreAdj  int = -999
	guaranteedOOMScoreAdj int = -997
	besteffortOOMScoreAdj int = 1000

	// oomScoreAdjTieBreakFloor is the smallest memory request the
	// KubeletOOMScoreAdjTieBreak nudge distinguishes. Nodes whose adj point
	// (capacity/1000) is at or below this size already separate requests well
	// enough, so the nudge is disabled there.
	oomScoreAdjTieBreakFloor int64 = 64 * 1024 * 1024
)

// GetContainerOOMScoreAdjust returns the amount by which the OOM score of all processes in the
// container should be adjusted.
// The OOM score of a process is the percentage of memory it consumes
// multiplied by 10 (barring exceptional cases) + a configurable quantity which is between -1000
// and 1000. Containers with higher OOM scores are killed if the system runs out of memory.
// See https://lwn.net/Articles/391222/ for more information.
// OOMScoreAdjust should be calculated based on the allocated resources, so the pod argument should
// contain the allocated resources in the spec.
func GetContainerOOMScoreAdjust(pod *v1.Pod, container *v1.Container, memoryCapacity int64) int {
	return containerOOMScoreAdjust(pod, container, memoryCapacity,
		utilfeature.DefaultFeatureGate.Enabled(features.KubeletOOMScoreAdjTieBreak))
}

// PossibleContainerOOMScoreAdjusts returns the oom_score_adj kubelet assigns to
// the container with KubeletOOMScoreAdjTieBreak disabled and enabled, in that
// order. It is meant for e2e tests, which cannot observe the node's feature
// gates.
func PossibleContainerOOMScoreAdjusts(pod *v1.Pod, container *v1.Container, memoryCapacity int64) []int {
	return []int{
		containerOOMScoreAdjust(pod, container, memoryCapacity, false),
		containerOOMScoreAdjust(pod, container, memoryCapacity, true),
	}
}

func containerOOMScoreAdjust(pod *v1.Pod, container *v1.Container, memoryCapacity int64, tieBreakEnabled bool) int {
	if types.IsNodeCriticalPod(pod) {
		// Only node critical pod should be the last to get killed.
		return guaranteedOOMScoreAdj
	}

	switch v1qos.GetPodQOS(pod) {
	case v1.PodQOSGuaranteed:
		// Guaranteed containers should be the last to get killed.
		return guaranteedOOMScoreAdj
	case v1.PodQOSBestEffort:
		return besteffortOOMScoreAdj
	}

	// Burstable containers are a middle tier, between Guaranteed and Best-Effort. Ideally,
	// we want to protect Burstable containers that consume less memory than requested.
	// The formula below is a heuristic. A container requesting for 10% of a system's
	// memory will have an OOM score adjust of 900. If a process in container Y
	// uses over 10% of memory, its OOM score will be 1000. The idea is that containers
	// which use more than their request will have an OOM score of 1000 and will be prime
	// targets for OOM kills.
	// Note that this is a heuristic, it won't work if a container has many small processes.
	containerMemReq := getEffectiveContainerMemoryRequest(pod, container)

	var oomScoreAdjust, remainingReqPerContainer int64
	// When PodLevelResources feature is enabled, the OOM score adjustment formula is modified
	// to account for pod-level memory requests. Any extra pod memory request that's
	// not allocated to the containers is divided equally among all containers and
	// added to their individual memory requests when calculating the OOM score
	// adjustment. Otherwise, only container-level memory requests are used. See
	// https://github.com/kubernetes/enhancements/blob/master/keps/sig-node/2837-pod-level-resource-spec/README.md#oom-score-adjustment
	// for more details.
	if utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResources) &&
		resourcehelper.IsPodLevelRequestsSet(pod) {
		// TODO(ndixita): Refactor to use this formula in all cases, as
		// remainingReqPerContainer will be 0 when pod-level resources are not set.
		remainingReqPerContainer = remainingPodMemReqPerContainer(pod)
		oomScoreAdjust = 1000 - (1000 * (containerMemReq + remainingReqPerContainer) / memoryCapacity)
	} else {
		oomScoreAdjust = 1000 - (1000*containerMemReq)/memoryCapacity
	}

	if tieBreakEnabled {
		oomScoreAdjust = tieBreakOOMScoreAdjust(oomScoreAdjust, containerMemReq+remainingReqPerContainer, enforcedMemoryLimit(pod, container), memoryCapacity)
	}

	// adapt the sidecarContainer memoryRequest for OOM ADJ calculation
	// calculate the oom score adjustment based on: max-memory( currentSideCarContainer , min-memory(regular containers) ) .
	if isSidecarContainer(pod, container) {
		var minMemoryOomScoreAdjust int64
		if tieBreakEnabled {
			// The tie-break makes the smallest request no longer imply the
			// highest score, so compare against every regular container.
			minMemoryOomScoreAdjust = maxRegularContainerOOMScoreAdjust(pod, remainingReqPerContainer, memoryCapacity)
		} else {
			// check min memory quantity in regular containers
			minMemoryRequest := minRegularContainerMemory(pod)

			// When calculating minMemoryOomScoreAdjust for sidecar containers with PodLevelResources enabled,
			// we add the per-container share of unallocated pod memory requests to the minimum memory request.
			// This ensures the OOM score adjustment i.e. minMemoryOomScoreAdjust
			// calculation remains consistent
			//  with how we handle pod-level memory requests for regular containers.
			if utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResources) &&
				resourcehelper.IsPodLevelRequestsSet(pod) {
				minMemoryRequest += remainingReqPerContainer
			}
			minMemoryOomScoreAdjust = 1000 - (1000*minMemoryRequest)/memoryCapacity
		}
		// the OOM adjustment for sidecar container will match
		// or fall below the OOM score adjustment of regular containers in the Pod.
		if oomScoreAdjust > minMemoryOomScoreAdjust {
			oomScoreAdjust = minMemoryOomScoreAdjust
		}
	}

	// A guaranteed pod using 100% of memory can have an OOM score of 10. Ensure
	// that burstable pods have a higher OOM score adjustment.
	if int(oomScoreAdjust) < (1000 + guaranteedOOMScoreAdj) {
		return (1000 + guaranteedOOMScoreAdj)
	}
	// Give burstable pods a higher chance of survival over besteffort pods.
	if int(oomScoreAdjust) == besteffortOOMScoreAdj {
		return int(oomScoreAdjust - 1)
	}
	return int(oomScoreAdjust)
}

// tieBreakOOMScoreAdjust refines the linear Burstable oom_score_adj for
// KubeletOOMScoreAdjTieBreak. memoryRequest is the effective request the linear
// value was computed from, and memoryLimit the limit enforced on the
// container's cgroup, or 0 if unlimited. The linear term is kept as is: one adj point is
// worth capacity/1000 bytes of usage to the kernel, which is what makes
// containers furthest over their request get killed first. Every adjustment
// here is bounded by a few points so it cannot override that ordering by more
// than a few adj points worth of memory.
func tieBreakOOMScoreAdjust(linearOOMScoreAdjust, memoryRequest, memoryLimit, memoryCapacity int64) int64 {
	// Requests smaller than one adj point collapse to the same score. Spend
	// one point per doubling of request inside that range, and no more.
	maxNudge := log2Floor(memoryCapacity / 1000 / oomScoreAdjTieBreakFloor)
	adj := linearOOMScoreAdjust - min(log2Floor(memoryRequest/oomScoreAdjTieBreakFloor), maxNudge)

	// A container whose enforced limit is within its request can never be
	// over its request, so ranking it below others does not conflict with
	// killing over-request containers first. The spec limit is not enough:
	// a shared DRA memory claim raises the cgroup limit above the request.
	if memoryLimit > 0 && memoryLimit <= memoryRequest {
		adj -= maxNudge + 1
	}
	return adj
}

// maxRegularContainerOOMScoreAdjust returns the highest tie-broken
// oom_score_adj among the regular containers in pod, before clamping.
func maxRegularContainerOOMScoreAdjust(pod *v1.Pod, remainingReqPerContainer, memoryCapacity int64) int64 {
	var maxAdj int64
	for i := range pod.Spec.Containers {
		c := &pod.Spec.Containers[i]
		memReq := getEffectiveContainerMemoryRequest(pod, c) + remainingReqPerContainer
		adj := tieBreakOOMScoreAdjust(1000-(1000*memReq)/memoryCapacity, memReq, enforcedMemoryLimit(pod, c), memoryCapacity)
		if i == 0 || adj > maxAdj {
			maxAdj = adj
		}
	}
	return maxAdj
}

// enforcedMemoryLimit returns the memory limit kubelet sets on the container's
// cgroup in bytes, or 0 if unlimited.
func enforcedMemoryLimit(pod *v1.Pod, container *v1.Container) int64 {
	draAllocations := v1.ResourceList{}
	if utilfeature.DefaultFeatureGate.Enabled(features.DRANodeAllocatableResources) {
		draAllocations = resourcehelper.GetContainerDRAAllocations(pod, container.Name)
	}
	return GetContainerMemoryLimit(pod, container, draAllocations).Value()
}

// log2Floor returns floor(log2(x)), or 0 for x < 1.
func log2Floor(x int64) int64 {
	if x < 1 {
		return 0
	}
	return int64(bits.Len64(uint64(x))) - 1
}

// isSidecarContainer returns a boolean indicating whether a container is a sidecar or not.
// Since v1.Container does not directly specify whether a container is a sidecar,
// this function uses available indicators (container.RestartPolicy == v1.ContainerRestartPolicyAlways)
// to make that determination.
func isSidecarContainer(pod *v1.Pod, container *v1.Container) bool {
	if container.RestartPolicy != nil && *container.RestartPolicy == v1.ContainerRestartPolicyAlways {
		for _, initContainer := range pod.Spec.InitContainers {
			if initContainer.Name == container.Name {
				return true
			}
		}
	}
	return false
}

// getEffectiveContainerMemoryRequest returns the effective container memory request,
// including DRA memory allocations if the feature is enabled.
// If multiple containers share a single DRA memory claim, the claim's memory quantity
// is divided equally among the sharing containers.
func getEffectiveContainerMemoryRequest(pod *v1.Pod, container *v1.Container) int64 {
	memReq := container.Resources.Requests.Memory().Value()
	if !utilfeature.DefaultFeatureGate.Enabled(features.DRANodeAllocatableResources) {
		return memReq
	}

	var draMemoryShare int64
	for _, claimStatus := range pod.Status.NodeAllocatableResourceClaimStatuses {
		// Check if this container references the claim
		if !slices.Contains(claimStatus.Containers, container.Name) {
			continue
		}

		numRefs := int64(len(claimStatus.Containers))

		// Add Mapping memory resources share. Since mapping resources represent a fixed allocation
		// shared by all referencing containers in the pod, we divide it equally.
		for _, mapping := range claimStatus.Mapping {
			if mapping.Name == v1.ResourceMemory && mapping.Quantity != nil {
				draMemoryShare += mapping.Quantity.Value() / numRefs
			}
		}

		// Add Overhead memory resources share.
		for _, overhead := range claimStatus.Overhead {
			if overhead.Name == v1.ResourceMemory {
				// PerPod overhead is applied once per pod. We split it equally among the
				// referencing containers.
				if overhead.PerPod != nil {
					draMemoryShare += overhead.PerPod.Value() / numRefs
				}
				// PerContainer overhead is applied per container
				// reference. Therefore, each container gets its full share without division.
				if overhead.PerContainer != nil {
					draMemoryShare += overhead.PerContainer.Value()
				}
			}
		}
	}

	return memReq + draMemoryShare
}

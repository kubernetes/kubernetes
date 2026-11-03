/*
Copyright 2018 The Kubernetes Authors.

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

package cm

import (
	"context"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	resourcehelper "k8s.io/component-helpers/resource"
	internalapi "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	v1qos "k8s.io/kubernetes/pkg/apis/core/v1/helper/qos"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm/containermap"
	evictionapi "k8s.io/kubernetes/pkg/kubelet/eviction/api"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

// for typecheck across platforms
var _ func(int64, int64) int64 = MilliCPUToQuota
var _ func(int64) uint64 = MilliCPUToShares
var _ func(uint64, uint64) bool = CPUSharesEqualAfterV2RoundTrip
var _ func(*v1.Pod, bool, uint64, bool, kubeletconfig.MemoryReservationPolicy) *ResourceConfig = ResourceConfigForPod
var _ func() (*CgroupSubsystems, error) = GetCgroupSubsystems
var _ func(string) ([]int, error) = getCgroupProcs
var _ func(types.UID) string = GetPodCgroupNameSuffix
var _ func(string, bool, string) string = NodeAllocatableRoot
var _ func(klog.Logger, string) (string, error) = GetKubeletContainer

func isBurstableNonCriticalPod(pod *v1.Pod) bool {
	return pod != nil && !kubetypes.IsCriticalPod(pod) && v1qos.GetPodQOS(pod) == v1.PodQOSBurstable
}

func isMemoryRangeEligibleForSwap(memRequest, memLimit int64) bool {
	return memRequest > 0 && memRequest != memLimit
}

// IsPodEligibleForSwap reports whether the pod is eligible for swap under LimitedSwap
// and returns its effective memory request. Pod overhead is excluded because KEP-2400
// treats overhead as guaranteed memory without swap.
func IsPodEligibleForSwap(pod *v1.Pod) (bool, int64) {
	if !isBurstableNonCriticalPod(pod) {
		return false, 0
	}
	opts := resourcehelper.PodResourcesOptions{
		ExcludeOverhead:       true,
		SkipPodLevelResources: !utilfeature.DefaultFeatureGate.Enabled(kubefeatures.PodLevelResources),
	}
	reqs := resourcehelper.PodRequests(pod, opts)
	limits := resourcehelper.PodLimits(pod, opts)

	podMemoryRequest := reqs.Memory().Value()
	podMemoryLimit := limits.Memory().Value()
	if !isMemoryRangeEligibleForSwap(podMemoryRequest, podMemoryLimit) {
		return false, 0
	}
	return true, podMemoryRequest
}

// IsContainerEligibleForSwap reports whether the container's pod/container resource spec
// is eligible for swap under LimitedSwap (assuming node-level LimitedSwap is enabled).
func IsContainerEligibleForSwap(pod *v1.Pod, container *v1.Container) bool {
	if container == nil || !isBurstableNonCriticalPod(pod) {
		return false
	}

	plrActive := utilfeature.DefaultFeatureGate.Enabled(kubefeatures.PodLevelResources) && resourcehelper.IsPodLevelResourcesSet(pod)
	if plrActive {
		// Ephemeral debug containers cannot specify resource requests/limits in the API,
		// and ephemeralContainerStartSpec casts EphemeralContainerCommon to v1.Container
		// with empty Resources, so guard against treating them as pod-level unbounded swap.
		for i := range pod.Spec.EphemeralContainers {
			if pod.Spec.EphemeralContainers[i].Name == container.Name {
				return false
			}
		}
		if eligible, _ := IsPodEligibleForSwap(pod); !eligible {
			return false
		}
	}

	memReq := container.Resources.Requests.Memory().Value()
	memLimit := container.Resources.Limits.Memory().Value()
	if memReq == 0 && memLimit == 0 {
		return plrActive && !pod.Spec.Resources.Requests.Memory().IsZero()
	}
	return isMemoryRangeEligibleForSwap(memReq, memLimit)
}

// hardEvictionReservation returns a resourcelist that includes reservation of resources based on hard eviction thresholds.
func hardEvictionReservation(thresholds []evictionapi.Threshold, capacity v1.ResourceList) v1.ResourceList {
	if len(thresholds) == 0 {
		return nil
	}
	ret := v1.ResourceList{}
	for _, threshold := range thresholds {
		if threshold.Operator != evictionapi.OpLessThan {
			continue
		}
		switch threshold.Signal {
		case evictionapi.SignalMemoryAvailable:
			memoryCapacity := capacity[v1.ResourceMemory]
			value := evictionapi.GetThresholdQuantity(threshold.Value, &memoryCapacity)
			ret[v1.ResourceMemory] = *value
		case evictionapi.SignalNodeFsAvailable:
			storageCapacity := capacity[v1.ResourceEphemeralStorage]
			value := evictionapi.GetThresholdQuantity(threshold.Value, &storageCapacity)
			ret[v1.ResourceEphemeralStorage] = *value
		}
	}
	return ret
}

func buildContainerMapAndRunningSetFromRuntime(ctx context.Context, runtimeService internalapi.RuntimeService) (containermap.ContainerMap, sets.Set[string]) {
	logger := klog.FromContext(ctx)
	podSandboxMap := make(map[string]string)
	podSandboxList, _ := runtimeService.ListPodSandbox(ctx, nil)
	for _, p := range podSandboxList {
		podSandboxMap[p.Id] = p.Metadata.Uid
	}

	runningSet := sets.New[string]()
	containerMap := containermap.NewContainerMap()
	containerList, _ := runtimeService.ListContainers(ctx, nil)
	for _, c := range containerList {
		if _, exists := podSandboxMap[c.PodSandboxId]; !exists {
			logger.Info("No PodSandBox found for the container", "podSandboxId", c.PodSandboxId, "containerName", c.Metadata.Name, "containerId", c.Id)
			continue
		}
		podUID := podSandboxMap[c.PodSandboxId]
		containerMap.Add(podUID, c.Metadata.Name, c.Id)
		if c.State == runtimeapi.ContainerState_CONTAINER_RUNNING {
			logger.V(4).Info("Container reported running", "podSandboxId", c.PodSandboxId, "podUID", podUID, "containerName", c.Metadata.Name, "containerId", c.Id)
			runningSet.Insert(c.Id)
		}
	}
	return containerMap, runningSet
}

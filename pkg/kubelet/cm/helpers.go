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
	"sort"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	internalapi "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/cm/containermap"
	evictionapi "k8s.io/kubernetes/pkg/kubelet/eviction/api"
)

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
	podSandboxMap := make(map[string]string)
	podSandboxList, _ := runtimeService.ListPodSandbox(ctx, nil)
	for _, p := range podSandboxList {
		podSandboxMap[p.Id] = p.Metadata.Uid
	}

	runningSet := sets.New[string]()
	containerMap := containermap.NewContainerMap()
	containerList, _ := runtimeService.ListContainers(ctx, nil)
	sort.SliceStable(containerList, func(i, j int) bool {
		return containerList[i].CreatedAt < containerList[j].CreatedAt
	})
	for _, c := range containerList {
		if _, exists := podSandboxMap[c.PodSandboxId]; !exists {
			klog.InfoS("No PodSandBox found for the container", "podSandboxId", c.PodSandboxId, "containerName", c.Metadata.Name, "containerId", c.Id)
			continue
		}
		podUID := podSandboxMap[c.PodSandboxId]
		exist, err := containerMap.GetContainerID(podUID, c.Metadata.Name)
		if err == nil {
			// ContainerMap key is containerId, value is podUID&containerName.
			// GetContainerID will iterator key-value pair, compare with value.
			// We need delete the same value to avoid get unexpected containerID.
			containerMap.RemoveByContainerID(exist)
			klog.V(4).InfoS("Remove duplicated entry from containerMap", "podUID", podUID, "containerName", c.Metadata.Name, "containerId", c.Id, "existContainerId", exist)
		}
		containerMap.Add(podUID, c.Metadata.Name, c.Id)
		if c.State == runtimeapi.ContainerState_CONTAINER_RUNNING {
			klog.V(4).InfoS("Container reported running", "podSandboxId", c.PodSandboxId, "podUID", podUID, "containerName", c.Metadata.Name, "containerId", c.Id)
			runningSet.Insert(c.Id)
		}
	}
	return containerMap, runningSet
}

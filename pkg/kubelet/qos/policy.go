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
	v1 "k8s.io/api/core/v1"
	resourceapimachinery "k8s.io/apimachinery/pkg/api/resource"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	resourcehelper "k8s.io/kubernetes/pkg/api/v1/resource"
	v1qos "k8s.io/kubernetes/pkg/apis/core/v1/helper/qos"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/types"
	schedulerutil "k8s.io/kubernetes/pkg/scheduler/util"
)

const (
	// KubeletOOMScoreAdj is the OOM score adjustment for Kubelet
	KubeletOOMScoreAdj int = -999
	// KubeProxyOOMScoreAdj is the OOM score adjustment for kube-proxy
	KubeProxyOOMScoreAdj  int = -999
	guaranteedOOMScoreAdj int = -997
	besteffortOOMScoreAdj int = 1000
)

// GetContainerOOMScoreAdjust returns the amount by which the OOM score of all processes in the
// container should be adjusted.
// The OOM score of a process is the percentage of memory it consumes
// multiplied by 10 (barring exceptional cases) + a configurable quantity which is between -1000
// and 1000. Containers with higher OOM scores are killed if the system runs out of memory.
// See https://lwn.net/Articles/391222/ for more information.
func GetContainerOOMScoreAdjust(pod *v1.Pod, container *v1.Container, memoryCapacity int64) int {
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
	containerMemoryRequest := container.Resources.Requests.Memory().Value()

	isInPlacePodVerticalScaling := utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScaling)
	if isInPlacePodVerticalScaling {
		if cs, ok := podutil.GetContainerStatus(pod.Status.ContainerStatuses, container.Name); ok {
			containerMemoryRequest = cs.AllocatedResources.Memory().Value()
		}
	}
	oomScoreAdjust := 1000 - (1000*containerMemoryRequest)/memoryCapacity

	// adapt the sidecarContainer memoryRequest for OOM ADJ calculation
	// calculate the oom score adjustment based on: min-memory( currentSideCarContainer , min-memory(regular containers) ) .
	if utilfeature.DefaultFeatureGate.Enabled(features.SidecarContainers) && isSidecarContainer(pod, container) {
		// check min resources with regular containers
		minResourceList := resourcehelper.MinRegularContainerResourceList(*pod, resourcehelper.PodResourcesOptions{
			InPlacePodVerticalScalingEnabled: isInPlacePodVerticalScaling,
			NonMissingContainerRequests: map[v1.ResourceName]resourceapimachinery.Quantity{
				v1.ResourceCPU:    *resourceapimachinery.NewMilliQuantity(schedulerutil.DefaultMilliCPURequest, resourceapimachinery.DecimalSI),
				v1.ResourceMemory: *resourceapimachinery.NewQuantity(schedulerutil.DefaultMemoryRequest, resourceapimachinery.DecimalSI),
			},
		})
		// check min resources with the current sidecarContainer
		// inPlacePodVerticalScaling is not possible for the moment in InitContainers
		minResourceList = resourcehelper.MinResourceList(container.Resources.Requests, minResourceList, resourcehelper.PodResourcesOptions{
			NonMissingContainerRequests: map[v1.ResourceName]resourceapimachinery.Quantity{
				v1.ResourceCPU:    *resourceapimachinery.NewMilliQuantity(schedulerutil.DefaultMilliCPURequest, resourceapimachinery.DecimalSI),
				v1.ResourceMemory: *resourceapimachinery.NewQuantity(schedulerutil.DefaultMemoryRequest, resourceapimachinery.DecimalSI),
			},
		})
		oomScoreAdjust = 1000 - (1000*minResourceList.Memory().Value())/memoryCapacity

		// when OOM score adj currentSideCarContainer <= Oom score adj min-memory(regular containers)
		// ==> add 1 to the sidecar container oom score adj (first to kill).
		if minResourceList.Memory().Cmp(*container.Resources.Requests.Memory()) <= 0 {
			oomScoreAdjust += 1
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

func isSidecarContainer(pod *v1.Pod, container *v1.Container) bool {
	for _, initContainer := range pod.Spec.InitContainers {
		if initContainer.Name == container.Name {
			return container.RestartPolicy != nil && *container.RestartPolicy == v1.ContainerRestartPolicyAlways
		}
	}
	return false
}

/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/controller/resourcequota"
)

const (
	DefaultDefaultContainerCPULimit = CPUShares(0.25) // CPUs allocated for pods without CPU limit
	DefaultDefaultContainerMemLimit = MegaBytes(64.0) // memory allocated for pods without memory limit
)

// CPUFromPodSpec computes the cpu shares that the pod is admitted to use. Containers
// without CPU limit are NOT taken into account.
func PodCPULimit(pod *api.Pod) CPUShares {
	cpuQuantity := resourcequotacontroller.PodCPU(pod)
	return CPUShares(float64(cpuQuantity.MilliValue()) / 1000.0)
}

// MemFromPodSpec computes the amount of memory that the pod is admitted to use. Containers
// without memory limit are NOT taken into account.
func PodMemLimit(pod *api.Pod) MegaBytes {
	memQuantity := resourcequotacontroller.PodMemory(pod)
	return MegaBytes(float64(memQuantity.Value()) / 1024.0 / 1024.0)
}

// limitPodResource sets the given default resource limit for each container that
// does not limit the given resource yet. limitPodResource returns true iff at least one
// container had no limit for that resource.
func limitPodResource(pod *api.Pod, resourceName api.ResourceName, defaultLimit resource.Quantity) bool {
	unlimited := false
	for j := range pod.Spec.Containers {
		container := &pod.Spec.Containers[j]
		if container.Resources.Limits == nil {
			container.Resources.Limits = api.ResourceList{}
		}
		_, ok := container.Resources.Limits[resourceName]
		if !ok {
			container.Resources.Limits[resourceName] = defaultLimit
			unlimited = true
		}
	}
	return unlimited
}

// unlimitedPodResources counts how many containers in the pod have no limit for the given resource
func unlimitedCountainerNum(pod *api.Pod, resourceName api.ResourceName) int {
	unlimited := 0
	for j := range pod.Spec.Containers {
		container := &pod.Spec.Containers[j]

		if container.Resources.Limits == nil {
			unlimited += 1
			continue
		}

		if _, ok := container.Resources.Limits[resourceName]; !ok {
			unlimited += 1
		}
	}
	return unlimited
}

// limitPodCPU sets DefaultContainerCPUs for the CPU limit of each container that
// does not limit its CPU resource yet. limitPodCPU returns true iff at least one
// container had no CPU limit set.
func LimitPodCPU(pod *api.Pod, defaultLimit CPUShares) bool {
	defaultCPUQuantity := resource.NewMilliQuantity(int64(float64(defaultLimit)*1000.0), resource.DecimalSI)
	return limitPodResource(pod, api.ResourceCPU, *defaultCPUQuantity)
}

// limitPodMem sets DefaultContainerMem for the memory limit of each container that
// does not limit its memory resource yet. limitPodMem returns true iff at least one
// container had no memory limit set.
func LimitPodMem(pod *api.Pod, defaultLimit MegaBytes) bool {
	defaultMemQuantity := resource.NewQuantity(int64(float64(defaultLimit)*1024.0*1024.0), resource.BinarySI)
	return limitPodResource(pod, api.ResourceMemory, *defaultMemQuantity)
}

// CPUForPod computes the limits from the spec plus the default CPU limit for unlimited containers
func CPUForPod(pod *api.Pod, defaultLimit CPUShares) CPUShares {
	return PodCPULimit(pod) + CPUShares(unlimitedCountainerNum(pod, api.ResourceCPU))*defaultLimit
}

// MemForPod computes the limits from the spec plus the default memory limit for unlimited containers
func MemForPod(pod *api.Pod, defaultLimit MegaBytes) MegaBytes {
	return PodMemLimit(pod) + MegaBytes(unlimitedCountainerNum(pod, api.ResourceMemory))*defaultLimit
}

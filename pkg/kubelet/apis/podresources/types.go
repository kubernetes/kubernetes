/*
Copyright 2020 The Kubernetes Authors.

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

//go:generate mockery
package podresources

import (
	v1 "k8s.io/api/core/v1"
	podresourcesapi "k8s.io/kubelet/pkg/apis/podresources/v1"
)

// DevicesProvider knows how to provide the devices used by the given container
type DevicesProvider interface {
	// UpdateAllocatedDevices frees any Devices that are bound to terminated pods.
	UpdateAllocatedDevices()
	// GetDevices returns information about the devices assigned to pods and containers
	GetDevices(podUID, containerName string) []*podresourcesapi.ContainerDevices
	// GetAllocatableDevices returns information about all the devices known to the manager
	GetAllocatableDevices() []*podresourcesapi.ContainerDevices
}

// PodsProvider knows how to provide the pods admitted by the node
type PodsProvider interface {
	GetActivePods() []*v1.Pod
	GetPods() []*v1.Pod
	GetPodByName(namespace, name string) (*v1.Pod, bool)
}

// CPUsProvider knows how to provide the cpus used by the given container
type CPUsProvider interface {
	// GetCPUs returns information about the cpus assigned to pods and containers
	GetCPUs(podUID, containerName string) []int64
	// GetAllocatableCPUs returns the allocatable (not allocated) CPUs
	GetAllocatableCPUs() []int64
}

type MemoryProvider interface {
	// GetMemory returns information about the memory assigned to containers
	GetMemory(podUID, containerName string) []*podresourcesapi.ContainerMemory
	// GetAllocatableMemory returns the allocatable memory from the node
	GetAllocatableMemory() []*podresourcesapi.ContainerMemory
}

type DynamicResourcesProvider interface {
	// GetDynamicResources returns information about dynamic resources assigned to pods and containers
	GetDynamicResources(pod *v1.Pod, container *v1.Container) []*podresourcesapi.DynamicResource
}

type PodResourcesProviders struct {
	Pods             PodsProvider
	Devices          DevicesProvider
	Cpus             CPUsProvider
	Memory           MemoryProvider
	DynamicResources DynamicResourcesProvider
}

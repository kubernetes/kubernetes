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

package podresources

import (
	"context"

	"k8s.io/api/core/v1"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
	podresourcesapi "k8s.io/kubelet/pkg/apis/podresources/v1alpha1"
)

// DevicesProvider knows how to provide the devices used by the given container
type DevicesProvider interface {
	GetDevices(podUID, containerName string) []*podresourcesapi.ContainerDevices
	UpdateAllocatedDevices()
	GetAllDevices() map[string]map[string]pluginapi.Device
}

// CPUsProvider knows how to provide the cpus accounting
type CPUsProvider interface {
	GetAllCPUs() []int64
}

// PodsProvider knows how to provide the pods admitted by the node
type PodsProvider interface {
	GetPods() []*v1.Pod
}

// podResourcesServer implements PodResourcesListerServer
type podResourcesServer struct {
	podsProvider    PodsProvider
	devicesProvider DevicesProvider
	cpusProvider    CPUsProvider
}

// NewPodResourcesServer returns a PodResourcesListerServer which lists pods provided by the PodsProvider
// with device information provided by the DevicesProvider
func NewPodResourcesServer(podsProvider PodsProvider, devicesProvider DevicesProvider, cpusProvider CPUsProvider) podresourcesapi.PodResourcesListerServer {
	return &podResourcesServer{
		podsProvider:    podsProvider,
		devicesProvider: devicesProvider,
		cpusProvider:    cpusProvider,
	}
}

// List returns information about the resources assigned to pods on the node
func (p *podResourcesServer) List(ctx context.Context, req *podresourcesapi.ListPodResourcesRequest) (*podresourcesapi.ListPodResourcesResponse, error) {
	pods := p.podsProvider.GetPods()
	podResources := make([]*podresourcesapi.PodResources, len(pods))
	p.devicesProvider.UpdateAllocatedDevices()

	for i, pod := range pods {
		pRes := podresourcesapi.PodResources{
			Name:       pod.Name,
			Namespace:  pod.Namespace,
			Containers: make([]*podresourcesapi.ContainerResources, len(pod.Spec.Containers)),
		}

		for j, container := range pod.Spec.Containers {
			pRes.Containers[j] = &podresourcesapi.ContainerResources{
				Name:    container.Name,
				Devices: p.devicesProvider.GetDevices(string(pod.UID), container.Name),
			}
		}
		podResources[i] = &pRes
	}

	return &podresourcesapi.ListPodResourcesResponse{
		PodResources: podResources,
	}, nil
}

// GetAllocatableResources returns information about all the devices known by the server
func (p *podResourcesServer) GetAllocatableResources(context.Context, *podresourcesapi.AllocatableResourcesRequest) (*podresourcesapi.AllocatableResourcesResponse, error) {
	allDevices := p.devicesProvider.GetAllDevices()
	var respDevs []*podresourcesapi.ContainerDevices

	for resourceName, resourceDevs := range allDevices {
		// group devices by NUMA node
		NUMADeviceIDs := map[int64][]string{}
		for devID, dev := range resourceDevs {
			for _, node := range dev.GetTopology().GetNodes() {
				numaNode := node.GetID()
				NUMADeviceIDs[numaNode] = append(NUMADeviceIDs[numaNode], devID)
			}
		}

		for numaNode, devIDs := range NUMADeviceIDs {
			respDevs = append(respDevs, &podresourcesapi.ContainerDevices{
				ResourceName: resourceName,
				DeviceIds:    devIDs, // these are all the instances of this resources on numaNode
				Topology: &podresourcesapi.TopologyInfo{
					Nodes: []*podresourcesapi.NUMANode{
						{ID: numaNode},
					},
				},
			})
		}

	}

	return &podresourcesapi.AllocatableResourcesResponse{
		Devices: respDevs,
		CpuIds:  p.cpusProvider.GetAllCPUs(),
	}, nil
}

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
	"fmt"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/metrics"

	"k8s.io/kubelet/pkg/apis/podresources/v1"
)

// v1PodResourcesServer implements PodResourcesListerServer
type v1PodResourcesServer struct {
	podsProvider             PodsProvider
	devicesProvider          DevicesProvider
	cpusProvider             CPUsProvider
	memoryProvider           MemoryProvider
	dynamicResourcesProvider DynamicResourcesProvider
}

// NewV1PodResourcesServer returns a PodResourcesListerServer which lists pods provided by the PodsProvider
// with device information provided by the DevicesProvider
func NewV1PodResourcesServer(providers PodResourcesProviders) v1.PodResourcesListerServer {
	return &v1PodResourcesServer{
		podsProvider:             providers.Pods,
		devicesProvider:          providers.Devices,
		cpusProvider:             providers.Cpus,
		memoryProvider:           providers.Memory,
		dynamicResourcesProvider: providers.DynamicResources,
	}
}

// List returns information about the resources assigned to pods on the node
func (p *v1PodResourcesServer) List(ctx context.Context, req *v1.ListPodResourcesRequest) (*v1.ListPodResourcesResponse, error) {
	metrics.PodResourcesEndpointRequestsTotalCount.WithLabelValues("v1").Inc()
	metrics.PodResourcesEndpointRequestsListCount.WithLabelValues("v1").Inc()

	pods := p.podsProvider.GetPods()
	podResources := make([]*v1.PodResources, len(pods))
	p.devicesProvider.UpdateAllocatedDevices()

	for i, pod := range pods {
		pRes := v1.PodResources{
			Name:       pod.Name,
			Namespace:  pod.Namespace,
			Containers: make([]*v1.ContainerResources, len(pod.Spec.Containers)),
		}

		for j, container := range pod.Spec.Containers {
			pRes.Containers[j] = &v1.ContainerResources{
				Name:    container.Name,
				Devices: p.devicesProvider.GetDevices(string(pod.UID), container.Name),
				CpuIds:  p.cpusProvider.GetCPUs(string(pod.UID), container.Name),
				Memory:  p.memoryProvider.GetMemory(string(pod.UID), container.Name),
			}
			if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.KubeletPodResourcesDynamicResources) {
				pRes.Containers[j].DynamicResources = p.dynamicResourcesProvider.GetDynamicResources(pod, &container)
			}

		}
		podResources[i] = &pRes
	}

	response := &v1.ListPodResourcesResponse{
		PodResources: podResources,
	}
	return response, nil
}

// GetAllocatableResources returns information about all the resources known by the server - this more like the capacity, not like the current amount of free resources.
func (p *v1PodResourcesServer) GetAllocatableResources(ctx context.Context, req *v1.AllocatableResourcesRequest) (*v1.AllocatableResourcesResponse, error) {
	metrics.PodResourcesEndpointRequestsTotalCount.WithLabelValues("v1").Inc()
	metrics.PodResourcesEndpointRequestsGetAllocatableCount.WithLabelValues("v1").Inc()

	response := &v1.AllocatableResourcesResponse{
		Devices: p.devicesProvider.GetAllocatableDevices(),
		CpuIds:  p.cpusProvider.GetAllocatableCPUs(),
		Memory:  p.memoryProvider.GetAllocatableMemory(),
	}

	return response, nil
}

// Get returns information about the resources assigned to a specific pod
func (p *v1PodResourcesServer) Get(ctx context.Context, req *v1.GetPodResourcesRequest) (*v1.GetPodResourcesResponse, error) {
	metrics.PodResourcesEndpointRequestsTotalCount.WithLabelValues("v1").Inc()
	metrics.PodResourcesEndpointRequestsGetCount.WithLabelValues("v1").Inc()

	if !utilfeature.DefaultFeatureGate.Enabled(kubefeatures.KubeletPodResourcesGet) {
		metrics.PodResourcesEndpointErrorsGetCount.WithLabelValues("v1").Inc()
		return nil, fmt.Errorf("PodResources API Get method disabled")
	}

	pod, exist := p.podsProvider.GetPodByName(req.PodNamespace, req.PodName)
	if !exist {
		metrics.PodResourcesEndpointErrorsGetCount.WithLabelValues("v1").Inc()
		return nil, fmt.Errorf("pod %s in namespace %s not found", req.PodName, req.PodNamespace)
	}

	podResources := &v1.PodResources{
		Name:       pod.Name,
		Namespace:  pod.Namespace,
		Containers: make([]*v1.ContainerResources, len(pod.Spec.Containers)),
	}

	for i, container := range pod.Spec.Containers {
		podResources.Containers[i] = &v1.ContainerResources{
			Name:    container.Name,
			Devices: p.devicesProvider.GetDevices(string(pod.UID), container.Name),
			CpuIds:  p.cpusProvider.GetCPUs(string(pod.UID), container.Name),
			Memory:  p.memoryProvider.GetMemory(string(pod.UID), container.Name),
		}
		if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.KubeletPodResourcesDynamicResources) {
			podResources.Containers[i].DynamicResources = p.dynamicResourcesProvider.GetDynamicResources(pod, &container)
		}
	}

	response := &v1.GetPodResourcesResponse{
		PodResources: podResources,
	}
	return response, nil
}

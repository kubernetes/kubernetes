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

	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/metrics"

	podresourcesv1 "k8s.io/kubelet/pkg/apis/podresources/v1"
)

// v1PodResourcesServer implements PodResourcesListerServer
type v1PodResourcesServer struct {
	podsProvider             PodsProvider
	devicesProvider          DevicesProvider
	cpusProvider             CPUsProvider
	memoryProvider           MemoryProvider
	dynamicResourcesProvider DynamicResourcesProvider
	useActivePods            bool
}

// NewV1PodResourcesServer returns a PodResourcesListerServer which lists pods provided by the PodsProvider
// with device information provided by the DevicesProvider
func NewV1PodResourcesServer(providers PodResourcesProviders) podresourcesv1.PodResourcesListerServer {
	useActivePods := true
	klog.InfoS("podresources", "method", "list", "useActivePods", useActivePods)
	return &v1PodResourcesServer{
		podsProvider:             providers.Pods,
		devicesProvider:          providers.Devices,
		cpusProvider:             providers.Cpus,
		memoryProvider:           providers.Memory,
		dynamicResourcesProvider: providers.DynamicResources,
		useActivePods:            useActivePods,
	}
}

// List returns information about the resources assigned to pods on the node
func (p *v1PodResourcesServer) List(ctx context.Context, req *podresourcesv1.ListPodResourcesRequest) (*podresourcesv1.ListPodResourcesResponse, error) {
	metrics.PodResourcesEndpointRequestsTotalCount.WithLabelValues("v1").Inc()
	metrics.PodResourcesEndpointRequestsListCount.WithLabelValues("v1").Inc()

	var pods []*v1.Pod
	if p.useActivePods {
		pods = p.podsProvider.GetActivePods()
	} else {
		pods = p.podsProvider.GetPods()
	}

	podResources := make([]*podresourcesv1.PodResources, len(pods))
	p.devicesProvider.UpdateAllocatedDevices()

	for i, pod := range pods {
		pRes := podresourcesv1.PodResources{
			Name:       pod.Name,
			Namespace:  pod.Namespace,
			Containers: make([]*podresourcesv1.ContainerResources, 0, len(pod.Spec.Containers)),
		}

		pRes.Containers = make([]*podresourcesv1.ContainerResources, 0, len(pod.Spec.InitContainers)+len(pod.Spec.Containers))
		for _, container := range pod.Spec.InitContainers {
			if !podutil.IsRestartableInitContainer(&container) {
				continue
			}

			pRes.Containers = append(pRes.Containers, p.getContainerResources(pod, &container))
		}

		for _, container := range pod.Spec.Containers {
			pRes.Containers = append(pRes.Containers, p.getContainerResources(pod, &container))
		}
		podResources[i] = &pRes
	}

	response := &podresourcesv1.ListPodResourcesResponse{
		PodResources: podResources,
	}
	return response, nil
}

// GetAllocatableResources returns information about all the resources known by the server - this more like the capacity, not like the current amount of free resources.
func (p *v1PodResourcesServer) GetAllocatableResources(ctx context.Context, req *podresourcesv1.AllocatableResourcesRequest) (*podresourcesv1.AllocatableResourcesResponse, error) {
	metrics.PodResourcesEndpointRequestsTotalCount.WithLabelValues("v1").Inc()
	metrics.PodResourcesEndpointRequestsGetAllocatableCount.WithLabelValues("v1").Inc()

	response := &podresourcesv1.AllocatableResourcesResponse{
		Devices: p.devicesProvider.GetAllocatableDevices(),
		CpuIds:  p.cpusProvider.GetAllocatableCPUs(),
		Memory:  p.memoryProvider.GetAllocatableMemory(),
	}

	return response, nil
}

// Get returns information about the resources assigned to a specific pod
func (p *v1PodResourcesServer) Get(ctx context.Context, req *podresourcesv1.GetPodResourcesRequest) (*podresourcesv1.GetPodResourcesResponse, error) {
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

	podResources := &podresourcesv1.PodResources{
		Name:       pod.Name,
		Namespace:  pod.Namespace,
		Containers: make([]*podresourcesv1.ContainerResources, 0, len(pod.Spec.Containers)),
	}

	podResources.Containers = make([]*podresourcesv1.ContainerResources, 0, len(pod.Spec.InitContainers)+len(pod.Spec.Containers))
	for _, container := range pod.Spec.InitContainers {
		if !podutil.IsRestartableInitContainer(&container) {
			continue
		}

		podResources.Containers = append(podResources.Containers, p.getContainerResources(pod, &container))
	}

	for _, container := range pod.Spec.Containers {
		podResources.Containers = append(podResources.Containers, p.getContainerResources(pod, &container))
	}

	response := &podresourcesv1.GetPodResourcesResponse{
		PodResources: podResources,
	}
	return response, nil
}

func (p *v1PodResourcesServer) getContainerResources(pod *v1.Pod, container *v1.Container) *podresourcesv1.ContainerResources {
	containerResources := &podresourcesv1.ContainerResources{
		Name:    container.Name,
		Devices: p.devicesProvider.GetDevices(string(pod.UID), container.Name),
		CpuIds:  p.cpusProvider.GetCPUs(string(pod.UID), container.Name),
		Memory:  p.memoryProvider.GetMemory(string(pod.UID), container.Name),
	}
	if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.KubeletPodResourcesDynamicResources) {
		containerResources.DynamicResources = p.dynamicResourcesProvider.GetDynamicResources(pod, container)
	}

	return containerResources
}

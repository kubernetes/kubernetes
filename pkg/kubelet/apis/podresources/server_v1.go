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

	"k8s.io/kubernetes/pkg/kubelet/metrics"

	"k8s.io/kubelet/pkg/apis/podresources/v1"
)

// podResourcesServerV1alpha1 implements PodResourcesListerServer
type v1PodResourcesServer struct {
	podsProvider    PodsProvider
	devicesProvider DevicesProvider
	cpusProvider    CPUsProvider
}

// NewV1PodResourcesServer returns a PodResourcesListerServer which lists pods provided by the PodsProvider
// with device information provided by the DevicesProvider
func NewV1PodResourcesServer(podsProvider PodsProvider, devicesProvider DevicesProvider, cpusProvider CPUsProvider) v1.PodResourcesListerServer {
	return &v1PodResourcesServer{
		podsProvider:    podsProvider,
		devicesProvider: devicesProvider,
		cpusProvider:    cpusProvider,
	}
}

// List returns information about the resources assigned to pods on the node
func (p *v1PodResourcesServer) List(ctx context.Context, req *v1.ListPodResourcesRequest) (*v1.ListPodResourcesResponse, error) {
	metrics.PodResourcesEndpointRequestsTotalCount.WithLabelValues("v1").Inc()

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
			}
		}
		podResources[i] = &pRes
	}

	return &v1.ListPodResourcesResponse{
		PodResources: podResources,
	}, nil
}

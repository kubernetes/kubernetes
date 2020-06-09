// +build !dockerless

/*
Copyright 2019 The Kubernetes Authors.

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

package dockershim

import (
	"context"

	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
)

// ContainerStats returns stats for a container stats request based on container id.
func (ds *dockerService) ContainerStats(_ context.Context, r *runtimeapi.ContainerStatsRequest) (*runtimeapi.ContainerStatsResponse, error) {
	stats, err := ds.getContainerStats(r.ContainerId)
	if err != nil {
		return nil, err
	}
	return &runtimeapi.ContainerStatsResponse{Stats: stats}, nil
}

// ListContainerStats returns stats for a list container stats request based on a filter.
func (ds *dockerService) ListContainerStats(ctx context.Context, r *runtimeapi.ListContainerStatsRequest) (*runtimeapi.ListContainerStatsResponse, error) {
	containerStatsFilter := r.GetFilter()
	filter := &runtimeapi.ContainerFilter{}

	if containerStatsFilter != nil {
		filter.Id = containerStatsFilter.Id
		filter.PodSandboxId = containerStatsFilter.PodSandboxId
		filter.LabelSelector = containerStatsFilter.LabelSelector
	}

	listResp, err := ds.ListContainers(ctx, &runtimeapi.ListContainersRequest{Filter: filter})
	if err != nil {
		return nil, err
	}

	var stats []*runtimeapi.ContainerStats
	for _, container := range listResp.Containers {
		containerStats, err := ds.getContainerStats(container.Id)
		if err != nil {
			return nil, err
		}
		if containerStats != nil {
			stats = append(stats, containerStats)
		}
	}

	return &runtimeapi.ListContainerStatsResponse{Stats: stats}, nil
}

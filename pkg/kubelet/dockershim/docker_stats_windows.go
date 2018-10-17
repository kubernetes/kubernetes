// +build windows

/*
Copyright 2017 The Kubernetes Authors.

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
	"time"

	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/runtime/v1alpha2"
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

		stats = append(stats, containerStats)
	}

	return &runtimeapi.ListContainerStatsResponse{Stats: stats}, nil
}

func (ds *dockerService) getContainerStats(containerID string) (*runtimeapi.ContainerStats, error) {
	info, err := ds.client.Info()
	if err != nil {
		return nil, err
	}

	statsJSON, err := ds.client.GetContainerStats(containerID)
	if err != nil {
		return nil, err
	}

	containerJSON, err := ds.client.InspectContainerWithSize(containerID)
	if err != nil {
		return nil, err
	}

	statusResp, err := ds.ContainerStatus(context.Background(), &runtimeapi.ContainerStatusRequest{ContainerId: containerID})
	if err != nil {
		return nil, err
	}
	status := statusResp.GetStatus()

	dockerStats := statsJSON.Stats
	timestamp := time.Now().UnixNano()
	containerStats := &runtimeapi.ContainerStats{
		Attributes: &runtimeapi.ContainerAttributes{
			Id:          containerID,
			Metadata:    status.Metadata,
			Labels:      status.Labels,
			Annotations: status.Annotations,
		},
		Cpu: &runtimeapi.CpuUsage{
			Timestamp: timestamp,
			// have to multiply cpu usage by 100 since docker stats units is in 100's of nano seconds for Windows
			// see https://github.com/moby/moby/blob/v1.13.1/api/types/stats.go#L22
			UsageCoreNanoSeconds: &runtimeapi.UInt64Value{Value: dockerStats.CPUStats.CPUUsage.TotalUsage * 100},
		},
		Memory: &runtimeapi.MemoryUsage{
			Timestamp:       timestamp,
			WorkingSetBytes: &runtimeapi.UInt64Value{Value: dockerStats.MemoryStats.PrivateWorkingSet},
		},
		WritableLayer: &runtimeapi.FilesystemUsage{
			Timestamp: timestamp,
			FsId:      &runtimeapi.FilesystemIdentifier{Mountpoint: info.DockerRootDir},
			UsedBytes: &runtimeapi.UInt64Value{Value: uint64(*containerJSON.SizeRw)},
		},
	}
	return containerStats, nil
}

// ListPodSandboxStats returns stats of all running PodSandboxes. Only for Windows.
func (ds *dockerService) ListPodSandboxStats(ctx context.Context, r *runtimeapi.ListPodSandboxStatsRequest) (*runtimeapi.ListPodSandboxStatsResponse, error) {
	listFilter := &runtimeapi.PodSandboxFilter{
		State: &runtimeapi.PodSandboxStateValue{State: runtimeapi.PodSandboxState_SANDBOX_READY},
	}
	if r.Filter != nil {
		listFilter.Id = r.Filter.Id
		listFilter.LabelSelector = r.Filter.LabelSelector
	}
	sandboxResp, err := ds.ListPodSandbox(ctx, &runtimeapi.ListPodSandboxRequest{
		Filter: listFilter,
	})
	if err != nil {
		return nil, err
	}

	var stats []*runtimeapi.PodSandboxStats
	for _, sandbox := range sandboxResp.Items {
		sandboxStats, err := ds.getPodSandboxStats(sandbox)
		if err != nil {
			return nil, err
		}

		stats = append(stats, sandboxStats)
	}

	return &runtimeapi.ListPodSandboxStatsResponse{Stats: stats}, nil
}

func (ds *dockerService) getPodSandboxStats(sandbox *runtimeapi.PodSandbox) (*runtimeapi.PodSandboxStats, error) {
	containerID := sandbox.Id
	statsJSON, err := ds.client.GetContainerStats(containerID)
	if err != nil {
		return nil, err
	}

	timestamp := time.Now().UnixNano()
	sandboxStat := &runtimeapi.PodSandboxStats{
		Id:       sandbox.Id,
		Metadata: sandbox.Metadata,
		Network: &runtimeapi.NetworkUsage{
			Timestamp:  timestamp,
			Interfaces: make([]*runtimeapi.InterfaceStats, 0),
		},
	}
	for name, stat := range statsJSON.Networks {
		criInterfaceStat := &runtimeapi.InterfaceStats{
			Name:     name,
			RxBytes:  &runtimeapi.UInt64Value{Value: stat.RxBytes},
			RxErrors: &runtimeapi.UInt64Value{Value: stat.RxErrors},
			TxBytes:  &runtimeapi.UInt64Value{Value: stat.TxBytes},
			TxErrors: &runtimeapi.UInt64Value{Value: stat.TxErrors},
		}
		// TODO(feiskyer): add support of multiple interfaces for getting default interface.
		if len(statsJSON.Networks) == 1 {
			sandboxStat.Network.Default = criInterfaceStat
		}
		sandboxStat.Network.Interfaces = append(sandboxStat.Network.Interfaces, criInterfaceStat)
	}

	return sandboxStat, nil
}

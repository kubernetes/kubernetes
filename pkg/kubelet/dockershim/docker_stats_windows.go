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

	"github.com/Microsoft/hcsshim"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	"k8s.io/klog"
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

	hcsshim_container, err := hcsshim.OpenContainer(containerID)
	if err != nil {
		return nil, err
	}
	defer func() {
		closeErr := hcsshim_container.Close()
		if closeErr != nil {
			klog.Errorf("Error closing container '%s': %v", containerID, closeErr)
		}
	}()

	stats, err := hcsshim_container.Statistics()
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
			// have to multiply cpu usage by 100 since stats units is in 100's of nano seconds for Windows
			UsageCoreNanoSeconds: &runtimeapi.UInt64Value{Value: stats.Processor.TotalRuntime100ns * 100},
		},
		Memory: &runtimeapi.MemoryUsage{
			Timestamp:       timestamp,
			WorkingSetBytes: &runtimeapi.UInt64Value{Value: stats.Memory.UsagePrivateWorkingSetBytes},
		},
		WritableLayer: &runtimeapi.FilesystemUsage{
			Timestamp: timestamp,
			FsId:      &runtimeapi.FilesystemIdentifier{Mountpoint: info.DockerRootDir},
			UsedBytes: &runtimeapi.UInt64Value{Value: uint64(*containerJSON.SizeRw)},
		},
	}
	return containerStats, nil
}

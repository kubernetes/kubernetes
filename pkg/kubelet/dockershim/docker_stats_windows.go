// +build windows,!dockerless

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
	"strings"
	"time"

	"github.com/Microsoft/hcsshim"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	"k8s.io/klog/v2"
)

func (ds *dockerService) getContainerStats(c *runtimeapi.Container) (*runtimeapi.ContainerStats, error) {
	hcsshimContainer, err := hcsshim.OpenContainer(c.Id)
	if err != nil {
		// As we moved from using Docker stats to hcsshim directly, we may query HCS with already exited container IDs.
		// That will typically happen with init-containers in Exited state. Docker still knows about them but the HCS does not.
		// As we don't want to block stats retrieval for other containers, we only log errors.
		if !hcsshim.IsNotExist(err) && !hcsshim.IsAlreadyStopped(err) {
			klog.V(4).InfoS("Error opening container (stats will be missing)", "containerID", c.Id, "err", err)
		}
		return nil, nil
	}
	defer func() {
		closeErr := hcsshimContainer.Close()
		if closeErr != nil {
			klog.ErrorS(closeErr, "Error closing container", "containerID", c.Id)
		}
	}()

	stats, err := hcsshimContainer.Statistics()
	if err != nil {
		if strings.Contains(err.Error(), "0x5") || strings.Contains(err.Error(), "0xc0370105") {
			// When the container is just created, querying for stats causes access errors because it hasn't started yet
			// This is transient; skip container for now
			//
			// These hcs errors do not have helpers exposed in public package so need to query for the known codes
			// https://github.com/microsoft/hcsshim/blob/master/internal/hcs/errors.go
			// PR to expose helpers in hcsshim: https://github.com/microsoft/hcsshim/pull/933
			klog.V(4).InfoS("Container is not in a state that stats can be accessed. This occurs when the container is created but not started.", "containerID", c.Id, "err", err)
			return nil, nil
		}
		return nil, err
	}

	timestamp := time.Now().UnixNano()
	containerStats := &runtimeapi.ContainerStats{
		Attributes: &runtimeapi.ContainerAttributes{
			Id:          c.Id,
			Metadata:    c.Metadata,
			Labels:      c.Labels,
			Annotations: c.Annotations,
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
			FsId:      &runtimeapi.FilesystemIdentifier{Mountpoint: ds.dockerRootDir},
			// used bytes from image are not implemented on Windows
			// don't query for it since it is expensive to call docker over named pipe
			// https://github.com/moby/moby/blob/1ba54a5fd0ba293db3bea46cd67604b593f2048b/daemon/images/image_windows.go#L11-L14
			UsedBytes: &runtimeapi.UInt64Value{Value: 0},
		},
	}
	return containerStats, nil
}

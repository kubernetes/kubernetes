// Copyright 2024 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package manager

import (
	"fmt"

	"github.com/google/cadvisor/lib/cache/memory"
	info "github.com/google/cadvisor/lib/model"

	"k8s.io/klog/v2"
)

// This file restores the read/query surface of the manager that the full
// cAdvisor binary's v1/v2 REST API and web UI need. The lean kubelet core
// (manager.go) does not use any of it, but none of these methods add
// dependencies: they are pure queries over the in-memory container registry
// and return model types. Methods that the upstream manager returned as
// info/v2 types instead return their model equivalents (model is a superset of
// v1); the root binary converts model -> v2 in its REST handlers, because the
// library cannot import info/v2 (that would be a dependency cycle).

func (m *manager) GetContainerInfo(containerName string, query *info.ContainerInfoRequest) (*info.ContainerInfo, error) {
	cont, err := m.getContainer(containerName)
	if err != nil {
		return nil, err
	}
	return m.containerDataToContainerInfo(cont, query)
}

func (m *manager) SubcontainersInfo(containerName string, query *info.ContainerInfoRequest) ([]*info.ContainerInfo, error) {
	containersMap := m.getSubcontainers(containerName)

	containers := make([]*containerData, 0, len(containersMap))
	for _, cont := range containersMap {
		containers = append(containers, cont)
	}
	return m.containerDataSliceToContainerInfoSlice(containers, query)
}

func (m *manager) AllDockerContainers(query *info.ContainerInfoRequest) (map[string]info.ContainerInfo, error) {
	containers := m.getAllNamespacedContainers(DockerNamespace)
	return m.containersInfo(containers, query)
}

func (m *manager) DockerContainer(containerName string, query *info.ContainerInfoRequest) (info.ContainerInfo, error) {
	container, err := m.namespacedContainer(containerName, DockerNamespace)
	if err != nil {
		return info.ContainerInfo{}, err
	}

	inf, err := m.containerDataToContainerInfo(container, query)
	if err != nil {
		return info.ContainerInfo{}, err
	}
	return *inf, nil
}

func (m *manager) AllPodmanContainers(query *info.ContainerInfoRequest) (map[string]info.ContainerInfo, error) {
	containers := m.getAllNamespacedContainers(PodmanNamespace)
	return m.containersInfo(containers, query)
}

func (m *manager) PodmanContainer(containerName string, query *info.ContainerInfoRequest) (info.ContainerInfo, error) {
	container, err := m.namespacedContainer(containerName, PodmanNamespace)
	if err != nil {
		return info.ContainerInfo{}, err
	}

	inf, err := m.containerDataToContainerInfo(container, query)
	if err != nil {
		return info.ContainerInfo{}, err
	}
	return *inf, nil
}

// GetContainerSpec returns model specs; the root binary converts each to
// v2.ContainerSpec (adding aliases/namespace from the container reference) in
// its REST handler.
func (m *manager) GetContainerSpec(containerName string, options info.RequestOptions) (map[string]info.ContainerSpec, error) {
	conts, err := m.getRequestedContainers(containerName, options)
	if err != nil {
		return nil, err
	}
	var errs partialFailure
	specs := make(map[string]info.ContainerSpec)
	for name, cont := range conts {
		cinfo, err := cont.GetInfo(false)
		if err != nil {
			errs.append(name, "GetInfo", err)
		}
		spec := m.getAdjustedSpec(cinfo)
		specs[name] = spec
	}
	return specs, errs.OrNil()
}

// GetFsInfoByFsUUID returns model.FsInfo, which the root binary's
// info/v2 aliases as v2.FsInfo (identical shape) — no conversion needed.
func (m *manager) GetFsInfoByFsUUID(uuid string) (info.FsInfo, error) {
	device, err := m.fsInfo.GetDeviceInfoByFsUUID(uuid)
	if err != nil {
		return info.FsInfo{}, err
	}
	return m.getFsInfoByDeviceName(device.Device)
}

func (m *manager) Exists(containerName string) bool {
	_, ok := m.containers.Load(namespacedContainerName{Name: containerName})
	return ok
}

func (m *manager) containersInfo(containers map[string]*containerData, query *info.ContainerInfoRequest) (map[string]info.ContainerInfo, error) {
	output := make(map[string]info.ContainerInfo, len(containers))
	for name, cont := range containers {
		inf, err := m.containerDataToContainerInfo(cont, query)
		if err != nil {
			// Ignore the error because of race condition and return best-effort result.
			if err == memory.ErrDataNotFound {
				klog.V(4).Infof("Error getting data for container %s because of race condition", name)
				continue
			}
			return nil, err
		}
		output[name] = *inf
	}
	return output, nil
}

// GetDerivedStats returns per-window usage percentiles for a container. The
// values come from each container's summary reader, which the root binary
// injects via SummaryReaderFactory; with no reader (the kubelet) each container
// reports a "derived stats not enabled" error.
func (m *manager) GetDerivedStats(containerName string, options info.RequestOptions) (map[string]info.DerivedStats, error) {
	conts, err := m.getRequestedContainers(containerName, options)
	if err != nil {
		return nil, err
	}
	var errs partialFailure
	stats := make(map[string]info.DerivedStats)
	for name, cont := range conts {
		d, err := cont.DerivedStats()
		if err != nil {
			errs.append(name, "DerivedStats", err)
		}
		stats[name] = d
	}
	return stats, errs.OrNil()
}

// GetProcessList returns the processes running in a container. The listing is
// done by an injected ProcessListProvider (the root binary's ps-based
// implementation); the kubelet leaves it nil and gets an empty list.
func (m *manager) GetProcessList(containerName string, options info.RequestOptions) ([]info.ProcessInfo, error) {
	if ProcessListProvider == nil {
		return []info.ProcessInfo{}, nil
	}
	// Only single-container listing is supported.
	options.Recursive = false
	conts, err := m.getRequestedContainers(containerName, options)
	if err != nil {
		return nil, err
	}
	if len(conts) != 1 {
		return nil, fmt.Errorf("expected the request to match only one container")
	}
	var cont *containerData
	for _, c := range conts {
		cont = c
	}
	return ProcessListProvider(cont.info.Name, cont.info.Name == "/", m.cadvisorContainer, m.inHostNamespace)
}

func (m *manager) containerDataSliceToContainerInfoSlice(containers []*containerData, query *info.ContainerInfoRequest) ([]*info.ContainerInfo, error) {
	if len(containers) == 0 {
		return nil, fmt.Errorf("no containers found")
	}

	// Get the info for each container.
	output := make([]*info.ContainerInfo, 0, len(containers))
	for i := range containers {
		cinfo, err := m.containerDataToContainerInfo(containers[i], query)
		if err != nil {
			// Skip containers with errors, we try to degrade gracefully.
			klog.V(4).Infof("convert container data to container info failed with error %s", err.Error())
			continue
		}
		output = append(output, cinfo)
	}

	return output, nil
}

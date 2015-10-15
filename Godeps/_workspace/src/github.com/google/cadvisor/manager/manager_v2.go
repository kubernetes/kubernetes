// Copyright 2015 Google Inc. All Rights Reserved.
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
	"strings"
	"time"

	"github.com/google/cadvisor/info/conversion"
	"github.com/google/cadvisor/info/v2"
)

// TODO: Export this interface once we're ready to make v2 the default interface.
type v2Manager interface {
	// Gets info for all containers based on request options.
	GetContainerInfo(containerName string, options v2.RequestOptions) (map[string]v2.ContainerInfo, error)

	// Gets spec for all containers based on request options.
	GetContainerSpec(containerName string, options v2.RequestOptions) (map[string]v2.ContainerSpec, error)

	// Gets summary stats for all containers based on request options.
	GetDerivedStats(containerName string, options v2.RequestOptions) (map[string]v2.DerivedStats, error)

	// Get filesystem information for a given label.
	// Returns information for all global filesystems if label is empty.
	GetFsInfo(label string) ([]v2.FsInfo, error)

	// Get ps output for a container.
	GetProcessList(containerName string, options v2.RequestOptions) ([]v2.ProcessInfo, error)
}

type v2manager manager

// v1 provides internal access to v1 methods.
func (m *v2manager) v1() *manager {
	return (*manager)(m)
}

func (m *v2manager) GetContainerInfo(containerName string, options v2.RequestOptions) (map[string]v2.ContainerInfo, error) {
	containers, err := m.getRequestedContainers(containerName, options)
	if err != nil {
		return nil, err
	}

	infos := make(map[string]v2.ContainerInfo, len(containers))
	for name, container := range containers {
		cinfo, err := container.GetInfo()
		if err != nil {
			return nil, err
		}

		stats, err := m.memoryCache.RecentStats(name, options.Start, options.End, options.Count)
		if err != nil {
			return nil, err
		}

		infos[name] = v2.ContainerInfo{
			Spec:  m.getV2Spec(cinfo),
			Stats: conversion.ConvertStats(stats, &cinfo.Spec),
		}
	}

	return infos, nil
}

func (m *v2manager) GetDerivedStats(containerName string, options v2.RequestOptions) (map[string]v2.DerivedStats, error) {
	conts, err := m.getRequestedContainers(containerName, options)
	if err != nil {
		return nil, err
	}
	stats := make(map[string]v2.DerivedStats)
	for name, cont := range conts {
		d, err := cont.DerivedStats()
		if err != nil {
			return nil, err
		}
		stats[name] = d
	}
	return stats, nil
}

func (m *v2manager) GetContainerSpec(containerName string, options v2.RequestOptions) (map[string]v2.ContainerSpec, error) {
	conts, err := m.getRequestedContainers(containerName, options)
	if err != nil {
		return nil, err
	}
	specs := make(map[string]v2.ContainerSpec)
	for name, cont := range conts {
		cinfo, err := cont.GetInfo()
		if err != nil {
			return nil, err
		}
		spec := m.getV2Spec(cinfo)
		specs[name] = spec
	}
	return specs, nil
}

// Get V2 container spec from v1 container info.
func (m *v2manager) getV2Spec(cinfo *containerInfo) v2.ContainerSpec {
	specV1 := m.v1().getAdjustedSpec(cinfo)
	return conversion.ConvertSpec(&specV1, cinfo.Aliases, cinfo.Namespace)
}

func (m *v2manager) getRequestedContainers(containerName string, options v2.RequestOptions) (map[string]*containerData, error) {
	containersMap := make(map[string]*containerData)
	switch options.IdType {
	case v2.TypeName:
		if options.Recursive == false {
			cont, err := m.getContainer(containerName)
			if err != nil {
				return containersMap, err
			}
			containersMap[cont.info.Name] = cont
		} else {
			containersMap = m.v1().getSubcontainers(containerName)
			if len(containersMap) == 0 {
				return containersMap, fmt.Errorf("unknown container: %q", containerName)
			}
		}
	case v2.TypeDocker:
		if options.Recursive == false {
			containerName = strings.TrimPrefix(containerName, "/")
			cont, err := m.v1().getDockerContainer(containerName)
			if err != nil {
				return containersMap, err
			}
			containersMap[cont.info.Name] = cont
		} else {
			if containerName != "/" {
				return containersMap, fmt.Errorf("invalid request for docker container %q with subcontainers", containerName)
			}
			containersMap = m.v1().getAllDockerContainers()
		}
	default:
		return containersMap, fmt.Errorf("invalid request type %q", options.IdType)
	}
	return containersMap, nil
}

func (m *v2manager) GetFsInfo(label string) ([]v2.FsInfo, error) {
	var empty time.Time
	// Get latest data from filesystems hanging off root container.
	stats, err := m.memoryCache.RecentStats("/", empty, empty, 1)
	if err != nil {
		return nil, err
	}
	dev := ""
	if len(label) != 0 {
		dev, err = m.fsInfo.GetDeviceForLabel(label)
		if err != nil {
			return nil, err
		}
	}
	fsInfo := []v2.FsInfo{}
	for _, fs := range stats[0].Filesystem {
		if len(label) != 0 && fs.Device != dev {
			continue
		}
		mountpoint, err := m.fsInfo.GetMountpointForDevice(fs.Device)
		if err != nil {
			return nil, err
		}
		labels, err := m.fsInfo.GetLabelsForDevice(fs.Device)
		if err != nil {
			return nil, err
		}
		fi := v2.FsInfo{
			Device:     fs.Device,
			Mountpoint: mountpoint,
			Capacity:   fs.Limit,
			Usage:      fs.Usage,
			Available:  fs.Available,
			Labels:     labels,
		}
		fsInfo = append(fsInfo, fi)
	}
	return fsInfo, nil
}

func (m *v2manager) GetProcessList(containerName string, options v2.RequestOptions) ([]v2.ProcessInfo, error) {
	// override recursive. Only support single container listing.
	options.Recursive = false
	conts, err := m.getRequestedContainers(containerName, options)
	if err != nil {
		return nil, err
	}
	if len(conts) != 1 {
		return nil, fmt.Errorf("Expected the request to match only one container")
	}
	// TODO(rjnagal): handle count? Only if we can do count by type (eg. top 5 cpu users)
	ps := []v2.ProcessInfo{}
	for _, cont := range conts {
		ps, err = cont.GetProcessList(m.cadvisorContainer, m.inHostNamespace)
		if err != nil {
			return nil, err
		}
	}
	return ps, nil
}

func (m *v2manager) getContainer(containerName string) (*containerData, error) {
	m.containersLock.RLock()
	defer m.containersLock.RUnlock()
	cont, ok := m.containers[namespacedContainerName{Name: containerName}]
	if !ok {
		return nil, fmt.Errorf("unknown container %q", containerName)
	}
	return cont, nil
}

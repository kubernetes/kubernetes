//go:build windows

/*
Copyright 2015 The Kubernetes Authors.

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

package cadvisor

import (
	"context"
	"time"

	cadvisorapi "github.com/google/cadvisor/lib/model"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/winstats"
)

type cadvisorClient struct {
	rootPath       string
	winStatsClient winstats.Client
	// containerEnumerator lists live CRI containers so per-container metrics
	// (e.g. container_oom_events_total) can be produced on Windows, where the
	// cAdvisor container discovery does not enumerate containers. It is
	// optional: without it GetRequestedContainersInfo returns nothing, matching
	// the historic behavior.
	containerEnumerator ContainerEnumerator
}

var _ Interface = new(cadvisorClient)

// New creates a cAdvisor and exports its API on the specified port if port > 0.
func New(logger klog.Logger, imageFsInfoProvider ImageFsInfoProvider, rootPath string, cgroupRoots []string, usingLegacyStats, localStorageCapacityIsolation, disableContainerDiscovery bool) (Interface, error) {
	client, err := winstats.NewPerfCounterClient(logger)
	return &cadvisorClient{
		rootPath:       rootPath,
		winStatsClient: client,
	}, err
}

func (cu *cadvisorClient) Start() error {
	return nil
}

// ContainerInfoV2 is only expected to be used for the root container. Returns info for all containers in the node.
func (cu *cadvisorClient) ContainerInfoV2(name string, options cadvisorapi.RequestOptions) (map[string]cadvisorapi.ContainerInfo, error) {
	return cu.winStatsClient.WinContainerInfos()
}

// SetContainerEnumerator wires the CRI runtime service so the client can
// enumerate live containers for per-container metrics. It is optional; if it is
// not called (or is nil) GetRequestedContainersInfo falls back to returning no
// per-container infos, preserving the historic behavior.
func (cu *cadvisorClient) SetContainerEnumerator(e ContainerEnumerator) {
	cu.containerEnumerator = e
}

// GetRequestedContainersInfo returns one minimal ContainerInfo per live CRI
// container, keyed by the container's ID. The metrics collector uses these to
// emit per-container metrics (such as container_oom_events_total) and the
// kubelet server reads OOMEventsForContainer(info.Name), so the container ID is
// the key the Windows OOM watcher must record the count under.
func (cu *cadvisorClient) GetRequestedContainersInfo(containerName string, options cadvisorapi.RequestOptions) (map[string]*cadvisorapi.ContainerInfo, error) {
	if cu.containerEnumerator == nil {
		return nil, nil
	}
	containers, err := cu.containerEnumerator.ListContainers(context.Background(), nil)
	if err != nil {
		return nil, err
	}
	now := time.Now()
	infos := make(map[string]*cadvisorapi.ContainerInfo, len(containers))
	for _, c := range containers {
		info := &cadvisorapi.ContainerInfo{
			Spec: cadvisorapi.ContainerSpec{
				CreationTime: time.Unix(0, c.GetCreatedAt()),
				HasCpu:       true,
				HasMemory:    true,
				Labels:       c.GetLabels(),
			},
			Stats: []*cadvisorapi.ContainerStats{{Timestamp: now}},
		}
		// Name is the promoted ContainerReference.Name (embedded into
		// ContainerInfo); assigning it by field access works on Go <1.27.
		info.Name = c.GetId()
		if c.GetMetadata() != nil && c.GetMetadata().GetName() != "" {
			info.Aliases = []string{c.GetMetadata().GetName()}
		}
		infos[c.GetId()] = info
	}
	return infos, nil
}

func (cu *cadvisorClient) MachineInfo(logger klog.Logger) (*cadvisorapi.MachineInfo, error) {
	return cu.winStatsClient.WinMachineInfo(logger)
}

func (cu *cadvisorClient) VersionInfo() (*cadvisorapi.VersionInfo, error) {
	return cu.winStatsClient.WinVersionInfo()
}

func (cu *cadvisorClient) ImagesFsInfo(context.Context) (cadvisorapi.FsInfo, error) {
	return cadvisorapi.FsInfo{}, nil
}

func (cu *cadvisorClient) ContainerFsInfo(context.Context) (cadvisorapi.FsInfo, error) {
	return cadvisorapi.FsInfo{}, nil
}

func (cu *cadvisorClient) RootFsInfo() (cadvisorapi.FsInfo, error) {
	return cu.GetDirFsInfo(cu.rootPath)
}

func (cu *cadvisorClient) GetDirFsInfo(path string) (cadvisorapi.FsInfo, error) {
	return cu.winStatsClient.GetDirFsInfo(path)
}

func IsPsiEnabled(_ klog.Logger) bool {
	return false
}

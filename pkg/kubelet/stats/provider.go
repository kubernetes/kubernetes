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

package stats

import (
	"fmt"

	cadvisorapiv1 "github.com/google/cadvisor/info/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	internalapi "k8s.io/cri-api/pkg/apis"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubepod "k8s.io/kubernetes/pkg/kubelet/pod"
	"k8s.io/kubernetes/pkg/kubelet/server/stats"
	"k8s.io/kubernetes/pkg/kubelet/stats/pidlimit"
	"k8s.io/kubernetes/pkg/kubelet/status"
)

// NewCRIStatsProvider returns a Provider that provides the node stats
// from cAdvisor and the container stats from CRI.
func NewCRIStatsProvider(
	cadvisor cadvisor.Interface,
	resourceAnalyzer stats.ResourceAnalyzer,
	podManager kubepod.Manager,
	runtimeCache kubecontainer.RuntimeCache,
	runtimeService internalapi.RuntimeService,
	imageService internalapi.ImageManagerService,
	logMetricsService LogMetricsService,
	osInterface kubecontainer.OSInterface,
) *Provider {
	return newStatsProvider(cadvisor, podManager, runtimeCache, newCRIStatsProvider(cadvisor, resourceAnalyzer,
		runtimeService, imageService, logMetricsService, osInterface))
}

// NewCadvisorStatsProvider returns a containerStatsProvider that provides both
// the node and the container stats from cAdvisor.
func NewCadvisorStatsProvider(
	cadvisor cadvisor.Interface,
	resourceAnalyzer stats.ResourceAnalyzer,
	podManager kubepod.Manager,
	runtimeCache kubecontainer.RuntimeCache,
	imageService kubecontainer.ImageService,
	statusProvider status.PodStatusProvider,
) *Provider {
	return newStatsProvider(cadvisor, podManager, runtimeCache, newCadvisorStatsProvider(cadvisor, resourceAnalyzer, imageService, statusProvider))
}

// newStatsProvider returns a new Provider that provides node stats from
// cAdvisor and the container stats using the containerStatsProvider.
func newStatsProvider(
	cadvisor cadvisor.Interface,
	podManager kubepod.Manager,
	runtimeCache kubecontainer.RuntimeCache,
	containerStatsProvider containerStatsProvider,
) *Provider {
	return &Provider{
		cadvisor:               cadvisor,
		podManager:             podManager,
		runtimeCache:           runtimeCache,
		containerStatsProvider: containerStatsProvider,
	}
}

// Provider provides the stats of the node and the pod-managed containers.
type Provider struct {
	cadvisor     cadvisor.Interface
	podManager   kubepod.Manager
	runtimeCache kubecontainer.RuntimeCache
	containerStatsProvider
	rlimitStatsProvider
}

// containerStatsProvider is an interface that provides the stats of the
// containers managed by pods.
type containerStatsProvider interface {
	ListPodStats() ([]statsapi.PodStats, error)
	ListPodStatsAndUpdateCPUNanoCoreUsage() ([]statsapi.PodStats, error)
	ListPodCPUAndMemoryStats() ([]statsapi.PodStats, error)
	ImageFsStats() (*statsapi.FsStats, error)
	ImageFsDevice() (string, error)
}

type rlimitStatsProvider interface {
	RlimitStats() (*statsapi.RlimitStats, error)
}

// RlimitStats returns base information about process count
func (p *Provider) RlimitStats() (*statsapi.RlimitStats, error) {
	return pidlimit.Stats()
}

// GetCgroupStats returns the stats of the cgroup with the cgroupName. Note that
// this function doesn't generate filesystem stats.
func (p *Provider) GetCgroupStats(cgroupName string, updateStats bool) (*statsapi.ContainerStats, *statsapi.NetworkStats, error) {
	info, err := getCgroupInfo(p.cadvisor, cgroupName, updateStats)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get cgroup stats for %q: %v", cgroupName, err)
	}
	// Rootfs and imagefs doesn't make sense for raw cgroup.
	s := cadvisorInfoToContainerStats(cgroupName, info, nil, nil)
	n := cadvisorInfoToNetworkStats(info)
	return s, n, nil
}

// GetCgroupCPUAndMemoryStats returns the CPU and memory stats of the cgroup with the cgroupName. Note that
// this function doesn't generate filesystem stats.
func (p *Provider) GetCgroupCPUAndMemoryStats(cgroupName string, updateStats bool) (*statsapi.ContainerStats, error) {
	info, err := getCgroupInfo(p.cadvisor, cgroupName, updateStats)
	if err != nil {
		return nil, fmt.Errorf("failed to get cgroup stats for %q: %v", cgroupName, err)
	}
	// Rootfs and imagefs doesn't make sense for raw cgroup.
	s := cadvisorInfoToContainerCPUAndMemoryStats(cgroupName, info)
	return s, nil
}

// RootFsStats returns the stats of the node root filesystem.
func (p *Provider) RootFsStats() (*statsapi.FsStats, error) {
	rootFsInfo, err := p.cadvisor.RootFsInfo()
	if err != nil {
		return nil, fmt.Errorf("failed to get rootFs info: %v", err)
	}

	var nodeFsInodesUsed *uint64
	if rootFsInfo.Inodes != nil && rootFsInfo.InodesFree != nil {
		nodeFsIU := *rootFsInfo.Inodes - *rootFsInfo.InodesFree
		nodeFsInodesUsed = &nodeFsIU
	}

	// Get the root container stats's timestamp, which will be used as the
	// imageFs stats timestamp.  Dont force a stats update, as we only want the timestamp.
	rootStats, err := getCgroupStats(p.cadvisor, "/", false)
	if err != nil {
		return nil, fmt.Errorf("failed to get root container stats: %v", err)
	}

	return &statsapi.FsStats{
		Time:           metav1.NewTime(rootStats.Timestamp),
		AvailableBytes: &rootFsInfo.Available,
		CapacityBytes:  &rootFsInfo.Capacity,
		UsedBytes:      &rootFsInfo.Usage,
		InodesFree:     rootFsInfo.InodesFree,
		Inodes:         rootFsInfo.Inodes,
		InodesUsed:     nodeFsInodesUsed,
	}, nil
}

// GetContainerInfo returns stats (from cAdvisor) for a container.
func (p *Provider) GetContainerInfo(podFullName string, podUID types.UID, containerName string, req *cadvisorapiv1.ContainerInfoRequest) (*cadvisorapiv1.ContainerInfo, error) {
	// Resolve and type convert back again.
	// We need the static pod UID but the kubecontainer API works with types.UID.
	podUID = types.UID(p.podManager.TranslatePodUID(podUID))

	pods, err := p.runtimeCache.GetPods()
	if err != nil {
		return nil, err
	}
	pod := kubecontainer.Pods(pods).FindPod(podFullName, podUID)
	container := pod.FindContainerByName(containerName)
	if container == nil {
		return nil, kubecontainer.ErrContainerNotFound
	}

	ci, err := p.cadvisor.DockerContainer(container.ID.ID, req)
	if err != nil {
		return nil, err
	}
	return &ci, nil
}

// GetRawContainerInfo returns the stats (from cadvisor) for a non-Kubernetes
// container.
func (p *Provider) GetRawContainerInfo(containerName string, req *cadvisorapiv1.ContainerInfoRequest, subcontainers bool) (map[string]*cadvisorapiv1.ContainerInfo, error) {
	if subcontainers {
		return p.cadvisor.SubcontainerInfo(containerName, req)
	}
	containerInfo, err := p.cadvisor.ContainerInfo(containerName, req)
	if err != nil {
		return nil, err
	}
	return map[string]*cadvisorapiv1.ContainerInfo{
		containerInfo.Name: containerInfo,
	}, nil
}

// HasDedicatedImageFs returns true if a dedicated image filesystem exists for storing images.
func (p *Provider) HasDedicatedImageFs() (bool, error) {
	device, err := p.containerStatsProvider.ImageFsDevice()
	if err != nil {
		return false, err
	}
	rootFsInfo, err := p.cadvisor.RootFsInfo()
	if err != nil {
		return false, err
	}
	return device != rootFsInfo.Device, nil
}

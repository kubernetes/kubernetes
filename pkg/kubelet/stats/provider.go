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
	"context"
	"errors"
	"fmt"

	cadvisormemory "github.com/google/cadvisor/cache/memory"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	internalapi "k8s.io/cri-api/pkg/apis"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/server/stats"
	"k8s.io/kubernetes/pkg/kubelet/stats/pidlimit"
	"k8s.io/kubernetes/pkg/kubelet/status"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/utils/ptr"
)

// PodManager is the subset of methods the manager needs to observe the actual state of the kubelet.
// See pkg/k8s.io/kubernetes/pkg/kubelet/pod.Manager for method godoc.
type PodManager interface {
	TranslatePodUID(uid types.UID) kubetypes.ResolvedPodUID
}

// NewCRIStatsProvider returns a Provider that provides the node stats
// from cAdvisor and the container stats from CRI.
func NewCRIStatsProvider(
	cadvisor cadvisor.Interface,
	resourceAnalyzer stats.ResourceAnalyzer,
	podManager PodManager,
	runtimeCache kubecontainer.RuntimeCache,
	runtimeService internalapi.RuntimeService,
	imageService internalapi.ImageManagerService,
	hostStatsProvider HostStatsProvider,
	podAndContainerStatsFromCRI bool,
) *Provider {
	return newStatsProvider(cadvisor, podManager, runtimeCache, newCRIStatsProvider(cadvisor, resourceAnalyzer,
		runtimeService, imageService, hostStatsProvider, podAndContainerStatsFromCRI))
}

// NewCadvisorStatsProvider returns a containerStatsProvider that provides both
// the node and the container stats from cAdvisor.
func NewCadvisorStatsProvider(
	cadvisor cadvisor.Interface,
	resourceAnalyzer stats.ResourceAnalyzer,
	podManager PodManager,
	runtimeCache kubecontainer.RuntimeCache,
	imageService kubecontainer.ImageService,
	statusProvider status.PodStatusProvider,
	hostStatsProvider HostStatsProvider,
) *Provider {
	return newStatsProvider(cadvisor, podManager, runtimeCache, newCadvisorStatsProvider(cadvisor, resourceAnalyzer, imageService, statusProvider, hostStatsProvider))
}

// newStatsProvider returns a new Provider that provides node stats from
// cAdvisor and the container stats using the containerStatsProvider.
func newStatsProvider(
	cadvisor cadvisor.Interface,
	podManager PodManager,
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
	podManager   PodManager
	runtimeCache kubecontainer.RuntimeCache
	containerStatsProvider
}

// containerStatsProvider is an interface that provides the stats of the
// containers managed by pods.
type containerStatsProvider interface {
	ListPodStats(ctx context.Context) ([]statsapi.PodStats, error)
	ListPodStatsAndUpdateCPUNanoCoreUsage(ctx context.Context) ([]statsapi.PodStats, error)
	ListPodCPUAndMemoryStats(ctx context.Context) ([]statsapi.PodStats, error)
	ImageFsStats(ctx context.Context) (*statsapi.FsStats, *statsapi.FsStats, error)
	ImageFsDevice(ctx context.Context) (string, error)
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
		if errors.Is(errors.Unwrap(err), cadvisormemory.ErrDataNotFound) {
			return nil, nil, fmt.Errorf("cgroup stats not found for %q: %w", cgroupName, cadvisormemory.ErrDataNotFound)
		}
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
		if errors.Is(errors.Unwrap(err), cadvisormemory.ErrDataNotFound) {
			return nil, fmt.Errorf("cgroup stats not found for %q: %w", cgroupName, cadvisormemory.ErrDataNotFound)
		}
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
	// imageFs stats timestamp.  Don't force a stats update, as we only want the timestamp.
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

// HasDedicatedImageFs returns true if a dedicated image filesystem exists for storing images.
// KEP Issue Number 4191: Enhanced this to allow for the containers to be separate from images.
func (p *Provider) HasDedicatedImageFs(ctx context.Context) (bool, error) {
	device, err := p.containerStatsProvider.ImageFsDevice(ctx)
	if err != nil {
		return false, err
	}
	rootFsInfo, err := p.cadvisor.RootFsInfo()
	if err != nil {
		return false, err
	}
	// KEP Enhancement: DedicatedImageFs can mean either container or image fs are separate from root
	// CAdvisor reports this a bit differently than Container runtimes
	if device == rootFsInfo.Device {
		imageFs, containerFs, err := p.ImageFsStats(ctx)
		if err != nil {
			return false, err
		}
		if !equalFileSystems(imageFs, containerFs) {
			return true, nil
		}
	}
	return device != rootFsInfo.Device, nil
}

func equalFileSystems(a, b *statsapi.FsStats) bool {
	if a == nil || b == nil {
		return false
	}
	if !ptr.Equal(a.AvailableBytes, b.AvailableBytes) {
		return false
	}
	if !ptr.Equal(a.CapacityBytes, b.CapacityBytes) {
		return false
	}
	if !ptr.Equal(a.InodesUsed, b.InodesUsed) {
		return false
	}
	if !ptr.Equal(a.InodesFree, b.InodesFree) {
		return false
	}
	if !ptr.Equal(a.Inodes, b.Inodes) {
		return false
	}
	return true
}

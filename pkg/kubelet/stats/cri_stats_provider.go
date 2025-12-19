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
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	cadvisormemory "github.com/google/cadvisor/cache/memory"
	cadvisorfs "github.com/google/cadvisor/fs"
	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	internalapi "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	kubetypes "k8s.io/kubelet/pkg/types"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/server/stats"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/utils/clock"
	"k8s.io/utils/ptr"
)

var (
	// defaultCachePeriod is the default cache period for each cpuUsage.
	defaultCachePeriod = 10 * time.Minute
)

// cpuUsageRecord holds the cpu usage stats and the calculated usageNanoCores.
type cpuUsageRecord struct {
	stats          *runtimeapi.CpuUsage
	usageNanoCores *uint64
}

// criStatsProvider implements the containerStatsProvider interface by getting
// the container stats from CRI.
type criStatsProvider struct {
	// cadvisor is used to get the node root filesystem's stats (such as the
	// capacity/available bytes/inodes) that will be populated in per container
	// filesystem stats.
	cadvisor cadvisor.Interface
	// resourceAnalyzer is used to get the volume stats of the pods.
	resourceAnalyzer stats.ResourceAnalyzer
	// runtimeService is used to get the status and stats of the pods and its
	// managed containers.
	runtimeService internalapi.RuntimeService
	// imageService is used to get the stats of the image filesystem.
	imageService internalapi.ImageManagerService
	// hostStatsProvider is used to get the status of the host filesystem consumed by pods.
	hostStatsProvider HostStatsProvider
	// windowsNetworkStatsProvider is used by kubelet to gather networking stats on Windows
	windowsNetworkStatsProvider interface{} //nolint:unused // U1000 We can't import hcsshim due to Build constraints in hcsshim
	// clock is used report current time
	clock clock.Clock
	// fallbackStatsProvider is used to fill in missing information incase the CRI
	// provides insufficient data.
	// TODO: A lot of the cadvisorStatsProvider logic is duplicated in this file, and should be read
	//       from the fallbackStatsProvider instead.
	// Remove this once the CRI stats migration is complete.
	fallbackStatsProvider containerStatsProvider

	// cpuUsageCache caches the cpu usage for containers.
	cpuUsageCache               map[string]*cpuUsageRecord
	mutex                       sync.RWMutex
	podAndContainerStatsFromCRI bool
}

// newCRIStatsProvider returns a containerStatsProvider implementation that
// provides container stats using CRI.
func newCRIStatsProvider(
	cadvisor cadvisor.Interface,
	resourceAnalyzer stats.ResourceAnalyzer,
	runtimeService internalapi.RuntimeService,
	imageService internalapi.ImageManagerService,
	hostStatsProvider HostStatsProvider,
	podAndContainerStatsFromCRI bool,
	fallbackStatsProvider containerStatsProvider,
) containerStatsProvider {
	return &criStatsProvider{
		cadvisor:                    cadvisor,
		resourceAnalyzer:            resourceAnalyzer,
		runtimeService:              runtimeService,
		imageService:                imageService,
		hostStatsProvider:           hostStatsProvider,
		cpuUsageCache:               make(map[string]*cpuUsageRecord),
		podAndContainerStatsFromCRI: podAndContainerStatsFromCRI,
		clock:                       clock.RealClock{},
		fallbackStatsProvider:       fallbackStatsProvider,
	}
}

// ListPodStats returns the stats of all the pod-managed containers.
func (p *criStatsProvider) ListPodStats(ctx context.Context) ([]statsapi.PodStats, error) {
	// Don't update CPU nano core usage.
	return p.listPodStats(ctx, false)
}

// ListPodStatsAndUpdateCPUNanoCoreUsage updates the cpu nano core usage for
// the containers and returns the stats for all the pod-managed containers.
// This is a workaround because CRI runtimes do not supply nano core usages,
// so this function calculate the difference between the current and the last
// (cached) cpu stats to calculate this metrics. The implementation assumes a
// single caller to periodically invoke this function to update the metrics. If
// there exist multiple callers, the period used to compute the cpu usage may
// vary and the usage could be incoherent (e.g., spiky). If no caller calls
// this function, the cpu usage will stay nil. Right now, eviction manager is
// the only caller, and it calls this function every 10s.
func (p *criStatsProvider) ListPodStatsAndUpdateCPUNanoCoreUsage(ctx context.Context) ([]statsapi.PodStats, error) {
	// Update CPU nano core usage.
	return p.listPodStats(ctx, true)
}

func (p *criStatsProvider) listPodStats(ctx context.Context, updateCPUNanoCoreUsage bool) ([]statsapi.PodStats, error) {
	// Gets node root filesystem information, which will be used to populate
	// the available and capacity bytes/inodes in container stats.
	rootFsInfo, err := p.cadvisor.RootFsInfo()
	if err != nil {
		return nil, fmt.Errorf("failed to get rootFs info: %v", err)
	}

	containerMap, podSandboxMap, err := p.getPodAndContainerMaps(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get pod or container map: %v", err)
	}

	logger := klog.FromContext(ctx)
	if p.podAndContainerStatsFromCRI {
		result, err := p.listPodStatsStrictlyFromCRI(ctx, updateCPUNanoCoreUsage, containerMap, podSandboxMap, &rootFsInfo)
		if err == nil {
			// Call succeeded
			return result, nil
		}
		s, ok := status.FromError(err)
		// Legitimate failure, rather than the CRI implementation does not support ListPodSandboxStats.
		if !ok || s.Code() != codes.Unimplemented {
			return nil, err
		}
		// CRI implementation doesn't support ListPodSandboxStats, warn and fallback.
		logger.V(5).Error(err,
			"CRI implementation must be updated to support ListPodSandboxStats if PodAndContainerStatsFromCRI feature gate is enabled. Falling back to populating with cAdvisor; this call will fail in the future.",
		)
	}
	return p.listPodStatsPartiallyFromCRI(ctx, updateCPUNanoCoreUsage, containerMap, podSandboxMap, &rootFsInfo)
}

func (p *criStatsProvider) listPodStatsPartiallyFromCRI(ctx context.Context, updateCPUNanoCoreUsage bool, containerMap map[string]*runtimeapi.Container, podSandboxMap map[string]*runtimeapi.PodSandbox, rootFsInfo *cadvisorapiv2.FsInfo) ([]statsapi.PodStats, error) {
	// fsIDtoInfo is a map from mountpoint to its stats. This will be used
	// as a cache to avoid querying cAdvisor for the filesystem stats with the
	// same filesystem id many times.
	fsIDtoInfo := make(map[string]*cadvisorapiv2.FsInfo)

	// sandboxIDToPodStats is a temporary map from sandbox ID to its pod stats.
	sandboxIDToPodStats := make(map[string]*statsapi.PodStats)

	resp, err := p.runtimeService.ListContainerStats(ctx, &runtimeapi.ContainerStatsFilter{})
	if err != nil {
		return nil, fmt.Errorf("failed to list all container stats: %v", err)
	}

	logger := klog.FromContext(ctx)
	allInfos, err := getCadvisorContainerInfo(logger, p.cadvisor)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch cadvisor stats: %v", err)
	}
	caInfos, allInfos := getCRICadvisorStats(logger, allInfos)

	// get network stats for containers.
	// This is only used on Windows. For other platforms, (nil, nil) should be returned.
	containerNetworkStats, err := p.listContainerNetworkStats(logger)
	if err != nil {
		return nil, fmt.Errorf("failed to list container network stats: %v", err)
	}
	for _, stats := range resp {
		containerID := stats.Attributes.Id
		container, found := containerMap[containerID]
		if !found {
			continue
		}

		podSandboxID := container.PodSandboxId
		podSandbox, found := podSandboxMap[podSandboxID]
		if !found {
			continue
		}

		// Creates the stats of the pod (if not created yet) which the
		// container belongs to.
		ps, found := sandboxIDToPodStats[podSandboxID]
		if !found {
			ps = buildPodStats(podSandbox)
			sandboxIDToPodStats[podSandboxID] = ps
		}

		// Fill available stats for full set of required pod stats
		cs, err := p.makeContainerStats(logger, stats, container, rootFsInfo, fsIDtoInfo, podSandbox.GetMetadata(), updateCPUNanoCoreUsage)
		if err != nil {
			return nil, fmt.Errorf("make container stats: %w", err)
		}
		p.addPodNetworkStats(logger, ps, podSandboxID, caInfos, cs, containerNetworkStats[podSandboxID])
		p.addPodCPUMemoryStats(ps, types.UID(podSandbox.Metadata.Uid), allInfos, cs)
		p.addSwapStats(ps, types.UID(podSandbox.Metadata.Uid), allInfos, cs)
		p.addIOStats(ps, types.UID(podSandbox.Metadata.Uid), allInfos, cs)

		// If cadvisor stats is available for the container, use it to populate
		// container stats
		caStats, caFound := caInfos[containerID]
		if !caFound {
			logger.V(5).Info("Unable to find cadvisor stats for container", "containerID", containerID)
		} else {
			p.addCadvisorContainerStats(logger, cs, &caStats)
			p.addProcessStats(ps, &caStats)
		}

		ps.Containers = append(ps.Containers, *cs)
	}
	// cleanup outdated caches.
	p.cleanupOutdatedCaches()

	result := make([]statsapi.PodStats, 0, len(sandboxIDToPodStats))
	for _, s := range sandboxIDToPodStats {
		makePodStorageStats(logger, s, rootFsInfo, p.resourceAnalyzer, p.hostStatsProvider, true)
		result = append(result, *s)
	}
	return result, nil
}

func (p *criStatsProvider) listPodStatsStrictlyFromCRI(ctx context.Context, updateCPUNanoCoreUsage bool, containerMap map[string]*runtimeapi.Container, podSandboxMap map[string]*runtimeapi.PodSandbox, rootFsInfo *cadvisorapiv2.FsInfo) ([]statsapi.PodStats, error) {
	criSandboxStats, err := p.runtimeService.ListPodSandboxStats(ctx, &runtimeapi.PodSandboxStatsFilter{})
	if err != nil {
		return nil, err
	}
	logger := klog.FromContext(ctx)

	fsIDtoInfo := make(map[string]*cadvisorapiv2.FsInfo)
	summarySandboxStats := make([]statsapi.PodStats, 0, len(podSandboxMap))
	for _, criSandboxStat := range criSandboxStats {
		if criSandboxStat == nil || criSandboxStat.Attributes == nil {
			logger.V(5).Info("Unable to find CRI stats for sandbox")
			continue
		}
		podSandbox, found := podSandboxMap[criSandboxStat.Attributes.Id]
		if !found {
			continue
		}
		ps := buildPodStats(podSandbox)
		if err := p.addCRIPodContainerStats(logger, criSandboxStat, ps, fsIDtoInfo, containerMap, podSandbox, rootFsInfo, updateCPUNanoCoreUsage); err != nil {
			return nil, fmt.Errorf("add CRI pod container stats: %w", err)
		}
		addCRIPodNetworkStats(ps, criSandboxStat)
		addCRIPodCPUStats(ps, criSandboxStat)
		addCRIPodMemoryStats(ps, criSandboxStat)
		addCRIPodProcessStats(ps, criSandboxStat)
		addCRIPodIOStats(ps, criSandboxStat)
		makePodStorageStats(logger, ps, rootFsInfo, p.resourceAnalyzer, p.hostStatsProvider, true)
		summarySandboxStats = append(summarySandboxStats, *ps)
	}
	return summarySandboxStats, nil
}

func (p *criStatsProvider) PodCPUAndMemoryStats(ctx context.Context, pod *v1.Pod, podStatus *kubecontainer.PodStatus) (*statsapi.PodStats, error) {
	if len(podStatus.SandboxStatuses) == 0 {
		return nil, fmt.Errorf("missing sandbox for pod %s", format.Pod(pod))
	}
	podSandbox := podStatus.SandboxStatuses[0]
	ps := &statsapi.PodStats{
		PodRef: statsapi.PodReference{
			Name:      podSandbox.Metadata.Name,
			UID:       podSandbox.Metadata.Uid,
			Namespace: podSandbox.Metadata.Namespace,
		},
		// The StartTime in the summary API is the pod creation time.
		StartTime: metav1.NewTime(time.Unix(0, podSandbox.CreatedAt)),
	}
	if p.podAndContainerStatsFromCRI {
		criSandboxStats, err := p.runtimeService.PodSandboxStats(ctx, podSandbox.Id)
		if err != nil {
			// Call failed, why?
			s, ok := status.FromError(err)
			// Legitimate failure, rather than the CRI implementation does not support PodSandboxStats.
			if !ok || s.Code() != codes.Unimplemented {
				return nil, err
			}
			// CRI implementation doesn't support PodSandboxStats, warn and fallback.
			klog.ErrorS(err,
				"CRI implementation must be updated to support PodSandboxStats if PodAndContainerStatsFromCRI feature gate is enabled. Falling back to populating with cAdvisor; this call will fail in the future.",
			)
		} else {
			addCRIPodCPUStats(ps, criSandboxStats)
			addCRIPodMemoryStats(ps, criSandboxStats)
		}
	}

	resp, err := p.runtimeService.ListContainerStats(ctx, &runtimeapi.ContainerStatsFilter{
		PodSandboxId: podSandbox.Id,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to list container stats from pod %s (sandbox %s): %w", format.Pod(pod), podSandbox.Id, err)
	}

	// Fallback if ListContainerStats doesn't return any results.
	useFallback := ps.CPU == nil || ps.Memory == nil || len(resp) == 0
	for _, stats := range resp {
		containerStatus := podStatus.FindContainerStatusByName(stats.Attributes.Metadata.Name)
		if containerStatus == nil {
			klog.V(4).InfoS("Received stats for unknown container", "pod", klog.KObj(pod), "container", stats.Attributes.Metadata)
			continue
		}

		// Fill available CPU and memory stats for full set of required pod stats
		cs := p.makeContainerCPUAndMemoryStats(stats, containerStatus.CreatedAt, false)
		useFallback = useFallback || cs.CPU == nil || cs.Memory == nil
		ps.Containers = append(ps.Containers, *cs)
	}

	if useFallback {
		fallbackStats, err := p.fallbackStatsProvider.PodCPUAndMemoryStats(ctx, pod, podStatus)
		if err != nil {
			return nil, fmt.Errorf("failed to fetch stats for pod %s from fallback provider: %w", format.Pod(pod), err)
		}
		if ps.CPU == nil {
			ps.CPU = fallbackStats.CPU
		}
		if ps.Memory == nil {
			ps.Memory = fallbackStats.Memory
		}

		for _, fb := range fallbackStats.Containers {
			var container *statsapi.ContainerStats
			for i, cs := range ps.Containers {
				if fb.Name == cs.Name {
					container = &ps.Containers[i]
					break
				}
			}
			if container != nil {
				if container.CPU == nil {
					container.CPU = fb.CPU
				}
				if container.Memory == nil {
					container.Memory = fb.Memory
				}
			} else {
				ps.Containers = append(ps.Containers, fb)
			}
		}
	}

	return ps, nil
}

// ListPodCPUAndMemoryStats returns the CPU and Memory stats of all the pod-managed containers.
func (p *criStatsProvider) ListPodCPUAndMemoryStats(ctx context.Context) ([]statsapi.PodStats, error) {
	// sandboxIDToPodStats is a temporary map from sandbox ID to its pod stats.
	sandboxIDToPodStats := make(map[string]*statsapi.PodStats)
	containerMap, podSandboxMap, err := p.getPodAndContainerMaps(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get pod or container map: %v", err)
	}
	logger := klog.FromContext(ctx)

	result := make([]statsapi.PodStats, 0, len(podSandboxMap))
	if p.podAndContainerStatsFromCRI {
		criSandboxStats, err := p.runtimeService.ListPodSandboxStats(ctx, &runtimeapi.PodSandboxStatsFilter{})
		// Call succeeded
		if err == nil {
			for _, criSandboxStat := range criSandboxStats {
				podSandbox, found := podSandboxMap[criSandboxStat.Attributes.Id]
				if !found {
					continue
				}
				ps := buildPodStats(podSandbox)
				// Add container-level CPU and memory stats from CRI
				p.addCRIPodContainerCPUAndMemoryStats(criSandboxStat, ps, containerMap)
				addCRIPodCPUStats(ps, criSandboxStat)
				addCRIPodMemoryStats(ps, criSandboxStat)
				// Aggregate pod swap from container swap stats (CRI doesn't have pod-level swap)
				aggregatePodSwapStats(ps)
				result = append(result, *ps)
			}
			return result, err
		}
		// Call failed, why?
		s, ok := status.FromError(err)
		// Legitimate failure, rather than the CRI implementation does not support ListPodSandboxStats.
		if !ok || s.Code() != codes.Unimplemented {
			return nil, err
		}
		// CRI implementation doesn't support ListPodSandboxStats, warn and fallback.
		logger.Error(err,
			"CRI implementation must be updated to support ListPodSandboxStats if PodAndContainerStatsFromCRI feature gate is enabled. Falling back to populating with cAdvisor; this call will fail in the future.",
		)
	}

	resp, err := p.runtimeService.ListContainerStats(ctx, &runtimeapi.ContainerStatsFilter{})
	if err != nil {
		return nil, fmt.Errorf("failed to list all container stats: %v", err)
	}

	allInfos, err := getCadvisorContainerInfo(logger, p.cadvisor)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch cadvisor stats: %v", err)
	}
	caInfos, allInfos := getCRICadvisorStats(logger, allInfos)

	for _, stats := range resp {
		containerID := stats.Attributes.Id
		container, found := containerMap[containerID]
		if !found {
			continue
		}

		podSandboxID := container.PodSandboxId
		podSandbox, found := podSandboxMap[podSandboxID]
		if !found {
			continue
		}

		// Creates the stats of the pod (if not created yet) which the
		// container belongs to.
		ps, found := sandboxIDToPodStats[podSandboxID]
		if !found {
			ps = buildPodStats(podSandbox)
			sandboxIDToPodStats[podSandboxID] = ps
		}

		// Fill available CPU and memory stats for full set of required pod stats
		cs := p.makeContainerCPUAndMemoryStats(stats, time.Unix(0, container.CreatedAt), true)
		p.addPodCPUMemoryStats(ps, types.UID(podSandbox.Metadata.Uid), allInfos, cs)
		p.addSwapStats(ps, types.UID(podSandbox.Metadata.Uid), allInfos, cs)

		// If cadvisor stats is available for the container, use it to populate
		// container stats
		caStats, caFound := caInfos[containerID]
		if !caFound {
			logger.V(4).Info("Unable to find cadvisor stats for container", "containerID", containerID)
		} else {
			p.addCadvisorContainerCPUAndMemoryStats(logger, cs, &caStats)
		}
		ps.Containers = append(ps.Containers, *cs)
	}
	// cleanup outdated caches.
	p.cleanupOutdatedCaches()

	for _, s := range sandboxIDToPodStats {
		result = append(result, *s)
	}
	return result, nil
}

func (p *criStatsProvider) getPodAndContainerMaps(ctx context.Context) (map[string]*runtimeapi.Container, map[string]*runtimeapi.PodSandbox, error) {
	containers, err := p.runtimeService.ListContainers(ctx, &runtimeapi.ContainerFilter{})
	if err != nil {
		return nil, nil, fmt.Errorf("failed to list all containers: %v", err)
	}

	// Creates pod sandbox map between the pod sandbox ID and the PodSandbox object.
	podSandboxMap := make(map[string]*runtimeapi.PodSandbox)
	podSandboxes, err := p.runtimeService.ListPodSandbox(ctx, &runtimeapi.PodSandboxFilter{})
	if err != nil {
		return nil, nil, fmt.Errorf("failed to list all pod sandboxes: %v", err)
	}
	podSandboxes = removeTerminatedPods(podSandboxes)
	for _, s := range podSandboxes {
		podSandboxMap[s.Id] = s
	}

	containers = removeTerminatedContainers(containers)
	// Creates container map between the container ID and the Container object.
	containerMap := make(map[string]*runtimeapi.Container)
	for _, c := range containers {
		containerMap[c.Id] = c
	}
	return containerMap, podSandboxMap, nil
}

// ImageFsStats returns the stats of the image filesystem.
func (p *criStatsProvider) ImageFsStats(ctx context.Context) (imageFsRet *statsapi.FsStats, containerFsRet *statsapi.FsStats, errRet error) {
	resp, err := p.imageService.ImageFsInfo(ctx)
	if err != nil {
		return nil, nil, err
	}

	// CRI may return the stats of multiple image filesystems but we only
	// return the first one.
	//
	// TODO(yguo0905): Support returning stats of multiple image filesystems.
	if len(resp.GetImageFilesystems()) == 0 {
		return nil, nil, fmt.Errorf("imageFs information is unavailable")
	}
	fs := resp.GetImageFilesystems()[0]
	imageFsRet = &statsapi.FsStats{
		Time:      metav1.NewTime(time.Unix(0, fs.Timestamp)),
		UsedBytes: &fs.UsedBytes.Value,
	}
	if fs.InodesUsed != nil {
		imageFsRet.InodesUsed = &fs.InodesUsed.Value
	}
	imageFsInfo, err := p.getFsInfo(klog.FromContext(ctx), fs.GetFsId())
	if err != nil {
		return nil, nil, fmt.Errorf("get filesystem info: %w", err)
	}
	if imageFsInfo != nil {
		// The image filesystem id is unknown to the local node or there's
		// an error on retrieving the stats. In these cases, we omit those
		// stats and return the best-effort partial result. See
		// https://github.com/kubernetes/heapster/issues/1793.
		imageFsRet.AvailableBytes = &imageFsInfo.Available
		imageFsRet.CapacityBytes = &imageFsInfo.Capacity
		imageFsRet.InodesFree = imageFsInfo.InodesFree
		imageFsRet.Inodes = imageFsInfo.Inodes
	}
	// TODO: For CRI Stats Provider we don't support separate disks yet.
	return imageFsRet, imageFsRet, nil
}

// ImageFsDevice returns name of the device where the image filesystem locates,
// e.g. /dev/sda1.
func (p *criStatsProvider) ImageFsDevice(ctx context.Context) (string, error) {
	resp, err := p.imageService.ImageFsInfo(ctx)
	if err != nil {
		return "", err
	}
	for _, fs := range resp.GetImageFilesystems() {
		fsInfo, err := p.getFsInfo(klog.FromContext(ctx), fs.GetFsId())
		if err != nil {
			return "", fmt.Errorf("get filesystem info: %w", err)
		}
		if fsInfo != nil {
			return fsInfo.Device, nil
		}
	}
	return "", errors.New("imagefs device is not found")
}

// getFsInfo returns the information of the filesystem with the specified
// fsID. If any error occurs, this function logs the error and returns
// nil.
func (p *criStatsProvider) getFsInfo(logger klog.Logger, fsID *runtimeapi.FilesystemIdentifier) (*cadvisorapiv2.FsInfo, error) {
	if fsID == nil {
		logger.V(2).Info("Failed to get filesystem info: fsID is nil")
		return nil, nil
	}
	mountpoint := fsID.GetMountpoint()
	fsInfo, err := p.cadvisor.GetDirFsInfo(mountpoint)
	if err != nil {
		msg := "Failed to get the info of the filesystem with mountpoint"
		if errors.Is(err, cadvisorfs.ErrNoSuchDevice) ||
			errors.Is(err, cadvisorfs.ErrDeviceNotInPartitionsMap) ||
			errors.Is(err, cadvisormemory.ErrDataNotFound) {
			logger.V(2).Info(msg, "mountpoint", mountpoint, "err", err)
		} else {
			logger.Error(err, msg, "mountpoint", mountpoint)
			return nil, fmt.Errorf("%s: %w", msg, err)
		}
		return nil, nil
	}
	return &fsInfo, nil
}

// buildPodStats returns a PodStats that identifies the Pod managing cinfo
func buildPodStats(podSandbox *runtimeapi.PodSandbox) *statsapi.PodStats {
	return &statsapi.PodStats{
		PodRef: statsapi.PodReference{
			Name:      podSandbox.Metadata.Name,
			UID:       podSandbox.Metadata.Uid,
			Namespace: podSandbox.Metadata.Namespace,
		},
		// The StartTime in the summary API is the pod creation time.
		StartTime: metav1.NewTime(time.Unix(0, podSandbox.CreatedAt)),
	}
}

func (p *criStatsProvider) addPodNetworkStats(
	logger klog.Logger,
	ps *statsapi.PodStats,
	podSandboxID string,
	caInfos map[string]cadvisorapiv2.ContainerInfo,
	cs *statsapi.ContainerStats,
	netStats *statsapi.NetworkStats,
) {
	caPodSandbox, found := caInfos[podSandboxID]
	// try get network stats from cadvisor first.
	if found {
		networkStats := cadvisorInfoToNetworkStats(&caPodSandbox)
		if networkStats != nil {
			ps.Network = networkStats
			return
		}
	}

	// Not found from cadvisor, get from netStats.
	if netStats != nil {
		ps.Network = netStats
		return
	}

	// TODO: sum Pod network stats from container stats.
	logger.V(4).Info("Unable to find network stats for sandbox", "sandboxID", podSandboxID)
}

func (p *criStatsProvider) addPodCPUMemoryStats(
	ps *statsapi.PodStats,
	podUID types.UID,
	allInfos map[string]cadvisorapiv2.ContainerInfo,
	cs *statsapi.ContainerStats,
) {
	// try get cpu and memory stats from cadvisor first.
	podCgroupInfo := getCadvisorPodInfoFromPodUID(podUID, allInfos)
	if podCgroupInfo != nil {
		cpu, memory := cadvisorInfoToCPUandMemoryStats(podCgroupInfo)
		ps.CPU = cpu
		ps.Memory = memory
		return
	}

	// Sum Pod cpu and memory stats from containers stats.
	if cs.CPU != nil {
		if ps.CPU == nil {
			ps.CPU = &statsapi.CPUStats{}
		}

		ps.CPU.Time = cs.CPU.Time
		usageCoreNanoSeconds := ptr.Deref(cs.CPU.UsageCoreNanoSeconds, 0) + ptr.Deref(ps.CPU.UsageCoreNanoSeconds, 0)
		usageNanoCores := ptr.Deref(cs.CPU.UsageNanoCores, 0) + ptr.Deref(ps.CPU.UsageNanoCores, 0)
		ps.CPU.UsageCoreNanoSeconds = &usageCoreNanoSeconds
		ps.CPU.UsageNanoCores = &usageNanoCores
		// Pod level PSI stats cannot be calculated from container level
	}

	if cs.Memory != nil {
		if ps.Memory == nil {
			ps.Memory = &statsapi.MemoryStats{}
		}

		ps.Memory.Time = cs.Memory.Time
		availableBytes := ptr.Deref(cs.Memory.AvailableBytes, 0) + ptr.Deref(ps.Memory.AvailableBytes, 0)
		usageBytes := ptr.Deref(cs.Memory.UsageBytes, 0) + ptr.Deref(ps.Memory.UsageBytes, 0)
		workingSetBytes := ptr.Deref(cs.Memory.WorkingSetBytes, 0) + ptr.Deref(ps.Memory.WorkingSetBytes, 0)
		rSSBytes := ptr.Deref(cs.Memory.RSSBytes, 0) + ptr.Deref(ps.Memory.RSSBytes, 0)
		pageFaults := ptr.Deref(cs.Memory.PageFaults, 0) + ptr.Deref(ps.Memory.PageFaults, 0)
		majorPageFaults := ptr.Deref(cs.Memory.MajorPageFaults, 0) + ptr.Deref(ps.Memory.MajorPageFaults, 0)
		ps.Memory.AvailableBytes = &availableBytes
		ps.Memory.UsageBytes = &usageBytes
		ps.Memory.WorkingSetBytes = &workingSetBytes
		ps.Memory.RSSBytes = &rSSBytes
		ps.Memory.PageFaults = &pageFaults
		ps.Memory.MajorPageFaults = &majorPageFaults
		// Pod level PSI stats cannot be calculated from container level
	}
}

func (p *criStatsProvider) addSwapStats(
	ps *statsapi.PodStats,
	podUID types.UID,
	allInfos map[string]cadvisorapiv2.ContainerInfo,
	cs *statsapi.ContainerStats,
) {
	// try get swap stats from cadvisor first.
	podCgroupInfo := getCadvisorPodInfoFromPodUID(podUID, allInfos)
	if podCgroupInfo != nil {
		ps.Swap = cadvisorInfoToSwapStats(podCgroupInfo)
		return
	}

	// Sum Pod swap stats from containers stats.
	if cs.Swap != nil {
		if ps.Swap == nil {
			ps.Swap = &statsapi.SwapStats{Time: cs.Swap.Time}
		}
		swapAvailableBytes := ptr.Deref(cs.Swap.SwapAvailableBytes, 0) + ptr.Deref(ps.Swap.SwapAvailableBytes, 0)
		swapUsageBytes := ptr.Deref(cs.Swap.SwapUsageBytes, 0) + ptr.Deref(ps.Swap.SwapUsageBytes, 0)
		ps.Swap.SwapAvailableBytes = &swapAvailableBytes
		ps.Swap.SwapUsageBytes = &swapUsageBytes
	}
}

// aggregatePodSwapStats aggregates pod-level swap stats from container swap stats.
// This is used when CRI doesn't provide pod-level swap stats (e.g., LinuxPodSandboxStats doesn't have a Swap field).
func aggregatePodSwapStats(ps *statsapi.PodStats) {
	if len(ps.Containers) == 0 {
		return
	}
	var swapAvailableBytes, swapUsageBytes uint64
	var hasSwapStats bool
	var swapTime metav1.Time
	for _, cs := range ps.Containers {
		if cs.Swap != nil {
			hasSwapStats = true
			// TODO: Consider picking the newest time across containers instead of just using the last one.
			swapTime = cs.Swap.Time
			swapAvailableBytes += ptr.Deref(cs.Swap.SwapAvailableBytes, 0)
			swapUsageBytes += ptr.Deref(cs.Swap.SwapUsageBytes, 0)
		}
	}
	if hasSwapStats {
		ps.Swap = &statsapi.SwapStats{
			Time:               swapTime,
			SwapAvailableBytes: &swapAvailableBytes,
			SwapUsageBytes:     &swapUsageBytes,
		}
	}
}

func (p *criStatsProvider) addIOStats(
	ps *statsapi.PodStats,
	podUID types.UID,
	allInfos map[string]cadvisorapiv2.ContainerInfo,
	cs *statsapi.ContainerStats,
) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.KubeletPSI) {
		return
	}
	// try get IO stats from cadvisor first.
	podCgroupInfo := getCadvisorPodInfoFromPodUID(podUID, allInfos)
	if podCgroupInfo != nil {
		ps.IO = cadvisorInfoToIOStats(podCgroupInfo)
		return
	}

	if cs.IO != nil {
		if ps.IO == nil {
			ps.IO = &statsapi.IOStats{Time: cs.IO.Time}
		}
		// Pod level PSI stats cannot be calculated from container level
	}
}

func (p *criStatsProvider) addProcessStats(
	ps *statsapi.PodStats,
	container *cadvisorapiv2.ContainerInfo,
) {
	processStats := cadvisorInfoToProcessStats(container)
	// Sum up all of the process stats for each of the containers to obtain the cumulative pod level process count
	ps.ProcessStats = mergeProcessStats(ps.ProcessStats, processStats)
}

func (p *criStatsProvider) makeContainerStats(
	logger klog.Logger,
	stats *runtimeapi.ContainerStats,
	container *runtimeapi.Container,
	rootFsInfo *cadvisorapiv2.FsInfo,
	fsIDtoInfo map[string]*cadvisorapiv2.FsInfo,
	meta *runtimeapi.PodSandboxMetadata,
	updateCPUNanoCoreUsage bool,
) (*statsapi.ContainerStats, error) {
	result := &statsapi.ContainerStats{
		Name: stats.Attributes.Metadata.Name,
		// The StartTime in the summary API is the container creation time.
		StartTime: metav1.NewTime(time.Unix(0, container.CreatedAt)),
		CPU:       &statsapi.CPUStats{},
		Memory:    &statsapi.MemoryStats{},
		Rootfs:    &statsapi.FsStats{},
		Swap:      &statsapi.SwapStats{},
		// UserDefinedMetrics is not supported by CRI.
	}
	if stats.Cpu != nil {
		result.CPU.Time = metav1.NewTime(time.Unix(0, stats.Cpu.Timestamp))
		if stats.Cpu.UsageCoreNanoSeconds != nil {
			result.CPU.UsageCoreNanoSeconds = &stats.Cpu.UsageCoreNanoSeconds.Value
		}
		var usageNanoCores *uint64
		if updateCPUNanoCoreUsage {
			usageNanoCores = p.getAndUpdateContainerUsageNanoCores(logger, stats)
		} else {
			usageNanoCores = p.getContainerUsageNanoCores(stats)
		}
		if usageNanoCores != nil {
			result.CPU.UsageNanoCores = usageNanoCores
		}
		result.CPU.PSI = makePSIStats(stats.Cpu.Psi)
	} else {
		result.CPU.Time = metav1.NewTime(time.Unix(0, time.Now().UnixNano()))
		result.CPU.UsageCoreNanoSeconds = ptr.To[uint64](0)
		result.CPU.UsageNanoCores = ptr.To[uint64](0)
	}
	if stats.Memory != nil {
		result.Memory.Time = metav1.NewTime(time.Unix(0, stats.Memory.Timestamp))
		if stats.Memory.WorkingSetBytes != nil {
			result.Memory.WorkingSetBytes = &stats.Memory.WorkingSetBytes.Value
		}
		if stats.Memory.UsageBytes != nil {
			result.Memory.UsageBytes = &stats.Memory.UsageBytes.Value
		}
		if stats.Memory.RssBytes != nil {
			result.Memory.RSSBytes = &stats.Memory.RssBytes.Value
		}
		if stats.Memory.AvailableBytes != nil {
			result.Memory.AvailableBytes = &stats.Memory.AvailableBytes.Value
		}
		if stats.Memory.PageFaults != nil {
			result.Memory.PageFaults = &stats.Memory.PageFaults.Value
		}
		if stats.Memory.MajorPageFaults != nil {
			result.Memory.MajorPageFaults = &stats.Memory.MajorPageFaults.Value
		}
		result.Memory.PSI = makePSIStats(stats.Memory.Psi)
	} else {
		result.Memory.Time = metav1.NewTime(time.Unix(0, time.Now().UnixNano()))
		result.Memory.WorkingSetBytes = ptr.To[uint64](0)
		result.Memory.UsageBytes = ptr.To[uint64](0)
		result.Memory.RSSBytes = ptr.To[uint64](0)
	}
	if stats.Swap != nil {
		result.Swap.Time = metav1.NewTime(time.Unix(0, stats.Swap.Timestamp))
		if stats.Swap.SwapUsageBytes != nil {
			result.Swap.SwapUsageBytes = &stats.Swap.SwapUsageBytes.Value
		}
		if stats.Swap.SwapAvailableBytes != nil {
			result.Swap.SwapAvailableBytes = &stats.Swap.SwapAvailableBytes.Value
		}
	} else {
		result.Swap.Time = metav1.NewTime(time.Unix(0, time.Now().UnixNano()))
		result.Swap.SwapUsageBytes = ptr.To[uint64](0)
		result.Swap.SwapAvailableBytes = ptr.To[uint64](0)
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.KubeletPSI) {
		result.IO = &statsapi.IOStats{}
		if stats.Io != nil {
			result.IO.Time = metav1.NewTime(time.Unix(0, stats.Io.Timestamp))
			result.IO.PSI = makePSIStats(stats.Io.Psi)
		} else {
			result.IO.Time = metav1.NewTime(time.Unix(0, time.Now().UnixNano()))
		}
	}
	if stats.WritableLayer != nil {
		result.Rootfs.Time = metav1.NewTime(time.Unix(0, stats.WritableLayer.Timestamp))
		if stats.WritableLayer.UsedBytes != nil {
			result.Rootfs.UsedBytes = &stats.WritableLayer.UsedBytes.Value
		}
		if stats.WritableLayer.InodesUsed != nil {
			result.Rootfs.InodesUsed = &stats.WritableLayer.InodesUsed.Value
		}
	}
	fsID := stats.GetWritableLayer().GetFsId()
	var err error
	if fsID != nil {
		imageFsInfo, found := fsIDtoInfo[fsID.Mountpoint]
		if !found {
			imageFsInfo, err = p.getFsInfo(logger, fsID)
			if err != nil {
				return nil, fmt.Errorf("get filesystem info: %w", err)
			}
			fsIDtoInfo[fsID.Mountpoint] = imageFsInfo
		}
		if imageFsInfo != nil {
			// The image filesystem id is unknown to the local node or there's
			// an error on retrieving the stats. In these cases, we omit those stats
			// and return the best-effort partial result. See
			// https://github.com/kubernetes/heapster/issues/1793.
			result.Rootfs.AvailableBytes = &imageFsInfo.Available
			result.Rootfs.CapacityBytes = &imageFsInfo.Capacity
			result.Rootfs.InodesFree = imageFsInfo.InodesFree
			result.Rootfs.Inodes = imageFsInfo.Inodes
		}
	}
	// NOTE: This doesn't support the old pod log path, `/var/log/pods/UID`. For containers
	// using old log path, empty log stats are returned. This is fine, because we don't
	// officially support in-place upgrade anyway.
	result.Logs, err = p.hostStatsProvider.getPodContainerLogStats(meta.GetNamespace(), meta.GetName(), types.UID(meta.GetUid()), container.GetMetadata().GetName(), rootFsInfo)
	if err != nil {
		logger.Error(err, "Unable to fetch container log stats", "containerName", container.GetMetadata().GetName())
	}
	return result, nil
}

func (p *criStatsProvider) makeContainerCPUAndMemoryStats(
	stats *runtimeapi.ContainerStats,
	startTime time.Time,
	zeroMissingValues bool, // whether to write zeros to missing values
) *statsapi.ContainerStats {
	result := &statsapi.ContainerStats{
		Name: stats.Attributes.Metadata.Name,
		// The StartTime in the summary API is the container creation time.
		StartTime: metav1.NewTime(startTime),
		// UserDefinedMetrics is not supported by CRI.
	}
	getUint64 := func(val *runtimeapi.UInt64Value) *uint64 {
		if val != nil {
			return &val.Value
		} else if zeroMissingValues {
			return ptr.To[uint64](0)
		} else {
			return nil
		}
	}
	if stats.Cpu != nil {
		result.CPU = &statsapi.CPUStats{
			Time:                 metav1.NewTime(time.Unix(0, stats.Cpu.Timestamp)),
			UsageNanoCores:       p.getContainerUsageNanoCores(stats),
			UsageCoreNanoSeconds: getUint64(stats.Cpu.UsageCoreNanoSeconds),
			PSI:                  makePSIStats(stats.Cpu.Psi),
		}
	} else if zeroMissingValues {
		result.CPU = &statsapi.CPUStats{
			Time:                 metav1.NewTime(time.Unix(0, time.Now().UnixNano())),
			UsageCoreNanoSeconds: ptr.To[uint64](0),
			UsageNanoCores:       ptr.To[uint64](0),
		}
	}
	if stats.Memory != nil {
		result.Memory = &statsapi.MemoryStats{
			Time:            metav1.NewTime(time.Unix(0, stats.Memory.Timestamp)),
			AvailableBytes:  getUint64(stats.Memory.AvailableBytes),
			UsageBytes:      getUint64(stats.Memory.UsageBytes),
			WorkingSetBytes: getUint64(stats.Memory.WorkingSetBytes),
			RSSBytes:        getUint64(stats.Memory.RssBytes),
			PageFaults:      getUint64(stats.Memory.PageFaults),
			MajorPageFaults: getUint64(stats.Memory.MajorPageFaults),
			PSI:             makePSIStats(stats.Memory.Psi),
		}
	} else if zeroMissingValues {
		result.Memory = &statsapi.MemoryStats{
			Time:            metav1.NewTime(time.Unix(0, time.Now().UnixNano())),
			AvailableBytes:  ptr.To[uint64](0),
			UsageBytes:      ptr.To[uint64](0),
			WorkingSetBytes: ptr.To[uint64](0),
			RSSBytes:        ptr.To[uint64](0),
			PageFaults:      ptr.To[uint64](0),
			MajorPageFaults: ptr.To[uint64](0),
		}
	}
	if stats.Swap != nil {
		result.Swap = &statsapi.SwapStats{
			Time:               metav1.NewTime(time.Unix(0, stats.Swap.Timestamp)),
			SwapUsageBytes:     getUint64(stats.Swap.SwapUsageBytes),
			SwapAvailableBytes: getUint64(stats.Swap.SwapAvailableBytes),
		}
	} else if zeroMissingValues {
		result.Swap = &statsapi.SwapStats{
			Time:               metav1.NewTime(time.Unix(0, time.Now().UnixNano())),
			SwapUsageBytes:     ptr.To[uint64](0),
			SwapAvailableBytes: ptr.To[uint64](0),
		}
	}

	return result
}

func makePSIStats(stats *runtimeapi.PsiStats) *statsapi.PSIStats {
	if !utilfeature.DefaultFeatureGate.Enabled(features.KubeletPSI) {
		return nil
	}
	if stats == nil {
		return nil
	}
	result := &statsapi.PSIStats{}
	if stats.Full != nil {
		result.Full = statsapi.PSIData{
			Total:  stats.Full.Total,
			Avg10:  stats.Full.Avg10,
			Avg60:  stats.Full.Avg60,
			Avg300: stats.Full.Avg300,
		}
	}
	if stats.Some != nil {
		result.Some = statsapi.PSIData{
			Total:  stats.Some.Total,
			Avg10:  stats.Some.Avg10,
			Avg60:  stats.Some.Avg60,
			Avg300: stats.Some.Avg300,
		}
	}
	return result
}

// getContainerUsageNanoCores first attempts to get the usage nano cores from the stats reported
// by the CRI. If it is unable to, it gets the information from the cache instead.
func (p *criStatsProvider) getContainerUsageNanoCores(stats *runtimeapi.ContainerStats) *uint64 {
	if stats == nil || stats.Attributes == nil {
		return nil
	}

	// Bypass the cache if the CRI implementation specified the UsageNanoCores.
	if stats.Cpu != nil && stats.Cpu.UsageNanoCores != nil {
		return &stats.Cpu.UsageNanoCores.Value
	}

	p.mutex.RLock()
	defer p.mutex.RUnlock()

	cached, ok := p.cpuUsageCache[stats.Attributes.Id]
	if !ok || cached.usageNanoCores == nil {
		return nil
	}
	// return a copy of the usage
	latestUsage := *cached.usageNanoCores
	return &latestUsage
}

// getAndUpdateContainerUsageNanoCores first attempts to get the usage nano cores from the stats reported
// by the CRI. If it is unable to, it computes usageNanoCores based on the given and the cached usageCoreNanoSeconds,
// updates the cache with the computed usageNanoCores, and returns the usageNanoCores.
func (p *criStatsProvider) getAndUpdateContainerUsageNanoCores(logger klog.Logger, stats *runtimeapi.ContainerStats) *uint64 {
	if stats == nil || stats.Attributes == nil || stats.Cpu == nil {
		return nil
	}
	// Bypass the cache if the CRI implementation specified the UsageNanoCores.
	if stats.Cpu.UsageNanoCores != nil {
		return &stats.Cpu.UsageNanoCores.Value
	}
	// If there is no UsageNanoCores, nor UsageCoreNanoSeconds, there is no information to use
	if stats.Cpu.UsageCoreNanoSeconds == nil {
		return nil
	}
	id := stats.Attributes.Id
	usage, err := func() (*uint64, error) {
		p.mutex.Lock()
		defer p.mutex.Unlock()

		cached, ok := p.cpuUsageCache[id]
		if !ok || cached.stats.UsageCoreNanoSeconds == nil || stats.Cpu.UsageCoreNanoSeconds.Value < cached.stats.UsageCoreNanoSeconds.Value {
			// Cannot compute the usage now, but update the cached stats anyway
			p.cpuUsageCache[id] = &cpuUsageRecord{stats: stats.Cpu, usageNanoCores: nil}
			return nil, nil
		}

		newStats := stats.Cpu
		cachedStats := cached.stats
		nanoSeconds := newStats.Timestamp - cachedStats.Timestamp
		if nanoSeconds <= 0 {
			return nil, fmt.Errorf("zero or negative interval (%v - %v)", newStats.Timestamp, cachedStats.Timestamp)
		}
		usageNanoCores := uint64(float64(newStats.UsageCoreNanoSeconds.Value-cachedStats.UsageCoreNanoSeconds.Value) /
			float64(nanoSeconds) * float64(time.Second/time.Nanosecond))

		// Update cache with new value.
		usageToUpdate := usageNanoCores
		p.cpuUsageCache[id] = &cpuUsageRecord{stats: newStats, usageNanoCores: &usageToUpdate}

		return &usageNanoCores, nil
	}()

	if err != nil {
		// This should not happen. Log now to raise visibility
		logger.Error(err, "Failed updating cpu usage nano core")
	}
	return usage
}

func (p *criStatsProvider) cleanupOutdatedCaches() {
	p.mutex.Lock()
	defer p.mutex.Unlock()

	for k, v := range p.cpuUsageCache {
		if v == nil {
			delete(p.cpuUsageCache, k)
			continue
		}

		if time.Since(time.Unix(0, v.stats.Timestamp)) > defaultCachePeriod {
			delete(p.cpuUsageCache, k)
		}
	}
}

// removeTerminatedPods returns pods with terminated ones removed.
// It only removes a terminated pod when there is a running instance
// of the pod with the same name and namespace.
// This is needed because:
// 1) PodSandbox may be recreated;
// 2) Pod may be recreated with the same name and namespace.
func removeTerminatedPods(pods []*runtimeapi.PodSandbox) []*runtimeapi.PodSandbox {
	podMap := make(map[statsapi.PodReference][]*runtimeapi.PodSandbox)
	// Sort order by create time
	sort.Slice(pods, func(i, j int) bool {
		return pods[i].CreatedAt < pods[j].CreatedAt
	})
	for _, pod := range pods {
		refID := statsapi.PodReference{
			Name:      pod.GetMetadata().GetName(),
			Namespace: pod.GetMetadata().GetNamespace(),
			// UID is intentionally left empty.
		}
		podMap[refID] = append(podMap[refID], pod)
	}

	result := make([]*runtimeapi.PodSandbox, 0)
	for _, refs := range podMap {
		if len(refs) == 1 {
			result = append(result, refs[0])
			continue
		}
		found := false
		for i := 0; i < len(refs); i++ {
			if refs[i].State == runtimeapi.PodSandboxState_SANDBOX_READY {
				found = true
				result = append(result, refs[i])
			}
		}
		if !found {
			result = append(result, refs[len(refs)-1])
		}
	}
	return result
}

// removeTerminatedContainers removes all terminated containers since they should
// not be used for usage calculations.
func removeTerminatedContainers(containers []*runtimeapi.Container) []*runtimeapi.Container {
	containerMap := make(map[containerID][]*runtimeapi.Container)
	// Sort order by create time
	sort.Slice(containers, func(i, j int) bool {
		return containers[i].CreatedAt < containers[j].CreatedAt
	})
	for _, container := range containers {
		refID := containerID{
			podRef:        buildPodRef(container.Labels),
			containerName: kubetypes.GetContainerName(container.Labels),
		}
		containerMap[refID] = append(containerMap[refID], container)
	}

	result := make([]*runtimeapi.Container, 0)
	for _, refs := range containerMap {
		for i := 0; i < len(refs); i++ {
			if refs[i].State == runtimeapi.ContainerState_CONTAINER_RUNNING {
				result = append(result, refs[i])
			}
		}
	}
	return result
}

func (p *criStatsProvider) addCadvisorContainerStats(
	logger klog.Logger,
	cs *statsapi.ContainerStats,
	caPodStats *cadvisorapiv2.ContainerInfo,
) {
	if caPodStats.Spec.HasCustomMetrics {
		cs.UserDefinedMetrics = cadvisorInfoToUserDefinedMetrics(logger, caPodStats)
	}

	cpu, memory := cadvisorInfoToCPUandMemoryStats(caPodStats)
	if cpu != nil {
		cs.CPU = cpu
	}
	if memory != nil {
		cs.Memory = memory
	}

	swap := cadvisorInfoToSwapStats(caPodStats)
	if swap != nil {
		cs.Swap = swap
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.KubeletPSI) {
		io := cadvisorInfoToIOStats(caPodStats)
		if io != nil {
			cs.IO = io
		}
	}
}

func (p *criStatsProvider) addCadvisorContainerCPUAndMemoryStats(
	logger klog.Logger,
	cs *statsapi.ContainerStats,
	caPodStats *cadvisorapiv2.ContainerInfo,
) {
	if caPodStats.Spec.HasCustomMetrics {
		cs.UserDefinedMetrics = cadvisorInfoToUserDefinedMetrics(logger, caPodStats)
	}

	cpu, memory := cadvisorInfoToCPUandMemoryStats(caPodStats)
	if cpu != nil {
		cs.CPU = cpu
	}
	if memory != nil {
		cs.Memory = memory
	}
}

func getCRICadvisorStats(logger klog.Logger, infos map[string]cadvisorapiv2.ContainerInfo) (map[string]cadvisorapiv2.ContainerInfo, map[string]cadvisorapiv2.ContainerInfo) {
	stats := make(map[string]cadvisorapiv2.ContainerInfo)
	filteredInfos, cinfosByPodCgroupKey := filterTerminatedContainerInfoAndAssembleByPodCgroupKey(logger, infos)
	for key, info := range filteredInfos {
		// On systemd using devicemapper each mount into the container has an
		// associated cgroup. We ignore them to ensure we do not get duplicate
		// entries in our summary. For details on .mount units:
		// http://man7.org/linux/man-pages/man5/systemd.mount.5.html
		if strings.HasSuffix(key, ".mount") {
			continue
		}
		// Build the Pod key if this container is managed by a Pod
		if !isPodManagedContainer(logger, &info) {
			continue
		}
		stats[extractIDFromCgroupPath(key)] = info
	}
	return stats, cinfosByPodCgroupKey
}

func extractIDFromCgroupPath(cgroupPath string) string {
	// case0 == cgroupfs: "/kubepods/burstable/pod2fc932ce-fdcc-454b-97bd-aadfdeb4c340/9be25294016e2dc0340dd605ce1f57b492039b267a6a618a7ad2a7a58a740f32"
	id := filepath.Base(cgroupPath)

	// case1 == systemd: "/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod2fc932ce_fdcc_454b_97bd_aadfdeb4c340.slice/cri-containerd-aaefb9d8feed2d453b543f6d928cede7a4dbefa6a0ae7c9b990dd234c56e93b9.scope"
	// trim anything before the final '-' and suffix .scope
	systemdSuffix := ".scope"
	if strings.HasSuffix(id, systemdSuffix) {
		id = strings.TrimSuffix(id, systemdSuffix)
		components := strings.Split(id, "-")
		if len(components) > 1 {
			id = components[len(components)-1]
		}
	}
	return id
}

func criInterfaceToSummary(criIface *runtimeapi.NetworkInterfaceUsage) statsapi.InterfaceStats {
	return statsapi.InterfaceStats{
		Name:     criIface.Name,
		RxBytes:  valueOfUInt64Value(criIface.RxBytes),
		RxErrors: valueOfUInt64Value(criIface.RxErrors),
		TxBytes:  valueOfUInt64Value(criIface.TxBytes),
		TxErrors: valueOfUInt64Value(criIface.TxErrors),
	}
}

func valueOfUInt64Value(value *runtimeapi.UInt64Value) *uint64 {
	if value == nil {
		return nil
	}
	return &value.Value
}

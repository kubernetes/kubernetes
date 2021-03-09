/*
Copyright 2018 The Kubernetes Authors.

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
	"path/filepath"

	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/kuberuntime"
	"k8s.io/kubernetes/pkg/volume"
)

// PodEtcHostsPathFunc is a function to fetch a etc hosts path by pod uid.
type PodEtcHostsPathFunc func(podUID types.UID) string

// statsProviderByPath maps a path to its stats provider
type statsProviderByPath map[string]volume.StatsProvider

// HostStatsProvider defines an interface for providing host stats associated with pod.
type HostStatsProvider interface {
	// getPodLogStats gets stats associated with pod log usage
	getPodLogStats(podNamespace, podName string, podUID types.UID, rootFsInfo *cadvisorapiv2.FsInfo) (*statsapi.FsStats, error)
	// getPodContainerLogStats gets stats associated with container log usage
	getPodContainerLogStats(podNamespace, podName string, podUID types.UID, containerName string, rootFsInfo *cadvisorapiv2.FsInfo) (*statsapi.FsStats, error)
	// getPodEtcHostsStats gets stats associated with pod etc-hosts usage
	getPodEtcHostsStats(podUID types.UID, rootFsInfo *cadvisorapiv2.FsInfo) (*statsapi.FsStats, error)
}

type hostStatsProvider struct {
	// osInterface is the interface for syscalls.
	osInterface kubecontainer.OSInterface
	// podEtcHostsPathFunc fetches a pod etc hosts path by uid.
	podEtcHostsPathFunc PodEtcHostsPathFunc
}

// NewHostStatsProvider returns a new HostStatsProvider type struct.
func NewHostStatsProvider(osInterface kubecontainer.OSInterface, podEtcHostsPathFunc PodEtcHostsPathFunc) HostStatsProvider {
	return hostStatsProvider{
		osInterface:         osInterface,
		podEtcHostsPathFunc: podEtcHostsPathFunc,
	}
}

func (h hostStatsProvider) getPodLogStats(podNamespace, podName string, podUID types.UID, rootFsInfo *cadvisorapiv2.FsInfo) (*statsapi.FsStats, error) {
	statsByPath, err := h.podLogMetrics(podNamespace, podName, podUID)
	if err != nil {
		return nil, err
	}
	return statsByPathToFsStats(statsByPath, rootFsInfo)
}

// getPodContainerLogStats gets stats for container
func (h hostStatsProvider) getPodContainerLogStats(podNamespace, podName string, podUID types.UID, containerName string, rootFsInfo *cadvisorapiv2.FsInfo) (*statsapi.FsStats, error) {
	statsByPath, err := h.podContainerLogMetrics(podNamespace, podName, podUID, containerName)
	if err != nil {
		return nil, err
	}
	return statsByPathToFsStats(statsByPath, rootFsInfo)
}

// getPodEtcHostsStats gets status for pod etc hosts usage
func (h hostStatsProvider) getPodEtcHostsStats(podUID types.UID, rootFsInfo *cadvisorapiv2.FsInfo) (*statsapi.FsStats, error) {
	stats := h.podEtcHostsMetrics(podUID)
	hostStats, err := stats.GetStats()
	if err != nil {
		return nil, fmt.Errorf("failed to get stats %v", err)
	}
	result := rootFsInfoToFsStats(rootFsInfo)
	usedBytes := uint64(hostStats.Used.Value())
	inodesUsed := uint64(hostStats.InodesUsed.Value())
	result.UsedBytes = addUsage(result.UsedBytes, &usedBytes)
	result.InodesUsed = addUsage(result.InodesUsed, &inodesUsed)
	result.Time = maxUpdateTime(&result.Time, &hostStats.Time)
	return result, nil
}

func (h hostStatsProvider) podLogMetrics(podNamespace, podName string, podUID types.UID) (statsProviderByPath, error) {
	podLogsDirectoryPath := kuberuntime.BuildPodLogsDirectory(podNamespace, podName, podUID)
	return h.fileMetricsByDir(podLogsDirectoryPath)
}

func (h hostStatsProvider) podContainerLogMetrics(podNamespace, podName string, podUID types.UID, containerName string) (statsProviderByPath, error) {
	podContainerLogsDirectoryPath := kuberuntime.BuildContainerLogsDirectory(podNamespace, podName, podUID, containerName)
	return h.fileMetricsByDir(podContainerLogsDirectoryPath)
}

func (h hostStatsProvider) podEtcHostsMetrics(podUID types.UID) volume.StatsProvider {
	podEtcHostsPath := h.podEtcHostsPathFunc(podUID)
	return volume.NewMetricsDu(podEtcHostsPath)
}

// fileMetricsByDir returns stats by path for each file under specified directory
func (h hostStatsProvider) fileMetricsByDir(dirname string) (statsProviderByPath, error) {
	files, err := h.osInterface.ReadDir(dirname)
	if err != nil {
		return nil, err
	}
	results := statsProviderByPath{}
	for _, f := range files {
		if f.IsDir() {
			continue
		}
		// Only include *files* under pod log directory.
		fpath := filepath.Join(dirname, f.Name())
		results[fpath] = volume.NewMetricsDu(fpath)
	}
	return results, nil
}

// statsByPathToFsStats converts a stats provider by path to fs stats
func statsByPathToFsStats(statsByPath statsProviderByPath, rootFsInfo *cadvisorapiv2.FsInfo) (*statsapi.FsStats, error) {
	result := rootFsInfoToFsStats(rootFsInfo)
	for fpath, stats := range statsByPath {
		hostStats, err := stats.GetStats()
		if err != nil {
			return nil, fmt.Errorf("failed to get fsstats for %q: %v", fpath, err)
		}
		usedBytes := uint64(hostStats.Used.Value())
		inodesUsed := uint64(hostStats.InodesUsed.Value())
		result.UsedBytes = addUsage(result.UsedBytes, &usedBytes)
		result.InodesUsed = addUsage(result.InodesUsed, &inodesUsed)
		result.Time = maxUpdateTime(&result.Time, &hostStats.Time)
	}
	return result, nil
}

// rootFsInfoToFsStats is a utility to convert rootFsInfo into statsapi.FsStats
func rootFsInfoToFsStats(rootFsInfo *cadvisorapiv2.FsInfo) *statsapi.FsStats {
	return &statsapi.FsStats{
		Time:           metav1.NewTime(rootFsInfo.Timestamp),
		AvailableBytes: &rootFsInfo.Available,
		CapacityBytes:  &rootFsInfo.Capacity,
		InodesFree:     rootFsInfo.InodesFree,
		Inodes:         rootFsInfo.Inodes,
	}
}

/*
Copyright 2016 The Kubernetes Authors.

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

	statsapi "k8s.io/kubernetes/pkg/kubelet/apis/stats/v1alpha1"
)

type SummaryProvider interface {
	// Get provides a new Summary with the stats from Kubelet,
	// and will update some stats if updateStats is true
	Get(updateStats bool) (*statsapi.Summary, error)
	// GetCPUAndMemoryStats provides a new Summary with the CPU and memory stats from Kubelet,
	GetCPUAndMemoryStats() (*statsapi.Summary, error)
}

// summaryProviderImpl implements the SummaryProvider interface.
type summaryProviderImpl struct {
	provider StatsProvider
}

var _ SummaryProvider = &summaryProviderImpl{}

// NewSummaryProvider returns a SummaryProvider using the stats provided by the
// specified statsProvider.
func NewSummaryProvider(statsProvider StatsProvider) SummaryProvider {
	return &summaryProviderImpl{statsProvider}
}

func (sp *summaryProviderImpl) Get(updateStats bool) (*statsapi.Summary, error) {
	// TODO(timstclair): Consider returning a best-effort response if any of
	// the following errors occur.
	node, err := sp.provider.GetNode()
	if err != nil {
		return nil, fmt.Errorf("failed to get node info: %v", err)
	}
	nodeConfig := sp.provider.GetNodeConfig()
	rootStats, networkStats, err := sp.provider.GetCgroupStats("/", updateStats)
	if err != nil {
		return nil, fmt.Errorf("failed to get root cgroup stats: %v", err)
	}
	rootFsStats, err := sp.provider.RootFsStats()
	if err != nil {
		return nil, fmt.Errorf("failed to get rootFs stats: %v", err)
	}
	imageFsStats, err := sp.provider.ImageFsStats()
	if err != nil {
		return nil, fmt.Errorf("failed to get imageFs stats: %v", err)
	}
	podStats, err := sp.provider.ListPodStats()
	if err != nil {
		return nil, fmt.Errorf("failed to list pod stats: %v", err)
	}
	rlimit, err := sp.provider.RlimitStats()
	if err != nil {
		return nil, fmt.Errorf("failed to get rlimit stats: %v", err)
	}

	nodeStats := statsapi.NodeStats{
		NodeName:         node.Name,
		CPU:              rootStats.CPU,
		Memory:           rootStats.Memory,
		Network:          networkStats,
		StartTime:        rootStats.StartTime,
		Fs:               rootFsStats,
		Runtime:          &statsapi.RuntimeStats{ImageFs: imageFsStats},
		Rlimit:           rlimit,
		SystemContainers: sp.GetSystemContainersStats(nodeConfig, podStats, updateStats),
	}
	summary := statsapi.Summary{
		Node: nodeStats,
		Pods: podStats,
	}
	return &summary, nil
}

func (sp *summaryProviderImpl) GetCPUAndMemoryStats() (*statsapi.Summary, error) {
	summary, err := sp.Get(false)
	if err != nil {
		return nil, err
	}
	summary.Node.Network = nil
	summary.Node.Fs = nil
	summary.Node.Runtime = nil
	summary.Node.Rlimit = nil
	for i := 0; i < len(summary.Node.SystemContainers); i++ {
		summary.Node.SystemContainers[i].Accelerators = nil
		summary.Node.SystemContainers[i].Rootfs = nil
		summary.Node.SystemContainers[i].Logs = nil
		summary.Node.SystemContainers[i].UserDefinedMetrics = nil
	}
	for i := 0; i < len(summary.Pods); i++ {
		summary.Pods[i].Network = nil
		summary.Pods[i].VolumeStats = nil
		summary.Pods[i].EphemeralStorage = nil
		for j := 0; j < len(summary.Pods[i].Containers); j++ {
			summary.Pods[i].Containers[j].Accelerators = nil
			summary.Pods[i].Containers[j].Rootfs = nil
			summary.Pods[i].Containers[j].Logs = nil
			summary.Pods[i].Containers[j].UserDefinedMetrics = nil
		}
	}
	return summary, nil
}

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

//go:generate mockery
package stats

import (
	"context"
	"fmt"

	"k8s.io/klog/v2"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/util"
)

// SummaryProvider provides summaries of the stats from Kubelet.
type SummaryProvider interface {
	// Get provides a new Summary with the stats from Kubelet,
	// and will update some stats if updateStats is true
	Get(ctx context.Context, updateStats bool) (*statsapi.Summary, error)
	// GetCPUAndMemoryStats provides a new Summary with the CPU and memory stats from Kubelet,
	GetCPUAndMemoryStats(ctx context.Context) (*statsapi.Summary, error)
}

// summaryProviderImpl implements the SummaryProvider interface.
type summaryProviderImpl struct {
	// kubeletCreationTime is the time at which the summaryProvider was created.
	kubeletCreationTime metav1.Time
	// systemBootTime is the time at which the system was started
	systemBootTime metav1.Time

	provider Provider
}

var _ SummaryProvider = &summaryProviderImpl{}

// NewSummaryProvider returns a SummaryProvider using the stats provided by the
// specified statsProvider.
func NewSummaryProvider(ctx context.Context, statsProvider Provider) SummaryProvider {
	logger := klog.FromContext(ctx)
	kubeletCreationTime := metav1.Now()
	bootTime, err := util.GetBootTime()
	if err != nil {
		// bootTime will be zero if we encounter an error getting the boot time.
		logger.Info("Error getting system boot time. Node metrics will have an incorrect start time", "err", err)
	}

	return &summaryProviderImpl{
		kubeletCreationTime: kubeletCreationTime,
		systemBootTime:      metav1.NewTime(bootTime),
		provider:            statsProvider,
	}
}

func (sp *summaryProviderImpl) Get(ctx context.Context, updateStats bool) (*statsapi.Summary, error) {
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
	imageFsStats, containerFsStats, err := sp.provider.ImageFsStats(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get imageFs stats: %v", err)
	}
	var podStats []statsapi.PodStats
	if updateStats {
		podStats, err = sp.provider.ListPodStatsAndUpdateCPUNanoCoreUsage(ctx)
	} else {
		podStats, err = sp.provider.ListPodStats(ctx)
	}
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
		Swap:             rootStats.Swap,
		Network:          networkStats,
		StartTime:        sp.systemBootTime,
		Fs:               rootFsStats,
		Runtime:          &statsapi.RuntimeStats{ContainerFs: containerFsStats, ImageFs: imageFsStats},
		Rlimit:           rlimit,
		SystemContainers: sp.GetSystemContainersStats(ctx, nodeConfig, podStats, updateStats),
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.KubeletPSI) {
		nodeStats.IO = rootStats.IO
	}
	summary := statsapi.Summary{
		Node: nodeStats,
		Pods: podStats,
	}
	return &summary, nil
}

func (sp *summaryProviderImpl) GetCPUAndMemoryStats(ctx context.Context) (*statsapi.Summary, error) {
	// TODO(timstclair): Consider returning a best-effort response if any of
	// the following errors occur.
	node, err := sp.provider.GetNode()
	if err != nil {
		return nil, fmt.Errorf("failed to get node info: %v", err)
	}
	nodeConfig := sp.provider.GetNodeConfig()
	rootStats, err := sp.provider.GetCgroupCPUAndMemoryStats("/", false)
	if err != nil {
		return nil, fmt.Errorf("failed to get root cgroup stats: %v", err)
	}

	podStats, err := sp.provider.ListPodCPUAndMemoryStats(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to list pod stats: %v", err)
	}

	nodeStats := statsapi.NodeStats{
		NodeName:         node.Name,
		CPU:              rootStats.CPU,
		Memory:           rootStats.Memory,
		Swap:             rootStats.Swap,
		StartTime:        rootStats.StartTime,
		SystemContainers: sp.GetSystemContainersCPUAndMemoryStats(ctx, nodeConfig, podStats, false),
	}
	summary := statsapi.Summary{
		Node: nodeStats,
		Pods: podStats,
	}
	return &summary, nil
}

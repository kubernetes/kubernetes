//go:build !windows
// +build !windows

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
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"sigs.k8s.io/randfill"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	statstest "k8s.io/kubernetes/pkg/kubelet/server/stats/testing"
)

var (
	imageFsStats = getFsStats()
	rootFsStats  = getFsStats()
	node         = &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "test-node"}}
	nodeConfig   = cm.NodeConfig{
		RuntimeCgroupsName: "/runtime",
		SystemCgroupsName:  "/misc",
		KubeletCgroupsName: "/kubelet",
	}
	cgroupRoot  = "/kubepods"
	rlimitStats = getRlimitStats()
)

func TestSummaryProviderGetStatsNoSplitFileSystem(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KubeletPSI, true)
	ctx := context.Background()
	assert := assert.New(t)

	podStats := []statsapi.PodStats{
		{
			PodRef:      statsapi.PodReference{Name: "test-pod", Namespace: "test-namespace", UID: "UID_test-pod"},
			StartTime:   metav1.NewTime(time.Now()),
			Containers:  []statsapi.ContainerStats{*getContainerStats()},
			Network:     getNetworkStats(),
			VolumeStats: []statsapi.VolumeStats{*getVolumeStats()},
		},
	}
	cgroupStatsMap := map[string]struct {
		cs *statsapi.ContainerStats
		ns *statsapi.NetworkStats
	}{
		"/":        {cs: getContainerStats(), ns: getNetworkStats()},
		"/runtime": {cs: getContainerStats(), ns: getNetworkStats()},
		"/misc":    {cs: getContainerStats(), ns: getNetworkStats()},
		"/kubelet": {cs: getContainerStats(), ns: getNetworkStats()},
		"/pods":    {cs: getContainerStats(), ns: getNetworkStats()},
	}

	mockStatsProvider := statstest.NewMockProvider(t)

	mockStatsProvider.EXPECT().GetNode().Return(node, nil)
	mockStatsProvider.EXPECT().GetNodeConfig().Return(nodeConfig)
	mockStatsProvider.EXPECT().GetPodCgroupRoot().Return(cgroupRoot)
	mockStatsProvider.EXPECT().ListPodStats(ctx).Return(podStats, nil).Maybe()
	mockStatsProvider.EXPECT().ListPodStatsAndUpdateCPUNanoCoreUsage(ctx).Return(podStats, nil)
	mockStatsProvider.EXPECT().ImageFsStats(ctx).Return(imageFsStats, imageFsStats, nil)
	mockStatsProvider.EXPECT().RootFsStats().Return(rootFsStats, nil)
	mockStatsProvider.EXPECT().RlimitStats().Return(rlimitStats, nil)
	mockStatsProvider.EXPECT().GetCgroupStats("/", true).Return(cgroupStatsMap["/"].cs, cgroupStatsMap["/"].ns, nil)
	mockStatsProvider.EXPECT().GetCgroupStats("/runtime", false).Return(cgroupStatsMap["/runtime"].cs, cgroupStatsMap["/runtime"].ns, nil)
	mockStatsProvider.EXPECT().GetCgroupStats("/misc", false).Return(cgroupStatsMap["/misc"].cs, cgroupStatsMap["/misc"].ns, nil)
	mockStatsProvider.EXPECT().GetCgroupStats("/kubelet", false).Return(cgroupStatsMap["/kubelet"].cs, cgroupStatsMap["/kubelet"].ns, nil)
	mockStatsProvider.EXPECT().GetCgroupStats("/kubepods", true).Return(cgroupStatsMap["/pods"].cs, cgroupStatsMap["/pods"].ns, nil)

	kubeletCreationTime := metav1.Now()
	systemBootTime := metav1.Now()
	provider := summaryProviderImpl{kubeletCreationTime: kubeletCreationTime, systemBootTime: systemBootTime, provider: mockStatsProvider}
	summary, err := provider.Get(ctx, true)
	assert.NoError(err)

	assert.Equal("test-node", summary.Node.NodeName)
	assert.Equal(summary.Node.StartTime, systemBootTime)
	assert.Equal(summary.Node.CPU, cgroupStatsMap["/"].cs.CPU)
	assert.Equal(summary.Node.Memory, cgroupStatsMap["/"].cs.Memory)
	assert.Equal(summary.Node.Swap, cgroupStatsMap["/"].cs.Swap)
	assert.Equal(summary.Node.IO, cgroupStatsMap["/"].cs.IO)
	assert.Equal(summary.Node.Network, cgroupStatsMap["/"].ns)
	assert.Equal(summary.Node.Fs, rootFsStats)
	assert.Equal(&statsapi.RuntimeStats{ContainerFs: imageFsStats, ImageFs: imageFsStats}, summary.Node.Runtime)

	assert.Len(summary.Node.SystemContainers, 4)
	assert.Contains(summary.Node.SystemContainers, statsapi.ContainerStats{
		Name:               "kubelet",
		StartTime:          kubeletCreationTime,
		CPU:                cgroupStatsMap["/kubelet"].cs.CPU,
		Memory:             cgroupStatsMap["/kubelet"].cs.Memory,
		Accelerators:       cgroupStatsMap["/kubelet"].cs.Accelerators,
		UserDefinedMetrics: cgroupStatsMap["/kubelet"].cs.UserDefinedMetrics,
		Swap:               cgroupStatsMap["/kubelet"].cs.Swap,
		IO:                 cgroupStatsMap["/kubelet"].cs.IO,
	})
	assert.Contains(summary.Node.SystemContainers, statsapi.ContainerStats{
		Name:               "misc",
		StartTime:          cgroupStatsMap["/misc"].cs.StartTime,
		CPU:                cgroupStatsMap["/misc"].cs.CPU,
		Memory:             cgroupStatsMap["/misc"].cs.Memory,
		Accelerators:       cgroupStatsMap["/misc"].cs.Accelerators,
		UserDefinedMetrics: cgroupStatsMap["/misc"].cs.UserDefinedMetrics,
		Swap:               cgroupStatsMap["/misc"].cs.Swap,
		IO:                 cgroupStatsMap["/misc"].cs.IO,
	})
	assert.Contains(summary.Node.SystemContainers, statsapi.ContainerStats{
		Name:               "runtime",
		StartTime:          cgroupStatsMap["/runtime"].cs.StartTime,
		CPU:                cgroupStatsMap["/runtime"].cs.CPU,
		Memory:             cgroupStatsMap["/runtime"].cs.Memory,
		Accelerators:       cgroupStatsMap["/runtime"].cs.Accelerators,
		UserDefinedMetrics: cgroupStatsMap["/runtime"].cs.UserDefinedMetrics,
		Swap:               cgroupStatsMap["/runtime"].cs.Swap,
		IO:                 cgroupStatsMap["/runtime"].cs.IO,
	})
	assert.Contains(summary.Node.SystemContainers, statsapi.ContainerStats{
		Name:               "pods",
		StartTime:          cgroupStatsMap["/pods"].cs.StartTime,
		CPU:                cgroupStatsMap["/pods"].cs.CPU,
		Memory:             cgroupStatsMap["/pods"].cs.Memory,
		Accelerators:       cgroupStatsMap["/pods"].cs.Accelerators,
		UserDefinedMetrics: cgroupStatsMap["/pods"].cs.UserDefinedMetrics,
		Swap:               cgroupStatsMap["/pods"].cs.Swap,
		IO:                 cgroupStatsMap["/pods"].cs.IO,
	})
	assert.Equal(summary.Pods, podStats)
}

func TestSummaryProviderGetStatsSplitImageFs(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KubeletPSI, true)
	ctx := context.Background()
	assert := assert.New(t)

	podStats := []statsapi.PodStats{
		{
			PodRef:      statsapi.PodReference{Name: "test-pod", Namespace: "test-namespace", UID: "UID_test-pod"},
			StartTime:   metav1.NewTime(time.Now()),
			Containers:  []statsapi.ContainerStats{*getContainerStats()},
			Network:     getNetworkStats(),
			VolumeStats: []statsapi.VolumeStats{*getVolumeStats()},
		},
	}
	cgroupStatsMap := map[string]struct {
		cs *statsapi.ContainerStats
		ns *statsapi.NetworkStats
	}{
		"/":        {cs: getContainerStats(), ns: getNetworkStats()},
		"/runtime": {cs: getContainerStats(), ns: getNetworkStats()},
		"/misc":    {cs: getContainerStats(), ns: getNetworkStats()},
		"/kubelet": {cs: getContainerStats(), ns: getNetworkStats()},
		"/pods":    {cs: getContainerStats(), ns: getNetworkStats()},
	}

	mockStatsProvider := statstest.NewMockProvider(t)

	mockStatsProvider.EXPECT().GetNode().Return(node, nil)
	mockStatsProvider.EXPECT().GetNodeConfig().Return(nodeConfig)
	mockStatsProvider.EXPECT().GetPodCgroupRoot().Return(cgroupRoot)
	mockStatsProvider.EXPECT().ListPodStats(ctx).Return(podStats, nil).Maybe()
	mockStatsProvider.EXPECT().ListPodStatsAndUpdateCPUNanoCoreUsage(ctx).Return(podStats, nil)
	mockStatsProvider.EXPECT().RootFsStats().Return(rootFsStats, nil)
	mockStatsProvider.EXPECT().RlimitStats().Return(rlimitStats, nil)
	mockStatsProvider.EXPECT().GetCgroupStats("/", true).Return(cgroupStatsMap["/"].cs, cgroupStatsMap["/"].ns, nil)
	mockStatsProvider.EXPECT().GetCgroupStats("/runtime", false).Return(cgroupStatsMap["/runtime"].cs, cgroupStatsMap["/runtime"].ns, nil)
	mockStatsProvider.EXPECT().GetCgroupStats("/misc", false).Return(cgroupStatsMap["/misc"].cs, cgroupStatsMap["/misc"].ns, nil)
	mockStatsProvider.EXPECT().GetCgroupStats("/kubelet", false).Return(cgroupStatsMap["/kubelet"].cs, cgroupStatsMap["/kubelet"].ns, nil)
	mockStatsProvider.EXPECT().GetCgroupStats("/kubepods", true).Return(cgroupStatsMap["/pods"].cs, cgroupStatsMap["/pods"].ns, nil)

	mockStatsProvider.EXPECT().ImageFsStats(ctx).Return(imageFsStats, rootFsStats, nil)

	kubeletCreationTime := metav1.Now()
	systemBootTime := metav1.Now()
	provider := summaryProviderImpl{kubeletCreationTime: kubeletCreationTime, systemBootTime: systemBootTime, provider: mockStatsProvider}
	summary, err := provider.Get(ctx, true)
	assert.NoError(err)

	assert.Equal("test-node", summary.Node.NodeName)
	assert.Equal(summary.Node.StartTime, systemBootTime)
	assert.Equal(summary.Node.CPU, cgroupStatsMap["/"].cs.CPU)
	assert.Equal(summary.Node.Memory, cgroupStatsMap["/"].cs.Memory)
	assert.Equal(summary.Node.Swap, cgroupStatsMap["/"].cs.Swap)
	assert.Equal(summary.Node.IO, cgroupStatsMap["/"].cs.IO)
	assert.Equal(summary.Node.Network, cgroupStatsMap["/"].ns)
	assert.Equal(summary.Node.Fs, rootFsStats)
	// Since we are a split filesystem we want root filesystem to be container fs and image to be image filesystem
	assert.Equal(&statsapi.RuntimeStats{ContainerFs: rootFsStats, ImageFs: imageFsStats}, summary.Node.Runtime)

	assert.Len(summary.Node.SystemContainers, 4)
	assert.Contains(summary.Node.SystemContainers, statsapi.ContainerStats{
		Name:               "kubelet",
		StartTime:          kubeletCreationTime,
		CPU:                cgroupStatsMap["/kubelet"].cs.CPU,
		Memory:             cgroupStatsMap["/kubelet"].cs.Memory,
		Accelerators:       cgroupStatsMap["/kubelet"].cs.Accelerators,
		UserDefinedMetrics: cgroupStatsMap["/kubelet"].cs.UserDefinedMetrics,
		Swap:               cgroupStatsMap["/kubelet"].cs.Swap,
		IO:                 cgroupStatsMap["/kubelet"].cs.IO,
	})
	assert.Contains(summary.Node.SystemContainers, statsapi.ContainerStats{
		Name:               "misc",
		StartTime:          cgroupStatsMap["/misc"].cs.StartTime,
		CPU:                cgroupStatsMap["/misc"].cs.CPU,
		Memory:             cgroupStatsMap["/misc"].cs.Memory,
		Accelerators:       cgroupStatsMap["/misc"].cs.Accelerators,
		UserDefinedMetrics: cgroupStatsMap["/misc"].cs.UserDefinedMetrics,
		Swap:               cgroupStatsMap["/misc"].cs.Swap,
		IO:                 cgroupStatsMap["/misc"].cs.IO,
	})
	assert.Contains(summary.Node.SystemContainers, statsapi.ContainerStats{
		Name:               "runtime",
		StartTime:          cgroupStatsMap["/runtime"].cs.StartTime,
		CPU:                cgroupStatsMap["/runtime"].cs.CPU,
		Memory:             cgroupStatsMap["/runtime"].cs.Memory,
		Accelerators:       cgroupStatsMap["/runtime"].cs.Accelerators,
		UserDefinedMetrics: cgroupStatsMap["/runtime"].cs.UserDefinedMetrics,
		Swap:               cgroupStatsMap["/runtime"].cs.Swap,
		IO:                 cgroupStatsMap["/runtime"].cs.IO,
	})
	assert.Contains(summary.Node.SystemContainers, statsapi.ContainerStats{
		Name:               "pods",
		StartTime:          cgroupStatsMap["/pods"].cs.StartTime,
		CPU:                cgroupStatsMap["/pods"].cs.CPU,
		Memory:             cgroupStatsMap["/pods"].cs.Memory,
		Accelerators:       cgroupStatsMap["/pods"].cs.Accelerators,
		UserDefinedMetrics: cgroupStatsMap["/pods"].cs.UserDefinedMetrics,
		Swap:               cgroupStatsMap["/pods"].cs.Swap,
		IO:                 cgroupStatsMap["/pods"].cs.IO,
	})
	assert.Equal(summary.Pods, podStats)
}

func TestSummaryProviderGetCPUAndMemoryStats(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KubeletPSI, true)
	ctx := context.Background()
	assert := assert.New(t)

	podStats := []statsapi.PodStats{
		{
			PodRef:     statsapi.PodReference{Name: "test-pod", Namespace: "test-namespace", UID: "UID_test-pod"},
			StartTime:  metav1.NewTime(time.Now()),
			Containers: []statsapi.ContainerStats{*getContainerStats()},
		},
	}
	cgroupStatsMap := map[string]struct {
		cs *statsapi.ContainerStats
	}{
		"/":        {cs: getVolumeCPUAndMemoryStats()},
		"/runtime": {cs: getVolumeCPUAndMemoryStats()},
		"/misc":    {cs: getVolumeCPUAndMemoryStats()},
		"/kubelet": {cs: getVolumeCPUAndMemoryStats()},
		"/pods":    {cs: getVolumeCPUAndMemoryStats()},
	}

	mockStatsProvider := statstest.NewMockProvider(t)

	mockStatsProvider.EXPECT().GetNode().Return(node, nil)
	mockStatsProvider.EXPECT().GetNodeConfig().Return(nodeConfig)
	mockStatsProvider.EXPECT().GetPodCgroupRoot().Return(cgroupRoot)
	mockStatsProvider.EXPECT().ListPodCPUAndMemoryStats(ctx).Return(podStats, nil)
	mockStatsProvider.EXPECT().GetCgroupCPUAndMemoryStats("/", false).Return(cgroupStatsMap["/"].cs, nil)
	mockStatsProvider.EXPECT().GetCgroupCPUAndMemoryStats("/runtime", false).Return(cgroupStatsMap["/runtime"].cs, nil)
	mockStatsProvider.EXPECT().GetCgroupCPUAndMemoryStats("/misc", false).Return(cgroupStatsMap["/misc"].cs, nil)
	mockStatsProvider.EXPECT().GetCgroupCPUAndMemoryStats("/kubelet", false).Return(cgroupStatsMap["/kubelet"].cs, nil)
	mockStatsProvider.EXPECT().GetCgroupCPUAndMemoryStats("/kubepods", false).Return(cgroupStatsMap["/pods"].cs, nil)

	provider := NewSummaryProvider(mockStatsProvider)
	summary, err := provider.GetCPUAndMemoryStats(ctx)
	assert.NoError(err)

	assert.Equal("test-node", summary.Node.NodeName)
	assert.Equal(summary.Node.StartTime, cgroupStatsMap["/"].cs.StartTime)
	assert.Equal(summary.Node.CPU, cgroupStatsMap["/"].cs.CPU)
	assert.Equal(summary.Node.Memory, cgroupStatsMap["/"].cs.Memory)
	assert.Nil(summary.Node.Network)
	assert.Nil(summary.Node.Fs)
	assert.Nil(summary.Node.Runtime)
	assert.Nil(summary.Node.IO)

	assert.Len(summary.Node.SystemContainers, 4)
	assert.Contains(summary.Node.SystemContainers, statsapi.ContainerStats{
		Name:      "kubelet",
		StartTime: cgroupStatsMap["/kubelet"].cs.StartTime,
		CPU:       cgroupStatsMap["/kubelet"].cs.CPU,
		Memory:    cgroupStatsMap["/kubelet"].cs.Memory,
	})
	assert.Contains(summary.Node.SystemContainers, statsapi.ContainerStats{
		Name:      "misc",
		StartTime: cgroupStatsMap["/misc"].cs.StartTime,
		CPU:       cgroupStatsMap["/misc"].cs.CPU,
		Memory:    cgroupStatsMap["/misc"].cs.Memory,
	})
	assert.Contains(summary.Node.SystemContainers, statsapi.ContainerStats{
		Name:      "runtime",
		StartTime: cgroupStatsMap["/runtime"].cs.StartTime,
		CPU:       cgroupStatsMap["/runtime"].cs.CPU,
		Memory:    cgroupStatsMap["/runtime"].cs.Memory,
	})
	assert.Contains(summary.Node.SystemContainers, statsapi.ContainerStats{
		Name:      "pods",
		StartTime: cgroupStatsMap["/pods"].cs.StartTime,
		CPU:       cgroupStatsMap["/pods"].cs.CPU,
		Memory:    cgroupStatsMap["/pods"].cs.Memory,
	})
	assert.Equal(summary.Pods, podStats)
}

func getFsStats() *statsapi.FsStats {
	f := randfill.New().NilChance(0)
	v := &statsapi.FsStats{}
	f.Fill(v)
	return v
}

func getContainerStats() *statsapi.ContainerStats {
	f := randfill.New().NilChance(0)
	v := &statsapi.ContainerStats{}
	f.Fill(v)
	return v
}
func getVolumeCPUAndMemoryStats() *statsapi.ContainerStats {
	f := randfill.New().NilChance(0)
	v := &statsapi.ContainerStats{}
	f.Fill(&v.Name)
	f.Fill(&v.StartTime)
	f.Fill(v.CPU)
	f.Fill(v.Memory)
	return v
}

func getVolumeStats() *statsapi.VolumeStats {
	f := randfill.New().NilChance(0)
	v := &statsapi.VolumeStats{}
	f.Fill(v)
	return v
}

func getNetworkStats() *statsapi.NetworkStats {
	f := randfill.New().NilChance(0)
	v := &statsapi.NetworkStats{}
	f.Fill(v)
	return v
}

func getRlimitStats() *statsapi.RlimitStats {
	f := randfill.New().NilChance(0)
	v := &statsapi.RlimitStats{}
	f.Fill(v)
	return v
}

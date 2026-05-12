//go:build windows

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
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"sigs.k8s.io/randfill"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	statstest "k8s.io/kubernetes/pkg/kubelet/server/stats/testing"
)

func TestSummaryProvider(t *testing.T) {
	var (
		ctx            = context.Background()
		podStats       = []statsapi.PodStats{*getPodStats()}
		imageFsStats   = getFsStats()
		rootFsStats    = getFsStats()
		node           = &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "test-node"}}
		nodeConfig     = cm.NodeConfig{}
		cgroupRoot     = "/kubepods"
		cgroupStatsMap = map[string]struct {
			cs *statsapi.ContainerStats
			ns *statsapi.NetworkStats
		}{
			"/":     {cs: getContainerStats()},
			"/pods": {cs: getContainerStats()},
		}
	)

	assert := assert.New(t)

	mockStatsProvider := statstest.NewMockProvider(t)
	mockStatsProvider.EXPECT().GetNode(ctx).Return(node, nil).Maybe()
	mockStatsProvider.EXPECT().GetNodeConfig().Return(nodeConfig).Maybe()
	mockStatsProvider.EXPECT().GetPodCgroupRoot().Return(cgroupRoot).Maybe()
	mockStatsProvider.EXPECT().ListPodStats(ctx).Return(podStats, nil).Maybe()
	mockStatsProvider.EXPECT().ListPodStatsAndUpdateCPUNanoCoreUsage(ctx).Return(podStats, nil).Maybe()
	mockStatsProvider.EXPECT().ImageFsStats(ctx).Return(imageFsStats, imageFsStats, nil).Maybe()
	mockStatsProvider.EXPECT().RootFsStats().Return(rootFsStats, nil).Maybe()
	mockStatsProvider.EXPECT().RlimitStats().Return(nil, nil).Maybe()
	mockStatsProvider.EXPECT().GetCgroupStats("/", true).Return(cgroupStatsMap["/"].cs, cgroupStatsMap["/"].ns, nil).Maybe()

	kubeletCreationTime := metav1.Now()
	systemBootTime := metav1.Now()
	provider := summaryProviderImpl{kubeletCreationTime: kubeletCreationTime, systemBootTime: systemBootTime, provider: mockStatsProvider}
	summary, err := provider.Get(ctx, true)
	assert.NoError(err)

	assert.Equal("test-node", summary.Node.NodeName)
	assert.Equal(systemBootTime, summary.Node.StartTime)
	assert.Equal(cgroupStatsMap["/"].cs.CPU, summary.Node.CPU)
	assert.Equal(cgroupStatsMap["/"].cs.Memory, summary.Node.Memory)
	assert.Equal(cgroupStatsMap["/"].ns, summary.Node.Network)
	assert.Equal(rootFsStats, summary.Node.Fs)
	assert.Equal(&statsapi.RuntimeStats{ContainerFs: imageFsStats, ImageFs: imageFsStats}, summary.Node.Runtime)

	assert.NoError(err)
	assert.Len(summary.Node.SystemContainers, 2)
	assert.Equal("pods", summary.Node.SystemContainers[0].Name)
	assert.Equal(podStats[0].CPU.UsageCoreNanoSeconds, summary.Node.SystemContainers[0].CPU.UsageCoreNanoSeconds)
	assert.Equal(podStats[0].CPU.UsageNanoCores, summary.Node.SystemContainers[0].CPU.UsageNanoCores)
	assert.Equal(podStats[0].Memory.WorkingSetBytes, summary.Node.SystemContainers[0].Memory.WorkingSetBytes)
	assert.Equal(podStats[0].Memory.UsageBytes, summary.Node.SystemContainers[0].Memory.UsageBytes)
	assert.Equal(podStats[0].Memory.AvailableBytes, summary.Node.SystemContainers[0].Memory.AvailableBytes)
	assert.Equal(statsapi.SystemContainerWindowsGlobalCommitMemory, summary.Node.SystemContainers[1].Name)
	assert.NotEqual(nil, summary.Node.SystemContainers[1].Memory)
	assert.NotEqual(nil, summary.Node.SystemContainers[1].Memory.AvailableBytes)
	assert.NotEqual(nil, summary.Node.SystemContainers[1].Memory.UsageBytes)
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

func getPodStats() *statsapi.PodStats {
	containerStats := getContainerStats()
	podStats := statsapi.PodStats{
		PodRef:     statsapi.PodReference{Name: "test-pod", Namespace: "test-namespace", UID: "UID_test-pod"},
		StartTime:  metav1.NewTime(time.Now()),
		Containers: []statsapi.ContainerStats{*containerStats},
		CPU:        containerStats.CPU,
		Memory:     containerStats.Memory,
	}

	return &podStats
}

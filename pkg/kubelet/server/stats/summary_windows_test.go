// +build windows

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
	"testing"
	"time"

	fuzz "github.com/google/gofuzz"
	"github.com/stretchr/testify/assert"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	statstest "k8s.io/kubernetes/pkg/kubelet/server/stats/testing"
)

func TestSummaryProvider(t *testing.T) {
	var (
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

	mockStatsProvider := new(statstest.StatsProvider)
	mockStatsProvider.
		On("GetNode").Return(node, nil).
		On("GetNodeConfig").Return(nodeConfig).
		On("GetPodCgroupRoot").Return(cgroupRoot).
		On("ListPodStats").Return(podStats, nil).
		On("ListPodStatsAndUpdateCPUNanoCoreUsage").Return(podStats, nil).
		On("ImageFsStats").Return(imageFsStats, nil).
		On("RootFsStats").Return(rootFsStats, nil).
		On("RlimitStats").Return(nil, nil).
		On("GetCgroupStats", "/", true).Return(cgroupStatsMap["/"].cs, cgroupStatsMap["/"].ns, nil)

	provider := NewSummaryProvider(mockStatsProvider)
	summary, err := provider.Get(true)
	assert.NoError(err)

	assert.Equal(summary.Node.NodeName, "test-node")
	assert.Equal(summary.Node.StartTime, cgroupStatsMap["/"].cs.StartTime)
	assert.Equal(summary.Node.CPU, cgroupStatsMap["/"].cs.CPU)
	assert.Equal(summary.Node.Memory, cgroupStatsMap["/"].cs.Memory)
	assert.Equal(summary.Node.Network, cgroupStatsMap["/"].ns)
	assert.Equal(summary.Node.Fs, rootFsStats)
	assert.Equal(summary.Node.Runtime, &statsapi.RuntimeStats{ImageFs: imageFsStats})

	assert.Equal(len(summary.Node.SystemContainers), 1)
	assert.Equal(summary.Node.SystemContainers[0].Name, "pods")
	assert.Equal(summary.Node.SystemContainers[0].CPU.UsageCoreNanoSeconds, podStats[0].CPU.UsageCoreNanoSeconds)
	assert.Equal(summary.Node.SystemContainers[0].CPU.UsageNanoCores, podStats[0].CPU.UsageNanoCores)
	assert.Equal(summary.Node.SystemContainers[0].Memory.WorkingSetBytes, podStats[0].Memory.WorkingSetBytes)
	assert.Equal(summary.Node.SystemContainers[0].Memory.UsageBytes, podStats[0].Memory.UsageBytes)
	assert.Equal(summary.Node.SystemContainers[0].Memory.AvailableBytes, podStats[0].Memory.AvailableBytes)
	assert.Equal(summary.Pods, podStats)
}

func getFsStats() *statsapi.FsStats {
	f := fuzz.New().NilChance(0)
	v := &statsapi.FsStats{}
	f.Fuzz(v)
	return v
}

func getContainerStats() *statsapi.ContainerStats {
	f := fuzz.New().NilChance(0)
	v := &statsapi.ContainerStats{}
	f.Fuzz(v)
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

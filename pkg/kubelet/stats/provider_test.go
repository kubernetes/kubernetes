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
	"fmt"
	"runtime"
	"strings"
	"testing"
	"time"

	cadvisorapiv1 "github.com/google/cadvisor/info/v1"
	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/randfill"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	cadvisortest "k8s.io/kubernetes/pkg/kubelet/cadvisor/testing"
	kubecontainertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	kubepodtest "k8s.io/kubernetes/pkg/kubelet/pod/testing"
	serverstats "k8s.io/kubernetes/pkg/kubelet/server/stats"
	"k8s.io/kubernetes/pkg/volume"
)

const (
	// Offsets from seed value in generated container stats.
	offsetCPUUsageCores = iota
	offsetCPUUsageCoreSeconds
	offsetMemPageFaults
	offsetMemMajorPageFaults
	offsetMemUsageBytes
	offsetMemRSSBytes
	offsetMemWorkingSetBytes
	offsetNetRxBytes
	offsetNetRxErrors
	offsetNetTxBytes
	offsetNetTxErrors
	offsetFsCapacity
	offsetFsAvailable
	offsetFsUsage
	offsetFsInodes
	offsetFsInodesFree
	offsetFsTotalUsageBytes
	offsetFsBaseUsageBytes
	offsetFsInodeUsage
	offsetAcceleratorDutyCycle
	offsetMemSwapUsageBytes
)

var (
	timestamp    = time.Now()
	creationTime = timestamp.Add(-5 * time.Minute)
)

func TestGetCgroupStats(t *testing.T) {
	const (
		cgroupName        = "test-cgroup-name"
		containerInfoSeed = 1000
		updateStats       = false
	)

	var (
		mockCadvisor     = cadvisortest.NewMockInterface(t)
		mockPodManager   = new(kubepodtest.MockManager)
		mockRuntimeCache = new(kubecontainertest.MockRuntimeCache)

		assert  = assert.New(t)
		options = cadvisorapiv2.RequestOptions{IdType: cadvisorapiv2.TypeName, Count: 2, Recursive: false}

		containerInfo    = getTestContainerInfo(containerInfoSeed, "test-pod", "test-ns", "test-container")
		containerInfoMap = map[string]cadvisorapiv2.ContainerInfo{cgroupName: containerInfo}
	)

	mockCadvisor.EXPECT().ContainerInfoV2(cgroupName, options).Return(containerInfoMap, nil)

	provider := newStatsProvider(mockCadvisor, mockPodManager, mockRuntimeCache, fakeContainerStatsProvider{})
	cs, ns, err := provider.GetCgroupStats(cgroupName, updateStats)
	assert.NoError(err)

	checkCPUStats(t, "", containerInfoSeed, cs.CPU)
	checkMemoryStats(t, "", containerInfoSeed, containerInfo, cs.Memory)
	checkNetworkStats(t, "", containerInfoSeed, ns)
	checkSwapStats(t, "", containerInfoSeed, containerInfo, cs.Swap)

	assert.Equal(cgroupName, cs.Name)
	assert.Equal(metav1.NewTime(containerInfo.Spec.CreationTime), cs.StartTime)
}

func TestGetCgroupCPUAndMemoryStats(t *testing.T) {
	const (
		cgroupName        = "test-cgroup-name"
		containerInfoSeed = 1000
		updateStats       = false
	)

	var (
		mockCadvisor     = cadvisortest.NewMockInterface(t)
		mockPodManager   = new(kubepodtest.MockManager)
		mockRuntimeCache = new(kubecontainertest.MockRuntimeCache)

		assert  = assert.New(t)
		options = cadvisorapiv2.RequestOptions{IdType: cadvisorapiv2.TypeName, Count: 2, Recursive: false}

		containerInfo    = getTestContainerInfo(containerInfoSeed, "test-pod", "test-ns", "test-container")
		containerInfoMap = map[string]cadvisorapiv2.ContainerInfo{cgroupName: containerInfo}
	)

	mockCadvisor.EXPECT().ContainerInfoV2(cgroupName, options).Return(containerInfoMap, nil)

	provider := newStatsProvider(mockCadvisor, mockPodManager, mockRuntimeCache, fakeContainerStatsProvider{})
	cs, err := provider.GetCgroupCPUAndMemoryStats(cgroupName, updateStats)
	assert.NoError(err)

	checkCPUStats(t, "", containerInfoSeed, cs.CPU)
	checkMemoryStats(t, "", containerInfoSeed, containerInfo, cs.Memory)

	assert.Equal(cgroupName, cs.Name)
	assert.Equal(metav1.NewTime(containerInfo.Spec.CreationTime), cs.StartTime)
}

func TestRootFsStats(t *testing.T) {
	const (
		rootFsInfoSeed    = 1000
		containerInfoSeed = 2000
	)

	var (
		mockCadvisor     = cadvisortest.NewMockInterface(t)
		mockPodManager   = new(kubepodtest.MockManager)
		mockRuntimeCache = new(kubecontainertest.MockRuntimeCache)

		assert  = assert.New(t)
		options = cadvisorapiv2.RequestOptions{IdType: cadvisorapiv2.TypeName, Count: 2, Recursive: false}

		rootFsInfo       = getTestFsInfo(rootFsInfoSeed)
		containerInfo    = getTestContainerInfo(containerInfoSeed, "test-pod", "test-ns", "test-container")
		containerInfoMap = map[string]cadvisorapiv2.ContainerInfo{"/": containerInfo}
	)

	mockCadvisor.EXPECT().RootFsInfo().Return(rootFsInfo, nil)
	mockCadvisor.EXPECT().ContainerInfoV2("/", options).Return(containerInfoMap, nil)

	provider := newStatsProvider(mockCadvisor, mockPodManager, mockRuntimeCache, fakeContainerStatsProvider{})
	stats, err := provider.RootFsStats()
	assert.NoError(err)

	checkFsStats(t, "", rootFsInfoSeed, stats)

	assert.Equal(metav1.NewTime(containerInfo.Stats[0].Timestamp), stats.Time)
	assert.Equal(rootFsInfo.Usage, *stats.UsedBytes)
	assert.Equal(*rootFsInfo.Inodes-*rootFsInfo.InodesFree, *stats.InodesUsed)
}

func TestHasDedicatedImageFs(t *testing.T) {
	ctx := context.Background()
	imageStatsExpected := &statsapi.FsStats{AvailableBytes: uint64Ptr(1)}

	for desc, test := range map[string]struct {
		rootfsDevice     string
		imagefsDevice    string
		dedicated        bool
		imageFsStats     *statsapi.FsStats
		containerFsStats *statsapi.FsStats
	}{
		"dedicated device for image filesystem": {
			rootfsDevice:  "root/device",
			imagefsDevice: "image/device",
			dedicated:     true,
			imageFsStats:  imageStatsExpected,
		},
		"shared device for image filesystem": {
			rootfsDevice:     "share/device",
			imagefsDevice:    "share/device",
			dedicated:        false,
			imageFsStats:     imageStatsExpected,
			containerFsStats: imageStatsExpected,
		},
		"split filesystem for images": {
			rootfsDevice:     "root/device",
			imagefsDevice:    "root/device",
			dedicated:        true,
			imageFsStats:     &statsapi.FsStats{AvailableBytes: uint64Ptr(1)},
			containerFsStats: &statsapi.FsStats{AvailableBytes: uint64Ptr(2)},
		},
	} {
		t.Logf("TestCase %q", desc)
		var (
			mockCadvisor     = cadvisortest.NewMockInterface(t)
			mockPodManager   = new(kubepodtest.MockManager)
			mockRuntimeCache = new(kubecontainertest.MockRuntimeCache)
		)
		mockCadvisor.EXPECT().RootFsInfo().Return(cadvisorapiv2.FsInfo{Device: test.rootfsDevice}, nil)
		provider := newStatsProvider(mockCadvisor, mockPodManager, mockRuntimeCache, fakeContainerStatsProvider{
			device:      test.imagefsDevice,
			imageFs:     test.imageFsStats,
			containerFs: test.containerFsStats,
		})

		dedicated, err := provider.HasDedicatedImageFs(ctx)
		assert.NoError(t, err)
		assert.Equal(t, test.dedicated, dedicated)
	}
}

func getTerminatedContainerInfo(seed int, podName string, podNamespace string, containerName string) cadvisorapiv2.ContainerInfo {
	cinfo := getTestContainerInfo(seed, podName, podNamespace, containerName)
	cinfo.Stats[0].Memory.RSS = 0
	cinfo.Stats[0].CpuInst.Usage.Total = 0
	cinfo.Stats[0].Network = &cadvisorapiv2.NetworkStats{
		Interfaces: []cadvisorapiv1.InterfaceStats{{
			Name:     "eth0",
			RxBytes:  0,
			RxErrors: 0,
			TxBytes:  0,
			TxErrors: 0,
		}, {
			Name:     "cbr0",
			RxBytes:  0,
			RxErrors: 0,
			TxBytes:  0,
			TxErrors: 0,
		}},
	}
	return cinfo
}

func getContainerInfoWithZeroCpuMem(seed int, podName string, podNamespace string, containerName string) cadvisorapiv2.ContainerInfo {
	cinfo := getTestContainerInfo(seed, podName, podNamespace, containerName)
	cinfo.Stats[0].Memory.RSS = 0
	cinfo.Stats[0].CpuInst.Usage.Total = 0
	return cinfo
}

func getTestContainerInfo(seed int, podName string, podNamespace string, containerName string) cadvisorapiv2.ContainerInfo {
	labels := map[string]string{}
	if podName != "" {
		labels = map[string]string{
			"io.kubernetes.pod.name":       podName,
			"io.kubernetes.pod.uid":        "UID" + podName,
			"io.kubernetes.pod.namespace":  podNamespace,
			"io.kubernetes.container.name": containerName,
		}
	}
	// by default, kernel will set memory.limit_in_bytes to 1 << 63 if not bounded
	unlimitedMemory := uint64(1 << 63)
	spec := cadvisorapiv2.ContainerSpec{
		CreationTime: testTime(creationTime, seed),
		HasCpu:       true,
		HasMemory:    true,
		HasNetwork:   true,
		Labels:       labels,
		Memory: cadvisorapiv2.MemorySpec{
			Limit:     unlimitedMemory,
			SwapLimit: unlimitedMemory,
		},
		CustomMetrics: generateCustomMetricSpec(),
	}

	totalUsageBytes := uint64(seed + offsetFsTotalUsageBytes)
	baseUsageBytes := uint64(seed + offsetFsBaseUsageBytes)
	inodeUsage := uint64(seed + offsetFsInodeUsage)

	stats := cadvisorapiv2.ContainerStats{
		Timestamp: testTime(timestamp, seed),
		Cpu:       &cadvisorapiv1.CpuStats{},
		CpuInst:   &cadvisorapiv2.CpuInstStats{},
		Memory: &cadvisorapiv1.MemoryStats{
			Usage:      uint64(seed + offsetMemUsageBytes),
			WorkingSet: uint64(seed + offsetMemWorkingSetBytes),
			RSS:        uint64(seed + offsetMemRSSBytes),
			ContainerData: cadvisorapiv1.MemoryStatsMemoryData{
				Pgfault:    uint64(seed + offsetMemPageFaults),
				Pgmajfault: uint64(seed + offsetMemMajorPageFaults),
			},
			Swap: uint64(seed + offsetMemSwapUsageBytes),
		},
		Network: &cadvisorapiv2.NetworkStats{
			Interfaces: []cadvisorapiv1.InterfaceStats{{
				Name:     "eth0",
				RxBytes:  uint64(seed + offsetNetRxBytes),
				RxErrors: uint64(seed + offsetNetRxErrors),
				TxBytes:  uint64(seed + offsetNetTxBytes),
				TxErrors: uint64(seed + offsetNetTxErrors),
			}, {
				Name:     "cbr0",
				RxBytes:  100,
				RxErrors: 100,
				TxBytes:  100,
				TxErrors: 100,
			}},
		},
		CustomMetrics: generateCustomMetrics(spec.CustomMetrics),
		Filesystem: &cadvisorapiv2.FilesystemStats{
			TotalUsageBytes: &totalUsageBytes,
			BaseUsageBytes:  &baseUsageBytes,
			InodeUsage:      &inodeUsage,
		},
		Accelerators: []cadvisorapiv1.AcceleratorStats{
			{
				Make:        "nvidia",
				Model:       "Tesla K80",
				ID:          "foobar",
				MemoryTotal: uint64(seed + offsetMemUsageBytes),
				MemoryUsed:  uint64(seed + offsetMemUsageBytes),
				DutyCycle:   uint64(seed + offsetAcceleratorDutyCycle),
			},
		},
	}
	stats.Cpu.Usage.Total = uint64(seed + offsetCPUUsageCoreSeconds)
	stats.CpuInst.Usage.Total = uint64(seed + offsetCPUUsageCores)
	return cadvisorapiv2.ContainerInfo{
		Spec:  spec,
		Stats: []*cadvisorapiv2.ContainerStats{&stats},
	}
}

func getTestFsInfo(seed int) cadvisorapiv2.FsInfo {
	var (
		inodes     = uint64(seed + offsetFsInodes)
		inodesFree = uint64(seed + offsetFsInodesFree)
	)
	return cadvisorapiv2.FsInfo{
		Timestamp:  time.Now(),
		Device:     "test-device",
		Mountpoint: "test-mount-point",
		Capacity:   uint64(seed + offsetFsCapacity),
		Available:  uint64(seed + offsetFsAvailable),
		Usage:      uint64(seed + offsetFsUsage),
		Inodes:     &inodes,
		InodesFree: &inodesFree,
	}
}

func getTestFsInfoWithDifferentMount(seed int, device string) cadvisorapiv2.FsInfo {
	var (
		inodes     = uint64(seed + offsetFsInodes)
		inodesFree = uint64(seed + offsetFsInodesFree)
	)
	return cadvisorapiv2.FsInfo{
		Timestamp:  time.Now(),
		Device:     device,
		Mountpoint: "test-mount-point",
		Capacity:   uint64(seed + offsetFsCapacity),
		Available:  uint64(seed + offsetFsAvailable),
		Usage:      uint64(seed + offsetFsUsage),
		Inodes:     &inodes,
		InodesFree: &inodesFree,
	}
}

func getPodVolumeStats(seed int, volumeName string) statsapi.VolumeStats {
	availableBytes := uint64(seed + offsetFsAvailable)
	capacityBytes := uint64(seed + offsetFsCapacity)
	usedBytes := uint64(seed + offsetFsUsage)
	inodes := uint64(seed + offsetFsInodes)
	inodesFree := uint64(seed + offsetFsInodesFree)
	inodesUsed := uint64(seed + offsetFsInodeUsage)
	fsStats := statsapi.FsStats{
		Time:           metav1.NewTime(time.Now()),
		AvailableBytes: &availableBytes,
		CapacityBytes:  &capacityBytes,
		UsedBytes:      &usedBytes,
		Inodes:         &inodes,
		InodesFree:     &inodesFree,
		InodesUsed:     &inodesUsed,
	}
	return statsapi.VolumeStats{
		FsStats: fsStats,
		Name:    volumeName,
	}
}

func generateCustomMetricSpec() []cadvisorapiv1.MetricSpec {
	f := randfill.New().NilChance(0).Funcs(
		func(e *cadvisorapiv1.MetricSpec, c randfill.Continue) {
			c.Fill(&e.Name)
			switch c.Intn(3) {
			case 0:
				e.Type = cadvisorapiv1.MetricGauge
			case 1:
				e.Type = cadvisorapiv1.MetricCumulative
			case 2:
				e.Type = cadvisorapiv1.MetricType("delta")
			}
			switch c.Intn(2) {
			case 0:
				e.Format = cadvisorapiv1.IntType
			case 1:
				e.Format = cadvisorapiv1.FloatType
			}
			c.Fill(&e.Units)
		})
	var ret []cadvisorapiv1.MetricSpec
	f.Fill(&ret)
	return ret
}

func generateCustomMetrics(spec []cadvisorapiv1.MetricSpec) map[string][]cadvisorapiv1.MetricVal {
	ret := map[string][]cadvisorapiv1.MetricVal{}
	for _, metricSpec := range spec {
		f := randfill.New().NilChance(0).Funcs(
			func(e *cadvisorapiv1.MetricVal, c randfill.Continue) {
				switch metricSpec.Format {
				case cadvisorapiv1.IntType:
					c.Fill(&e.IntValue)
				case cadvisorapiv1.FloatType:
					c.Fill(&e.FloatValue)
				}
			})

		var metrics []cadvisorapiv1.MetricVal
		f.Fill(&metrics)
		ret[metricSpec.Name] = metrics
	}
	return ret
}

func testTime(base time.Time, seed int) time.Time {
	return base.Add(time.Duration(seed) * time.Second)
}

func checkNetworkStats(t *testing.T, label string, seed int, stats *statsapi.NetworkStats) {
	assert.NotNil(t, stats)
	assert.EqualValues(t, testTime(timestamp, seed).Unix(), stats.Time.Time.Unix(), label+".Net.Time")
	assert.EqualValues(t, "eth0", stats.Name, "default interface name is not eth0")
	assert.EqualValues(t, seed+offsetNetRxBytes, *stats.RxBytes, label+".Net.RxBytes")
	assert.EqualValues(t, seed+offsetNetRxErrors, *stats.RxErrors, label+".Net.RxErrors")
	assert.EqualValues(t, seed+offsetNetTxBytes, *stats.TxBytes, label+".Net.TxBytes")
	assert.EqualValues(t, seed+offsetNetTxErrors, *stats.TxErrors, label+".Net.TxErrors")

	assert.Len(t, stats.Interfaces, 2, "network interfaces should contain 2 elements")

	assert.EqualValues(t, "eth0", stats.Interfaces[0].Name, "default interface name is not eth0")
	assert.EqualValues(t, seed+offsetNetRxBytes, *stats.Interfaces[0].RxBytes, label+".Net.TxErrors")
	assert.EqualValues(t, seed+offsetNetRxErrors, *stats.Interfaces[0].RxErrors, label+".Net.TxErrors")
	assert.EqualValues(t, seed+offsetNetTxBytes, *stats.Interfaces[0].TxBytes, label+".Net.TxErrors")
	assert.EqualValues(t, seed+offsetNetTxErrors, *stats.Interfaces[0].TxErrors, label+".Net.TxErrors")

	assert.EqualValues(t, "cbr0", stats.Interfaces[1].Name, "cbr0 interface name is not cbr0")
	assert.EqualValues(t, 100, *stats.Interfaces[1].RxBytes, label+".Net.TxErrors")
	assert.EqualValues(t, 100, *stats.Interfaces[1].RxErrors, label+".Net.TxErrors")
	assert.EqualValues(t, 100, *stats.Interfaces[1].TxBytes, label+".Net.TxErrors")
	assert.EqualValues(t, 100, *stats.Interfaces[1].TxErrors, label+".Net.TxErrors")

}

func checkCPUStats(t *testing.T, label string, seed int, stats *statsapi.CPUStats) {
	require.NotNil(t, stats.Time, label+".CPU.Time")
	require.NotNil(t, stats.UsageNanoCores, label+".CPU.UsageNanoCores")
	require.NotNil(t, stats.UsageNanoCores, label+".CPU.UsageCoreSeconds")
	assert.EqualValues(t, testTime(timestamp, seed).Unix(), stats.Time.Time.Unix(), label+".CPU.Time")
	assert.EqualValues(t, seed+offsetCPUUsageCores, *stats.UsageNanoCores, label+".CPU.UsageCores")
	assert.EqualValues(t, seed+offsetCPUUsageCoreSeconds, *stats.UsageCoreNanoSeconds, label+".CPU.UsageCoreSeconds")
}

func checkMemoryStats(t *testing.T, label string, seed int, info cadvisorapiv2.ContainerInfo, stats *statsapi.MemoryStats) {
	assert.EqualValues(t, testTime(timestamp, seed).Unix(), stats.Time.Time.Unix(), label+".Mem.Time")
	assert.EqualValues(t, seed+offsetMemUsageBytes, *stats.UsageBytes, label+".Mem.UsageBytes")
	assert.EqualValues(t, seed+offsetMemWorkingSetBytes, *stats.WorkingSetBytes, label+".Mem.WorkingSetBytes")
	assert.EqualValues(t, seed+offsetMemRSSBytes, *stats.RSSBytes, label+".Mem.RSSBytes")
	assert.EqualValues(t, seed+offsetMemPageFaults, *stats.PageFaults, label+".Mem.PageFaults")
	assert.EqualValues(t, seed+offsetMemMajorPageFaults, *stats.MajorPageFaults, label+".Mem.MajorPageFaults")
	if !info.Spec.HasMemory || isMemoryUnlimited(info.Spec.Memory.Limit) {
		assert.Nil(t, stats.AvailableBytes, label+".Mem.AvailableBytes")
	} else {
		expected := info.Spec.Memory.Limit - *stats.WorkingSetBytes
		assert.EqualValues(t, expected, *stats.AvailableBytes, label+".Mem.AvailableBytes")
	}
}

func checkSwapStats(t *testing.T, label string, seed int, info cadvisorapiv2.ContainerInfo, stats *statsapi.SwapStats) {
	label += ".Swap"

	assert.EqualValues(t, testTime(timestamp, seed).Unix(), stats.Time.Time.Unix(), label+".Time")
	assert.EqualValues(t, seed+offsetMemSwapUsageBytes, *stats.SwapUsageBytes, label+".SwapUsageBytes")

	if !info.Spec.HasMemory || isMemoryUnlimited(info.Spec.Memory.SwapLimit) {
		assert.Nil(t, stats.SwapAvailableBytes, label+".SwapAvailableBytes")
	} else {
		expected := info.Spec.Memory.Limit - *stats.SwapUsageBytes
		assert.EqualValues(t, expected, *stats.SwapAvailableBytes, label+".AvailableBytes")
	}
}

func checkFsStats(t *testing.T, label string, seed int, stats *statsapi.FsStats) {
	assert.EqualValues(t, seed+offsetFsCapacity, *stats.CapacityBytes, label+".CapacityBytes")
	assert.EqualValues(t, seed+offsetFsAvailable, *stats.AvailableBytes, label+".AvailableBytes")
	assert.EqualValues(t, seed+offsetFsInodes, *stats.Inodes, label+".Inodes")
	assert.EqualValues(t, seed+offsetFsInodesFree, *stats.InodesFree, label+".InodesFree")
}

func checkContainersSwapStats(t *testing.T, podStats statsapi.PodStats, containerStats ...cadvisorapiv2.ContainerInfo) {
	if runtime.GOOS != "linux" {
		return
	}

	podContainers := make(map[string]struct{}, len(podStats.Containers))
	for _, container := range podStats.Containers {
		podContainers[container.Name] = struct{}{}
	}

	for _, container := range containerStats {
		found := false
		containerName := container.Spec.Labels["io.kubernetes.container.name"]
		for _, containerPodStats := range podStats.Containers {
			if containerPodStats.Name == containerName {
				assert.Equal(t, container.Stats[0].Memory.Swap, *containerPodStats.Swap.SwapUsageBytes)
				found = true
			}
		}
		assert.True(t, found, "container %s not found in pod stats", container.Spec.Labels["io.kubernetes.container.name"])
		delete(podContainers, containerName)
	}

	var missingContainerNames []string
	for containerName := range podContainers {
		missingContainerNames = append(missingContainerNames, containerName)
	}
	assert.Emptyf(t, podContainers, "containers not found in pod stats: %v", strings.Join(missingContainerNames, " "))
	if len(missingContainerNames) > 0 {
		assert.FailNow(t, "containers not found in pod stats")
	}
}

func checkEphemeralStats(t *testing.T, label string, containerSeeds []int, volumeSeeds []int, containerLogStats []*volume.Metrics, stats *statsapi.FsStats) {
	var usedBytes, inodeUsage int
	for _, cseed := range containerSeeds {
		usedBytes += cseed + offsetFsBaseUsageBytes
		inodeUsage += cseed + offsetFsInodeUsage
		// If containerLogStats is nil, then the log stats calculated from cAdvisor
		// information is used. Since it's Total - Base, and these values are
		// set to the offset, we can use the calculated difference in the offset
		// to account for this.
		if containerLogStats == nil {
			usedBytes += offsetFsTotalUsageBytes - offsetFsBaseUsageBytes
		}
	}
	for _, vseed := range volumeSeeds {
		usedBytes += vseed + offsetFsUsage
		inodeUsage += vseed + offsetFsInodeUsage
	}
	for _, logStats := range containerLogStats {
		usedBytes += int(logStats.Used.Value())
		inodeUsage += int(logStats.InodesUsed.Value())
	}
	assert.EqualValues(t, usedBytes, int(*stats.UsedBytes), label+".UsedBytes")
	assert.EqualValues(t, inodeUsage, int(*stats.InodesUsed), label+".InodesUsed")
}

type fakeResourceAnalyzer struct {
	podVolumeStats serverstats.PodVolumeStats
}

func (o *fakeResourceAnalyzer) Start()                                               {}
func (o *fakeResourceAnalyzer) Get(context.Context, bool) (*statsapi.Summary, error) { return nil, nil }
func (o *fakeResourceAnalyzer) GetCPUAndMemoryStats(context.Context) (*statsapi.Summary, error) {
	return nil, nil
}
func (o *fakeResourceAnalyzer) GetPodVolumeStats(uid types.UID) (serverstats.PodVolumeStats, bool) {
	return o.podVolumeStats, true
}

type fakeContainerStatsProvider struct {
	device      string
	imageFs     *statsapi.FsStats
	containerFs *statsapi.FsStats
}

func (p fakeContainerStatsProvider) ListPodStats(context.Context) ([]statsapi.PodStats, error) {
	return nil, fmt.Errorf("not implemented")
}

func (p fakeContainerStatsProvider) ListPodStatsAndUpdateCPUNanoCoreUsage(context.Context) ([]statsapi.PodStats, error) {
	return nil, fmt.Errorf("not implemented")
}

func (p fakeContainerStatsProvider) ListPodCPUAndMemoryStats(context.Context) ([]statsapi.PodStats, error) {
	return nil, fmt.Errorf("not implemented")
}

func (p fakeContainerStatsProvider) ImageFsStats(context.Context) (*statsapi.FsStats, *statsapi.FsStats, error) {
	return p.imageFs, p.containerFs, nil
}

func (p fakeContainerStatsProvider) ImageFsDevice(context.Context) (string, error) {
	return p.device, nil
}

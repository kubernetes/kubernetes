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
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	cadvisorfs "github.com/google/cadvisor/fs"
	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	critest "k8s.io/cri-api/pkg/apis/testing"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	kubelettypes "k8s.io/kubelet/pkg/types"
	cadvisortest "k8s.io/kubernetes/pkg/kubelet/cadvisor/testing"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	kubecontainertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/kuberuntime"
	kubepodtest "k8s.io/kubernetes/pkg/kubelet/pod/testing"
	serverstats "k8s.io/kubernetes/pkg/kubelet/server/stats"
	"k8s.io/kubernetes/pkg/volume"
)

const (
	offsetInodeUsage = iota
	offsetUsage
)

const (
	// This offset offsetCRI is to distinguish it from Cadvisor stats
	offsetCRI = 1000
)

const (
	seedRoot       = 0
	seedKubelet    = 200
	seedMisc       = 300
	seedSandbox0   = 1000
	seedContainer0 = 2000
	seedSandbox1   = 3000
	seedContainer1 = 4000
	seedContainer2 = 5000
	seedSandbox2   = 6000
	seedContainer3 = 7000
	seedSandbox3   = 8000
)

const (
	pName0 = "pod0"
	pName1 = "pod1"
	pName2 = "pod2"
)

const (
	cName0 = "container0-name"
	cName1 = "container1-name"
	cName2 = "container2-name"
	cName3 = "container3-name"
	cName5 = "container5-name"
	cName6 = "container6-name"
	cName7 = "container7-name"
	cName8 = "container8-name"
	cName9 = "container9-name"
)

const testPodLogDirectory = "/var/log/kube/pods/" // Use non-default path to ensure stats are collected properly

func TestCRIListPodStats(t *testing.T) {
	ctx := context.Background()
	var (
		imageFsMountpoint = "/test/mount/point"
		unknownMountpoint = "/unknown/mount/point"
		imageFsInfo       = getTestFsInfo(2000)
		rootFsInfo        = getTestFsInfo(1000)

		sandbox0           = makeFakePodSandbox("sandbox0-name", "sandbox0-uid", "sandbox0-ns", false)
		sandbox0Cgroup     = "/" + cm.GetPodCgroupNameSuffix(types.UID(sandbox0.PodSandboxStatus.Metadata.Uid))
		container0         = makeFakeContainer(sandbox0, cName0, 0, false)
		containerStats0    = makeFakeContainerStats(container0, imageFsMountpoint)
		containerLogStats0 = makeFakeLogStats(1000)
		container1         = makeFakeContainer(sandbox0, cName1, 0, false)
		containerStats1    = makeFakeContainerStats(container1, unknownMountpoint)
		containerLogStats1 = makeFakeLogStats(2000)

		sandbox1           = makeFakePodSandbox("sandbox1-name", "sandbox1-uid", "sandbox1-ns", false)
		sandbox1Cgroup     = "/" + cm.GetPodCgroupNameSuffix(types.UID(sandbox1.PodSandboxStatus.Metadata.Uid))
		container2         = makeFakeContainer(sandbox1, cName2, 0, false)
		containerStats2    = makeFakeContainerStats(container2, imageFsMountpoint)
		containerLogStats2 = makeFakeLogStats(3000)

		sandbox2           = makeFakePodSandbox("sandbox2-name", "sandbox2-uid", "sandbox2-ns", false)
		sandbox2Cgroup     = "/" + cm.GetPodCgroupNameSuffix(types.UID(sandbox2.PodSandboxStatus.Metadata.Uid))
		container3         = makeFakeContainer(sandbox2, cName3, 0, true)
		containerStats3    = makeFakeContainerStats(container3, imageFsMountpoint)
		container4         = makeFakeContainer(sandbox2, cName3, 1, false)
		containerStats4    = makeFakeContainerStats(container4, imageFsMountpoint)
		containerLogStats4 = makeFakeLogStats(4000)

		// Running pod with a terminated container and a running container
		sandbox3           = makeFakePodSandbox("sandbox3-name", "sandbox3-uid", "sandbox3-ns", false)
		sandbox3Cgroup     = "/" + cm.GetPodCgroupNameSuffix(types.UID(sandbox3.PodSandboxStatus.Metadata.Uid))
		container5         = makeFakeContainer(sandbox3, cName5, 0, true)
		containerStats5    = makeFakeContainerStats(container5, imageFsMountpoint)
		containerLogStats5 = makeFakeLogStats(5000)
		container8         = makeFakeContainer(sandbox3, cName8, 0, false)
		containerStats8    = makeFakeContainerStats(container8, imageFsMountpoint)
		containerLogStats8 = makeFakeLogStats(6000)

		// Terminated pod sandbox
		sandbox4        = makeFakePodSandbox("sandbox1-name", "sandbox1-uid", "sandbox1-ns", true)
		container6      = makeFakeContainer(sandbox4, cName6, 0, true)
		containerStats6 = makeFakeContainerStats(container6, imageFsMountpoint)

		// Terminated pod
		sandbox5        = makeFakePodSandbox("sandbox1-name", "sandbox5-uid", "sandbox1-ns", true)
		container7      = makeFakeContainer(sandbox5, cName7, 0, true)
		containerStats7 = makeFakeContainerStats(container7, imageFsMountpoint)

		podLogName0  = "pod-log-0"
		podLogName1  = "pod-log-1"
		podLogStats0 = makeFakeLogStats(5000)
		podLogStats1 = makeFakeLogStats(6000)
	)

	var (
		mockCadvisor       = cadvisortest.NewMockInterface(t)
		mockRuntimeCache   = new(kubecontainertest.MockRuntimeCache)
		mockPodManager     = new(kubepodtest.MockManager)
		resourceAnalyzer   = new(fakeResourceAnalyzer)
		fakeRuntimeService = critest.NewFakeRuntimeService()
		fakeImageService   = critest.NewFakeImageService()
	)

	infos := map[string]cadvisorapiv2.ContainerInfo{
		"/":                           getTestContainerInfo(seedRoot, "", "", ""),
		"/kubelet":                    getTestContainerInfo(seedKubelet, "", "", ""),
		"/system":                     getTestContainerInfo(seedMisc, "", "", ""),
		sandbox0.PodSandboxStatus.Id:  getTestContainerInfo(seedSandbox0, pName0, sandbox0.PodSandboxStatus.Metadata.Namespace, kubelettypes.PodInfraContainerName),
		sandbox0Cgroup:                getTestContainerInfo(seedSandbox0, "", "", ""),
		container0.ContainerStatus.Id: getTestContainerInfo(seedContainer0, pName0, sandbox0.PodSandboxStatus.Metadata.Namespace, cName0),
		container1.ContainerStatus.Id: getTestContainerInfo(seedContainer1, pName0, sandbox0.PodSandboxStatus.Metadata.Namespace, cName1),
		sandbox1.PodSandboxStatus.Id:  getTestContainerInfo(seedSandbox1, pName1, sandbox1.PodSandboxStatus.Metadata.Namespace, kubelettypes.PodInfraContainerName),
		sandbox1Cgroup:                getTestContainerInfo(seedSandbox1, "", "", ""),
		container2.ContainerStatus.Id: getTestContainerInfo(seedContainer2, pName1, sandbox1.PodSandboxStatus.Metadata.Namespace, cName2),
		sandbox2.PodSandboxStatus.Id:  getTestContainerInfo(seedSandbox2, pName2, sandbox2.PodSandboxStatus.Metadata.Namespace, kubelettypes.PodInfraContainerName),
		sandbox2Cgroup:                getTestContainerInfo(seedSandbox2, "", "", ""),
		container4.ContainerStatus.Id: getTestContainerInfo(seedContainer3, pName2, sandbox2.PodSandboxStatus.Metadata.Namespace, cName3),
		sandbox3Cgroup:                getTestContainerInfo(seedSandbox3, "", "", ""),
	}

	options := cadvisorapiv2.RequestOptions{
		IdType:    cadvisorapiv2.TypeName,
		Count:     2,
		Recursive: true,
	}

	mockCadvisor.EXPECT().ContainerInfoV2("/", options).Return(infos, nil)
	mockCadvisor.EXPECT().RootFsInfo().Return(rootFsInfo, nil)
	mockCadvisor.EXPECT().GetDirFsInfo(imageFsMountpoint).Return(imageFsInfo, nil)
	mockCadvisor.EXPECT().GetDirFsInfo(unknownMountpoint).Return(cadvisorapiv2.FsInfo{}, cadvisorfs.ErrNoSuchDevice)

	fakeRuntimeService.SetFakeSandboxes([]*critest.FakePodSandbox{
		sandbox0, sandbox1, sandbox2, sandbox3, sandbox4, sandbox5,
	})
	fakeRuntimeService.SetFakeContainers([]*critest.FakeContainer{
		container0, container1, container2, container3, container4, container5, container6, container7, container8,
	})
	fakeRuntimeService.SetFakeContainerStats([]*runtimeapi.ContainerStats{
		containerStats0, containerStats1, containerStats2, containerStats3, containerStats4, containerStats5, containerStats6, containerStats7, containerStats8,
	})

	ephemeralVolumes := makeFakeVolumeStats([]string{"ephVolume1, ephVolumes2"})
	persistentVolumes := makeFakeVolumeStats([]string{"persisVolume1, persisVolumes2"})
	resourceAnalyzer.podVolumeStats = serverstats.PodVolumeStats{
		EphemeralVolumes:  ephemeralVolumes,
		PersistentVolumes: persistentVolumes,
	}

	fakeStats := map[string]*volume.Metrics{
		kuberuntime.BuildContainerLogsDirectory(testPodLogDirectory, "sandbox0-ns", "sandbox0-name", types.UID("sandbox0-uid"), cName0):               containerLogStats0,
		kuberuntime.BuildContainerLogsDirectory(testPodLogDirectory, "sandbox0-ns", "sandbox0-name", types.UID("sandbox0-uid"), cName1):               containerLogStats1,
		kuberuntime.BuildContainerLogsDirectory(testPodLogDirectory, "sandbox1-ns", "sandbox1-name", types.UID("sandbox1-uid"), cName2):               containerLogStats2,
		kuberuntime.BuildContainerLogsDirectory(testPodLogDirectory, "sandbox2-ns", "sandbox2-name", types.UID("sandbox2-uid"), cName3):               containerLogStats4,
		kuberuntime.BuildContainerLogsDirectory(testPodLogDirectory, "sandbox3-ns", "sandbox3-name", types.UID("sandbox3-uid"), cName5):               containerLogStats5,
		kuberuntime.BuildContainerLogsDirectory(testPodLogDirectory, "sandbox3-ns", "sandbox3-name", types.UID("sandbox3-uid"), cName8):               containerLogStats8,
		filepath.Join(kuberuntime.BuildPodLogsDirectory(testPodLogDirectory, "sandbox0-ns", "sandbox0-name", types.UID("sandbox0-uid")), podLogName0): podLogStats0,
		filepath.Join(kuberuntime.BuildPodLogsDirectory(testPodLogDirectory, "sandbox1-ns", "sandbox1-name", types.UID("sandbox1-uid")), podLogName1): podLogStats1,
	}

	fakeOS := &kubecontainertest.FakeOS{}
	fakeOS.ReadDirFn = func(path string) ([]os.DirEntry, error) {
		var dirEntries []os.DirEntry
		mockDE := kubecontainertest.NewMockDirEntry(t)
		switch path {
		case kuberuntime.BuildPodLogsDirectory(testPodLogDirectory, "sandbox0-ns", "sandbox0-name", types.UID("sandbox0-uid")):
			mockDE.EXPECT().Name().Return(podLogName0)
		case kuberuntime.BuildPodLogsDirectory(testPodLogDirectory, "sandbox1-ns", "sandbox1-name", types.UID("sandbox1-uid")):
			mockDE.EXPECT().Name().Return(podLogName1)
		default:
			return nil, nil
		}
		mockDE.EXPECT().IsDir().Return(false)
		dirEntries = append(dirEntries, mockDE)
		return dirEntries, nil
	}

	provider := NewCRIStatsProvider(
		mockCadvisor,
		resourceAnalyzer,
		mockPodManager,
		mockRuntimeCache,
		fakeRuntimeService,
		fakeImageService,
		NewFakeHostStatsProviderWithData(fakeStats, fakeOS),
		false,
	)

	stats, err := provider.ListPodStats(ctx)
	assert := assert.New(t)
	assert.NoError(err)
	assert.Len(stats, 4)

	podStatsMap := make(map[statsapi.PodReference]statsapi.PodStats)
	for _, s := range stats {
		podStatsMap[s.PodRef] = s
	}

	p0 := podStatsMap[statsapi.PodReference{Name: "sandbox0-name", UID: "sandbox0-uid", Namespace: "sandbox0-ns"}]
	assert.Equal(sandbox0.CreatedAt, p0.StartTime.UnixNano())
	assert.Len(p0.Containers, 2)

	checkEphemeralStorageStats(assert, p0, ephemeralVolumes, []*runtimeapi.ContainerStats{containerStats0, containerStats1},
		[]*volume.Metrics{containerLogStats0, containerLogStats1}, podLogStats0)

	containerStatsMap := make(map[string]statsapi.ContainerStats)
	for _, s := range p0.Containers {
		containerStatsMap[s.Name] = s
	}

	c0 := containerStatsMap[cName0]
	assert.Equal(container0.CreatedAt, c0.StartTime.UnixNano())
	checkCRICPUAndMemoryStats(assert, c0, infos[container0.ContainerStatus.Id].Stats[0])
	assert.Nil(c0.Accelerators)
	checkCRIRootfsStats(assert, c0, containerStats0, &imageFsInfo)
	checkCRILogsStats(assert, c0, &rootFsInfo, containerLogStats0)

	c1 := containerStatsMap[cName1]
	assert.Equal(container1.CreatedAt, c1.StartTime.UnixNano())
	checkCRICPUAndMemoryStats(assert, c1, infos[container1.ContainerStatus.Id].Stats[0])
	assert.Nil(c0.Accelerators)
	checkCRIRootfsStats(assert, c1, containerStats1, nil)
	checkCRILogsStats(assert, c1, &rootFsInfo, containerLogStats1)
	checkCRINetworkStats(assert, p0.Network, infos[sandbox0.PodSandboxStatus.Id].Stats[0].Network)
	checkCRIPodCPUAndMemoryStats(assert, p0, infos[sandbox0Cgroup].Stats[0])
	checkCRIPodSwapStats(assert, p0, infos[sandbox0Cgroup].Stats[0])

	checkContainersSwapStats(t, p0, infos[container0.Id], infos[container1.Id])

	p1 := podStatsMap[statsapi.PodReference{Name: "sandbox1-name", UID: "sandbox1-uid", Namespace: "sandbox1-ns"}]
	assert.Equal(sandbox1.CreatedAt, p1.StartTime.UnixNano())
	assert.Len(p1.Containers, 1)

	checkEphemeralStorageStats(assert, p1, ephemeralVolumes, []*runtimeapi.ContainerStats{containerStats2},
		[]*volume.Metrics{containerLogStats2}, podLogStats1)
	c2 := p1.Containers[0]
	assert.Equal(cName2, c2.Name)
	assert.Equal(container2.CreatedAt, c2.StartTime.UnixNano())
	checkCRICPUAndMemoryStats(assert, c2, infos[container2.ContainerStatus.Id].Stats[0])
	assert.Nil(c0.Accelerators)
	checkCRIRootfsStats(assert, c2, containerStats2, &imageFsInfo)
	checkCRILogsStats(assert, c2, &rootFsInfo, containerLogStats2)
	checkCRINetworkStats(assert, p1.Network, infos[sandbox1.PodSandboxStatus.Id].Stats[0].Network)
	checkCRIPodCPUAndMemoryStats(assert, p1, infos[sandbox1Cgroup].Stats[0])
	checkCRIPodSwapStats(assert, p1, infos[sandbox1Cgroup].Stats[0])

	checkContainersSwapStats(t, p1, infos[container2.Id])

	p2 := podStatsMap[statsapi.PodReference{Name: "sandbox2-name", UID: "sandbox2-uid", Namespace: "sandbox2-ns"}]
	assert.Equal(sandbox2.CreatedAt, p2.StartTime.UnixNano())
	assert.Len(p2.Containers, 1)

	checkEphemeralStorageStats(assert, p2, ephemeralVolumes, []*runtimeapi.ContainerStats{containerStats4},
		[]*volume.Metrics{containerLogStats4}, nil)

	c3 := p2.Containers[0]
	assert.Equal(cName3, c3.Name)
	assert.Equal(container4.CreatedAt, c3.StartTime.UnixNano())
	checkCRICPUAndMemoryStats(assert, c3, infos[container4.ContainerStatus.Id].Stats[0])
	assert.Nil(c0.Accelerators)
	checkCRIRootfsStats(assert, c3, containerStats4, &imageFsInfo)

	checkCRILogsStats(assert, c3, &rootFsInfo, containerLogStats4)
	checkCRINetworkStats(assert, p2.Network, infos[sandbox2.PodSandboxStatus.Id].Stats[0].Network)
	checkCRIPodCPUAndMemoryStats(assert, p2, infos[sandbox2Cgroup].Stats[0])
	checkCRIPodSwapStats(assert, p2, infos[sandbox2Cgroup].Stats[0])

	checkContainersSwapStats(t, p2, infos[container4.Id])

	p3 := podStatsMap[statsapi.PodReference{Name: "sandbox3-name", UID: "sandbox3-uid", Namespace: "sandbox3-ns"}]
	assert.Equal(sandbox3.CreatedAt, p3.StartTime.UnixNano())
	assert.Len(p3.Containers, 1)

	c8 := p3.Containers[0]
	assert.Equal(cName8, c8.Name)
	assert.Equal(container8.CreatedAt, c8.StartTime.UnixNano())
	assert.NotNil(c8.CPU.Time)
	assert.NotNil(c8.Memory.Time)
	checkCRIPodCPUAndMemoryStats(assert, p3, infos[sandbox3Cgroup].Stats[0])
	checkCRIPodSwapStats(assert, p3, infos[sandbox3Cgroup].Stats[0])
}

func TestListPodStatsStrictlyFromCRI(t *testing.T) {
	if runtime.GOOS == "windows" {
		// TODO: remove skip once the failing test has been fixed.
		t.Skip("Skip failing test on Windows.")
	}
	ctx := context.Background()
	var (
		imageFsMountpoint = "/test/mount/point"
		unknownMountpoint = "/unknown/mount/point"
		imageFsInfo       = getTestFsInfo(2000)
		rootFsInfo        = getTestFsInfo(1000)

		// A pod that CRI returns stats and cadvisor returns stats
		// The pod stats from CRI stats
		sandbox0           = makeFakePodSandbox("sandbox0-name", "sandbox0-uid", "sandbox0-ns", false)
		sandbox0Cgroup     = "/" + cm.GetPodCgroupNameSuffix(types.UID(sandbox0.PodSandboxStatus.Metadata.Uid))
		container0         = makeFakeContainer(sandbox0, cName0, 0, false)
		containerStats0    = makeFakeContainerStatsStrictlyFromCRI(seedContainer0, container0, imageFsMountpoint)
		containerLogStats0 = makeFakeLogStats(1000)
		container1         = makeFakeContainer(sandbox0, cName1, 0, false)
		containerStats1    = makeFakeContainerStatsStrictlyFromCRI(seedContainer1, container1, unknownMountpoint)
		containerLogStats1 = makeFakeLogStats(2000)
		sandboxPodStats0   = makeFakePodSandboxStatsStrictlyFromCRI(seedSandbox0, sandbox0, containerStats0, containerStats1)

		// A pod that CRI returns stats and cadvisor returns no stats
		// The pod stats from CRI stats
		sandbox1           = makeFakePodSandbox("sandbox1-name", "sandbox1-uid", "sandbox1-ns", false)
		sandbox1Cgroup     = "/" + cm.GetPodCgroupNameSuffix(types.UID(sandbox1.PodSandboxStatus.Metadata.Uid))
		container2         = makeFakeContainer(sandbox1, cName2, 0, false)
		containerStats2    = makeFakeContainerStatsStrictlyFromCRI(seedContainer2, container2, imageFsMountpoint)
		containerLogStats2 = makeFakeLogStats(3000)
		sandboxPodStats1   = makeFakePodSandboxStatsStrictlyFromCRI(seedSandbox1, sandbox1, containerStats2)

		podLogName0  = "pod-log-0"
		podLogName1  = "pod-log-1"
		podLogStats0 = makeFakeLogStats(5000)
		podLogStats1 = makeFakeLogStats(6000)
	)
	var (
		mockCadvisor       = cadvisortest.NewMockInterface(t)
		mockRuntimeCache   = new(kubecontainertest.MockRuntimeCache)
		mockPodManager     = new(kubepodtest.MockManager)
		resourceAnalyzer   = new(fakeResourceAnalyzer)
		fakeRuntimeService = critest.NewFakeRuntimeService()
		fakeImageService   = critest.NewFakeImageService()
	)
	infos := map[string]cadvisorapiv2.ContainerInfo{
		"/":                           getTestContainerInfo(seedRoot, "", "", ""),
		"/kubelet":                    getTestContainerInfo(seedKubelet, "", "", ""),
		"/system":                     getTestContainerInfo(seedMisc, "", "", ""),
		sandbox0.PodSandboxStatus.Id:  getTestContainerInfo(seedSandbox0, pName0, sandbox0.PodSandboxStatus.Metadata.Namespace, kubelettypes.PodInfraContainerName),
		sandbox0Cgroup:                getTestContainerInfo(seedSandbox0, "", "", ""),
		container0.ContainerStatus.Id: getTestContainerInfo(seedContainer0, pName0, sandbox0.PodSandboxStatus.Metadata.Namespace, cName0),
		container1.ContainerStatus.Id: getTestContainerInfo(seedContainer1, pName0, sandbox0.PodSandboxStatus.Metadata.Namespace, cName1),
	}

	exceptedContainerStatsMap := map[string]statsapi.ContainerStats{
		cName0: getCRIContainerStatsStrictlyFromCRI(seedContainer0, cName0),
		cName1: getCRIContainerStatsStrictlyFromCRI(seedContainer1, cName1),
		cName2: getCRIContainerStatsStrictlyFromCRI(seedContainer2, cName2),
	}

	prf0 := statsapi.PodReference{Name: "sandbox0-name", UID: "sandbox0-uid", Namespace: "sandbox0-ns"}
	prf1 := statsapi.PodReference{Name: "sandbox1-name", UID: "sandbox1-uid", Namespace: "sandbox1-ns"}

	exceptedPodStatsMap := map[statsapi.PodReference]statsapi.PodStats{
		prf0: getPodSandboxStatsStrictlyFromCRI(seedSandbox0, sandbox0),
		prf1: getPodSandboxStatsStrictlyFromCRI(seedSandbox1, sandbox1),
	}

	options := cadvisorapiv2.RequestOptions{
		IdType:    cadvisorapiv2.TypeName,
		Count:     2,
		Recursive: true,
	}
	mockCadvisor.EXPECT().ContainerInfoV2("/", options).Return(infos, nil)
	mockCadvisor.EXPECT().RootFsInfo().Return(rootFsInfo, nil)
	mockCadvisor.EXPECT().GetDirFsInfo(imageFsMountpoint).Return(imageFsInfo, nil)
	mockCadvisor.EXPECT().GetDirFsInfo(unknownMountpoint).Return(cadvisorapiv2.FsInfo{}, cadvisorfs.ErrNoSuchDevice)
	fakeRuntimeService.SetFakeSandboxes([]*critest.FakePodSandbox{
		sandbox0, sandbox1,
	})
	fakeRuntimeService.SetFakeContainers([]*critest.FakeContainer{
		container0, container1, container2,
	})
	fakeRuntimeService.SetFakeContainerStats([]*runtimeapi.ContainerStats{
		containerStats0, containerStats1, containerStats2,
	})

	fakeRuntimeService.SetFakePodSandboxStats([]*runtimeapi.PodSandboxStats{
		sandboxPodStats0, sandboxPodStats1,
	})

	ephemeralVolumes := makeFakeVolumeStats([]string{"ephVolume1, ephVolumes2"})
	persistentVolumes := makeFakeVolumeStats([]string{"persisVolume1, persisVolumes2"})
	resourceAnalyzer.podVolumeStats = serverstats.PodVolumeStats{
		EphemeralVolumes:  ephemeralVolumes,
		PersistentVolumes: persistentVolumes,
	}
	fakeStats := map[string]*volume.Metrics{
		kuberuntime.BuildContainerLogsDirectory(testPodLogDirectory, "sandbox0-ns", "sandbox0-name", types.UID("sandbox0-uid"), cName0):               containerLogStats0,
		kuberuntime.BuildContainerLogsDirectory(testPodLogDirectory, "sandbox0-ns", "sandbox0-name", types.UID("sandbox0-uid"), cName1):               containerLogStats1,
		kuberuntime.BuildContainerLogsDirectory(testPodLogDirectory, "sandbox1-ns", "sandbox1-name", types.UID("sandbox1-uid"), cName2):               containerLogStats2,
		filepath.Join(kuberuntime.BuildPodLogsDirectory(testPodLogDirectory, "sandbox0-ns", "sandbox0-name", types.UID("sandbox0-uid")), podLogName0): podLogStats0,
		filepath.Join(kuberuntime.BuildPodLogsDirectory(testPodLogDirectory, "sandbox1-ns", "sandbox1-name", types.UID("sandbox1-uid")), podLogName1): podLogStats1,
	}
	fakeOS := &kubecontainertest.FakeOS{}
	fakeOS.ReadDirFn = func(path string) ([]os.DirEntry, error) {
		var dirEntries []os.DirEntry
		mockDE := kubecontainertest.NewMockDirEntry(t)
		switch path {
		case kuberuntime.BuildPodLogsDirectory(testPodLogDirectory, "sandbox0-ns", "sandbox0-name", types.UID("sandbox0-uid")):
			mockDE.EXPECT().Name().Return(podLogName0)
		case kuberuntime.BuildPodLogsDirectory(testPodLogDirectory, "sandbox1-ns", "sandbox1-name", types.UID("sandbox1-uid")):
			mockDE.EXPECT().Name().Return(podLogName1)
		default:
			return nil, nil
		}
		mockDE.EXPECT().IsDir().Return(false)
		dirEntries = append(dirEntries, mockDE)
		return dirEntries, nil
	}
	provider := NewCRIStatsProvider(
		mockCadvisor,
		resourceAnalyzer,
		mockPodManager,
		mockRuntimeCache,
		fakeRuntimeService,
		fakeImageService,
		NewFakeHostStatsProviderWithData(fakeStats, fakeOS),
		true,
	)

	cadvisorInfos, err := getCadvisorContainerInfo(mockCadvisor)
	if err != nil {
		t.Errorf("failed to get container info from cadvisor: %v", err)
	}
	stats, err := provider.ListPodStats(ctx)
	assert := assert.New(t)
	assert.NoError(err)
	assert.Len(stats, 2)
	podStatsMap := make(map[statsapi.PodReference]statsapi.PodStats)
	for _, s := range stats {
		podStatsMap[s.PodRef] = s
	}
	p0 := podStatsMap[prf0]
	assert.Equal(sandbox0.CreatedAt, p0.StartTime.UnixNano())
	assert.Len(p0.Containers, 2)

	checkEphemeralStorageStats(assert, p0, ephemeralVolumes, []*runtimeapi.ContainerStats{containerStats0, containerStats1},
		[]*volume.Metrics{containerLogStats0, containerLogStats1}, podLogStats0)

	containerStatsMap := make(map[string]statsapi.ContainerStats)
	for _, s := range p0.Containers {
		containerStatsMap[s.Name] = s
	}

	c0 := containerStatsMap[cName0]
	assert.Equal(container0.CreatedAt, c0.StartTime.UnixNano())
	checkCRICPUAndMemoryStatsForStrictlyFromCRI(assert, c0, exceptedContainerStatsMap[cName0])
	assert.Nil(c0.Accelerators)
	checkCRIRootfsStats(assert, c0, containerStats0, &imageFsInfo)
	checkCRILogsStats(assert, c0, &rootFsInfo, containerLogStats0)

	c1 := containerStatsMap[cName1]
	assert.Equal(container1.CreatedAt, c1.StartTime.UnixNano())
	checkCRICPUAndMemoryStatsForStrictlyFromCRI(assert, c1, exceptedContainerStatsMap[cName1])
	assert.Nil(c0.Accelerators)
	checkCRIRootfsStats(assert, c1, containerStats1, nil)
	checkCRILogsStats(assert, c1, &rootFsInfo, containerLogStats1)
	checkCRIPodCPUAndMemoryStatsStrictlyFromCRI(assert, p0, exceptedPodStatsMap[prf0])
	assert.NotNil(cadvisorInfos[sandbox0Cgroup].Stats[0].Cpu)
	assert.NotNil(cadvisorInfos[sandbox0Cgroup].Stats[0].Memory)

	p1 := podStatsMap[prf1]
	assert.Equal(sandbox1.CreatedAt, p1.StartTime.UnixNano())
	assert.Len(p1.Containers, 1)

	checkEphemeralStorageStats(assert, p1, ephemeralVolumes, []*runtimeapi.ContainerStats{containerStats2},
		[]*volume.Metrics{containerLogStats2}, podLogStats1)
	c2 := p1.Containers[0]
	assert.Equal(cName2, c2.Name)
	assert.Equal(container2.CreatedAt, c2.StartTime.UnixNano())
	checkCRICPUAndMemoryStatsForStrictlyFromCRI(assert, c2, exceptedContainerStatsMap[cName2])
	assert.Nil(c0.Accelerators)
	checkCRIRootfsStats(assert, c2, containerStats2, &imageFsInfo)
	checkCRILogsStats(assert, c2, &rootFsInfo, containerLogStats2)
	checkCRIPodCPUAndMemoryStatsStrictlyFromCRI(assert, p1, exceptedPodStatsMap[prf1])

	if runtime.GOOS == "linux" {
		if _, ok := cadvisorInfos[sandbox1Cgroup]; ok {
			t.Errorf("expect no cadvisor stats for pod %v", prf1)
		}
	}
}
func TestCRIListPodCPUAndMemoryStats(t *testing.T) {
	ctx := context.Background()

	var (
		imageFsMountpoint = "/test/mount/point"
		unknownMountpoint = "/unknown/mount/point"

		sandbox0        = makeFakePodSandbox("sandbox0-name", "sandbox0-uid", "sandbox0-ns", false)
		sandbox0Cgroup  = "/" + cm.GetPodCgroupNameSuffix(types.UID(sandbox0.PodSandboxStatus.Metadata.Uid))
		container0      = makeFakeContainer(sandbox0, cName0, 0, false)
		containerStats0 = makeFakeContainerStats(container0, imageFsMountpoint)
		container1      = makeFakeContainer(sandbox0, cName1, 0, false)
		containerStats1 = makeFakeContainerStats(container1, unknownMountpoint)

		sandbox1        = makeFakePodSandbox("sandbox1-name", "sandbox1-uid", "sandbox1-ns", false)
		sandbox1Cgroup  = "/" + cm.GetPodCgroupNameSuffix(types.UID(sandbox1.PodSandboxStatus.Metadata.Uid))
		container2      = makeFakeContainer(sandbox1, cName2, 0, false)
		containerStats2 = makeFakeContainerStats(container2, imageFsMountpoint)

		sandbox2        = makeFakePodSandbox("sandbox2-name", "sandbox2-uid", "sandbox2-ns", false)
		sandbox2Cgroup  = "/" + cm.GetPodCgroupNameSuffix(types.UID(sandbox2.PodSandboxStatus.Metadata.Uid))
		container3      = makeFakeContainer(sandbox2, cName3, 0, true)
		containerStats3 = makeFakeContainerStats(container3, imageFsMountpoint)
		container4      = makeFakeContainer(sandbox2, cName3, 1, false)
		containerStats4 = makeFakeContainerStats(container4, imageFsMountpoint)

		// Running pod with a terminated container and a running container
		sandbox3        = makeFakePodSandbox("sandbox3-name", "sandbox3-uid", "sandbox3-ns", false)
		sandbox3Cgroup  = "/" + cm.GetPodCgroupNameSuffix(types.UID(sandbox3.PodSandboxStatus.Metadata.Uid))
		container5      = makeFakeContainer(sandbox3, cName5, 0, true)
		containerStats5 = makeFakeContainerStats(container5, imageFsMountpoint)
		container8      = makeFakeContainer(sandbox3, cName8, 0, false)
		containerStats8 = makeFakeContainerStats(container8, imageFsMountpoint)

		// Terminated pod sandbox
		sandbox4        = makeFakePodSandbox("sandbox1-name", "sandbox1-uid", "sandbox1-ns", true)
		container6      = makeFakeContainer(sandbox4, cName6, 0, true)
		containerStats6 = makeFakeContainerStats(container6, imageFsMountpoint)

		// Terminated pod
		sandbox5        = makeFakePodSandbox("sandbox1-name", "sandbox5-uid", "sandbox1-ns", true)
		container7      = makeFakeContainer(sandbox5, cName7, 0, true)
		containerStats7 = makeFakeContainerStats(container7, imageFsMountpoint)

		// A pod that cadvisor returns no stats
		sandbox6        = makeFakePodSandbox("sandbox6-name", "sandbox6-uid", "sandbox6-ns", false)
		container9      = makeFakeContainer(sandbox6, cName9, 0, false)
		containerStats9 = makeFakeContainerStats(container9, imageFsMountpoint)
	)

	var (
		mockCadvisor       = cadvisortest.NewMockInterface(t)
		mockRuntimeCache   = new(kubecontainertest.MockRuntimeCache)
		mockPodManager     = new(kubepodtest.MockManager)
		resourceAnalyzer   = new(fakeResourceAnalyzer)
		fakeRuntimeService = critest.NewFakeRuntimeService()
	)

	infos := map[string]cadvisorapiv2.ContainerInfo{
		"/":                           getTestContainerInfo(seedRoot, "", "", ""),
		"/kubelet":                    getTestContainerInfo(seedKubelet, "", "", ""),
		"/system":                     getTestContainerInfo(seedMisc, "", "", ""),
		sandbox0.PodSandboxStatus.Id:  getTestContainerInfo(seedSandbox0, pName0, sandbox0.PodSandboxStatus.Metadata.Namespace, kubelettypes.PodInfraContainerName),
		sandbox0Cgroup:                getTestContainerInfo(seedSandbox0, "", "", ""),
		container0.ContainerStatus.Id: getTestContainerInfo(seedContainer0, pName0, sandbox0.PodSandboxStatus.Metadata.Namespace, cName0),
		container1.ContainerStatus.Id: getTestContainerInfo(seedContainer1, pName0, sandbox0.PodSandboxStatus.Metadata.Namespace, cName1),
		sandbox1.PodSandboxStatus.Id:  getTestContainerInfo(seedSandbox1, pName1, sandbox1.PodSandboxStatus.Metadata.Namespace, kubelettypes.PodInfraContainerName),
		sandbox1Cgroup:                getTestContainerInfo(seedSandbox1, "", "", ""),
		container2.ContainerStatus.Id: getTestContainerInfo(seedContainer2, pName1, sandbox1.PodSandboxStatus.Metadata.Namespace, cName2),
		sandbox2.PodSandboxStatus.Id:  getTestContainerInfo(seedSandbox2, pName2, sandbox2.PodSandboxStatus.Metadata.Namespace, kubelettypes.PodInfraContainerName),
		sandbox2Cgroup:                getTestContainerInfo(seedSandbox2, "", "", ""),
		container4.ContainerStatus.Id: getTestContainerInfo(seedContainer3, pName2, sandbox2.PodSandboxStatus.Metadata.Namespace, cName3),
		sandbox3Cgroup:                getTestContainerInfo(seedSandbox3, "", "", ""),
	}

	options := cadvisorapiv2.RequestOptions{
		IdType:    cadvisorapiv2.TypeName,
		Count:     2,
		Recursive: true,
	}

	mockCadvisor.EXPECT().ContainerInfoV2("/", options).Return(infos, nil)

	fakeRuntimeService.SetFakeSandboxes([]*critest.FakePodSandbox{
		sandbox0, sandbox1, sandbox2, sandbox3, sandbox4, sandbox5, sandbox6,
	})
	fakeRuntimeService.SetFakeContainers([]*critest.FakeContainer{
		container0, container1, container2, container3, container4, container5, container6, container7, container8, container9,
	})
	fakeRuntimeService.SetFakeContainerStats([]*runtimeapi.ContainerStats{
		containerStats0, containerStats1, containerStats2, containerStats3, containerStats4, containerStats5, containerStats6, containerStats7, containerStats8, containerStats9,
	})

	ephemeralVolumes := makeFakeVolumeStats([]string{"ephVolume1, ephVolumes2"})
	persistentVolumes := makeFakeVolumeStats([]string{"persisVolume1, persisVolumes2"})
	resourceAnalyzer.podVolumeStats = serverstats.PodVolumeStats{
		EphemeralVolumes:  ephemeralVolumes,
		PersistentVolumes: persistentVolumes,
	}

	provider := NewCRIStatsProvider(
		mockCadvisor,
		resourceAnalyzer,
		mockPodManager,
		mockRuntimeCache,
		fakeRuntimeService,
		nil,
		NewFakeHostStatsProvider(),
		false,
	)

	stats, err := provider.ListPodCPUAndMemoryStats(ctx)
	assert := assert.New(t)
	assert.NoError(err)
	assert.Len(stats, 5)

	podStatsMap := make(map[statsapi.PodReference]statsapi.PodStats)
	for _, s := range stats {
		podStatsMap[s.PodRef] = s
	}

	p0 := podStatsMap[statsapi.PodReference{Name: "sandbox0-name", UID: "sandbox0-uid", Namespace: "sandbox0-ns"}]
	assert.Equal(sandbox0.CreatedAt, p0.StartTime.UnixNano())
	assert.Len(p0.Containers, 2)
	assert.Nil(p0.EphemeralStorage)
	assert.Nil(p0.VolumeStats)
	assert.Nil(p0.Network)
	checkCRIPodCPUAndMemoryStats(assert, p0, infos[sandbox0Cgroup].Stats[0])

	containerStatsMap := make(map[string]statsapi.ContainerStats)
	for _, s := range p0.Containers {
		containerStatsMap[s.Name] = s
	}

	c0 := containerStatsMap[cName0]
	assert.Equal(container0.CreatedAt, c0.StartTime.UnixNano())
	checkCRICPUAndMemoryStats(assert, c0, infos[container0.ContainerStatus.Id].Stats[0])
	assert.Nil(c0.Rootfs)
	assert.Nil(c0.Logs)
	assert.Nil(c0.Accelerators)
	assert.Nil(c0.UserDefinedMetrics)
	c1 := containerStatsMap[cName1]
	assert.Equal(container1.CreatedAt, c1.StartTime.UnixNano())
	checkCRICPUAndMemoryStats(assert, c1, infos[container1.ContainerStatus.Id].Stats[0])
	assert.Nil(c1.Rootfs)
	assert.Nil(c1.Logs)
	assert.Nil(c1.Accelerators)
	assert.Nil(c1.UserDefinedMetrics)

	p1 := podStatsMap[statsapi.PodReference{Name: "sandbox1-name", UID: "sandbox1-uid", Namespace: "sandbox1-ns"}]
	assert.Equal(sandbox1.CreatedAt, p1.StartTime.UnixNano())
	assert.Len(p1.Containers, 1)
	assert.Nil(p1.EphemeralStorage)
	assert.Nil(p1.VolumeStats)
	assert.Nil(p1.Network)
	checkCRIPodCPUAndMemoryStats(assert, p1, infos[sandbox1Cgroup].Stats[0])

	c2 := p1.Containers[0]
	assert.Equal(cName2, c2.Name)
	assert.Equal(container2.CreatedAt, c2.StartTime.UnixNano())
	checkCRICPUAndMemoryStats(assert, c2, infos[container2.ContainerStatus.Id].Stats[0])
	assert.Nil(c2.Rootfs)
	assert.Nil(c2.Logs)
	assert.Nil(c2.Accelerators)
	assert.Nil(c2.UserDefinedMetrics)

	p2 := podStatsMap[statsapi.PodReference{Name: "sandbox2-name", UID: "sandbox2-uid", Namespace: "sandbox2-ns"}]
	assert.Equal(sandbox2.CreatedAt, p2.StartTime.UnixNano())
	assert.Len(p2.Containers, 1)
	assert.Nil(p2.EphemeralStorage)
	assert.Nil(p2.VolumeStats)
	assert.Nil(p2.Network)
	checkCRIPodCPUAndMemoryStats(assert, p2, infos[sandbox2Cgroup].Stats[0])

	c3 := p2.Containers[0]
	assert.Equal(cName3, c3.Name)
	assert.Equal(container4.CreatedAt, c3.StartTime.UnixNano())
	checkCRICPUAndMemoryStats(assert, c3, infos[container4.ContainerStatus.Id].Stats[0])
	assert.Nil(c2.Rootfs)
	assert.Nil(c2.Logs)
	assert.Nil(c2.Accelerators)
	assert.Nil(c2.UserDefinedMetrics)

	p3 := podStatsMap[statsapi.PodReference{Name: "sandbox3-name", UID: "sandbox3-uid", Namespace: "sandbox3-ns"}]
	assert.Equal(sandbox3.CreatedAt, p3.StartTime.UnixNano())
	assert.Len(p3.Containers, 1)

	c8 := p3.Containers[0]
	assert.Equal(cName8, c8.Name)
	assert.Equal(container8.CreatedAt, c8.StartTime.UnixNano())
	assert.NotNil(c8.CPU.Time)
	assert.NotNil(c8.Memory.Time)
	checkCRIPodCPUAndMemoryStats(assert, p3, infos[sandbox3Cgroup].Stats[0])

	p6 := podStatsMap[statsapi.PodReference{Name: "sandbox6-name", UID: "sandbox6-uid", Namespace: "sandbox6-ns"}]
	assert.Equal(sandbox6.CreatedAt, p6.StartTime.UnixNano())
	assert.Len(p6.Containers, 1)

	c9 := p6.Containers[0]
	assert.Equal(cName9, c9.Name)
	assert.Equal(container9.CreatedAt, c9.StartTime.UnixNano())
	assert.NotNil(c9.CPU.Time)
	assert.Equal(containerStats9.Cpu.Timestamp, p6.CPU.Time.UnixNano())
	assert.NotNil(c9.Memory.Time)
	assert.Equal(containerStats9.Memory.Timestamp, p6.Memory.Time.UnixNano())
}

func TestCRIImagesFsStats(t *testing.T) {
	ctx := context.Background()
	var (
		imageFsMountpoint = "/test/mount/point"
		imageFsInfo       = getTestFsInfo(2000)
		imageFsUsage      = makeFakeImageFsUsage(imageFsMountpoint)
	)

	var (
		mockCadvisor       = cadvisortest.NewMockInterface(t)
		mockRuntimeCache   = new(kubecontainertest.MockRuntimeCache)
		mockPodManager     = new(kubepodtest.MockManager)
		resourceAnalyzer   = new(fakeResourceAnalyzer)
		fakeRuntimeService = critest.NewFakeRuntimeService()
		fakeImageService   = critest.NewFakeImageService()
	)
	mockCadvisor.EXPECT().GetDirFsInfo(imageFsMountpoint).Return(imageFsInfo, nil)
	fakeImageService.SetFakeFilesystemUsage([]*runtimeapi.FilesystemUsage{
		imageFsUsage,
	})

	provider := NewCRIStatsProvider(
		mockCadvisor,
		resourceAnalyzer,
		mockPodManager,
		mockRuntimeCache,
		fakeRuntimeService,
		fakeImageService,
		NewFakeHostStatsProvider(),
		false,
	)

	stats, containerStats, err := provider.ImageFsStats(ctx)
	assert := assert.New(t)
	assert.NoError(err)

	assert.Equal(imageFsUsage.Timestamp, stats.Time.UnixNano())
	assert.Equal(imageFsInfo.Available, *stats.AvailableBytes)
	assert.Equal(imageFsInfo.Capacity, *stats.CapacityBytes)
	assert.Equal(imageFsInfo.InodesFree, stats.InodesFree)
	assert.Equal(imageFsInfo.Inodes, stats.Inodes)
	assert.Equal(imageFsUsage.UsedBytes.Value, *stats.UsedBytes)
	assert.Equal(imageFsUsage.InodesUsed.Value, *stats.InodesUsed)

	assert.Equal(imageFsUsage.Timestamp, containerStats.Time.UnixNano())
	assert.Equal(imageFsInfo.Available, *containerStats.AvailableBytes)
	assert.Equal(imageFsInfo.Capacity, *containerStats.CapacityBytes)
	assert.Equal(imageFsInfo.InodesFree, containerStats.InodesFree)
	assert.Equal(imageFsInfo.Inodes, containerStats.Inodes)
	assert.Equal(imageFsUsage.UsedBytes.Value, *containerStats.UsedBytes)
	assert.Equal(imageFsUsage.InodesUsed.Value, *containerStats.InodesUsed)

}

func makeFakePodSandbox(name, uid, namespace string, terminated bool) *critest.FakePodSandbox {
	p := &critest.FakePodSandbox{
		PodSandboxStatus: runtimeapi.PodSandboxStatus{
			Metadata: &runtimeapi.PodSandboxMetadata{
				Name:      name,
				Uid:       uid,
				Namespace: namespace,
			},
			State:     runtimeapi.PodSandboxState_SANDBOX_READY,
			CreatedAt: time.Now().UnixNano(),
		},
	}
	if terminated {
		p.PodSandboxStatus.State = runtimeapi.PodSandboxState_SANDBOX_NOTREADY
	}
	p.PodSandboxStatus.Id = strings.ReplaceAll(string(uuid.NewUUID()), "-", "")
	return p
}

func makeFakeContainer(sandbox *critest.FakePodSandbox, name string, attempt uint32, terminated bool) *critest.FakeContainer {
	sandboxID := sandbox.PodSandboxStatus.Id
	c := &critest.FakeContainer{
		SandboxID: sandboxID,
		ContainerStatus: runtimeapi.ContainerStatus{
			Metadata:  &runtimeapi.ContainerMetadata{Name: name, Attempt: attempt},
			Image:     &runtimeapi.ImageSpec{},
			ImageRef:  "fake-image-ref",
			CreatedAt: time.Now().UnixNano(),
		},
	}
	c.ContainerStatus.Labels = map[string]string{
		"io.kubernetes.pod.name":       sandbox.Metadata.Name,
		"io.kubernetes.pod.uid":        sandbox.Metadata.Uid,
		"io.kubernetes.pod.namespace":  sandbox.Metadata.Namespace,
		"io.kubernetes.container.name": name,
	}
	if terminated {
		c.ContainerStatus.State = runtimeapi.ContainerState_CONTAINER_EXITED
	} else {
		c.ContainerStatus.State = runtimeapi.ContainerState_CONTAINER_RUNNING
	}
	c.ContainerStatus.Id = strings.ReplaceAll(string(uuid.NewUUID()), "-", "")
	return c
}

func makeFakeContainerStats(container *critest.FakeContainer, imageFsMountpoint string) *runtimeapi.ContainerStats {
	containerStats := &runtimeapi.ContainerStats{
		Attributes: &runtimeapi.ContainerAttributes{
			Id:       container.ContainerStatus.Id,
			Metadata: container.ContainerStatus.Metadata,
		},
		WritableLayer: &runtimeapi.FilesystemUsage{
			Timestamp:  time.Now().UnixNano(),
			FsId:       &runtimeapi.FilesystemIdentifier{Mountpoint: imageFsMountpoint},
			UsedBytes:  &runtimeapi.UInt64Value{Value: rand.Uint64() / 100},
			InodesUsed: &runtimeapi.UInt64Value{Value: rand.Uint64() / 100},
		},
	}
	if container.State == runtimeapi.ContainerState_CONTAINER_EXITED {
		containerStats.Cpu = nil
		containerStats.Memory = nil
	} else {
		containerStats.Cpu = &runtimeapi.CpuUsage{
			Timestamp:            time.Now().UnixNano(),
			UsageCoreNanoSeconds: &runtimeapi.UInt64Value{Value: rand.Uint64()},
		}
		containerStats.Memory = &runtimeapi.MemoryUsage{
			Timestamp:       time.Now().UnixNano(),
			WorkingSetBytes: &runtimeapi.UInt64Value{Value: rand.Uint64()},
		}
	}
	return containerStats
}

// makeFakeContainerStatsStrictlyFromCRI use CRI offset to fake CRI container stats to distinguish cadvisor stats.
func makeFakeContainerStatsStrictlyFromCRI(seed int, container *critest.FakeContainer, imageFsMountpoint string) *runtimeapi.ContainerStats {
	containerStats := &runtimeapi.ContainerStats{
		Attributes: &runtimeapi.ContainerAttributes{
			Id:       container.ContainerStatus.Id,
			Metadata: container.ContainerStatus.Metadata,
		},
		WritableLayer: &runtimeapi.FilesystemUsage{
			Timestamp:  timestamp.UnixNano(),
			FsId:       &runtimeapi.FilesystemIdentifier{Mountpoint: imageFsMountpoint},
			UsedBytes:  &runtimeapi.UInt64Value{Value: uint64(seed + offsetCRI + offsetFsUsage)},
			InodesUsed: &runtimeapi.UInt64Value{Value: uint64(seed + offsetCRI + offsetFsInodeUsage)},
		},
	}
	if container.State == runtimeapi.ContainerState_CONTAINER_EXITED {
		containerStats.Cpu = nil
		containerStats.Memory = nil
	} else {
		containerStats.Cpu = &runtimeapi.CpuUsage{
			Timestamp:            timestamp.UnixNano(),
			UsageCoreNanoSeconds: &runtimeapi.UInt64Value{Value: uint64(seed + offsetCRI + offsetCPUUsageCoreSeconds)},
		}
		containerStats.Memory = &runtimeapi.MemoryUsage{
			Timestamp:       timestamp.UnixNano(),
			WorkingSetBytes: &runtimeapi.UInt64Value{Value: uint64(seed + offsetCRI + offsetMemWorkingSetBytes)},
		}
	}
	return containerStats
}

func makeFakePodSandboxStatsStrictlyFromCRI(seed int, podSandbox *critest.FakePodSandbox, podContainerStats ...*runtimeapi.ContainerStats) *runtimeapi.PodSandboxStats {
	podSandboxStats := &runtimeapi.PodSandboxStats{
		Attributes: &runtimeapi.PodSandboxAttributes{
			Id:       podSandbox.Id,
			Metadata: podSandbox.Metadata,
		},
		Linux: &runtimeapi.LinuxPodSandboxStats{},
	}
	podSandboxStats.Linux.Containers = append(podSandboxStats.Linux.Containers, podContainerStats...)
	if podSandbox.State == runtimeapi.PodSandboxState_SANDBOX_NOTREADY {
		podSandboxStats.Linux.Cpu = nil
		podSandboxStats.Linux.Memory = nil
	} else {
		podSandboxStats.Linux.Cpu = &runtimeapi.CpuUsage{
			Timestamp:            timestamp.UnixNano(),
			UsageCoreNanoSeconds: &runtimeapi.UInt64Value{Value: uint64(seed + offsetCRI + offsetCPUUsageCoreSeconds)},
		}
		podSandboxStats.Linux.Memory = &runtimeapi.MemoryUsage{
			Timestamp:       timestamp.UnixNano(),
			WorkingSetBytes: &runtimeapi.UInt64Value{Value: uint64(seed + offsetCRI + offsetMemWorkingSetBytes)},
		}
	}
	return podSandboxStats
}
func getPodSandboxStatsStrictlyFromCRI(seed int, podSandbox *critest.FakePodSandbox) statsapi.PodStats {
	podStats := statsapi.PodStats{
		PodRef: statsapi.PodReference{
			Name:      podSandbox.Metadata.Name,
			UID:       podSandbox.Metadata.Uid,
			Namespace: podSandbox.Metadata.Namespace,
		},
		// The StartTime in the summary API is the pod creation time.
		StartTime: metav1.NewTime(time.Unix(0, podSandbox.CreatedAt)),
	}
	if podSandbox.State == runtimeapi.PodSandboxState_SANDBOX_NOTREADY {
		podStats.CPU = nil
		podStats.Memory = nil
	} else {
		usageCoreNanoSeconds := uint64(seed + offsetCRI + offsetCPUUsageCoreSeconds)
		workingSetBytes := uint64(seed + offsetCRI + offsetMemWorkingSetBytes)
		podStats.CPU = &statsapi.CPUStats{
			Time:                 metav1.NewTime(timestamp),
			UsageCoreNanoSeconds: &usageCoreNanoSeconds,
		}
		podStats.Memory = &statsapi.MemoryStats{
			Time:            metav1.NewTime(timestamp),
			WorkingSetBytes: &workingSetBytes,
		}
	}

	return podStats
}

func makeFakeImageFsUsage(fsMountpoint string) *runtimeapi.FilesystemUsage {
	return &runtimeapi.FilesystemUsage{
		Timestamp:  time.Now().UnixNano(),
		FsId:       &runtimeapi.FilesystemIdentifier{Mountpoint: fsMountpoint},
		UsedBytes:  &runtimeapi.UInt64Value{Value: rand.Uint64()},
		InodesUsed: &runtimeapi.UInt64Value{Value: rand.Uint64()},
	}
}

func makeFakeVolumeStats(volumeNames []string) []statsapi.VolumeStats {
	volumes := make([]statsapi.VolumeStats, len(volumeNames))
	availableBytes := rand.Uint64()
	capacityBytes := rand.Uint64()
	usedBytes := rand.Uint64() / 100
	inodes := rand.Uint64()
	inodesFree := rand.Uint64()
	inodesUsed := rand.Uint64() / 100
	for i, name := range volumeNames {
		fsStats := statsapi.FsStats{
			Time:           metav1.NewTime(time.Now()),
			AvailableBytes: &availableBytes,
			CapacityBytes:  &capacityBytes,
			UsedBytes:      &usedBytes,
			Inodes:         &inodes,
			InodesFree:     &inodesFree,
			InodesUsed:     &inodesUsed,
		}
		volumes[i] = statsapi.VolumeStats{
			FsStats: fsStats,
			Name:    name,
		}
	}
	return volumes
}

func checkCRICPUAndMemoryStats(assert *assert.Assertions, actual statsapi.ContainerStats, cs *cadvisorapiv2.ContainerStats) {
	assert.Equal(cs.Timestamp.UnixNano(), actual.CPU.Time.UnixNano())
	assert.Equal(cs.Cpu.Usage.Total, *actual.CPU.UsageCoreNanoSeconds)
	assert.Equal(cs.CpuInst.Usage.Total, *actual.CPU.UsageNanoCores)

	assert.Equal(cs.Memory.Usage, *actual.Memory.UsageBytes)
	assert.Equal(cs.Memory.WorkingSet, *actual.Memory.WorkingSetBytes)
	assert.Equal(cs.Memory.RSS, *actual.Memory.RSSBytes)
	assert.Equal(cs.Memory.ContainerData.Pgfault, *actual.Memory.PageFaults)
	assert.Equal(cs.Memory.ContainerData.Pgmajfault, *actual.Memory.MajorPageFaults)
}

func checkCRICPUAndMemoryStatsForStrictlyFromCRI(assert *assert.Assertions, actual statsapi.ContainerStats, excepted statsapi.ContainerStats) {
	assert.Equal(excepted.CPU.Time.UnixNano(), actual.CPU.Time.UnixNano())
	assert.Equal(*excepted.CPU.UsageCoreNanoSeconds, *actual.CPU.UsageCoreNanoSeconds)
	assert.Equal(*excepted.Memory.WorkingSetBytes, *actual.Memory.WorkingSetBytes)
}

func checkCRIRootfsStats(assert *assert.Assertions, actual statsapi.ContainerStats, cs *runtimeapi.ContainerStats, imageFsInfo *cadvisorapiv2.FsInfo) {
	assert.Equal(cs.WritableLayer.Timestamp, actual.Rootfs.Time.UnixNano())
	if imageFsInfo != nil {
		assert.Equal(imageFsInfo.Available, *actual.Rootfs.AvailableBytes)
		assert.Equal(imageFsInfo.Capacity, *actual.Rootfs.CapacityBytes)
		assert.Equal(*imageFsInfo.InodesFree, *actual.Rootfs.InodesFree)
		assert.Equal(*imageFsInfo.Inodes, *actual.Rootfs.Inodes)
	} else {
		assert.Nil(actual.Rootfs.AvailableBytes)
		assert.Nil(actual.Rootfs.CapacityBytes)
		assert.Nil(actual.Rootfs.InodesFree)
		assert.Nil(actual.Rootfs.Inodes)
	}
	assert.Equal(cs.WritableLayer.UsedBytes.Value, *actual.Rootfs.UsedBytes)
	assert.Equal(cs.WritableLayer.InodesUsed.Value, *actual.Rootfs.InodesUsed)
}

func checkCRILogsStats(assert *assert.Assertions, actual statsapi.ContainerStats, rootFsInfo *cadvisorapiv2.FsInfo, logStats *volume.Metrics) {
	assert.Equal(rootFsInfo.Timestamp, actual.Logs.Time.Time)
	assert.Equal(rootFsInfo.Available, *actual.Logs.AvailableBytes)
	assert.Equal(rootFsInfo.Capacity, *actual.Logs.CapacityBytes)
	assert.Equal(*rootFsInfo.InodesFree, *actual.Logs.InodesFree)
	assert.Equal(*rootFsInfo.Inodes, *actual.Logs.Inodes)
	assert.Equal(uint64(logStats.Used.Value()), *actual.Logs.UsedBytes)
	assert.Equal(uint64(logStats.InodesUsed.Value()), *actual.Logs.InodesUsed)
}

func checkEphemeralStorageStats(assert *assert.Assertions,
	actual statsapi.PodStats,
	volumes []statsapi.VolumeStats,
	containers []*runtimeapi.ContainerStats,
	containerLogStats []*volume.Metrics,
	podLogStats *volume.Metrics) {
	var totalUsed, inodesUsed uint64
	for _, container := range containers {
		totalUsed = totalUsed + container.WritableLayer.UsedBytes.Value
		inodesUsed = inodesUsed + container.WritableLayer.InodesUsed.Value
	}

	for _, volume := range volumes {
		totalUsed = totalUsed + *volume.FsStats.UsedBytes
		inodesUsed = inodesUsed + *volume.FsStats.InodesUsed
	}

	for _, logStats := range containerLogStats {
		totalUsed = totalUsed + uint64(logStats.Used.Value())
		inodesUsed = inodesUsed + uint64(logStats.InodesUsed.Value())
	}

	if podLogStats != nil {
		totalUsed = totalUsed + uint64(podLogStats.Used.Value())
		inodesUsed = inodesUsed + uint64(podLogStats.InodesUsed.Value())
	}

	assert.Equal(int(totalUsed), int(*actual.EphemeralStorage.UsedBytes))
	assert.Equal(int(inodesUsed), int(*actual.EphemeralStorage.InodesUsed))
}

func checkCRINetworkStats(assert *assert.Assertions, actual *statsapi.NetworkStats, expected *cadvisorapiv2.NetworkStats) {
	assert.Equal(expected.Interfaces[0].RxBytes, *actual.RxBytes)
	assert.Equal(expected.Interfaces[0].RxErrors, *actual.RxErrors)
	assert.Equal(expected.Interfaces[0].TxBytes, *actual.TxBytes)
	assert.Equal(expected.Interfaces[0].TxErrors, *actual.TxErrors)
}

func checkCRIPodCPUAndMemoryStats(assert *assert.Assertions, actual statsapi.PodStats, cs *cadvisorapiv2.ContainerStats) {
	if runtime.GOOS != "linux" {
		return
	}
	assert.Equal(cs.Timestamp.UnixNano(), actual.CPU.Time.UnixNano())
	assert.Equal(cs.Cpu.Usage.Total, *actual.CPU.UsageCoreNanoSeconds)
	assert.Equal(cs.CpuInst.Usage.Total, *actual.CPU.UsageNanoCores)

	assert.Equal(cs.Memory.Usage, *actual.Memory.UsageBytes)
	assert.Equal(cs.Memory.WorkingSet, *actual.Memory.WorkingSetBytes)
	assert.Equal(cs.Memory.RSS, *actual.Memory.RSSBytes)
	assert.Equal(cs.Memory.ContainerData.Pgfault, *actual.Memory.PageFaults)
	assert.Equal(cs.Memory.ContainerData.Pgmajfault, *actual.Memory.MajorPageFaults)
}

func checkCRIPodSwapStats(assert *assert.Assertions, actual statsapi.PodStats, cs *cadvisorapiv2.ContainerStats) {
	if runtime.GOOS != "linux" {
		return
	}

	assert.Equal(cs.Timestamp.UnixNano(), actual.Swap.Time.UnixNano())
	assert.Equal(cs.Memory.Swap, *actual.Swap.SwapUsageBytes)
}

func checkCRIPodCPUAndMemoryStatsStrictlyFromCRI(assert *assert.Assertions, actual statsapi.PodStats, excepted statsapi.PodStats) {
	if runtime.GOOS != "linux" {
		return
	}
	assert.Equal(excepted.CPU.Time.UnixNano(), actual.CPU.Time.UnixNano())
	assert.Equal(*excepted.CPU.UsageCoreNanoSeconds, *actual.CPU.UsageCoreNanoSeconds)
	assert.Equal(*excepted.Memory.WorkingSetBytes, *actual.Memory.WorkingSetBytes)
}

func makeFakeLogStats(seed int) *volume.Metrics {
	m := &volume.Metrics{}
	m.Used = resource.NewQuantity(int64(seed+offsetUsage), resource.BinarySI)
	m.InodesUsed = resource.NewQuantity(int64(seed+offsetInodeUsage), resource.BinarySI)
	return m
}

func TestGetContainerUsageNanoCores(t *testing.T) {
	var value0 uint64
	var value1 uint64 = 10000000000

	// Test with a large container of 100+ CPUs
	var value2 uint64 = 188427786383

	tests := []struct {
		desc          string
		cpuUsageCache map[string]*cpuUsageRecord
		stats         *runtimeapi.ContainerStats
		expected      *uint64
	}{
		{
			desc:          "should return nil if stats is nil",
			cpuUsageCache: map[string]*cpuUsageRecord{},
		},
		{
			desc:          "should return nil if cpu stats is nil",
			cpuUsageCache: map[string]*cpuUsageRecord{},
			stats: &runtimeapi.ContainerStats{
				Attributes: &runtimeapi.ContainerAttributes{
					Id: "1",
				},
				Cpu: nil,
			},
		},
		{
			desc:          "should return nil if usageCoreNanoSeconds is nil",
			cpuUsageCache: map[string]*cpuUsageRecord{},
			stats: &runtimeapi.ContainerStats{
				Attributes: &runtimeapi.ContainerAttributes{
					Id: "1",
				},
				Cpu: &runtimeapi.CpuUsage{
					Timestamp:            1,
					UsageCoreNanoSeconds: nil,
				},
			},
		},
		{
			desc:          "should return nil if cpu stats is not cached yet",
			cpuUsageCache: map[string]*cpuUsageRecord{},
			stats: &runtimeapi.ContainerStats{
				Attributes: &runtimeapi.ContainerAttributes{
					Id: "1",
				},
				Cpu: &runtimeapi.CpuUsage{
					Timestamp: 1,
					UsageCoreNanoSeconds: &runtimeapi.UInt64Value{
						Value: 10000000000,
					},
				},
			},
		},
		{
			desc: "should return zero value if cached cpu stats is equal to current value",
			stats: &runtimeapi.ContainerStats{
				Attributes: &runtimeapi.ContainerAttributes{
					Id: "1",
				},
				Cpu: &runtimeapi.CpuUsage{
					Timestamp: 1,
					UsageCoreNanoSeconds: &runtimeapi.UInt64Value{
						Value: 10000000000,
					},
				},
			},
			cpuUsageCache: map[string]*cpuUsageRecord{
				"1": {
					stats: &runtimeapi.CpuUsage{
						Timestamp: 0,
						UsageCoreNanoSeconds: &runtimeapi.UInt64Value{
							Value: 10000000000,
						},
					},
				},
			},
			expected: &value0,
		},
		{
			desc: "should return correct value if cached cpu stats is not equal to current value",
			stats: &runtimeapi.ContainerStats{
				Attributes: &runtimeapi.ContainerAttributes{
					Id: "1",
				},
				Cpu: &runtimeapi.CpuUsage{
					Timestamp: int64(time.Second / time.Nanosecond),
					UsageCoreNanoSeconds: &runtimeapi.UInt64Value{
						Value: 20000000000,
					},
				},
			},
			cpuUsageCache: map[string]*cpuUsageRecord{
				"1": {
					stats: &runtimeapi.CpuUsage{
						Timestamp: 0,
						UsageCoreNanoSeconds: &runtimeapi.UInt64Value{
							Value: 10000000000,
						},
					},
				},
			},
			expected: &value1,
		},
		{
			desc: "should return correct value if elapsed UsageCoreNanoSeconds exceeds 18446744073",
			stats: &runtimeapi.ContainerStats{
				Attributes: &runtimeapi.ContainerAttributes{
					Id: "1",
				},
				Cpu: &runtimeapi.CpuUsage{
					Timestamp: int64(time.Second / time.Nanosecond),
					UsageCoreNanoSeconds: &runtimeapi.UInt64Value{
						Value: 68172016162105,
					},
				},
			},
			cpuUsageCache: map[string]*cpuUsageRecord{
				"1": {
					stats: &runtimeapi.CpuUsage{
						Timestamp: 0,
						UsageCoreNanoSeconds: &runtimeapi.UInt64Value{
							Value: 67983588375722,
						},
					},
				},
			},
			expected: &value2,
		},
		{
			desc: "should return nil if cpuacct is reset to 0 in a live container",
			stats: &runtimeapi.ContainerStats{
				Attributes: &runtimeapi.ContainerAttributes{
					Id: "1",
				},
				Cpu: &runtimeapi.CpuUsage{
					Timestamp: 2,
					UsageCoreNanoSeconds: &runtimeapi.UInt64Value{
						Value: 0,
					},
				},
			},
			cpuUsageCache: map[string]*cpuUsageRecord{
				"1": {
					stats: &runtimeapi.CpuUsage{
						Timestamp: 1,
						UsageCoreNanoSeconds: &runtimeapi.UInt64Value{
							Value: 10000000000,
						},
					},
				},
			},
			expected: nil,
		},
	}
	for _, test := range tests {
		provider := &criStatsProvider{cpuUsageCache: test.cpuUsageCache}
		// Before the update, the cached value should be nil
		cached := provider.getContainerUsageNanoCores(test.stats)
		assert.Nil(t, cached)

		// Update the cache and get the latest value.
		real := provider.getAndUpdateContainerUsageNanoCores(test.stats)
		assert.Equal(t, test.expected, real, test.desc)

		// After the update, the cached value should be up-to-date
		cached = provider.getContainerUsageNanoCores(test.stats)
		assert.Equal(t, test.expected, cached, test.desc)
	}
}

func TestExtractIDFromCgroupPath(t *testing.T) {
	tests := []struct {
		cgroupPath string
		expected   string
	}{
		{
			cgroupPath: "/kubepods/burstable/pod2fc932ce-fdcc-454b-97bd-aadfdeb4c340/9be25294016e2dc0340dd605ce1f57b492039b267a6a618a7ad2a7a58a740f32",
			expected:   "9be25294016e2dc0340dd605ce1f57b492039b267a6a618a7ad2a7a58a740f32",
		},
		{
			cgroupPath: "/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod2fc932ce_fdcc_454b_97bd_aadfdeb4c340.slice/cri-containerd-aaefb9d8feed2d453b543f6d928cede7a4dbefa6a0ae7c9b990dd234c56e93b9.scope",
			expected:   "aaefb9d8feed2d453b543f6d928cede7a4dbefa6a0ae7c9b990dd234c56e93b9",
		},
		{
			cgroupPath: "/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod2fc932ce_fdcc_454b_97bd_aadfdeb4c340.slice/cri-o-aaefb9d8feed2d453b543f6d928cede7a4dbefa6a0ae7c9b990dd234c56e93b9.scope",
			expected:   "aaefb9d8feed2d453b543f6d928cede7a4dbefa6a0ae7c9b990dd234c56e93b9",
		},
	}

	for _, test := range tests {
		id := extractIDFromCgroupPath(test.cgroupPath)
		assert.Equal(t, test.expected, id)
	}
}

func getCRIContainerStatsStrictlyFromCRI(seed int, containerName string) statsapi.ContainerStats {
	result := statsapi.ContainerStats{
		Name:      containerName,
		StartTime: metav1.NewTime(timestamp),
		CPU:       &statsapi.CPUStats{},
		Memory:    &statsapi.MemoryStats{},
		// UserDefinedMetrics is not supported by CRI.
		Rootfs: &statsapi.FsStats{},
	}

	result.CPU.Time = metav1.NewTime(timestamp)
	usageCoreNanoSeconds := uint64(seed + offsetCRI + offsetCPUUsageCoreSeconds)
	result.CPU.UsageCoreNanoSeconds = &usageCoreNanoSeconds

	result.Memory.Time = metav1.NewTime(timestamp)
	workingSetBytes := uint64(seed + offsetCRI + offsetMemWorkingSetBytes)
	result.Memory.WorkingSetBytes = &workingSetBytes

	result.Rootfs.Time = metav1.NewTime(timestamp)
	usedBytes := uint64(seed + offsetCRI + offsetFsUsage)
	result.Rootfs.UsedBytes = &usedBytes

	inodesUsed := uint64(seed + offsetCRI + offsetFsInodeUsage)
	result.Rootfs.InodesUsed = &inodesUsed

	return result
}

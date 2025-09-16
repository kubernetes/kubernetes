//go:build linux
// +build linux

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

package kuberuntime

import (
	"context"
	"fmt"
	"math"
	"os"
	"reflect"
	"strconv"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/pkg/kubelet/types"

	libcontainercgroups "github.com/opencontainers/cgroups"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

func makeExpectedConfig(_ *testing.T, tCtx context.Context, m *kubeGenericRuntimeManager, pod *v1.Pod, containerIndex int, enforceMemoryQoS bool) *runtimeapi.ContainerConfig {
	container := &pod.Spec.Containers[containerIndex]
	podIP := ""
	restartCount := 0
	opts, _, _ := m.runtimeHelper.GenerateRunContainerOptions(tCtx, pod, container, podIP, []string{podIP}, nil)
	containerLogsPath := buildContainerLogsPath(container.Name, restartCount)
	stopsignal := getContainerConfigStopSignal(container)
	restartCountUint32 := uint32(restartCount)
	envs := make([]*runtimeapi.KeyValue, len(opts.Envs))

	l, _ := m.generateLinuxContainerConfig(tCtx, container, pod, new(int64), "", nil, enforceMemoryQoS)

	expectedConfig := &runtimeapi.ContainerConfig{
		Metadata: &runtimeapi.ContainerMetadata{
			Name:    container.Name,
			Attempt: restartCountUint32,
		},
		Image:       &runtimeapi.ImageSpec{Image: container.Image, UserSpecifiedImage: container.Image},
		Command:     container.Command,
		Args:        []string(nil),
		WorkingDir:  container.WorkingDir,
		Labels:      newContainerLabels(container, pod),
		Annotations: newContainerAnnotations(tCtx, container, pod, restartCount, opts),
		Devices:     makeDevices(opts),
		Mounts:      m.makeMounts(opts, container),
		LogPath:     containerLogsPath,
		Stdin:       container.Stdin,
		StdinOnce:   container.StdinOnce,
		Tty:         container.TTY,
		Linux:       l,
		Envs:        envs,
		CDIDevices:  makeCDIDevices(opts),
	}
	if stopsignal != nil {
		expectedConfig.StopSignal = *stopsignal
	}
	return expectedConfig
}

func TestGenerateContainerConfig(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, imageService, m, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	runAsUser := int64(1000)
	runAsGroup := int64(2000)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "bar",
			Namespace: "new",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "foo",
					Image:           "busybox",
					ImagePullPolicy: v1.PullIfNotPresent,
					Command:         []string{"testCommand"},
					WorkingDir:      "testWorkingDir",
					SecurityContext: &v1.SecurityContext{
						RunAsUser:  &runAsUser,
						RunAsGroup: &runAsGroup,
					},
				},
			},
		},
	}

	expectedConfig := makeExpectedConfig(t, tCtx, m, pod, 0, false)
	containerConfig, _, err := m.generateContainerConfig(tCtx, &pod.Spec.Containers[0], pod, 0, "", pod.Spec.Containers[0].Image, []string{}, nil, nil)
	assert.NoError(t, err)
	assert.Equal(t, expectedConfig, containerConfig, "generate container config for kubelet runtime v1.")
	assert.Equal(t, runAsUser, containerConfig.GetLinux().GetSecurityContext().GetRunAsUser().GetValue(), "RunAsUser should be set")
	assert.Equal(t, runAsGroup, containerConfig.GetLinux().GetSecurityContext().GetRunAsGroup().GetValue(), "RunAsGroup should be set")

	runAsRoot := int64(0)
	runAsNonRootTrue := true
	podWithContainerSecurityContext := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "bar",
			Namespace: "new",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "foo",
					Image:           "busybox",
					ImagePullPolicy: v1.PullIfNotPresent,
					Command:         []string{"testCommand"},
					WorkingDir:      "testWorkingDir",
					SecurityContext: &v1.SecurityContext{
						RunAsNonRoot: &runAsNonRootTrue,
						RunAsUser:    &runAsRoot,
					},
				},
			},
		},
	}

	_, _, err = m.generateContainerConfig(tCtx, &podWithContainerSecurityContext.Spec.Containers[0], podWithContainerSecurityContext, 0, "", podWithContainerSecurityContext.Spec.Containers[0].Image, []string{}, nil, nil)
	assert.Error(t, err)

	imageID, _ := imageService.PullImage(tCtx, &runtimeapi.ImageSpec{Image: "busybox"}, nil, nil)
	resp, _ := imageService.ImageStatus(tCtx, &runtimeapi.ImageSpec{Image: imageID}, false)

	resp.Image.Uid = nil
	resp.Image.Username = "test"

	podWithContainerSecurityContext.Spec.Containers[0].SecurityContext.RunAsUser = nil
	podWithContainerSecurityContext.Spec.Containers[0].SecurityContext.RunAsNonRoot = &runAsNonRootTrue

	_, _, err = m.generateContainerConfig(tCtx, &podWithContainerSecurityContext.Spec.Containers[0], podWithContainerSecurityContext, 0, "", podWithContainerSecurityContext.Spec.Containers[0].Image, []string{}, nil, nil)
	assert.Error(t, err, "RunAsNonRoot should fail for non-numeric username")
}

func TestGenerateLinuxContainerConfigResources(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, _, m, err := createTestRuntimeManager(tCtx)
	m.cpuCFSQuota = true

	assert.NoError(t, err)

	tests := []struct {
		name               string
		containerResources v1.ResourceRequirements
		podResources       *v1.ResourceRequirements
		expected           *runtimeapi.LinuxContainerResources
	}{
		{
			name: "Request 128M/1C, Limit 256M/3C",
			containerResources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceMemory: resource.MustParse("128Mi"),
					v1.ResourceCPU:    resource.MustParse("1"),
				},
				Limits: v1.ResourceList{
					v1.ResourceMemory: resource.MustParse("256Mi"),
					v1.ResourceCPU:    resource.MustParse("3"),
				},
			},
			expected: &runtimeapi.LinuxContainerResources{
				CpuPeriod:          100000,
				CpuQuota:           300000,
				CpuShares:          1024,
				MemoryLimitInBytes: 256 * 1024 * 1024,
			},
		},
		{
			name: "Request 128M/2C, No Limit",
			containerResources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceMemory: resource.MustParse("128Mi"),
					v1.ResourceCPU:    resource.MustParse("2"),
				},
			},
			expected: &runtimeapi.LinuxContainerResources{
				CpuPeriod:          100000,
				CpuQuota:           0,
				CpuShares:          2048,
				MemoryLimitInBytes: 0,
			},
		},
		{
			name: "Container Level Request 128M/1C, Pod Level Limit 256M/3C",
			containerResources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceMemory: resource.MustParse("128Mi"),
					v1.ResourceCPU:    resource.MustParse("1"),
				},
			},
			podResources: &v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceMemory: resource.MustParse("256Mi"),
					v1.ResourceCPU:    resource.MustParse("3"),
				},
			},
			expected: &runtimeapi.LinuxContainerResources{
				CpuPeriod:          100000,
				CpuQuota:           300000,
				CpuShares:          1024,
				MemoryLimitInBytes: 256 * 1024 * 1024,
			},
		},
	}

	for _, test := range tests {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				UID:       "12345678",
				Name:      "bar",
				Namespace: "new",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:            "foo",
						Image:           "busybox",
						ImagePullPolicy: v1.PullIfNotPresent,
						Command:         []string{"testCommand"},
						WorkingDir:      "testWorkingDir",
						Resources:       test.containerResources,
					},
				},
			},
		}

		if test.podResources != nil {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, true)
			pod.Spec.Resources = test.podResources
		}

		linuxConfig, err := m.generateLinuxContainerConfig(tCtx, &pod.Spec.Containers[0], pod, new(int64), "", nil, false)
		assert.NoError(t, err)
		assert.Equal(t, test.expected.CpuPeriod, linuxConfig.GetResources().CpuPeriod, test.name)
		assert.Equal(t, test.expected.CpuQuota, linuxConfig.GetResources().CpuQuota, test.name)
		assert.Equal(t, test.expected.CpuShares, linuxConfig.GetResources().CpuShares, test.name)
		assert.Equal(t, test.expected.MemoryLimitInBytes, linuxConfig.GetResources().MemoryLimitInBytes, test.name)
	}
}

func TestCalculateLinuxResources(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, _, m, err := createTestRuntimeManager(tCtx)
	m.cpuCFSQuota = true

	assert.NoError(t, err)

	generateResourceQuantity := func(str string) *resource.Quantity {
		quantity := resource.MustParse(str)
		return &quantity
	}

	tests := []struct {
		name                 string
		cpuReq               *resource.Quantity
		cpuLim               *resource.Quantity
		memLim               *resource.Quantity
		expected             *runtimeapi.LinuxContainerResources
		cgroupVersion        CgroupVersion
		singleProcessOOMKill bool
	}{
		{
			name:   "Request128MBLimit256MB",
			cpuReq: generateResourceQuantity("1"),
			cpuLim: generateResourceQuantity("2"),
			memLim: generateResourceQuantity("128Mi"),
			expected: &runtimeapi.LinuxContainerResources{
				CpuPeriod:          100000,
				CpuQuota:           200000,
				CpuShares:          1024,
				MemoryLimitInBytes: 134217728,
			},
			cgroupVersion: cgroupV1,
		},
		{
			name:   "RequestNoMemory",
			cpuReq: generateResourceQuantity("2"),
			cpuLim: generateResourceQuantity("8"),
			memLim: generateResourceQuantity("0"),
			expected: &runtimeapi.LinuxContainerResources{
				CpuPeriod:          100000,
				CpuQuota:           800000,
				CpuShares:          2048,
				MemoryLimitInBytes: 0,
			},
			cgroupVersion: cgroupV1,
		},
		{
			name:   "RequestNilCPU",
			cpuLim: generateResourceQuantity("2"),
			memLim: generateResourceQuantity("0"),
			expected: &runtimeapi.LinuxContainerResources{
				CpuPeriod:          100000,
				CpuQuota:           200000,
				CpuShares:          2048,
				MemoryLimitInBytes: 0,
			},
			cgroupVersion: cgroupV1,
		},
		{
			name:   "RequestZeroCPU",
			cpuReq: generateResourceQuantity("0"),
			cpuLim: generateResourceQuantity("2"),
			memLim: generateResourceQuantity("0"),
			expected: &runtimeapi.LinuxContainerResources{
				CpuPeriod:          100000,
				CpuQuota:           200000,
				CpuShares:          2,
				MemoryLimitInBytes: 0,
			},
			cgroupVersion: cgroupV1,
		},
		{
			name:   "Request128MBLimit256MB",
			cpuReq: generateResourceQuantity("1"),
			cpuLim: generateResourceQuantity("2"),
			memLim: generateResourceQuantity("128Mi"),
			expected: &runtimeapi.LinuxContainerResources{
				CpuPeriod:          100000,
				CpuQuota:           200000,
				CpuShares:          1024,
				MemoryLimitInBytes: 134217728,
				Unified:            map[string]string{"memory.oom.group": "1"},
			},
			cgroupVersion: cgroupV2,
		},
		{
			name:   "Request128MBLimit256MBSingleProcess",
			cpuReq: generateResourceQuantity("1"),
			cpuLim: generateResourceQuantity("2"),
			memLim: generateResourceQuantity("128Mi"),
			expected: &runtimeapi.LinuxContainerResources{
				CpuPeriod:          100000,
				CpuQuota:           200000,
				CpuShares:          1024,
				MemoryLimitInBytes: 134217728,
			},
			cgroupVersion:        cgroupV2,
			singleProcessOOMKill: true,
		},
		{
			name:   "RequestNoMemory",
			cpuReq: generateResourceQuantity("2"),
			cpuLim: generateResourceQuantity("8"),
			memLim: generateResourceQuantity("0"),
			expected: &runtimeapi.LinuxContainerResources{
				CpuPeriod:          100000,
				CpuQuota:           800000,
				CpuShares:          2048,
				MemoryLimitInBytes: 0,
				Unified:            map[string]string{"memory.oom.group": "1"},
			},
			cgroupVersion: cgroupV2,
		},
		{
			name:   "RequestNilCPU",
			cpuLim: generateResourceQuantity("2"),
			memLim: generateResourceQuantity("0"),
			expected: &runtimeapi.LinuxContainerResources{
				CpuPeriod:          100000,
				CpuQuota:           200000,
				CpuShares:          2048,
				MemoryLimitInBytes: 0,
				Unified:            map[string]string{"memory.oom.group": "1"},
			},
			cgroupVersion: cgroupV2,
		},
		{
			name:   "RequestZeroCPU",
			cpuReq: generateResourceQuantity("0"),
			cpuLim: generateResourceQuantity("2"),
			memLim: generateResourceQuantity("0"),
			expected: &runtimeapi.LinuxContainerResources{
				CpuPeriod:          100000,
				CpuQuota:           200000,
				CpuShares:          2,
				MemoryLimitInBytes: 0,
				Unified:            map[string]string{"memory.oom.group": "1"},
			},
			cgroupVersion: cgroupV2,
		},
	}
	for _, test := range tests {
		setCgroupVersionDuringTest(test.cgroupVersion)
		m.singleProcessOOMKill = ptr.To(test.singleProcessOOMKill)
		linuxContainerResources := m.calculateLinuxResources(test.cpuReq, test.cpuLim, test.memLim, false)
		assert.Equal(t, test.expected, linuxContainerResources)
	}
}

func TestGenerateContainerConfigWithMemoryQoSEnforced(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, _, m, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	podRequestMemory := resource.MustParse("128Mi")
	pod1LimitMemory := resource.MustParse("256Mi")
	pod1 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "bar",
			Namespace: "new",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "foo",
					Image:           "busybox",
					ImagePullPolicy: v1.PullIfNotPresent,
					Command:         []string{"testCommand"},
					WorkingDir:      "testWorkingDir",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceMemory: podRequestMemory,
						},
						Limits: v1.ResourceList{
							v1.ResourceMemory: pod1LimitMemory,
						},
					},
				},
			},
		},
	}

	pod2 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "bar",
			Namespace: "new",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "foo",
					Image:           "busybox",
					ImagePullPolicy: v1.PullIfNotPresent,
					Command:         []string{"testCommand"},
					WorkingDir:      "testWorkingDir",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceMemory: podRequestMemory,
						},
					},
				},
			},
		},
	}
	pageSize := int64(os.Getpagesize())
	memoryNodeAllocatable := resource.MustParse(fakeNodeAllocatableMemory)
	pod1MemoryHigh := int64(math.Floor(
		float64(podRequestMemory.Value())+
			(float64(pod1LimitMemory.Value())-float64(podRequestMemory.Value()))*float64(m.memoryThrottlingFactor))/float64(pageSize)) * pageSize
	pod2MemoryHigh := int64(math.Floor(
		float64(podRequestMemory.Value())+
			(float64(memoryNodeAllocatable.Value())-float64(podRequestMemory.Value()))*float64(m.memoryThrottlingFactor))/float64(pageSize)) * pageSize

	type expectedResult struct {
		containerConfig *runtimeapi.LinuxContainerConfig
		memoryLow       int64
		memoryHigh      int64
	}
	l1, _ := m.generateLinuxContainerConfig(tCtx, &pod1.Spec.Containers[0], pod1, new(int64), "", nil, true)
	l2, _ := m.generateLinuxContainerConfig(tCtx, &pod2.Spec.Containers[0], pod2, new(int64), "", nil, true)
	tests := []struct {
		name     string
		pod      *v1.Pod
		expected *expectedResult
	}{
		{
			name: "Request128MBLimit256MB",
			pod:  pod1,
			expected: &expectedResult{
				l1,
				128 * 1024 * 1024,
				int64(pod1MemoryHigh),
			},
		},
		{
			name: "Request128MBWithoutLimit",
			pod:  pod2,
			expected: &expectedResult{
				l2,
				128 * 1024 * 1024,
				int64(pod2MemoryHigh),
			},
		},
	}

	for _, test := range tests {
		linuxConfig, err := m.generateLinuxContainerConfig(tCtx, &test.pod.Spec.Containers[0], test.pod, new(int64), "", nil, true)
		assert.NoError(t, err)
		assert.Equal(t, test.expected.containerConfig, linuxConfig, test.name)
		assert.Equal(t, linuxConfig.GetResources().GetUnified()["memory.min"], strconv.FormatInt(test.expected.memoryLow, 10), test.name)
		assert.Equal(t, linuxConfig.GetResources().GetUnified()["memory.high"], strconv.FormatInt(test.expected.memoryHigh, 10), test.name)
	}
}

func TestGetHugepageLimitsFromResources(t *testing.T) {
	tCtx := ktesting.Init(t)
	var baseHugepage []*runtimeapi.HugepageLimit

	// For each page size, limit to 0.
	for _, pageSize := range libcontainercgroups.HugePageSizes() {
		baseHugepage = append(baseHugepage, &runtimeapi.HugepageLimit{
			PageSize: pageSize,
			Limit:    uint64(0),
		})
	}

	tests := []struct {
		name                     string
		podResources             v1.ResourceRequirements
		resources                v1.ResourceRequirements
		expected                 []*runtimeapi.HugepageLimit
		podLevelResourcesEnabled bool
	}{
		{
			name: "Success2MB",
			resources: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					"hugepages-2Mi": resource.MustParse("2Mi"),
				},
			},
			expected: []*runtimeapi.HugepageLimit{
				{
					PageSize: "2MB",
					Limit:    2097152,
				},
			},
		},
		{
			name: "Success1GB",
			resources: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					"hugepages-1Gi": resource.MustParse("2Gi"),
				},
			},
			expected: []*runtimeapi.HugepageLimit{
				{
					PageSize: "1GB",
					Limit:    2147483648,
				},
			},
		},
		{
			name: "Skip2MB",
			resources: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					"hugepages-2MB": resource.MustParse("2Mi"),
				},
			},
			expected: []*runtimeapi.HugepageLimit{
				{
					PageSize: "2MB",
					Limit:    0,
				},
			},
		},
		{
			name: "Skip1GB",
			resources: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					"hugepages-1GB": resource.MustParse("2Gi"),
				},
			},
			expected: []*runtimeapi.HugepageLimit{
				{
					PageSize: "1GB",
					Limit:    0,
				},
			},
		},
		{
			name: "Success2MBand1GB",
			resources: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					"hugepages-2Mi": resource.MustParse("2Mi"),
					"hugepages-1Gi": resource.MustParse("2Gi"),
				},
			},
			expected: []*runtimeapi.HugepageLimit{
				{
					PageSize: "2MB",
					Limit:    2097152,
				},
				{
					PageSize: "1GB",
					Limit:    2147483648,
				},
			},
		},
		{
			name: "Skip2MBand1GB",
			resources: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					"hugepages-2MB": resource.MustParse("2Mi"),
					"hugepages-1GB": resource.MustParse("2Gi"),
				},
			},
			expected: []*runtimeapi.HugepageLimit{
				{
					PageSize: "2MB",
					Limit:    0,
				},
				{
					PageSize: "1GB",
					Limit:    0,
				},
			},
		},
		// Pod level hugepage set 2MB, Container level hugepage unset
		{
			name: "PodLevel2MB",
			podResources: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					"hugepages-2Mi": resource.MustParse("2Mi"),
				},
			},
			resources: v1.ResourceRequirements{},
			expected: []*runtimeapi.HugepageLimit{
				{
					PageSize: "2MB",
					Limit:    2097152,
				},
			},
			podLevelResourcesEnabled: true,
		},
		// Pod level hugepage set 1GB, Container level hugepage unset
		{
			name: "PodLevel1GB",
			podResources: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					"hugepages-1Gi": resource.MustParse("2Gi"),
				},
			},
			resources: v1.ResourceRequirements{},
			expected: []*runtimeapi.HugepageLimit{
				{
					PageSize: "1GB",
					Limit:    2147483648,
				},
			},
			podLevelResourcesEnabled: true,
		},
		// Pod level hugepage set 2MB 1GB, Container level hugepage unset
		{
			name: "PodLevel2MBand1GB",
			podResources: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					"hugepages-2Mi": resource.MustParse("2Mi"),
					"hugepages-1Gi": resource.MustParse("2Gi"),
				},
			},
			resources: v1.ResourceRequirements{},
			expected: []*runtimeapi.HugepageLimit{
				{
					PageSize: "2MB",
					Limit:    2097152,
				},
				{
					PageSize: "1GB",
					Limit:    2147483648,
				},
			},
			podLevelResourcesEnabled: true,
		},
		// Pod level hugepage set 2MB, Container level hugepage set
		{
			name: "PodLevel2MBContainerLevel2MB",
			podResources: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					"hugepages-2Mi": resource.MustParse("4Mi"),
				},
			},
			resources: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					"hugepages-2Mi": resource.MustParse("2Mi"),
				},
			},
			expected: []*runtimeapi.HugepageLimit{
				{
					PageSize: "2MB",
					Limit:    2097152,
				},
			},
			podLevelResourcesEnabled: true,
		},
		// Pod level hugepage set 1GB, Container level hugepage set
		{
			name: "PodLevel1GBContainerLevel1GB",
			podResources: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					"hugepages-1Gi": resource.MustParse("4Gi"),
				},
			},
			resources: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					"hugepages-1Gi": resource.MustParse("2Gi"),
				},
			},
			expected: []*runtimeapi.HugepageLimit{
				{
					PageSize: "1GB",
					Limit:    2147483648,
				},
			},
			podLevelResourcesEnabled: true,
		},
		// Pod level hugepage set 2MB 1GB, Container level hugepage set
		{
			name: "PodLevel2MBand1GBContainerLevel2MBand1GB",
			podResources: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					"hugepages-2Mi": resource.MustParse("4Mi"),
					"hugepages-1Gi": resource.MustParse("4Gi"),
				},
			},
			resources: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					"hugepages-2Mi": resource.MustParse("2Mi"),
					"hugepages-1Gi": resource.MustParse("2Gi"),
				},
			},
			expected: []*runtimeapi.HugepageLimit{
				{
					PageSize: "2MB",
					Limit:    2097152,
				},
				{
					PageSize: "1GB",
					Limit:    2147483648,
				},
			},
			podLevelResourcesEnabled: true,
		},
	}

	for _, test := range tests {
		// Validate if machine supports hugepage size that used in test case.
		machineHugepageSupport := true
		for _, hugepageLimit := range test.expected {
			hugepageSupport := false
			for _, pageSize := range libcontainercgroups.HugePageSizes() {
				if pageSize == hugepageLimit.PageSize {
					hugepageSupport = true
					break
				}
			}

			if !hugepageSupport {
				machineHugepageSupport = false
				break
			}
		}

		// Case of machine can't support hugepage size
		if !machineHugepageSupport {
			continue
		}

		expectedHugepages := baseHugepage
		for _, hugepage := range test.expected {
			for _, expectedHugepage := range expectedHugepages {
				if expectedHugepage.PageSize == hugepage.PageSize {
					expectedHugepage.Limit = hugepage.Limit
				}
			}
		}

		testPod := &v1.Pod{}
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, test.podLevelResourcesEnabled)
		if test.podLevelResourcesEnabled {
			testPod.Spec.Resources = &test.podResources
		}

		results := GetHugepageLimitsFromResources(tCtx, testPod, test.resources)
		if !reflect.DeepEqual(expectedHugepages, results) {
			t.Errorf("%s test failed. Expected %v but got %v", test.name, expectedHugepages, results)
		}

		for _, hugepage := range baseHugepage {
			hugepage.Limit = uint64(0)
		}
	}
}

func TestGenerateLinuxContainerConfigNamespaces(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, _, m, err := createTestRuntimeManager(tCtx)
	if err != nil {
		t.Fatalf("error creating test RuntimeManager: %v", err)
	}

	for _, tc := range []struct {
		name   string
		pod    *v1.Pod
		target *kubecontainer.ContainerID
		want   *runtimeapi.NamespaceOption
	}{
		{
			"Default namespaces",
			&v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{Name: "test"},
					},
				},
			},
			nil,
			&runtimeapi.NamespaceOption{
				Pid: runtimeapi.NamespaceMode_CONTAINER,
			},
		},
		{
			"PID Namespace POD",
			&v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{Name: "test"},
					},
					ShareProcessNamespace: &[]bool{true}[0],
				},
			},
			nil,
			&runtimeapi.NamespaceOption{
				Pid: runtimeapi.NamespaceMode_POD,
			},
		},
		{
			"PID Namespace TARGET",
			&v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{Name: "test"},
					},
				},
			},
			&kubecontainer.ContainerID{Type: "docker", ID: "really-long-id-string"},
			&runtimeapi.NamespaceOption{
				Pid:      runtimeapi.NamespaceMode_TARGET,
				TargetId: "really-long-id-string",
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := m.generateLinuxContainerConfig(tCtx, &tc.pod.Spec.Containers[0], tc.pod, nil, "", tc.target, false)
			assert.NoError(t, err)
			if !proto.Equal(tc.want, got.SecurityContext.NamespaceOptions) {
				t.Errorf("%v: want %q, got %q", t.Name(), tc.want, got.SecurityContext.NamespaceOptions)
			}
		})
	}
}

var (
	supplementalGroupsPolicyUnSupported = v1.SupplementalGroupsPolicy("UnSupported")
	supplementalGroupsPolicyMerge       = v1.SupplementalGroupsPolicyMerge
	supplementalGroupsPolicyStrict      = v1.SupplementalGroupsPolicyStrict
)

func TestGenerateLinuxConfigSupplementalGroupsPolicy(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, _, m, err := createTestRuntimeManager(tCtx)
	if err != nil {
		t.Fatalf("error creating test RuntimeManager: %v", err)
	}

	containerName := "test"
	for _, tc := range []struct {
		name           string
		pod            *v1.Pod
		expected       runtimeapi.SupplementalGroupsPolicy
		expectErr      bool
		expectedErrMsg string
	}{{
		name: "Merge SupplementalGroupsPolicy should convert to Merge",
		pod: &v1.Pod{
			Spec: v1.PodSpec{
				SecurityContext: &v1.PodSecurityContext{
					SupplementalGroupsPolicy: &supplementalGroupsPolicyMerge,
				},
				Containers: []v1.Container{
					{Name: containerName},
				},
			},
		},
		expected: runtimeapi.SupplementalGroupsPolicy_Merge,
	}, {
		name: "Strict SupplementalGroupsPolicy should convert to Strict",
		pod: &v1.Pod{
			Spec: v1.PodSpec{
				SecurityContext: &v1.PodSecurityContext{
					SupplementalGroupsPolicy: &supplementalGroupsPolicyStrict,
				},
				Containers: []v1.Container{
					{Name: containerName},
				},
			},
		},
		expected: runtimeapi.SupplementalGroupsPolicy_Strict,
	}, {
		name: "nil SupplementalGroupsPolicy should convert to Merge",
		pod: &v1.Pod{
			Spec: v1.PodSpec{
				SecurityContext: &v1.PodSecurityContext{},
				Containers: []v1.Container{
					{Name: containerName},
				},
			},
		},
		expected: runtimeapi.SupplementalGroupsPolicy_Merge,
	}, {
		name: "unsupported SupplementalGroupsPolicy should raise an error",
		pod: &v1.Pod{
			Spec: v1.PodSpec{
				SecurityContext: &v1.PodSecurityContext{
					SupplementalGroupsPolicy: &supplementalGroupsPolicyUnSupported,
				},
				Containers: []v1.Container{
					{Name: containerName},
				},
			},
		},
		expectErr:      true,
		expectedErrMsg: "unsupported supplementalGroupsPolicy: UnSupported",
	},
	} {
		t.Run(tc.name, func(t *testing.T) {
			actual, err := m.generateLinuxContainerConfig(tCtx, &tc.pod.Spec.Containers[0], tc.pod, nil, "", nil, false)
			if !tc.expectErr {
				assert.Emptyf(t, err, "Unexpected error")
				assert.EqualValuesf(t, tc.expected, actual.SecurityContext.SupplementalGroupsPolicy, "SupplementalGroupPolicy for %s", tc.name)
			} else {
				assert.NotEmpty(t, err, "Unexpected success")
				assert.Empty(t, actual, "Unexpected non empty value")
				assert.ErrorContainsf(t, err, tc.expectedErrMsg, "Error for %s", tc.name)
			}
		})
	}
}

func TestGenerateLinuxContainerResources(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, _, m, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)
	m.machineInfo.MemoryCapacity = 17179860387 // 16GB

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "bar",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "c1",
					Image: "busybox",
				},
			},
		},
	}

	for _, tc := range []struct {
		name                 string
		limits               v1.ResourceList
		requests             v1.ResourceList
		singleProcessOOMKill bool
		expected             *runtimeapi.LinuxContainerResources
		cgroupVersion        CgroupVersion
	}{
		{
			"requests & limits, cpu & memory, guaranteed qos",
			v1.ResourceList{v1.ResourceCPU: resource.MustParse("250m"), v1.ResourceMemory: resource.MustParse("500Mi")},
			v1.ResourceList{v1.ResourceCPU: resource.MustParse("250m"), v1.ResourceMemory: resource.MustParse("500Mi")},
			true,
			&runtimeapi.LinuxContainerResources{CpuShares: 256, MemoryLimitInBytes: 524288000, OomScoreAdj: -997},
			cgroupV1,
		},
		{
			"requests & limits, cpu & memory, burstable qos",
			v1.ResourceList{v1.ResourceCPU: resource.MustParse("500m"), v1.ResourceMemory: resource.MustParse("750Mi")},
			v1.ResourceList{v1.ResourceCPU: resource.MustParse("250m"), v1.ResourceMemory: resource.MustParse("500Mi")},
			true,
			&runtimeapi.LinuxContainerResources{CpuShares: 256, MemoryLimitInBytes: 786432000, OomScoreAdj: 970},
			cgroupV1,
		},
		{
			"best-effort qos",
			nil,
			nil,
			true,
			&runtimeapi.LinuxContainerResources{CpuShares: 2, OomScoreAdj: 1000},
			cgroupV1,
		},
		{
			"requests & limits, cpu & memory, guaranteed qos",
			v1.ResourceList{v1.ResourceCPU: resource.MustParse("250m"), v1.ResourceMemory: resource.MustParse("500Mi")},
			v1.ResourceList{v1.ResourceCPU: resource.MustParse("250m"), v1.ResourceMemory: resource.MustParse("500Mi")},
			false,
			&runtimeapi.LinuxContainerResources{CpuShares: 256, MemoryLimitInBytes: 524288000, OomScoreAdj: -997, Unified: map[string]string{"memory.oom.group": "1"}},
			cgroupV2,
		},
		{
			"requests & limits, cpu & memory, burstable qos",
			v1.ResourceList{v1.ResourceCPU: resource.MustParse("500m"), v1.ResourceMemory: resource.MustParse("750Mi")},
			v1.ResourceList{v1.ResourceCPU: resource.MustParse("250m"), v1.ResourceMemory: resource.MustParse("500Mi")},
			false,
			&runtimeapi.LinuxContainerResources{CpuShares: 256, MemoryLimitInBytes: 786432000, OomScoreAdj: 970, Unified: map[string]string{"memory.oom.group": "1"}},
			cgroupV2,
		},
		{
			"best-effort qos",
			nil,
			nil,
			false,
			&runtimeapi.LinuxContainerResources{CpuShares: 2, OomScoreAdj: 1000, Unified: map[string]string{"memory.oom.group": "1"}},
			cgroupV2,
		},
	} {
		t.Run(fmt.Sprintf("cgroup%s:%s", tc.cgroupVersion, tc.name), func(t *testing.T) {
			setCgroupVersionDuringTest(tc.cgroupVersion)
			m.getSwapControllerAvailable = func() bool { return false }

			pod.Spec.Containers[0].Resources = v1.ResourceRequirements{Limits: tc.limits, Requests: tc.requests}

			m.singleProcessOOMKill = ptr.To(tc.singleProcessOOMKill)

			resources := m.generateLinuxContainerResources(tCtx, pod, &pod.Spec.Containers[0], false)
			tc.expected.HugepageLimits = resources.HugepageLimits
			assert.Equal(t, tc.expected, resources)
		})
	}
}

func TestGetContainerSwapBehavior(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "bar",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("1Gi")},
						Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse("2Gi")},
					},
				},
			},
		},
		Status: v1.PodStatus{},
	}
	tests := []struct {
		name                      string
		configuredMemorySwap      types.SwapBehavior
		isSwapControllerAvailable bool
		cgroupVersion             CgroupVersion
		isCriticalPod             bool
		qosClass                  v1.PodQOSClass
		containerResourceOverride func(container *v1.Container)
		expected                  types.SwapBehavior
	}{
		{
			name:                 "NoSwap, user set NoSwap behavior",
			configuredMemorySwap: types.NoSwap,
			expected:             types.NoSwap,
		},
		{
			name:                      "NoSwap, swap controller unavailable",
			configuredMemorySwap:      types.LimitedSwap,
			isSwapControllerAvailable: false,
			expected:                  types.NoSwap,
		},
		{
			name:                      "NoSwap, cgroup v1",
			configuredMemorySwap:      types.LimitedSwap,
			isSwapControllerAvailable: true,
			cgroupVersion:             cgroupV1,
			expected:                  types.NoSwap,
		},
		{
			name:                      "NoSwap, qos is Best-Effort",
			configuredMemorySwap:      types.LimitedSwap,
			isSwapControllerAvailable: true,
			cgroupVersion:             cgroupV2,
			qosClass:                  v1.PodQOSBestEffort,
			expected:                  types.NoSwap,
		},
		{
			name:                      "NoSwap, qos is Guaranteed",
			configuredMemorySwap:      types.LimitedSwap,
			isSwapControllerAvailable: true,
			cgroupVersion:             cgroupV2,
			qosClass:                  v1.PodQOSGuaranteed,
			expected:                  types.NoSwap,
		},
		{
			name:                      "NoSwap, zero memory",
			configuredMemorySwap:      types.LimitedSwap,
			isSwapControllerAvailable: true,
			cgroupVersion:             cgroupV2,
			qosClass:                  v1.PodQOSBurstable,
			containerResourceOverride: func(c *v1.Container) {
				c.Resources.Requests = v1.ResourceList{
					v1.ResourceMemory: resource.MustParse("0"),
				}
				c.Resources.Limits = v1.ResourceList{
					v1.ResourceMemory: resource.MustParse("0"),
				}
			},
			expected: types.NoSwap,
		},
		{
			name:                      "NoSwap, memory request equal to limit",
			configuredMemorySwap:      types.LimitedSwap,
			isSwapControllerAvailable: true,
			cgroupVersion:             cgroupV2,
			qosClass:                  v1.PodQOSBurstable,
			containerResourceOverride: func(c *v1.Container) {
				c.Resources.Requests = v1.ResourceList{
					v1.ResourceMemory: resource.MustParse("100Mi"),
				}
				c.Resources.Limits = v1.ResourceList{
					v1.ResourceMemory: resource.MustParse("100Mi"),
				}
			},
			expected: types.NoSwap,
		},
		{
			name:                      "LimitedSwap, cgroup v2",
			configuredMemorySwap:      types.LimitedSwap,
			isSwapControllerAvailable: true,
			cgroupVersion:             cgroupV2,
			qosClass:                  v1.PodQOSBurstable,
			expected:                  types.LimitedSwap,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m.memorySwapBehavior = string(tt.configuredMemorySwap)
			setCgroupVersionDuringTest(tt.cgroupVersion)
			m.getSwapControllerAvailable = func() bool { return tt.isSwapControllerAvailable }
			testpod := pod.DeepCopy()
			testpod.Status.QOSClass = tt.qosClass
			if tt.containerResourceOverride != nil {
				tt.containerResourceOverride(&testpod.Spec.Containers[0])
			}
			assert.Equal(t, tt.expected, m.GetContainerSwapBehavior(testpod, &testpod.Spec.Containers[0]))
		})
	}
}

func TestGenerateLinuxContainerResourcesWithSwap(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, _, m, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)
	m.machineInfo.MemoryCapacity = 42949672960 // 40Gb == 40 * 1024^3
	m.machineInfo.SwapCapacity = 5368709120    // 5Gb == 5 * 1024^3

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "bar",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: "c1",
				},
				{
					Name: "c2",
				},
			},
		},
		Status: v1.PodStatus{},
	}

	expectSwapDisabled := func(cgroupVersion CgroupVersion, resources ...*runtimeapi.LinuxContainerResources) {
		const msg = "container is expected to not have swap configured"

		for _, r := range resources {
			switch cgroupVersion {
			case cgroupV1:
				assert.Equal(t, int64(0), r.MemorySwapLimitInBytes, msg)
			case cgroupV2:
				assert.NotContains(t, r.Unified, cm.Cgroup2MaxSwapFilename, msg)
			}
		}
	}

	expectNoSwap := func(cgroupVersion CgroupVersion, resources ...*runtimeapi.LinuxContainerResources) {
		const msg = "container is expected to not have swap access"

		for _, r := range resources {
			switch cgroupVersion {
			case cgroupV1:
				assert.Equal(t, r.MemoryLimitInBytes, r.MemorySwapLimitInBytes, msg)
			case cgroupV2:
				assert.Equal(t, "0", r.Unified[cm.Cgroup2MaxSwapFilename], msg)
			}
		}
	}

	expectSwap := func(cgroupVersion CgroupVersion, swapBytesExpected int64, resources *runtimeapi.LinuxContainerResources) {
		msg := fmt.Sprintf("container swap is expected to be limited by %d bytes", swapBytesExpected)

		switch cgroupVersion {
		case cgroupV1:
			assert.Equal(t, resources.MemoryLimitInBytes+swapBytesExpected, resources.MemorySwapLimitInBytes, msg)
		case cgroupV2:
			assert.Equal(t, fmt.Sprintf("%d", swapBytesExpected), resources.Unified[cm.Cgroup2MaxSwapFilename], msg)
		}
	}

	calcSwapForBurstablePods := func(containerMemoryRequest int64) int64 {
		swapSize, err := calcSwapForBurstablePods(containerMemoryRequest, int64(m.machineInfo.MemoryCapacity), int64(m.machineInfo.SwapCapacity))
		assert.NoError(t, err)

		return swapSize
	}

	for _, tc := range []struct {
		name                        string
		cgroupVersion               CgroupVersion
		qosClass                    v1.PodQOSClass
		swapDisabledOnNode          bool
		swapBehavior                types.SwapBehavior
		addContainerWithoutRequests bool
		addGuaranteedContainer      bool
		isCriticalPod               bool
	}{
		// With cgroup v1
		{
			name:          "cgroups v1, LimitedSwap, Burstable QoS",
			cgroupVersion: cgroupV1,
			qosClass:      v1.PodQOSBurstable,
			swapBehavior:  types.LimitedSwap,
		},
		{
			name:          "cgroups v1, LimitedSwap, Best-effort QoS",
			cgroupVersion: cgroupV1,
			qosClass:      v1.PodQOSBestEffort,
			swapBehavior:  types.LimitedSwap,
		},

		// With no swapBehavior, NoSwap should be the default
		{
			name:          "With no swapBehavior - NoSwap should be the default",
			cgroupVersion: cgroupV2,
			qosClass:      v1.PodQOSBestEffort,
			swapBehavior:  "",
		},

		// With Guaranteed and Best-effort QoS
		{
			name:          "Best-effort QoS, cgroups v2, NoSwap",
			cgroupVersion: cgroupV2,
			qosClass:      v1.PodQOSBestEffort,
			swapBehavior:  "NoSwap",
		},
		{
			name:          "Best-effort QoS, cgroups v2, LimitedSwap",
			cgroupVersion: cgroupV2,
			qosClass:      v1.PodQOSBurstable,
			swapBehavior:  types.LimitedSwap,
		},
		{
			name:          "Guaranteed QoS, cgroups v2, LimitedSwap",
			cgroupVersion: cgroupV2,
			qosClass:      v1.PodQOSGuaranteed,
			swapBehavior:  types.LimitedSwap,
		},

		// With a "guaranteed" container (when memory requests equal to limits)
		{
			name:                        "Burstable QoS, cgroups v2, LimitedSwap, with a guaranteed container",
			cgroupVersion:               cgroupV2,
			qosClass:                    v1.PodQOSBurstable,
			swapBehavior:                types.LimitedSwap,
			addContainerWithoutRequests: false,
			addGuaranteedContainer:      true,
		},

		// Swap is expected to be allocated
		{
			name:                        "Burstable QoS, cgroups v2, LimitedSwap",
			cgroupVersion:               cgroupV2,
			qosClass:                    v1.PodQOSBurstable,
			swapBehavior:                types.LimitedSwap,
			addContainerWithoutRequests: false,
			addGuaranteedContainer:      false,
		},
		{
			name:                        "Burstable QoS, cgroups v2, LimitedSwap, with a container with no requests",
			cgroupVersion:               cgroupV2,
			qosClass:                    v1.PodQOSBurstable,
			swapBehavior:                types.LimitedSwap,
			addContainerWithoutRequests: true,
			addGuaranteedContainer:      false,
		},
		// All the above examples with Swap disabled on node
		{
			name:               "Swap disabled on node, cgroups v1, LimitedSwap, Burstable QoS",
			swapDisabledOnNode: true,
			cgroupVersion:      cgroupV1,
			qosClass:           v1.PodQOSBurstable,
			swapBehavior:       types.LimitedSwap,
		},
		{
			name:               "Swap disabled on node, cgroups v1, LimitedSwap, Best-effort QoS",
			swapDisabledOnNode: true,
			cgroupVersion:      cgroupV1,
			qosClass:           v1.PodQOSBestEffort,
			swapBehavior:       types.LimitedSwap,
		},

		// With no swapBehavior, NoSwap should be the default
		{
			name:               "Swap disabled on node, With no swapBehavior - NoSwap should be the default",
			swapDisabledOnNode: true,
			cgroupVersion:      cgroupV2,
			qosClass:           v1.PodQOSBestEffort,
			swapBehavior:       "",
		},

		// With Guaranteed and Best-effort QoS
		{
			name:               "Swap disabled on node, Best-effort QoS, cgroups v2, LimitedSwap",
			swapDisabledOnNode: true,
			cgroupVersion:      cgroupV2,
			qosClass:           v1.PodQOSBurstable,
			swapBehavior:       types.LimitedSwap,
		},
		{
			name:               "Swap disabled on node, Guaranteed QoS, cgroups v2, LimitedSwap",
			swapDisabledOnNode: true,
			cgroupVersion:      cgroupV2,
			qosClass:           v1.PodQOSGuaranteed,
			swapBehavior:       types.LimitedSwap,
		},

		// With a "guaranteed" container (when memory requests equal to limits)
		{
			name:                        "Swap disabled on node, Burstable QoS, cgroups v2, LimitedSwap, with a guaranteed container",
			swapDisabledOnNode:          true,
			cgroupVersion:               cgroupV2,
			qosClass:                    v1.PodQOSBurstable,
			swapBehavior:                types.LimitedSwap,
			addContainerWithoutRequests: false,
			addGuaranteedContainer:      true,
		},

		// Swap is expected to be allocated
		{
			name:                        "Swap disabled on node, Burstable QoS, cgroups v2, LimitedSwap",
			swapDisabledOnNode:          true,
			cgroupVersion:               cgroupV2,
			qosClass:                    v1.PodQOSBurstable,
			swapBehavior:                types.LimitedSwap,
			addContainerWithoutRequests: false,
			addGuaranteedContainer:      false,
		},
		{
			name:                        "Swap disabled on node, Burstable QoS, cgroups v2, LimitedSwap, with a container with no requests",
			swapDisabledOnNode:          true,
			cgroupVersion:               cgroupV2,
			qosClass:                    v1.PodQOSBurstable,
			swapBehavior:                types.LimitedSwap,
			addContainerWithoutRequests: true,
			addGuaranteedContainer:      false,
		},

		// When the pod is considered critical, disallow swap access
		{
			name:          "Best-effort QoS, cgroups v2, LimitedSwap, critical pod",
			cgroupVersion: cgroupV2,
			qosClass:      v1.PodQOSBurstable,
			swapBehavior:  types.LimitedSwap,
			isCriticalPod: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			setCgroupVersionDuringTest(tc.cgroupVersion)
			m.getSwapControllerAvailable = func() bool { return !tc.swapDisabledOnNode }
			m.memorySwapBehavior = string(tc.swapBehavior)

			var resourceReqsC1, resourceReqsC2 v1.ResourceRequirements
			switch tc.qosClass {
			case v1.PodQOSBurstable:
				resourceReqsC1 = v1.ResourceRequirements{
					Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("1Gi")},
				}

				if !tc.addContainerWithoutRequests {
					resourceReqsC2 = v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("2Gi")},
					}

					if tc.addGuaranteedContainer {
						resourceReqsC2.Limits = v1.ResourceList{v1.ResourceMemory: resource.MustParse("2Gi")}
					}
				}
			case v1.PodQOSGuaranteed:
				resourceReqsC1 = v1.ResourceRequirements{
					Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("1Gi"), v1.ResourceCPU: resource.MustParse("1")},
					Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse("1Gi"), v1.ResourceCPU: resource.MustParse("1")},
				}
				resourceReqsC2 = v1.ResourceRequirements{
					Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("2Gi"), v1.ResourceCPU: resource.MustParse("1")},
					Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse("2Gi"), v1.ResourceCPU: resource.MustParse("1")},
				}
			}
			pod.Spec.Containers[0].Resources = resourceReqsC1
			pod.Spec.Containers[1].Resources = resourceReqsC2

			if tc.isCriticalPod {
				pod.Spec.Priority = ptr.To(scheduling.SystemCriticalPriority)
				assert.True(t, types.IsCriticalPod(pod), "pod is expected to be critical")
			}

			resourcesC1 := m.generateLinuxContainerResources(tCtx, pod, &pod.Spec.Containers[0], false)
			resourcesC2 := m.generateLinuxContainerResources(tCtx, pod, &pod.Spec.Containers[1], false)

			if tc.swapDisabledOnNode {
				expectSwapDisabled(tc.cgroupVersion, resourcesC1, resourcesC2)
				return
			}

			if tc.isCriticalPod || tc.cgroupVersion == cgroupV1 || (tc.swapBehavior == types.LimitedSwap && tc.qosClass != v1.PodQOSBurstable) {
				expectNoSwap(tc.cgroupVersion, resourcesC1, resourcesC2)
				return
			}

			if tc.swapBehavior == types.NoSwap || tc.swapBehavior == "" {
				expectNoSwap(tc.cgroupVersion, resourcesC1, resourcesC2)
				return
			}

			c1ExpectedSwap := calcSwapForBurstablePods(resourceReqsC1.Requests.Memory().Value())
			c2ExpectedSwap := int64(0)
			if !tc.addContainerWithoutRequests && !tc.addGuaranteedContainer {
				c2ExpectedSwap = calcSwapForBurstablePods(resourceReqsC2.Requests.Memory().Value())
			}

			expectSwap(tc.cgroupVersion, c1ExpectedSwap, resourcesC1)
			expectSwap(tc.cgroupVersion, c2ExpectedSwap, resourcesC2)
		})
	}
}

func TestGenerateUpdatePodSandboxResourcesRequest(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)

	podRequestCPU := resource.MustParse("400m")
	podLimitCPU := resource.MustParse("800m")
	podRequestMemory := resource.MustParse("128Mi")
	podLimitMemory := resource.MustParse("256Mi")
	podOverheadCPU := resource.MustParse("100m")
	podOverheadMemory := resource.MustParse("64Mi")
	enforceCPULimits := true
	m.cpuCFSQuota = true

	for _, tc := range []struct {
		name             string
		qosClass         v1.PodQOSClass
		pod              *v1.Pod
		sandboxID        string
		enforceCPULimits bool
	}{
		// Best effort pod (no resources defined)
		{
			name:             "Best effort",
			qosClass:         v1.PodQOSBestEffort,
			enforceCPULimits: enforceCPULimits,
			sandboxID:        "1",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "foo",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{},
								Limits:   v1.ResourceList{},
							},
						},
					},
					Overhead: v1.ResourceList{},
				},
			},
		},
		{
			name:             "Best effort with overhead",
			qosClass:         v1.PodQOSBestEffort,
			enforceCPULimits: enforceCPULimits,
			sandboxID:        "2",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "foo",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{},
								Limits:   v1.ResourceList{},
							},
						},
					},
					Overhead: v1.ResourceList{
						v1.ResourceCPU:    podOverheadCPU,
						v1.ResourceMemory: podOverheadMemory,
					},
				},
			},
		},
		// Guaranteed pod
		{
			name:             "Guaranteed",
			qosClass:         v1.PodQOSGuaranteed,
			enforceCPULimits: enforceCPULimits,
			sandboxID:        "3",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "foo",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    podRequestCPU,
									v1.ResourceMemory: podLimitMemory,
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    podRequestCPU,
									v1.ResourceMemory: podLimitMemory,
								},
							},
						},
					},
				},
			},
		},
		{
			name:             "Guaranteed with overhead",
			qosClass:         v1.PodQOSGuaranteed,
			enforceCPULimits: enforceCPULimits,
			sandboxID:        "4",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "foo",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    podRequestCPU,
									v1.ResourceMemory: podLimitMemory,
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    podRequestCPU,
									v1.ResourceMemory: podLimitMemory,
								},
							},
						},
					},
					Overhead: v1.ResourceList{
						v1.ResourceCPU:    podOverheadCPU,
						v1.ResourceMemory: podOverheadMemory,
					},
				},
			},
		},
		// Burstable pods that leave some resources unspecified (e.g. only CPU/Mem requests)
		{
			name:             "Burstable only cpu",
			qosClass:         v1.PodQOSBurstable,
			enforceCPULimits: enforceCPULimits,
			sandboxID:        "5",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "foo",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU: podRequestCPU,
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU: podLimitCPU,
								},
							},
						},
					},
				},
			},
		},
		{
			name:             "Burstable only cpu with overhead",
			qosClass:         v1.PodQOSBurstable,
			enforceCPULimits: enforceCPULimits,
			sandboxID:        "6",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "foo",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU: podRequestCPU,
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU: podLimitCPU,
								},
							},
						},
					},
					Overhead: v1.ResourceList{
						v1.ResourceCPU: podOverheadCPU,
					},
				},
			},
		},
		{
			name:             "Burstable only memory",
			qosClass:         v1.PodQOSBurstable,
			enforceCPULimits: enforceCPULimits,
			sandboxID:        "7",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "foo",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceMemory: podRequestMemory,
								},
								Limits: v1.ResourceList{
									v1.ResourceMemory: podLimitMemory,
								},
							},
						},
					},
				},
			},
		},
		{
			name:             "Burstable only memory with overhead",
			qosClass:         v1.PodQOSBurstable,
			enforceCPULimits: enforceCPULimits,
			sandboxID:        "8",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "foo",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceMemory: podRequestMemory,
								},
								Limits: v1.ResourceList{
									v1.ResourceMemory: podLimitMemory,
								},
							},
						},
					},
					Overhead: v1.ResourceList{
						v1.ResourceMemory: podOverheadMemory,
					},
				},
			},
		},
		// With init container
		{
			name:             "Pod with init container",
			qosClass:         v1.PodQOSGuaranteed,
			enforceCPULimits: enforceCPULimits,
			sandboxID:        "9",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					InitContainers: []v1.Container{
						{
							Name: "init",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    podOverheadCPU,
									v1.ResourceMemory: podOverheadMemory,
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    podOverheadCPU,
									v1.ResourceMemory: podOverheadMemory,
								},
							},
						},
					},
					Containers: []v1.Container{
						{
							Name: "foo",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    podRequestCPU,
									v1.ResourceMemory: podLimitMemory,
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    podRequestCPU,
									v1.ResourceMemory: podLimitMemory,
								},
							},
						},
					},
				},
			},
		},
		{
			name:             "Pod with init container and overhead",
			qosClass:         v1.PodQOSGuaranteed,
			enforceCPULimits: enforceCPULimits,
			sandboxID:        "10",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					InitContainers: []v1.Container{
						{
							Name: "init",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    podOverheadCPU,
									v1.ResourceMemory: podOverheadMemory,
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    podOverheadCPU,
									v1.ResourceMemory: podOverheadMemory,
								},
							},
						},
					},
					Containers: []v1.Container{
						{
							Name: "foo",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    podRequestCPU,
									v1.ResourceMemory: podLimitMemory,
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    podRequestCPU,
									v1.ResourceMemory: podLimitMemory,
								},
							},
						},
					},
					Overhead: v1.ResourceList{
						v1.ResourceCPU:    podOverheadCPU,
						v1.ResourceMemory: podOverheadMemory,
					},
				},
			},
		},
		// With a sidecar container
		{
			name:             "Pod with sidecar container",
			qosClass:         v1.PodQOSBurstable,
			enforceCPULimits: enforceCPULimits,
			sandboxID:        "11",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "foo",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    podRequestCPU,
									v1.ResourceMemory: podRequestMemory,
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    podLimitCPU,
									v1.ResourceMemory: podLimitMemory,
								},
							},
						},
						{
							Name: "bar",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    podRequestCPU,
									v1.ResourceMemory: podRequestMemory,
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    podLimitCPU,
									v1.ResourceMemory: podLimitMemory,
								},
							},
						},
					},
				},
			},
		},
		{
			name:             "Pod with sidecar container and overhead",
			qosClass:         v1.PodQOSBurstable,
			enforceCPULimits: enforceCPULimits,
			sandboxID:        "11",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "foo",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    podRequestCPU,
									v1.ResourceMemory: podRequestMemory,
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    podLimitCPU,
									v1.ResourceMemory: podLimitMemory,
								},
							},
						},
						{
							Name: "bar",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    podRequestCPU,
									v1.ResourceMemory: podRequestMemory,
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    podLimitCPU,
									v1.ResourceMemory: podLimitMemory,
								},
							},
						},
					},
					Overhead: v1.ResourceList{
						v1.ResourceCPU:    podOverheadCPU,
						v1.ResourceMemory: podOverheadMemory,
					},
				},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			expectedLcr := m.calculateSandboxResources(tCtx, tc.pod)
			expectedLcrOverhead := m.convertOverheadToLinuxResources(tc.pod)

			podResourcesCfg := cm.ResourceConfigForPod(tc.pod, tc.enforceCPULimits, uint64((m.cpuCFSQuotaPeriod.Duration)/time.Microsecond), false)
			assert.NotNil(t, podResourcesCfg, "podResourcesCfg is expected to be not nil")

			if podResourcesCfg.CPUPeriod == nil {
				expectedLcr.CpuPeriod = 0
			}

			updatePodSandboxRequest := m.generateUpdatePodSandboxResourcesRequest("123", tc.pod, podResourcesCfg)
			assert.NotNil(t, updatePodSandboxRequest, "updatePodSandboxRequest is expected to be not nil")

			assert.Equal(t, expectedLcr, updatePodSandboxRequest.Resources, "expectedLcr need to be equal then updatePodSandboxRequest.Resources")
			assert.Equal(t, expectedLcrOverhead, updatePodSandboxRequest.Overhead, "expectedLcrOverhead need to be equal then updatePodSandboxRequest.Overhead")
		})
	}
}

func TestUpdatePodSandboxResources(t *testing.T) {
	tCtx := ktesting.Init(t)
	fakeRuntime, _, m, errCreate := createTestRuntimeManager(tCtx)
	require.NoError(t, errCreate)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "bar",
			Namespace: "new",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "foo",
					Image:           "busybox",
					ImagePullPolicy: v1.PullIfNotPresent,
				},
			},
		},
	}

	// Create fake sandbox and container
	fakeSandbox, fakeContainers := makeAndSetFakePod(t, m, fakeRuntime, pod)
	assert.Len(t, fakeContainers, 1)

	_, _, err := m.getPodContainerStatuses(tCtx, pod.UID, pod.Name, pod.Namespace, "")
	require.NoError(t, err)

	resourceConfig := &cm.ResourceConfig{}

	err = m.updatePodSandboxResources(tCtx, fakeSandbox.Id, pod, resourceConfig)
	require.NoError(t, err)

	// Verify sandbox is updated
	assert.Contains(t, fakeRuntime.Called, "UpdatePodSandboxResources")
}

type CgroupVersion string

const (
	cgroupV1 CgroupVersion = "v1"
	cgroupV2 CgroupVersion = "v2"
)

func setCgroupVersionDuringTest(version CgroupVersion) {
	isCgroup2UnifiedMode = func() bool {
		return version == cgroupV2
	}
}

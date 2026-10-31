//go:build linux

/*
Copyright The Kubernetes Authors.

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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	apitest "k8s.io/cri-api/pkg/apis/testing"
	"k8s.io/klog/v2"
	"k8s.io/ktesting"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	cmtesting "k8s.io/kubernetes/pkg/kubelet/cm/testing"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

// Fixed machine shape so LimitedSwap values are predictable:
// swap = request / memCapacity * swapCapacity = request / 8.
const (
	testMemCapacity  = uint64(8 << 30) // 8Gi
	testSwapCapacity = uint64(1 << 30) // 1Gi
)

func limitedSwapFor(request int64) int64 {
	v, err := calcSwapForBurstablePods(request, int64(testMemCapacity), int64(testSwapCapacity))
	if err != nil {
		panic(err)
	}
	return v
}

func newSwapTestManager(t *testing.T, tCtx context.Context, behavior kubetypes.SwapBehavior) *kubeGenericRuntimeManager {
	t.Helper()
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScaling, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WorkloadControlledSwap, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, true)
	_, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)
	m.machineInfo.MemoryCapacity = testMemCapacity
	m.machineInfo.SwapCapacity = testSwapCapacity
	m.memorySwapBehavior = string(behavior)
	m.getSwapControllerAvailable = func() bool { return true }
	setCgroupVersionDuringTest(cgroupV2)
	return m
}

func burstableContainer(name string, memRequest, memLimit string) v1.Container {
	c := v1.Container{Name: name, Image: "busybox", Resources: v1.ResourceRequirements{
		Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse(memRequest)},
	}}
	if memLimit != "" {
		c.Resources.Limits = v1.ResourceList{v1.ResourceMemory: resource.MustParse(memLimit)}
	}
	return c
}

func withSwapLimit(c v1.Container, swap string) v1.Container {
	if c.Resources.Limits == nil {
		c.Resources.Limits = v1.ResourceList{}
	}
	c.Resources.Limits[v1.ResourceSwap] = resource.MustParse(swap)
	return c
}

func TestEffectiveSwapLimit(t *testing.T) {
	tCtx := ktesting.Init(t)
	giQ := resource.MustParse("1Gi")
	mi := giQ.Value()

	for _, tc := range []struct {
		name           string
		behavior       kubetypes.SwapBehavior
		cgroupV1       bool
		noController   bool
		podSwap        string
		container      v1.Container
		expectedLimit  int64
		expectedSource swapLimitSource
	}{
		{
			name:     "NoSwap node ignores explicit limit",
			behavior: kubetypes.NoSwap, container: withSwapLimit(burstableContainer("c", "1Gi", "2Gi"), "512Mi"),
			expectedLimit: 0, expectedSource: swapSourceNone,
		},
		{
			name:     "cgroup v1 ignores explicit limit",
			behavior: kubetypes.WorkloadControlledSwap, cgroupV1: true, container: withSwapLimit(burstableContainer("c", "1Gi", "2Gi"), "512Mi"),
			expectedLimit: 0, expectedSource: swapSourceNone,
		},
		{
			name:     "no swap controller",
			behavior: kubetypes.WorkloadControlledSwap, noController: true, container: withSwapLimit(burstableContainer("c", "1Gi", "2Gi"), "512Mi"),
			expectedLimit: 0, expectedSource: swapSourceNone,
		},
		{
			name:     "WorkloadControlledSwap default is 0",
			behavior: kubetypes.WorkloadControlledSwap, container: burstableContainer("c", "1Gi", "2Gi"),
			expectedLimit: 0, expectedSource: swapSourceModeDefault,
		},
		{
			name:     "WorkloadControlledSwap explicit container limit",
			behavior: kubetypes.WorkloadControlledSwap, container: withSwapLimit(burstableContainer("c", "1Gi", "2Gi"), "512Mi"),
			expectedLimit: mi / 2, expectedSource: swapSourceContainer,
		},
		{
			name:     "WorkloadControlledSwap explicit container 0 disables",
			behavior: kubetypes.WorkloadControlledSwap, podSwap: "1Gi", container: withSwapLimit(burstableContainer("c", "1Gi", "2Gi"), "0"),
			expectedLimit: 0, expectedSource: swapSourceContainer,
		},
		{
			name:     "WorkloadControlledSwap inherits pod-level limit",
			behavior: kubetypes.WorkloadControlledSwap, podSwap: "1Gi", container: burstableContainer("c", "1Gi", "2Gi"),
			expectedLimit: mi, expectedSource: swapSourcePod,
		},
		{
			name:     "LimitedSwap Burstable recalculates from request",
			behavior: kubetypes.LimitedSwap, container: burstableContainer("c", "1Gi", "2Gi"),
			expectedLimit: limitedSwapFor(mi), expectedSource: swapSourceLimitedSwap,
		},
		{
			name:     "LimitedSwap request equals limit gets 0",
			behavior: kubetypes.LimitedSwap, container: burstableContainer("c", "2Gi", "2Gi"),
			expectedLimit: 0, expectedSource: swapSourceModeDefault,
		},
		{
			name:     "LimitedSwap explicit container limit wins",
			behavior: kubetypes.LimitedSwap, container: withSwapLimit(burstableContainer("c", "1Gi", "2Gi"), "512Mi"),
			expectedLimit: mi / 2, expectedSource: swapSourceContainer,
		},
		{
			name:     "LimitedSwap pod-level limit wins over recalculation",
			behavior: kubetypes.LimitedSwap, podSwap: "256Mi", container: burstableContainer("c", "1Gi", "2Gi"),
			expectedLimit: mi / 4, expectedSource: swapSourcePod,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := newSwapTestManager(t, tCtx, tc.behavior)
			if tc.cgroupV1 {
				setCgroupVersionDuringTest(cgroupV1)
			}
			if tc.noController {
				m.getSwapControllerAvailable = func() bool { return false }
			}
			// Burstable pod: a second container without limits keeps QoS Burstable
			// even when the container under test has request == limit.
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{UID: "uid", Name: "p", Namespace: "ns"},
				Spec:       v1.PodSpec{Containers: []v1.Container{tc.container, burstableContainer("other", "64Mi", "")}},
			}
			if tc.podSwap != "" {
				pod.Spec.Resources = &v1.ResourceRequirements{Limits: v1.ResourceList{v1.ResourceSwap: resource.MustParse(tc.podSwap)}}
			}
			limit, source := m.effectiveSwapLimit(pod, &pod.Spec.Containers[0])
			assert.Equal(t, tc.expectedLimit, limit, "limit")
			assert.Equal(t, tc.expectedSource, source, "source")
		})
	}
}

// TestComputePodActionsForSwapResize covers how swap deltas are detected:
// user changes, pod-level changes, LimitedSwap recalculation, and the
// invariant that a cpu-only change never queues a swap update.
func TestComputePodActionsForSwapResize(t *testing.T) {
	logger, tCtx := ktesting.NewTestContext(t)
	giQ := resource.MustParse("1Gi")
	gi := giQ.Value()

	type expectation struct {
		swapUpdates   int   // entries in ContainersToUpdate[swap]
		swapDesired   int64 // desired swapLimit on the first swap update
		swapCurrent   int64
		memoryUpdates int
		cpuUpdates    int
		podLevel      bool
	}
	for _, tc := range []struct {
		name          string
		behavior      kubetypes.SwapBehavior
		podLevelGate  bool
		allocated     func() *v1.Pod // what was actuated (checkpoint)
		desired       func(*v1.Pod)  // mutate into the desired spec
		expectation   expectation
		containerName string
	}{
		{
			name:     "swap-only increase",
			behavior: kubetypes.WorkloadControlledSwap,
			allocated: func() *v1.Pod {
				return swapTestPod(withSwapLimit(burstableContainer("c", "1Gi", "2Gi"), "256Mi"))
			},
			desired:     func(p *v1.Pod) { p.Spec.Containers[0].Resources.Limits[v1.ResourceSwap] = resource.MustParse("512Mi") },
			expectation: expectation{swapUpdates: 1, swapDesired: gi / 2, swapCurrent: gi / 4},
		},
		{
			name:     "swap-only decrease",
			behavior: kubetypes.WorkloadControlledSwap,
			allocated: func() *v1.Pod {
				return swapTestPod(withSwapLimit(burstableContainer("c", "1Gi", "2Gi"), "512Mi"))
			},
			desired:     func(p *v1.Pod) { p.Spec.Containers[0].Resources.Limits[v1.ResourceSwap] = resource.MustParse("0") },
			expectation: expectation{swapUpdates: 1, swapDesired: 0, swapCurrent: gi / 2},
		},
		{
			name:     "cpu-only change does not touch swap",
			behavior: kubetypes.WorkloadControlledSwap,
			allocated: func() *v1.Pod {
				return swapTestPod(withSwapLimit(burstableContainer("c", "1Gi", "2Gi"), "512Mi"))
			},
			desired:     func(p *v1.Pod) { p.Spec.Containers[0].Resources.Requests[v1.ResourceCPU] = resource.MustParse("200m") },
			expectation: expectation{cpuUpdates: 1},
		},
		{
			name:      "LimitedSwap memory request change surfaces recalculated swap delta",
			behavior:  kubetypes.LimitedSwap,
			allocated: func() *v1.Pod { return swapTestPod(burstableContainer("c", "1Gi", "4Gi")) },
			desired: func(p *v1.Pod) {
				p.Spec.Containers[0].Resources.Requests[v1.ResourceMemory] = resource.MustParse("2Gi")
			},
			expectation: expectation{
				memoryUpdates: 1,
				swapUpdates:   1, swapDesired: limitedSwapFor(2 * gi), swapCurrent: limitedSwapFor(gi),
			},
		},
		{
			name:     "LimitedSwap with explicit swap: memory request change leaves swap alone",
			behavior: kubetypes.LimitedSwap,
			allocated: func() *v1.Pod {
				return swapTestPod(withSwapLimit(burstableContainer("c", "1Gi", "4Gi"), "128Mi"))
			},
			desired: func(p *v1.Pod) {
				p.Spec.Containers[0].Resources.Requests[v1.ResourceMemory] = resource.MustParse("2Gi")
			},
			expectation: expectation{memoryUpdates: 1},
		},
		{
			name:         "pod-level swap change fans out to inheriting container",
			behavior:     kubetypes.WorkloadControlledSwap,
			podLevelGate: true,
			allocated: func() *v1.Pod {
				p := swapTestPod(burstableContainer("c", "1Gi", "2Gi"))
				p.Spec.Resources = &v1.ResourceRequirements{Limits: v1.ResourceList{v1.ResourceSwap: resource.MustParse("256Mi")}}
				return p
			},
			desired: func(p *v1.Pod) { p.Spec.Resources.Limits[v1.ResourceSwap] = resource.MustParse("1Gi") },
			// Both containers inherit the pod limit, so both get an update.
			expectation: expectation{swapUpdates: 2, swapDesired: gi, swapCurrent: gi / 4, podLevel: true},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := newSwapTestManager(t, tCtx, tc.behavior)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodLevelResourcesVerticalScaling, tc.podLevelGate)

			pod := tc.allocated()
			status := runningStatusFor(pod)
			m.InitializeActuatedPod(logger, pod)
			tc.desired(pod)

			actions := m.computePodActions(tCtx, pod, status, false)

			assert.Len(t, actions.ContainersToUpdate[v1.ResourceSwap], tc.expectation.swapUpdates, "swap updates")
			assert.Len(t, actions.ContainersToUpdate[v1.ResourceMemory], tc.expectation.memoryUpdates, "memory updates")
			assert.Len(t, actions.ContainersToUpdate[v1.ResourceCPU], tc.expectation.cpuUpdates, "cpu updates")
			assert.Equal(t, tc.expectation.podLevel, actions.UpdatePodLevelResources, "UpdatePodLevelResources")
			if tc.expectation.swapUpdates > 0 {
				u := actions.ContainersToUpdate[v1.ResourceSwap][0]
				assert.Equal(t, tc.expectation.swapDesired, u.desiredContainerResources.swapLimit, "desired swap")
				assert.Equal(t, tc.expectation.swapCurrent, u.currentContainerResources.swapLimit, "current swap")
			}
			assert.Empty(t, actions.ContainersToKill, "swap changes never restart")
		})
	}
}

// TestUpdatePodContainerResourcesPinsSwap is the regression test for the
// synthetic-container hazard: a cpu-only UpdateContainerResources must send
// the current swap limit, and a swap update must send the desired one.
func TestUpdatePodContainerResourcesPinsSwap(t *testing.T) {
	tCtx := ktesting.Init(t)
	giQ := resource.MustParse("1Gi")
	gi := giQ.Value()

	for _, tc := range []struct {
		name         string
		behavior     kubetypes.SwapBehavior
		resourceName v1.ResourceName
		current      resourceRequirements
		desired      resourceRequirements
		expectedSwap string
	}{
		{
			name: "WorkloadControlledSwap cpu update keeps current swap", behavior: kubetypes.WorkloadControlledSwap,
			resourceName: v1.ResourceCPU,
			current:      resourceRequirements{cpuRequest: 100, cpuLimit: 100, memoryRequest: gi, memoryLimit: 2 * gi, swapLimit: gi / 2},
			desired:      resourceRequirements{cpuRequest: 200, cpuLimit: 200, memoryRequest: gi, memoryLimit: 2 * gi, swapLimit: gi / 2},
			expectedSwap: fmt.Sprintf("%d", gi/2),
		},
		{
			name: "WorkloadControlledSwap memory update keeps current swap", behavior: kubetypes.WorkloadControlledSwap,
			resourceName: v1.ResourceMemory,
			current:      resourceRequirements{cpuRequest: 100, cpuLimit: 100, memoryRequest: gi, memoryLimit: 2 * gi, swapLimit: gi / 2},
			desired:      resourceRequirements{cpuRequest: 100, cpuLimit: 100, memoryRequest: gi, memoryLimit: 3 * gi, swapLimit: gi},
			expectedSwap: fmt.Sprintf("%d", gi/2),
		},
		{
			name: "WorkloadControlledSwap swap update sends desired swap", behavior: kubetypes.WorkloadControlledSwap,
			resourceName: v1.ResourceSwap,
			current:      resourceRequirements{cpuRequest: 100, cpuLimit: 100, memoryRequest: gi, memoryLimit: 2 * gi, swapLimit: gi / 2},
			desired:      resourceRequirements{cpuRequest: 100, cpuLimit: 100, memoryRequest: gi, memoryLimit: 2 * gi, swapLimit: gi},
			expectedSwap: fmt.Sprintf("%d", gi),
		},
		{
			// Without pinning, LimitedSwap would recompute from the synthetic
			// container's memory request (2Gi here) during the memory pass.
			name: "LimitedSwap memory update does not recompute swap", behavior: kubetypes.LimitedSwap,
			resourceName: v1.ResourceMemory,
			current:      resourceRequirements{cpuRequest: 100, cpuLimit: 100, memoryRequest: gi, memoryLimit: 4 * gi, swapLimit: limitedSwapFor(gi)},
			desired:      resourceRequirements{cpuRequest: 100, cpuLimit: 100, memoryRequest: 2 * gi, memoryLimit: 4 * gi, swapLimit: limitedSwapFor(2 * gi)},
			expectedSwap: fmt.Sprintf("%d", limitedSwapFor(gi)),
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := newSwapTestManager(t, tCtx, tc.behavior)
			runtimeSvc := m.runtimeService.(*apitest.FakeRuntimeService)

			pod := swapTestPod(burstableContainer("c", "1Gi", "2Gi"))
			makeAndSetFakePod(tCtx, m, runtimeSvc, pod)

			var cid kubecontainer.ContainerID
			for id, c := range runtimeSvc.Containers {
				if c.Metadata.Name == "c" {
					cid = kubecontainer.ContainerID{Type: "testRuntime", ID: id}
				}
			}
			require.NotEmpty(t, cid.ID)

			current := tc.current
			err := m.updatePodContainerResources(tCtx, pod, tc.resourceName, []containerToUpdateInfo{{
				container:                 &pod.Spec.Containers[0],
				containerType:             podutil.Containers,
				kubeContainerID:           cid,
				desiredContainerResources: tc.desired,
				currentContainerResources: &current,
			}})
			require.NoError(t, err)

			c := runtimeSvc.Containers[cid.ID]
			require.NotNil(t, c.LinuxResources)
			assert.Equal(t, tc.expectedSwap, c.LinuxResources.Unified[cm.Cgroup2MaxSwapFilename], "memory.swap.max sent to runtime")

			// The actuated checkpoint must carry the pinned value so effective
			// comparisons stay stable.
			actuated, found := m.actuatedState.GetContainerResources(pod.UID, "c")
			require.True(t, found)
			q := actuated.Limits[v1.ResourceSwap]
			assert.Equal(t, tc.expectedSwap, fmt.Sprintf("%d", q.Value()), "actuated limits.swap")
		})
	}
}

// TestValidateSwapResizeAction: swap decreases (container and pod) are
// deferred while usage is above the new limit; 0 with 0 usage is allowed.
func TestValidateSwapResizeAction(t *testing.T) {
	tCtx := ktesting.Init(t)
	miQ := resource.MustParse("1Mi")
	mi := miQ.Value()

	for _, tc := range []struct {
		name                   string
		currentSwap, wantSwap  int64
		containerSwapUsage     *uint64
		currentPodSwap         *int64 // nil: unset (max)
		desiredPodSwap         *int64
		podSwapUsage           *uint64
		podLevelResourcesResiz bool
		expectErr              bool
	}{
		{name: "increase never checked", currentSwap: 100 * mi, wantSwap: 200 * mi},
		{name: "decrease above usage ok", currentSwap: 200 * mi, wantSwap: 100 * mi, containerSwapUsage: new(uint64(50 * mi))},
		{name: "decrease below usage deferred", currentSwap: 200 * mi, wantSwap: 100 * mi, containerSwapUsage: new(uint64(150 * mi)), expectErr: true},
		{name: "decrease to 0 with 0 usage ok", currentSwap: 200 * mi, wantSwap: 0, containerSwapUsage: new(uint64(0))},
		{name: "decrease to 0 with usage deferred", currentSwap: 200 * mi, wantSwap: 0, containerSwapUsage: new(uint64(1)), expectErr: true},
		{name: "decrease with missing usage deferred", currentSwap: 200 * mi, wantSwap: 100 * mi, expectErr: true},
		{
			name: "pod-level swap added below usage deferred", currentSwap: 100 * mi, wantSwap: 100 * mi,
			containerSwapUsage: new(uint64(0)), podLevelResourcesResiz: true,
			desiredPodSwap: new(50 * mi), podSwapUsage: new(uint64(80 * mi)), expectErr: true,
		},
		{
			name: "pod-level swap decrease above usage ok", currentSwap: 100 * mi, wantSwap: 100 * mi,
			containerSwapUsage: new(uint64(0)), podLevelResourcesResiz: true,
			currentPodSwap: new(500 * mi), desiredPodSwap: new(200 * mi), podSwapUsage: new(uint64(80 * mi)),
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := newSwapTestManager(t, tCtx, kubetypes.WorkloadControlledSwap)
			pod := swapTestPod(withSwapLimit(burstableContainer("c", "1Gi", "2Gi"), "100Mi"))
			kps := runningStatusFor(pod)

			current := resourceRequirements{memoryRequest: 1, memoryLimit: 2, swapLimit: tc.currentSwap}
			desired := resourceRequirements{memoryRequest: 1, memoryLimit: 2, swapLimit: tc.wantSwap}
			actions := podActions{
				ContainersToUpdate:      map[v1.ResourceName][]containerToUpdateInfo{},
				UpdatePodLevelResources: tc.podLevelResourcesResiz,
			}
			if tc.currentSwap != tc.wantSwap {
				actions.ContainersToUpdate[v1.ResourceSwap] = []containerToUpdateInfo{{
					container: &pod.Spec.Containers[0], kubeContainerID: kps.ContainerStatuses[0].ID,
					desiredContainerResources: desired, currentContainerResources: &current,
				}}
			}

			podStats := &statsapi.PodStats{
				PodRef: statsapi.PodReference{Name: pod.Name, Namespace: pod.Namespace, UID: string(pod.UID)},
				Memory: &statsapi.MemoryStats{UsageBytes: new(uint64(1))},
				Containers: []statsapi.ContainerStats{{
					Name:   "c",
					Memory: &statsapi.MemoryStats{UsageBytes: new(uint64(1))},
				}},
			}
			if tc.containerSwapUsage != nil {
				podStats.Containers[0].Swap = &statsapi.SwapStats{SwapUsageBytes: tc.containerSwapUsage}
			}
			if tc.podSwapUsage != nil {
				podStats.Swap = &statsapi.SwapStats{SwapUsageBytes: tc.podSwapUsage}
			}
			m.runtimeHelper = &containertest.FakeRuntimeHelper{PodStats: map[types.UID]*statsapi.PodStats{pod.UID: podStats}}

			currentPod := &cm.ResourceConfig{}
			cm.SetSwapLimit(currentPod, tc.currentPodSwap)
			desiredPod := &cm.ResourceConfig{}
			if tc.desiredPodSwap != nil {
				cm.SetSwapLimit(desiredPod, tc.desiredPodSwap)
			}

			err := m.validatePodResizeAction(tCtx, pod, kps, currentPod, desiredPod, actions)
			if tc.expectErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
		})
	}
}

// TestIsPodResizeInProgressSwap: the in-progress condition must track swap.
func TestIsPodResizeInProgressSwap(t *testing.T) {
	logger, tCtx := ktesting.NewTestContext(t)
	m := newSwapTestManager(t, tCtx, kubetypes.WorkloadControlledSwap)

	pod := swapTestPod(withSwapLimit(burstableContainer("c", "1Gi", "2Gi"), "256Mi"))
	status := runningStatusFor(pod)
	m.InitializeActuatedPod(logger, pod)
	assert.False(t, m.IsPodResizeInProgress(pod, status), "allocated == actuated")

	pod.Spec.Containers[0].Resources.Limits[v1.ResourceSwap] = resource.MustParse("512Mi")
	assert.True(t, m.IsPodResizeInProgress(pod, status), "swap-only change must be in progress")

	// Simulate actuation through the resize path (pinned explicit value).
	actuated := pod.Spec.Containers[0].DeepCopy()
	require.NoError(t, m.setActuatedContainerResources(logger, pod, actuated, podutil.Containers))
	assert.False(t, m.IsPodResizeInProgress(pod, status), "actuated swap matches allocated")
}

// TestDoPodResizeActionSwapOrdering verifies the 2a (swap increases) -> 2b (memory with UnifiedWithoutSwap) -> 2c (swap decreases)
// ordering across pod-level cgroups and container CRI updates in doPodResizeAction.
func TestDoPodResizeActionSwapOrdering(t *testing.T) {
	tCtx := ktesting.Init(t)
	logger := tCtx.Logger()
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScaling, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodLevelResourcesVerticalScaling, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WorkloadControlledSwap, true)
	setCgroupVersionDuringTest(cgroupV2)

	mi := int64(1024 * 1024)

	t.Run("swap increase updates pod cgroup before container (2a)", func(t *testing.T) {
		fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
		require.NoError(t, err)
		m.machineInfo.MemoryCapacity = testMemCapacity
		m.machineInfo.SwapCapacity = testSwapCapacity
		m.memorySwapBehavior = string(kubetypes.WorkloadControlledSwap)
		m.getSwapControllerAvailable = func() bool { return true }

		var podCgroupCalls []*cm.ResourceConfig
		mockCM := cmtesting.NewMockContainerManager(t)
		mockCM.EXPECT().PodHasExclusiveCPUs(logger, mock.Anything).Return(false).Maybe()
		mockCM.EXPECT().ContainerHasExclusiveCPUs(logger, mock.Anything, mock.Anything).Return(false).Maybe()
		m.containerManager = mockCM
		mockPCM := cmtesting.NewMockPodContainerManager(t)
		mockCM.EXPECT().NewPodContainerManager().Return(mockPCM)
		mockPCM.EXPECT().GetPodCgroupConfig(mock.Anything, v1.ResourceMemory).Return(&cm.ResourceConfig{
			Memory: new(400 * mi),
		}, nil)
		mockPCM.EXPECT().GetPodCgroupConfig(mock.Anything, v1.ResourceCPU).Return(&cm.ResourceConfig{
			CPUShares: new(cm.MilliCPUToShares(100)),
			CPUQuota:  new(int64(-1)),
		}, nil)
		mockPCM.EXPECT().GetPodCgroupConfig(mock.Anything, v1.ResourceSwap).Return(&cm.ResourceConfig{
			Unified: map[string]string{cm.Cgroup2MaxSwapFilename: fmt.Sprintf("%d", 200*mi)},
		}, nil)
		mockPCM.EXPECT().SetPodCgroupConfig(logger, mock.Anything, mock.Anything).Run(func(_ klog.Logger, _ *v1.Pod, rc *cm.ResourceConfig) {
			// At the moment pod cgroup swap is raised, the container CRI call must not have happened yet.
			assert.NotContains(t, fakeRuntime.Called, "UpdateContainerResources")
			podCgroupCalls = append(podCgroupCalls, rc)
		}).Return(nil).Once()

		pod := swapTestPod(withSwapLimit(burstableContainer("c", "100Mi", "200Mi"), "200Mi"))
		pod.Spec.Resources = &v1.ResourceRequirements{
			Limits: v1.ResourceList{
				v1.ResourceMemory: resource.MustParse("400Mi"),
				v1.ResourceSwap:   resource.MustParse("400Mi"),
			},
		}
		kps := runningStatusFor(pod)
		_, fakeContainers := makeAndSetFakePod(tCtx, m, fakeRuntime, pod)
		for idx, fc := range fakeContainers {
			kps.ContainerStatuses[idx].ID = kubecontainer.ContainerID{Type: "testRuntime", ID: fc.Id}
		}
		fakeRuntime.Called = nil

		require.NoError(t, m.actuatedState.SetPodLevelResources(logger, pod.UID, &v1.ResourceRequirements{
			Limits: v1.ResourceList{
				v1.ResourceMemory: resource.MustParse("400Mi"),
				v1.ResourceSwap:   resource.MustParse("200Mi"),
			},
		}))

		actions := podActions{
			UpdatePodLevelResources: true,
			SandboxID:               "sandboxID",
			ContainersToUpdate: map[v1.ResourceName][]containerToUpdateInfo{
				v1.ResourceSwap: {{
					container:                 &pod.Spec.Containers[0],
					containerType:             podutil.Containers,
					kubeContainerID:           kps.ContainerStatuses[0].ID,
					currentContainerResources: &resourceRequirements{memoryRequest: 100 * mi, memoryLimit: 200 * mi, swapLimit: 100 * mi},
					desiredContainerResources: resourceRequirements{memoryRequest: 100 * mi, memoryLimit: 200 * mi, swapLimit: 200 * mi},
				}},
			},
		}

		res := m.doPodResizeAction(tCtx, pod, kps, actions)
		require.NoError(t, res.Error)
		require.Len(t, podCgroupCalls, 1)
		assert.Equal(t, fmt.Sprintf("%d", 400*mi), podCgroupCalls[0].Unified[cm.Cgroup2MaxSwapFilename])
		assert.Contains(t, fakeRuntime.Called, "UpdateContainerResources")

		actuatedPodRes, found := m.actuatedState.GetPodLevelResources(pod.UID)
		require.True(t, found)
		actuatedSwap := actuatedPodRes.Limits[v1.ResourceSwap]
		assert.Equal(t, int64(400*mi), actuatedSwap.Value())
	})

	t.Run("combined memory increase (2b) and swap decrease (2c) isolates UnifiedWithoutSwap and lowers container swap before pod cgroup", func(t *testing.T) {
		fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
		require.NoError(t, err)
		m.machineInfo.MemoryCapacity = testMemCapacity
		m.machineInfo.SwapCapacity = testSwapCapacity
		m.memorySwapBehavior = string(kubetypes.WorkloadControlledSwap)
		m.getSwapControllerAvailable = func() bool { return true }

		var podCgroupSteps []string
		mockCM := cmtesting.NewMockContainerManager(t)
		mockCM.EXPECT().PodHasExclusiveCPUs(logger, mock.Anything).Return(false).Maybe()
		mockCM.EXPECT().ContainerHasExclusiveCPUs(logger, mock.Anything, mock.Anything).Return(false).Maybe()
		m.containerManager = mockCM
		mockPCM := cmtesting.NewMockPodContainerManager(t)
		mockCM.EXPECT().NewPodContainerManager().Return(mockPCM)
		mockPCM.EXPECT().GetPodCgroupConfig(mock.Anything, v1.ResourceMemory).Return(&cm.ResourceConfig{
			Memory: new(400 * mi),
		}, nil)
		mockPCM.EXPECT().GetPodCgroupConfig(mock.Anything, v1.ResourceCPU).Return(&cm.ResourceConfig{
			CPUShares: new(cm.MilliCPUToShares(100)),
			CPUQuota:  new(int64(-1)),
		}, nil)
		mockPCM.EXPECT().GetPodCgroupConfig(mock.Anything, v1.ResourceSwap).Return(&cm.ResourceConfig{
			Unified: map[string]string{cm.Cgroup2MaxSwapFilename: fmt.Sprintf("%d", 400*mi)},
		}, nil)
		mockPCM.EXPECT().SetPodCgroupConfig(logger, mock.Anything, mock.Anything).Run(func(_ klog.Logger, _ *v1.Pod, rc *cm.ResourceConfig) {
			if rc.Memory != nil {
				// Pass 2b: memory cgroup update must strip memory.swap.max via UnifiedWithoutSwap.
				assert.NotContains(t, rc.Unified, cm.Cgroup2MaxSwapFilename)
				podCgroupSteps = append(podCgroupSteps, fmt.Sprintf("pod-mem:%d", *rc.Memory))
			}
			if v, ok := rc.Unified[cm.Cgroup2MaxSwapFilename]; ok {
				// Pass 2c: pod cgroup swap decrease must happen AFTER container UpdateContainerResources.
				assert.Contains(t, fakeRuntime.Called, "UpdateContainerResources")
				podCgroupSteps = append(podCgroupSteps, "pod-swap:"+v)
			}
		}).Return(nil).Times(2)

		pod := swapTestPod(withSwapLimit(burstableContainer("c", "100Mi", "300Mi"), "100Mi"))
		pod.Spec.Resources = &v1.ResourceRequirements{
			Limits: v1.ResourceList{
				v1.ResourceMemory: resource.MustParse("600Mi"),
				v1.ResourceSwap:   resource.MustParse("200Mi"),
			},
		}
		kps := runningStatusFor(pod)
		_, fakeContainers := makeAndSetFakePod(tCtx, m, fakeRuntime, pod)
		for idx, fc := range fakeContainers {
			kps.ContainerStatuses[idx].ID = kubecontainer.ContainerID{Type: "testRuntime", ID: fc.Id}
		}
		fakeRuntime.Called = nil

		m.runtimeHelper = &containertest.FakeRuntimeHelper{
			PodStats: map[types.UID]*statsapi.PodStats{
				pod.UID: {
					Swap: &statsapi.SwapStats{SwapUsageBytes: new(uint64(10 * mi))},
					Containers: []statsapi.ContainerStats{{
						Name:   "c",
						Memory: &statsapi.MemoryStats{UsageBytes: new(uint64(50 * mi))},
						Swap:   &statsapi.SwapStats{SwapUsageBytes: new(uint64(10 * mi))},
					}},
				},
			},
		}

		require.NoError(t, m.actuatedState.SetPodLevelResources(logger, pod.UID, &v1.ResourceRequirements{
			Limits: v1.ResourceList{
				v1.ResourceMemory: resource.MustParse("400Mi"),
				v1.ResourceSwap:   resource.MustParse("400Mi"),
			},
		}))

		cur := resourceRequirements{memoryRequest: 100 * mi, memoryLimit: 200 * mi, swapLimit: 200 * mi}
		des := resourceRequirements{memoryRequest: 100 * mi, memoryLimit: 300 * mi, swapLimit: 100 * mi}
		info := containerToUpdateInfo{
			container:                 &pod.Spec.Containers[0],
			containerType:             podutil.Containers,
			kubeContainerID:           kps.ContainerStatuses[0].ID,
			currentContainerResources: &cur,
			desiredContainerResources: des,
		}
		actions := podActions{
			UpdatePodLevelResources: true,
			SandboxID:               "sandboxID",
			ContainersToUpdate: map[v1.ResourceName][]containerToUpdateInfo{
				v1.ResourceMemory: {info},
				v1.ResourceSwap:   {info},
			},
		}

		res := m.doPodResizeAction(tCtx, pod, kps, actions)
		require.NoError(t, res.Error)
		assert.Equal(t, []string{
			fmt.Sprintf("pod-mem:%d", 600*mi),
			fmt.Sprintf("pod-swap:%d", 200*mi),
		}, podCgroupSteps)

		actuatedPodRes, found := m.actuatedState.GetPodLevelResources(pod.UID)
		require.True(t, found)
		actuatedMem := actuatedPodRes.Limits[v1.ResourceMemory]
		actuatedSwap := actuatedPodRes.Limits[v1.ResourceSwap]
		assert.Equal(t, int64(600*mi), actuatedMem.Value())
		assert.Equal(t, int64(200*mi), actuatedSwap.Value())
	})
}

// swapTestPod builds a Burstable pod with the given container plus a
// limit-less helper so QoS stays Burstable regardless of the first container.
func swapTestPod(c v1.Container) *v1.Pod {
	if c.Resources.Requests == nil {
		c.Resources.Requests = v1.ResourceList{}
	}
	if _, ok := c.Resources.Requests[v1.ResourceCPU]; !ok {
		c.Resources.Requests[v1.ResourceCPU] = resource.MustParse("100m")
	}
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: "12345678", Name: "foo", Namespace: "new"},
		Spec: v1.PodSpec{
			Containers: []v1.Container{c, burstableContainer("helper", "64Mi", "")},
		},
	}
}

func runningStatusFor(pod *v1.Pod) *kubecontainer.PodStatus {
	ps := &kubecontainer.PodStatus{
		ID: pod.UID, Name: pod.Name, Namespace: pod.Namespace,
		SandboxStatuses: []*runtimeapi.PodSandboxStatus{{
			Id:       "sandboxID",
			State:    runtimeapi.PodSandboxState_SANDBOX_READY,
			Metadata: &runtimeapi.PodSandboxMetadata{Name: pod.Name, Namespace: pod.Namespace, Uid: "sandboxuid", Attempt: uint32(0)},
			Network:  &runtimeapi.PodSandboxNetworkStatus{Ip: "10.0.0.1"},
		}},
	}
	for i := range pod.Spec.Containers {
		c := &pod.Spec.Containers[i]
		ps.ContainerStatuses = append(ps.ContainerStatuses, &kubecontainer.Status{
			ID:    kubecontainer.ContainerID{ID: c.Name + "-id"},
			Name:  c.Name,
			State: kubecontainer.ContainerStateRunning,
			Hash:  kubecontainer.HashContainer(c),
		})
	}
	return ps
}

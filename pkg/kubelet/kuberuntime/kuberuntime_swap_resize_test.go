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
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
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
	v, err := cm.CalcSwapForBurstablePods(request, int64(testMemCapacity), int64(testSwapCapacity))
	if err != nil {
		panic(err)
	}
	return v
}

func newSwapTestManager(t *testing.T, tCtx context.Context, behavior kubetypes.SwapBehavior) *kubeGenericRuntimeManager {
	t.Helper()
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

func podLevelMemory(request, limit string) *v1.ResourceRequirements {
	return &v1.ResourceRequirements{
		Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse(request)},
		Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse(limit)},
	}
}

func TestEffectiveSwapLimit(t *testing.T) {
	tCtx := ktesting.Init(t)
	giQ := resource.MustParse("1Gi")
	mi := giQ.Value()

	for _, tc := range []struct {
		name          string
		behavior      kubetypes.SwapBehavior
		cgroupV1      bool
		noController  bool
		podSwap       string
		podMemRequest string
		podMemLimit   string
		container     v1.Container
		expectedLimit int64
	}{
		{
			name:     "NoSwap node ignores explicit limit",
			behavior: kubetypes.NoSwap, container: withSwapLimit(burstableContainer("c", "1Gi", "2Gi"), "512Mi"),
			expectedLimit: 0,
		},
		{
			name:     "cgroup v1 ignores explicit limit",
			behavior: kubetypes.WorkloadControlledSwap, cgroupV1: true, container: withSwapLimit(burstableContainer("c", "1Gi", "2Gi"), "512Mi"),
			expectedLimit: 0,
		},
		{
			name:     "no swap controller",
			behavior: kubetypes.WorkloadControlledSwap, noController: true, container: withSwapLimit(burstableContainer("c", "1Gi", "2Gi"), "512Mi"),
			expectedLimit: 0,
		},
		{
			name:     "WorkloadControlledSwap default is 0",
			behavior: kubetypes.WorkloadControlledSwap, container: burstableContainer("c", "1Gi", "2Gi"),
			expectedLimit: 0,
		},
		{
			name:     "WorkloadControlledSwap explicit container limit",
			behavior: kubetypes.WorkloadControlledSwap, container: withSwapLimit(burstableContainer("c", "1Gi", "2Gi"), "512Mi"),
			expectedLimit: mi / 2,
		},
		{
			name:     "WorkloadControlledSwap explicit container 0 disables",
			behavior: kubetypes.WorkloadControlledSwap, podSwap: "1Gi", container: withSwapLimit(burstableContainer("c", "1Gi", "2Gi"), "0"),
			expectedLimit: 0,
		},
		{
			name:     "WorkloadControlledSwap inherits pod-level limit",
			behavior: kubetypes.WorkloadControlledSwap, podSwap: "1Gi", container: burstableContainer("c", "1Gi", "2Gi"),
			expectedLimit: mi,
		},
		{
			name:     "LimitedSwap Burstable recalculates from request",
			behavior: kubetypes.LimitedSwap, container: burstableContainer("c", "1Gi", "2Gi"),
			expectedLimit: limitedSwapFor(mi),
		},
		{
			name:     "LimitedSwap request equals limit gets 0",
			behavior: kubetypes.LimitedSwap, container: burstableContainer("c", "2Gi", "2Gi"),
			expectedLimit: 0,
		},
		{
			name:     "LimitedSwap explicit container limit wins",
			behavior: kubetypes.LimitedSwap, container: withSwapLimit(burstableContainer("c", "1Gi", "2Gi"), "512Mi"),
			expectedLimit: mi / 2,
		},
		{
			name:     "LimitedSwap pod-level limit wins over recalculation",
			behavior: kubetypes.LimitedSwap, podSwap: "256Mi", container: burstableContainer("c", "1Gi", "2Gi"),
			expectedLimit: mi / 4,
		},
		{
			name:     "LimitedSwap request equals inherited pod-level memory limit gets 0",
			behavior: kubetypes.LimitedSwap, podMemRequest: "1Gi", podMemLimit: "1Gi", container: burstableContainer("c", "1Gi", ""),
			expectedLimit: 0,
		},
		{
			name:     "LimitedSwap request below inherited pod-level memory limit recalculates from request",
			behavior: kubetypes.LimitedSwap, podMemRequest: "512Mi", podMemLimit: "1Gi", container: burstableContainer("c", "512Mi", ""),
			expectedLimit: limitedSwapFor(mi / 2),
		},
		{
			name:     "LimitedSwap container without a memory request under pod-level resources is unbounded",
			behavior: kubetypes.LimitedSwap, podMemRequest: "512Mi", podMemLimit: "1Gi", container: v1.Container{Name: "c", Image: "busybox"},
			expectedLimit: swapLimitMax,
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
			if tc.podMemLimit != "" {
				// The container under test owns the whole pod-level memory request.
				pod.Spec.Containers = pod.Spec.Containers[:1]
				pod.Spec.Resources = podLevelMemory(tc.podMemRequest, tc.podMemLimit)
			}
			assert.Equal(t, tc.expectedLimit, m.effectiveSwapLimit(pod, &pod.Spec.Containers[0]), "limit")
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
			name:     "pod-level swap change fans out to inheriting container",
			behavior: kubetypes.WorkloadControlledSwap,
			allocated: func() *v1.Pod {
				p := swapTestPod(burstableContainer("c", "1Gi", "2Gi"))
				p.Spec.Resources = &v1.ResourceRequirements{Limits: v1.ResourceList{v1.ResourceSwap: resource.MustParse("256Mi")}}
				return p
			},
			desired: func(p *v1.Pod) { p.Spec.Resources.Limits[v1.ResourceSwap] = resource.MustParse("1Gi") },
			// Both containers inherit the pod limit, so both get an update.
			expectation: expectation{swapUpdates: 2, swapDesired: gi, swapCurrent: gi / 4, podLevel: true},
		},
		{
			// The pod cgroup held the LimitedSwap value. Adding a pod-level limit
			// replaces it, and the containers move from their own LimitedSwap value
			// to the pod limit.
			name:     "adding pod-level limits.swap to a LimitedSwap pod with pod-level resources",
			behavior: kubetypes.LimitedSwap,
			allocated: func() *v1.Pod {
				p := swapTestPod(burstableContainer("c", "1Gi", "2Gi"))
				p.Spec.Resources = podLevelMemory("2Gi", "4Gi")
				return p
			},
			desired:     func(p *v1.Pod) { p.Spec.Resources.Limits[v1.ResourceSwap] = resource.MustParse("1Gi") },
			expectation: expectation{swapUpdates: 2, swapDesired: gi, swapCurrent: limitedSwapFor(gi), podLevel: true},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := newSwapTestManager(t, tCtx, tc.behavior)

			pod := tc.allocated()
			status := runningStatusFor(pod)
			m.InitializeActuatedPod(logger, pod, nil)
			tc.desired(pod)

			actions := m.computePodActions(tCtx, pod, status, false)

			assert.Len(t, actions.ContainersToUpdate[v1.ResourceSwap], tc.expectation.swapUpdates, "swap updates")
			assert.Len(t, actions.ContainersToUpdate[v1.ResourceMemory], tc.expectation.memoryUpdates, "memory updates")
			assert.Len(t, actions.ContainersToUpdate[v1.ResourceCPU], tc.expectation.cpuUpdates, "cpu updates")
			assert.Equal(t, tc.expectation.podLevel, actions.PodCgroupResourcesToUpdate.Has(v1.ResourceSwap), "pod-level swap update")
			if tc.expectation.swapUpdates > 0 {
				u := actions.ContainersToUpdate[v1.ResourceSwap][0]
				assert.Equal(t, tc.expectation.swapDesired, u.desiredContainerResources.swapLimit, "desired swap")
				assert.Equal(t, tc.expectation.swapCurrent, u.currentContainerResources.swapLimit, "current swap")
			}
			assert.Empty(t, actions.ContainersToKill, "swap changes never restart")
		})
	}
}

// TestLimitedSwapInheritedMemoryLimitDoesNotResize is the regression test for
// a container that runs under the pod-level memory limit. The actuated record
// stores that inherited limit while the spec has none, and LimitedSwap
// eligibility has to give the same answer for both. Otherwise a freshly
// started container looks like it needs a swap resize on every sync.
func TestLimitedSwapInheritedMemoryLimitDoesNotResize(t *testing.T) {
	logger, tCtx := ktesting.NewTestContext(t)
	giQ := resource.MustParse("1Gi")
	gi := giQ.Value()

	for _, tc := range []struct {
		name          string
		container     v1.Container
		podMemRequest string
		podMemLimit   string
		expectedSwap  int64
	}{
		{
			name:      "request equals pod-level limit",
			container: burstableContainer("c", "1Gi", ""), podMemRequest: "1Gi", podMemLimit: "1Gi",
			expectedSwap: 0,
		},
		{
			name:      "request below pod-level limit",
			container: burstableContainer("c", "512Mi", ""), podMemRequest: "512Mi", podMemLimit: "1Gi",
			expectedSwap: limitedSwapFor(gi / 2),
		},
		{
			name:      "no memory request",
			container: v1.Container{Name: "c", Image: "busybox"}, podMemRequest: "512Mi", podMemLimit: "1Gi",
			expectedSwap: swapLimitMax,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := newSwapTestManager(t, tCtx, kubetypes.LimitedSwap)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodLevelResourcesVerticalScaling, true)

			pod := swapTestPod(tc.container)
			pod.Spec.Containers = pod.Spec.Containers[:1]
			pod.Spec.Resources = podLevelMemory(tc.podMemRequest, tc.podMemLimit)
			status := runningStatusFor(pod)
			m.InitializeActuatedPod(logger, pod, nil)
			// What starting the container records.
			require.NoError(t, m.setActuatedContainer(logger, pod, &pod.Spec.Containers[0], podutil.Containers))

			assert.Equal(t, tc.expectedSwap, m.effectiveSwapLimit(pod, &pod.Spec.Containers[0]), "swap limit at creation")
			actions := m.computePodActions(tCtx, pod, status, false)
			assert.Empty(t, actions.ContainersToUpdate, "no update without a resize request")
			assert.False(t, m.IsPodResizeInProgress(pod, status), "resize in progress")
		})
	}
}

// TestLimitedSwapUnchangedWithoutSwapLimit: for pods that set no limits.swap,
// a LimitedSwap node reports the same swap behavior and writes the same
// memory.swap.max with the WorkloadControlledSwap gate on and off.
func TestLimitedSwapUnchangedWithoutSwapLimit(t *testing.T) {
	tCtx := ktesting.Init(t)
	giQ := resource.MustParse("1Gi")
	gi := giQ.Value()
	swapFor := func(request int64) string { return strconv.FormatInt(limitedSwapFor(request), 10) }

	type expectation struct {
		behavior kubetypes.SwapBehavior
		swapMax  string
	}
	for _, tc := range []struct {
		name       string
		podMemory  *v1.ResourceRequirements
		containers []v1.Container
		expected   []expectation
	}{
		{
			name:       "container-level resources only",
			containers: []v1.Container{burstableContainer("c1", "1Gi", "2Gi"), burstableContainer("c2", "1Gi", "1Gi")},
			expected:   []expectation{{kubetypes.LimitedSwap, swapFor(gi)}, {kubetypes.NoSwap, "0"}},
		},
		{
			name:       "pod-level request below limit",
			podMemory:  podLevelMemory("1Gi", "2Gi"),
			containers: []v1.Container{burstableContainer("c1", "512Mi", ""), {Name: "c2", Image: "busybox"}},
			expected:   []expectation{{kubetypes.LimitedSwap, swapFor(gi / 2)}, {kubetypes.LimitedSwap, cm.Cgroup2MaxSwapLimit}},
		},
		{
			name:       "pod-level request equals limit",
			podMemory:  podLevelMemory("1Gi", "1Gi"),
			containers: []v1.Container{burstableContainer("c1", "512Mi", ""), {Name: "c2", Image: "busybox"}},
			expected:   []expectation{{kubetypes.NoSwap, "0"}, {kubetypes.NoSwap, "0"}},
		},
	} {
		for _, gate := range []bool{true, false} {
			t.Run(fmt.Sprintf("%s, gate %t", tc.name, gate), func(t *testing.T) {
				m := newSwapTestManager(t, tCtx, kubetypes.LimitedSwap)
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WorkloadControlledSwap, gate)

				pod := swapTestPod(tc.containers[0])
				// A CPU request on the second container keeps the pod Burstable in every case.
				pod.Spec.Containers = append(pod.Spec.Containers[:1], tc.containers[1:]...)
				pod.Spec.Resources = tc.podMemory
				for i := range pod.Spec.Containers {
					container := &pod.Spec.Containers[i]
					assert.Equal(t, tc.expected[i].behavior, m.GetContainerSwapBehavior(pod, container), "behavior of %s", container.Name)
					resources := m.generateLinuxContainerResources(tCtx, pod, container, false)
					assert.Equal(t, tc.expected[i].swapMax, resources.Unified[cm.Cgroup2MaxSwapFilename], "memory.swap.max of %s", container.Name)
				}
			})
		}
	}
}

// TestGetContainerSwapBehaviorWorkloadControlledNode: on a
// WorkloadControlledSwap node only containers with a swap limit set at
// container or pod level count as swap-enabled. The others get
// memory.swap.max=0 and must report NoSwap, or resize admission would reject
// adding or removing them.
func TestGetContainerSwapBehaviorWorkloadControlledNode(t *testing.T) {
	tCtx := ktesting.Init(t)

	for _, tc := range []struct {
		name         string
		gateOff      bool
		container    v1.Container
		podSwap      string
		nilContainer bool
		expected     kubetypes.SwapBehavior
	}{
		{
			name:      "no swap limit",
			container: burstableContainer("c", "1Gi", "2Gi"),
			expected:  kubetypes.NoSwap,
		},
		{
			name:      "container swap limit",
			container: withSwapLimit(burstableContainer("c", "1Gi", "2Gi"), "512Mi"),
			expected:  kubetypes.WorkloadControlledSwap,
		},
		{
			name:      "pod-level swap limit",
			container: burstableContainer("c", "1Gi", "2Gi"), podSwap: "1Gi",
			expected: kubetypes.WorkloadControlledSwap,
		},
		{
			name:      "node-level query",
			container: burstableContainer("c", "1Gi", "2Gi"), nilContainer: true,
			expected: kubetypes.WorkloadControlledSwap,
		},
		{
			name:      "gate off",
			container: withSwapLimit(burstableContainer("c", "1Gi", "2Gi"), "512Mi"), gateOff: true,
			expected: kubetypes.NoSwap,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := newSwapTestManager(t, tCtx, kubetypes.WorkloadControlledSwap)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WorkloadControlledSwap, !tc.gateOff)

			pod := swapTestPod(tc.container)
			if tc.podSwap != "" {
				pod.Spec.Resources = &v1.ResourceRequirements{Limits: v1.ResourceList{v1.ResourceSwap: resource.MustParse(tc.podSwap)}}
			}
			container := &pod.Spec.Containers[0]
			if tc.nilContainer {
				container = nil
			}
			assert.Equal(t, tc.expected, m.GetContainerSwapBehavior(pod, container))
		})
	}
}

func drainEvents(recorder *record.FakeRecorder) []string {
	var events []string
	for {
		select {
		case e := <-recorder.Events:
			events = append(events, e)
		default:
			return events
		}
	}
}

// TestSwapLimitAbovePodLimitEvent: on a LimitedSwap node, a pod with pod-level
// resources gets a calculated swap limit on its pod cgroup. A container
// limits.swap above it is reported with SwapLimitNotEnforced when the
// container is created.
func TestSwapLimitAbovePodLimitEvent(t *testing.T) {
	tCtx := ktesting.Init(t)
	guaranteed := &v1.ResourceRequirements{
		Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("1"), v1.ResourceMemory: resource.MustParse("2Gi")},
		Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse("1"), v1.ResourceMemory: resource.MustParse("2Gi")},
	}
	// 2Gi of the 8Gi test node memory, of 1Gi swap: 256Mi for the pod.
	burstable := podLevelMemory("2Gi", "4Gi")
	burstableWithPodSwap := podLevelMemory("2Gi", "4Gi")
	burstableWithPodSwap.Limits[v1.ResourceSwap] = resource.MustParse("1Gi")

	for _, tc := range []struct {
		name          string
		behavior      kubetypes.SwapBehavior
		gateOff       bool
		podResources  *v1.ResourceRequirements
		containerSwap string
		expectEvent   string
	}{
		{
			name:     "pod not eligible for LimitedSwap",
			behavior: kubetypes.LimitedSwap, podResources: guaranteed, containerSwap: "512Mi",
			expectEvent: "resources.limits.swap (512Mi) exceeds the pod swap limit (0)",
		},
		{
			name:     "above the calculated pod limit",
			behavior: kubetypes.LimitedSwap, podResources: burstable, containerSwap: "512Mi",
			expectEvent: "resources.limits.swap (512Mi) exceeds the pod swap limit (256Mi)",
		},
		{
			name:     "within the calculated pod limit",
			behavior: kubetypes.LimitedSwap, podResources: burstable, containerSwap: "256Mi",
		},
		{
			name:     "pod-level limits.swap set",
			behavior: kubetypes.LimitedSwap, podResources: burstableWithPodSwap, containerSwap: "512Mi",
		},
		{
			name:     "no pod-level resources",
			behavior: kubetypes.LimitedSwap, containerSwap: "512Mi",
		},
		{
			name:     "WorkloadControlledSwap node",
			behavior: kubetypes.WorkloadControlledSwap, podResources: guaranteed, containerSwap: "512Mi",
		},
		{
			name:     "gate off",
			behavior: kubetypes.LimitedSwap, gateOff: true, podResources: guaranteed, containerSwap: "512Mi",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := newSwapTestManager(t, tCtx, tc.behavior)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WorkloadControlledSwap, !tc.gateOff)
			recorder := record.NewFakeRecorder(10)
			m.recorder = recorder

			pod := swapTestPod(withSwapLimit(burstableContainer("c", "1Gi", "2Gi"), tc.containerSwap))
			pod.Spec.Containers = pod.Spec.Containers[:1]
			pod.Spec.Resources = tc.podResources

			// Container creation passes the container from the pod spec.
			m.generateLinuxContainerResources(tCtx, pod, &pod.Spec.Containers[0], false)
			events := drainEvents(recorder)
			if tc.expectEvent == "" {
				assert.Empty(t, events)
			} else {
				require.Len(t, events, 1)
				assert.Contains(t, events[0], "SwapLimitNotEnforced")
				assert.Contains(t, events[0], tc.expectEvent)
			}

			// cpu and memory resize passes hand over a copy and must not repeat it.
			m.generateLinuxContainerResources(tCtx, pod, pod.Spec.Containers[0].DeepCopy(), false)
			assert.Empty(t, drainEvents(recorder), "events for a synthetic container")
		})
	}
}

// TestSwapResizeAbovePodLimitEvent: a swap resize that leaves a container
// limits.swap above the calculated pod limit reports it too.
func TestSwapResizeAbovePodLimitEvent(t *testing.T) {
	tCtx := ktesting.Init(t)
	giQ := resource.MustParse("1Gi")
	gi := giQ.Value()

	m := newSwapTestManager(t, tCtx, kubetypes.LimitedSwap)
	runtimeSvc := m.runtimeService.(*apitest.FakeRuntimeService)
	pod := swapTestPod(withSwapLimit(burstableContainer("c", "1Gi", "2Gi"), "512Mi"))
	pod.Spec.Containers = pod.Spec.Containers[:1]
	pod.Spec.Resources = podLevelMemory("2Gi", "4Gi")
	makeAndSetFakePod(tCtx, m, runtimeSvc, pod)

	var cid kubecontainer.ContainerID
	for id, c := range runtimeSvc.Containers {
		if c.Metadata.Name == "c" {
			cid = kubecontainer.ContainerID{Type: "testRuntime", ID: id}
		}
	}
	require.NotEmpty(t, cid.ID)
	recorder := record.NewFakeRecorder(10)
	m.recorder = recorder

	current := resourceRequirements{cpuRequest: 100, cpuLimit: 100, memoryRequest: gi, memoryLimit: 2 * gi, swapLimit: gi / 8}
	desired := resourceRequirements{cpuRequest: 100, cpuLimit: 100, memoryRequest: gi, memoryLimit: 2 * gi, swapLimit: gi / 2}
	update := []containerToUpdateInfo{{
		container:                 &pod.Spec.Containers[0],
		containerType:             podutil.Containers,
		kubeContainerID:           cid,
		desiredContainerResources: desired,
		currentContainerResources: &current,
	}}

	require.NoError(t, m.updatePodContainerResources(tCtx, pod, v1.ResourceMemory, update))
	assert.Empty(t, drainEvents(recorder), "events for the memory pass")

	require.NoError(t, m.updatePodContainerResources(tCtx, pod, v1.ResourceSwap, update))
	events := drainEvents(recorder)
	require.Len(t, events, 1)
	assert.Contains(t, events[0], "SwapLimitNotEnforced")
	assert.Contains(t, events[0], "resources.limits.swap (512Mi) exceeds the pod swap limit (256Mi)")
}

// TestUpdatePodContainerResourcesPinsSwap is the regression test for the
// synthetic-container hazard: a cpu-only UpdateContainerResources must send
// the current swap limit, and a swap update must send the desired one.
func TestUpdatePodContainerResourcesPinsSwap(t *testing.T) {
	tCtx := ktesting.Init(t)
	logger := tCtx.Logger()
	giQ := resource.MustParse("1Gi")
	gi := giQ.Value()

	for _, tc := range []struct {
		name         string
		behavior     kubetypes.SwapBehavior
		declaredSwap string // container limits.swap in the spec; empty means inherited
		addsSwap     bool   // the resize adds declaredSwap: the container started without it
		resourceName v1.ResourceName
		current      resourceRequirements
		desired      resourceRequirements
		expectedSwap string
	}{
		{
			name: "WorkloadControlledSwap cpu update keeps current swap", behavior: kubetypes.WorkloadControlledSwap,
			declaredSwap: "512Mi",
			resourceName: v1.ResourceCPU,
			current:      resourceRequirements{cpuRequest: 100, cpuLimit: 100, memoryRequest: gi, memoryLimit: 2 * gi, swapLimit: gi / 2},
			desired:      resourceRequirements{cpuRequest: 200, cpuLimit: 200, memoryRequest: gi, memoryLimit: 2 * gi, swapLimit: gi / 2},
			expectedSwap: fmt.Sprintf("%d", gi/2),
		},
		{
			name: "WorkloadControlledSwap memory update keeps current swap", behavior: kubetypes.WorkloadControlledSwap,
			declaredSwap: "512Mi",
			resourceName: v1.ResourceMemory,
			current:      resourceRequirements{cpuRequest: 100, cpuLimit: 100, memoryRequest: gi, memoryLimit: 2 * gi, swapLimit: gi / 2},
			desired:      resourceRequirements{cpuRequest: 100, cpuLimit: 100, memoryRequest: gi, memoryLimit: 3 * gi, swapLimit: gi},
			expectedSwap: fmt.Sprintf("%d", gi/2),
		},
		{
			name: "WorkloadControlledSwap swap update sends desired swap", behavior: kubetypes.WorkloadControlledSwap,
			declaredSwap: "1Gi",
			resourceName: v1.ResourceSwap,
			current:      resourceRequirements{cpuRequest: 100, cpuLimit: 100, memoryRequest: gi, memoryLimit: 2 * gi, swapLimit: gi / 2},
			desired:      resourceRequirements{cpuRequest: 100, cpuLimit: 100, memoryRequest: gi, memoryLimit: 2 * gi, swapLimit: gi},
			expectedSwap: fmt.Sprintf("%d", gi),
		},
		{
			// The container inherits the pod-level limit: CRI gets the pinned
			// value, but the checkpoint must not gain an explicit limits.swap.
			name: "WorkloadControlledSwap cpu update of inheriting container", behavior: kubetypes.WorkloadControlledSwap,
			resourceName: v1.ResourceCPU,
			current:      resourceRequirements{cpuRequest: 100, cpuLimit: 100, memoryRequest: gi, memoryLimit: 2 * gi, swapLimit: gi / 4},
			desired:      resourceRequirements{cpuRequest: 200, cpuLimit: 200, memoryRequest: gi, memoryLimit: 2 * gi, swapLimit: gi / 4},
			expectedSwap: fmt.Sprintf("%d", gi/4),
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
		{
			// The resize adds limits.swap and changes memory. The memory pass runs
			// before the swap decrease and must not record the inherited value.
			name: "LimitedSwap memory update of a resize that adds limits.swap", behavior: kubetypes.LimitedSwap,
			declaredSwap: "64Mi", addsSwap: true,
			resourceName: v1.ResourceMemory,
			current:      resourceRequirements{cpuRequest: 100, cpuLimit: 100, memoryRequest: gi, memoryLimit: 4 * gi, swapLimit: limitedSwapFor(gi)},
			desired:      resourceRequirements{cpuRequest: 100, cpuLimit: 100, memoryRequest: gi, memoryLimit: 2 * gi, swapLimit: gi / 16},
			expectedSwap: fmt.Sprintf("%d", limitedSwapFor(gi)),
		},
		{
			name: "LimitedSwap memory update of a resize that adds limits.swap to an unbounded container", behavior: kubetypes.LimitedSwap,
			declaredSwap: "64Mi", addsSwap: true,
			resourceName: v1.ResourceMemory,
			current:      resourceRequirements{cpuRequest: 100, cpuLimit: 100, memoryRequest: gi, memoryLimit: 4 * gi, swapLimit: swapLimitMax},
			desired:      resourceRequirements{cpuRequest: 100, cpuLimit: 100, memoryRequest: gi, memoryLimit: 2 * gi, swapLimit: gi / 16},
			expectedSwap: cm.Cgroup2MaxSwapLimit,
		},
		{
			name: "LimitedSwap swap update of a resize that adds limits.swap", behavior: kubetypes.LimitedSwap,
			declaredSwap: "64Mi", addsSwap: true,
			resourceName: v1.ResourceSwap,
			current:      resourceRequirements{cpuRequest: 100, cpuLimit: 100, memoryRequest: gi, memoryLimit: 2 * gi, swapLimit: limitedSwapFor(gi)},
			desired:      resourceRequirements{cpuRequest: 100, cpuLimit: 100, memoryRequest: gi, memoryLimit: 2 * gi, swapLimit: gi / 16},
			expectedSwap: fmt.Sprintf("%d", gi/16),
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := newSwapTestManager(t, tCtx, tc.behavior)
			runtimeSvc := m.runtimeService.(*apitest.FakeRuntimeService)

			startedContainer := burstableContainer("c", "1Gi", "2Gi")
			specContainer := burstableContainer("c", "1Gi", "2Gi")
			if tc.declaredSwap != "" {
				specContainer = withSwapLimit(specContainer, tc.declaredSwap)
				if !tc.addsSwap {
					startedContainer = withSwapLimit(startedContainer, tc.declaredSwap)
				}
			}
			pod := swapTestPod(specContainer)
			makeAndSetFakePod(tCtx, m, runtimeSvc, pod)
			// What starting the container recorded.
			require.NoError(t, m.setActuatedContainer(logger, pod, &startedContainer, podutil.Containers))

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

			// The checkpoint records limits.swap only when the spec declares it and
			// the swap pass has applied it. An inherited value is never recorded.
			actuated, found := m.actuatedState.GetContainerResources(pod.UID, "c")
			require.True(t, found)
			q, hasSwap := actuated.Limits[v1.ResourceSwap]
			if tc.declaredSwap == "" || (tc.addsSwap && tc.resourceName != v1.ResourceSwap) {
				assert.False(t, hasSwap, "actuated limits.swap must not be recorded yet, got %s", q.String())
			} else {
				require.True(t, hasSwap, "actuated limits.swap")
				assert.Equal(t, tc.expectedSwap, fmt.Sprintf("%d", q.Value()), "actuated limits.swap")
			}
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
				ContainersToUpdate: map[v1.ResourceName][]containerToUpdateInfo{},
			}
			if tc.podLevelResourcesResiz {
				actions.PodCgroupResourcesToUpdate = sets.New(v1.ResourceSwap)
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

// TestPodLevelResizeUsageCheckFollowsSwapLimits: a pod-level swap limit update
// runs the usage check only where workload swap limits are honored. Elsewhere
// it is not checked, as before KEP-5359.
func TestPodLevelResizeUsageCheckFollowsSwapLimits(t *testing.T) {
	tCtx := ktesting.Init(t)
	miQ := resource.MustParse("1Mi")
	mi := miQ.Value()

	for _, tc := range []struct {
		name      string
		gate      bool
		behavior  kubetypes.SwapBehavior
		expectErr string
	}{
		{name: "gate off", gate: false, behavior: kubetypes.LimitedSwap},
		{name: "gate on, NoSwap node", gate: true, behavior: kubetypes.NoSwap},
		{name: "gate on, WorkloadControlledSwap node", gate: true, behavior: kubetypes.WorkloadControlledSwap, expectErr: "cannot decrease memory limits"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := newSwapTestManager(t, tCtx, tc.behavior)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WorkloadControlledSwap, tc.gate)
			pod := swapTestPod(burstableContainer("c", "1Gi", "2Gi"))
			kps := runningStatusFor(pod)

			// Pod memory usage is above the new pod-level memory limit.
			m.runtimeHelper = &containertest.FakeRuntimeHelper{PodStats: map[types.UID]*statsapi.PodStats{pod.UID: {
				PodRef: statsapi.PodReference{Name: pod.Name, Namespace: pod.Namespace, UID: string(pod.UID)},
				Memory: &statsapi.MemoryStats{UsageBytes: new(uint64(150 * mi))},
			}}}
			actions := podActions{
				ContainersToUpdate:         map[v1.ResourceName][]containerToUpdateInfo{},
				PodCgroupResourcesToUpdate: sets.New(v1.ResourceSwap),
			}
			err := m.validatePodResizeAction(tCtx, pod, kps, &cm.ResourceConfig{Memory: new(200 * mi)}, &cm.ResourceConfig{Memory: new(100 * mi)}, actions)
			if tc.expectErr == "" {
				require.NoError(t, err)
			} else {
				require.ErrorContains(t, err, tc.expectErr)
			}
		})
	}
}

// TestSwapUsageReadOnlyForSwapDecrease: when CRI stats omit swap, the extra
// swap stats read happens only for swap decreases, not memory-only downsizes.
func TestSwapUsageReadOnlyForSwapDecrease(t *testing.T) {
	tCtx := ktesting.Init(t)
	miQ := resource.MustParse("1Mi")
	mi := miQ.Value()

	for _, tc := range []struct {
		name          string
		resourceName  v1.ResourceName
		current       resourceRequirements
		desired       resourceRequirements
		swapUsage     *uint64 // returned by PodSwapStats; nil means no swap stats
		expectErr     bool
		expectedCalls int
	}{
		{
			name:         "swap decrease above usage from swap stats",
			resourceName: v1.ResourceSwap,
			current:      resourceRequirements{memoryRequest: 1, memoryLimit: 2, swapLimit: 200 * mi},
			desired:      resourceRequirements{memoryRequest: 1, memoryLimit: 2, swapLimit: 100 * mi},
			swapUsage:    new(uint64(50 * mi)), expectedCalls: 1,
		},
		{
			name:         "swap decrease below usage from swap stats is deferred",
			resourceName: v1.ResourceSwap,
			current:      resourceRequirements{memoryRequest: 1, memoryLimit: 2, swapLimit: 200 * mi},
			desired:      resourceRequirements{memoryRequest: 1, memoryLimit: 2, swapLimit: 100 * mi},
			swapUsage:    new(uint64(150 * mi)), expectErr: true, expectedCalls: 1,
		},
		{
			name:         "swap decrease with no swap stats anywhere is deferred",
			resourceName: v1.ResourceSwap,
			current:      resourceRequirements{memoryRequest: 1, memoryLimit: 2, swapLimit: 200 * mi},
			desired:      resourceRequirements{memoryRequest: 1, memoryLimit: 2, swapLimit: 100 * mi},
			expectErr:    true, expectedCalls: 1,
		},
		{
			name:          "memory-only decrease does not read swap stats",
			resourceName:  v1.ResourceMemory,
			current:       resourceRequirements{memoryRequest: 1, memoryLimit: 200 * mi},
			desired:       resourceRequirements{memoryRequest: 1, memoryLimit: 100 * mi},
			expectedCalls: 0,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := newSwapTestManager(t, tCtx, kubetypes.WorkloadControlledSwap)
			pod := swapTestPod(withSwapLimit(burstableContainer("c", "1Gi", "2Gi"), "100Mi"))
			kps := runningStatusFor(pod)

			current := tc.current
			actions := podActions{ContainersToUpdate: map[v1.ResourceName][]containerToUpdateInfo{
				tc.resourceName: {{
					container: &pod.Spec.Containers[0], kubeContainerID: kps.ContainerStatuses[0].ID,
					desiredContainerResources: tc.desired, currentContainerResources: &current,
				}},
			}}

			// CRI stats without any swap, as containerd reports today.
			criStats := &statsapi.PodStats{
				PodRef: statsapi.PodReference{Name: pod.Name, Namespace: pod.Namespace, UID: string(pod.UID)},
				Memory: &statsapi.MemoryStats{UsageBytes: new(uint64(1))},
				Containers: []statsapi.ContainerStats{{
					Name:   "c",
					Memory: &statsapi.MemoryStats{UsageBytes: new(uint64(1))},
				}},
			}
			helper := &containertest.FakeRuntimeHelper{PodStats: map[types.UID]*statsapi.PodStats{pod.UID: criStats}}
			if tc.swapUsage != nil {
				helper.PodSwapStatsByUID = map[types.UID]*statsapi.PodStats{pod.UID: {
					Containers: []statsapi.ContainerStats{{Name: "c", Swap: &statsapi.SwapStats{SwapUsageBytes: tc.swapUsage}}},
				}}
			}
			m.runtimeHelper = helper

			err := m.validatePodResizeAction(tCtx, pod, kps, &cm.ResourceConfig{}, &cm.ResourceConfig{}, actions)
			if tc.expectErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
			assert.Equal(t, tc.expectedCalls, helper.PodSwapStatsCalls, "PodSwapStats calls")
		})
	}
}

// TestIsPodResizeInProgressSwap: the in-progress condition must track swap.
func TestIsPodResizeInProgressSwap(t *testing.T) {
	logger, tCtx := ktesting.NewTestContext(t)
	m := newSwapTestManager(t, tCtx, kubetypes.WorkloadControlledSwap)

	pod := swapTestPod(withSwapLimit(burstableContainer("c", "1Gi", "2Gi"), "256Mi"))
	status := runningStatusFor(pod)
	m.InitializeActuatedPod(logger, pod, nil)
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
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodLevelResourcesVerticalScaling, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WorkloadControlledSwap, true)
	setCgroupVersionDuringTest(cgroupV2)

	mi := int64(1024 * 1024)

	// The pod cgroup is only written when the pod-level swap limit is part of the pod cgroup
	// update. Otherwise the container update runs alone, whatever the pod cgroup holds.
	for _, podSwapResize := range []bool{true, false} {
		t.Run(fmt.Sprintf("swap increase updates pod cgroup before container (2a), pod-level swap resize %t", podSwapResize), func(t *testing.T) {
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
			if podSwapResize {
				mockPCM.EXPECT().SetPodCgroupConfig(logger, mock.Anything, mock.Anything).Run(func(_ klog.Logger, _ *v1.Pod, rc *cm.ResourceConfig) {
					// At the moment pod cgroup swap is raised, the container CRI call must not have happened yet.
					assert.NotContains(t, fakeRuntime.Called, "UpdateContainerResources")
					podCgroupCalls = append(podCgroupCalls, rc)
				}).Return(nil).Once()
			}

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
				SandboxID: "sandboxID",
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

			if podSwapResize {
				actions.PodCgroupResourcesToUpdate = sets.New(v1.ResourceSwap)
			}

			res := m.doPodResizeAction(tCtx, pod, kps, actions)
			require.NoError(t, res.Error)
			assert.Contains(t, fakeRuntime.Called, "UpdateContainerResources")

			actuatedPodRes, found := m.actuatedState.GetPodLevelResources(pod.UID)
			require.True(t, found)
			actuatedSwap := actuatedPodRes.Limits[v1.ResourceSwap]
			if podSwapResize {
				require.Len(t, podCgroupCalls, 1)
				assert.Equal(t, fmt.Sprintf("%d", 400*mi), podCgroupCalls[0].Unified[cm.Cgroup2MaxSwapFilename])
				assert.Equal(t, int64(400*mi), actuatedSwap.Value())
			} else {
				assert.Empty(t, podCgroupCalls)
				assert.Equal(t, int64(200*mi), actuatedSwap.Value())
			}
		})
	}

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
			PodCgroupResourcesToUpdate: sets.New(v1.ResourceSwap),
			SandboxID:                  "sandboxID",
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

// TestActuatedPodLevelResourcesKeepSwapLimit: the actuated pod-level record
// holds the pod-level swap limit next to the effective cpu and memory, and is
// the plain effective resources for a pod without one.
func TestActuatedPodLevelResourcesKeepSwapLimit(t *testing.T) {
	logger, tCtx := ktesting.NewTestContext(t)
	podWithSwap := swapTestPod(burstableContainer("c", "1Gi", "2Gi"))
	podWithSwap.Spec.Resources = &v1.ResourceRequirements{Limits: v1.ResourceList{
		v1.ResourceMemory: resource.MustParse("4Gi"),
		v1.ResourceSwap:   resource.MustParse("1Gi"),
	}}
	withSwap := cm.EffectivePodResources(podWithSwap)
	withSwap.Limits[v1.ResourceSwap] = resource.MustParse("1Gi")
	podWithoutSwap := swapTestPod(burstableContainer("c", "1Gi", "2Gi"))

	for _, tc := range []struct {
		name     string
		pod      *v1.Pod
		setup    func(m *kubeGenericRuntimeManager, pod *v1.Pod) error
		expected *v1.ResourceRequirements
	}{
		{
			name: "new pod with a pod-level swap limit",
			pod:  podWithSwap, expected: withSwap,
		},
		{
			name: "pod cgroup created before the pod is initialized",
			pod:  podWithSwap, expected: withSwap,
			setup: func(m *kubeGenericRuntimeManager, pod *v1.Pod) error {
				return m.UpdateActuatedPodLevelResources(logger, pod)
			},
		},
		{
			name: "new pod without a pod-level swap limit",
			pod:  podWithoutSwap, expected: cm.EffectivePodResources(podWithoutSwap),
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := newSwapTestManager(t, tCtx, kubetypes.WorkloadControlledSwap)
			if tc.setup != nil {
				require.NoError(t, tc.setup(m, tc.pod))
			}
			m.InitializeActuatedPod(logger, tc.pod, nil)

			actuated, found := m.actuatedState.GetPodLevelResources(tc.pod.UID)
			require.True(t, found)
			assert.Equal(t, tc.expected, actuated)
			assert.Empty(t, m.computePodLevelResourcesResizeAction(tCtx, tc.pod), "pod cgroup resources to update")
		})
	}
}

// TestPodLevelSwapResizeInProgressFollowsNodeSwap: pod-level swap is only
// compared when the node manages swap, matching what doPodResizeAction actuates.
func TestPodLevelSwapResizeInProgressFollowsNodeSwap(t *testing.T) {
	for _, tc := range []struct {
		name         string
		behavior     kubetypes.SwapBehavior
		wantProgress bool
	}{
		{name: "NoSwap node ignores pod-level swap drift", behavior: kubetypes.NoSwap, wantProgress: false},
		{name: "WorkloadControlledSwap node tracks pod-level swap drift", behavior: kubetypes.WorkloadControlledSwap, wantProgress: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			logger, tCtx := ktesting.NewTestContext(t)
			m := newSwapTestManager(t, tCtx, tc.behavior)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodLevelResourcesVerticalScaling, true)

			pod := swapTestPod(burstableContainer("c", "1Gi", "2Gi"))
			pod.Spec.Resources = &v1.ResourceRequirements{Limits: v1.ResourceList{
				v1.ResourceMemory: resource.MustParse("4Gi"),
				v1.ResourceSwap:   resource.MustParse("1Gi"),
			}}
			// Actuated cpu/memory match the spec; only pod-level swap differs.
			actuated := actuatedPodLevelResources(pod)
			actuated.Limits[v1.ResourceSwap] = resource.MustParse("512Mi")
			require.NoError(t, m.actuatedState.SetPodLevelResources(logger, pod.UID, actuated))

			assert.Equal(t, tc.wantProgress, m.isPodLevelResourcesResizeInProgress(pod, runningStatusFor(pod)), "isPodLevelResourcesResizeInProgress")
			toUpdate := m.computePodLevelResourcesResizeAction(tCtx, pod)
			assert.Equal(t, tc.wantProgress, toUpdate.Has(v1.ResourceSwap), "computePodLevelResourcesResizeAction swap")
			assert.False(t, toUpdate.HasAny(v1.ResourceCPU, v1.ResourceMemory), "computePodLevelResourcesResizeAction cpu and memory")
		})
	}
}

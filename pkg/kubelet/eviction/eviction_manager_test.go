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

package eviction

import (
	"fmt"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/clock"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	kubeapi "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
	evictionapi "k8s.io/kubernetes/pkg/kubelet/eviction/api"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	kubelettypes "k8s.io/kubernetes/pkg/kubelet/types"
)

const (
	lowPriority     = -1
	defaultPriority = 0
	highPriority    = 1
)

// mockPodKiller is used to testing which pod is killed
type mockPodKiller struct {
	pod                 *v1.Pod
	status              v1.PodStatus
	gracePeriodOverride *int64
}

// killPodNow records the pod that was killed
func (m *mockPodKiller) killPodNow(pod *v1.Pod, status v1.PodStatus, gracePeriodOverride *int64) error {
	m.pod = pod
	m.status = status
	m.gracePeriodOverride = gracePeriodOverride
	return nil
}

// mockDiskInfoProvider is used to simulate testing.
type mockDiskInfoProvider struct {
	dedicatedImageFs bool
}

// HasDedicatedImageFs returns the mocked value
func (m *mockDiskInfoProvider) HasDedicatedImageFs() (bool, error) {
	return m.dedicatedImageFs, nil
}

// mockDiskGC is used to simulate invoking image and container garbage collection.
type mockDiskGC struct {
	err                 error
	imageGCInvoked      bool
	containerGCInvoked  bool
	fakeSummaryProvider *fakeSummaryProvider
	summaryAfterGC      *statsapi.Summary
}

// DeleteUnusedImages returns the mocked values.
func (m *mockDiskGC) DeleteUnusedImages() error {
	m.imageGCInvoked = true
	if m.summaryAfterGC != nil && m.fakeSummaryProvider != nil {
		m.fakeSummaryProvider.result = m.summaryAfterGC
	}
	return m.err
}

// DeleteAllUnusedContainers returns the mocked value
func (m *mockDiskGC) DeleteAllUnusedContainers() error {
	m.containerGCInvoked = true
	if m.summaryAfterGC != nil && m.fakeSummaryProvider != nil {
		m.fakeSummaryProvider.result = m.summaryAfterGC
	}
	return m.err
}

func makePodWithMemoryStats(name string, priority int32, requests v1.ResourceList, limits v1.ResourceList, memoryWorkingSet string) (*v1.Pod, statsapi.PodStats) {
	pod := newPod(name, priority, []v1.Container{
		newContainer(name, requests, limits),
	}, nil)
	podStats := newPodMemoryStats(pod, resource.MustParse(memoryWorkingSet))
	return pod, podStats
}

func makePodWithDiskStats(name string, priority int32, requests v1.ResourceList, limits v1.ResourceList, rootFsUsed, logsUsed, perLocalVolumeUsed string) (*v1.Pod, statsapi.PodStats) {
	pod := newPod(name, priority, []v1.Container{
		newContainer(name, requests, limits),
	}, nil)
	podStats := newPodDiskStats(pod, parseQuantity(rootFsUsed), parseQuantity(logsUsed), parseQuantity(perLocalVolumeUsed))
	return pod, podStats
}

func makeMemoryStats(nodeAvailableBytes string, podStats map[*v1.Pod]statsapi.PodStats) *statsapi.Summary {
	val := resource.MustParse(nodeAvailableBytes)
	availableBytes := uint64(val.Value())
	WorkingSetBytes := uint64(val.Value())
	result := &statsapi.Summary{
		Node: statsapi.NodeStats{
			Memory: &statsapi.MemoryStats{
				AvailableBytes:  &availableBytes,
				WorkingSetBytes: &WorkingSetBytes,
			},
			SystemContainers: []statsapi.ContainerStats{
				{
					Name: statsapi.SystemContainerPods,
					Memory: &statsapi.MemoryStats{
						AvailableBytes:  &availableBytes,
						WorkingSetBytes: &WorkingSetBytes,
					},
				},
			},
		},
		Pods: []statsapi.PodStats{},
	}
	for _, podStat := range podStats {
		result.Pods = append(result.Pods, podStat)
	}
	return result
}

func makeDiskStats(rootFsAvailableBytes, imageFsAvailableBytes string, podStats map[*v1.Pod]statsapi.PodStats) *statsapi.Summary {
	rootFsVal := resource.MustParse(rootFsAvailableBytes)
	rootFsBytes := uint64(rootFsVal.Value())
	rootFsCapacityBytes := uint64(rootFsVal.Value() * 2)
	imageFsVal := resource.MustParse(imageFsAvailableBytes)
	imageFsBytes := uint64(imageFsVal.Value())
	imageFsCapacityBytes := uint64(imageFsVal.Value() * 2)
	result := &statsapi.Summary{
		Node: statsapi.NodeStats{
			Fs: &statsapi.FsStats{
				AvailableBytes: &rootFsBytes,
				CapacityBytes:  &rootFsCapacityBytes,
			},
			Runtime: &statsapi.RuntimeStats{
				ImageFs: &statsapi.FsStats{
					AvailableBytes: &imageFsBytes,
					CapacityBytes:  &imageFsCapacityBytes,
				},
			},
		},
		Pods: []statsapi.PodStats{},
	}
	for _, podStat := range podStats {
		result.Pods = append(result.Pods, podStat)
	}
	return result
}

type podToMake struct {
	name                     string
	priority                 int32
	requests                 v1.ResourceList
	limits                   v1.ResourceList
	memoryWorkingSet         string
	rootFsUsed               string
	logsFsUsed               string
	logsFsInodesUsed         string
	rootFsInodesUsed         string
	perLocalVolumeUsed       string
	perLocalVolumeInodesUsed string
}

// TestMemoryPressure
func TestMemoryPressure(t *testing.T) {
	podMaker := makePodWithMemoryStats
	summaryStatsMaker := makeMemoryStats
	podsToMake := []podToMake{
		{name: "guaranteed-low-priority-high-usage", priority: lowPriority, requests: newResourceList("100m", "1Gi", ""), limits: newResourceList("100m", "1Gi", ""), memoryWorkingSet: "900Mi"},
		{name: "burstable-below-requests", priority: defaultPriority, requests: newResourceList("100m", "100Mi", ""), limits: newResourceList("200m", "1Gi", ""), memoryWorkingSet: "50Mi"},
		{name: "burstable-above-requests", priority: defaultPriority, requests: newResourceList("100m", "100Mi", ""), limits: newResourceList("200m", "1Gi", ""), memoryWorkingSet: "400Mi"},
		{name: "best-effort-high-priority-high-usage", priority: highPriority, requests: newResourceList("", "", ""), limits: newResourceList("", "", ""), memoryWorkingSet: "400Mi"},
		{name: "best-effort-low-priority-low-usage", priority: lowPriority, requests: newResourceList("", "", ""), limits: newResourceList("", "", ""), memoryWorkingSet: "100Mi"},
	}
	pods := []*v1.Pod{}
	podStats := map[*v1.Pod]statsapi.PodStats{}
	for _, podToMake := range podsToMake {
		pod, podStat := podMaker(podToMake.name, podToMake.priority, podToMake.requests, podToMake.limits, podToMake.memoryWorkingSet)
		pods = append(pods, pod)
		podStats[pod] = podStat
	}
	podToEvict := pods[4]
	activePodsFunc := func() []*v1.Pod {
		return pods
	}

	fakeClock := clock.NewFakeClock(time.Now())
	podKiller := &mockPodKiller{}
	diskInfoProvider := &mockDiskInfoProvider{dedicatedImageFs: false}
	diskGC := &mockDiskGC{err: nil}
	nodeRef := &v1.ObjectReference{Kind: "Node", Name: "test", UID: types.UID("test"), Namespace: ""}

	config := Config{
		MaxPodGracePeriodSeconds: 5,
		PressureTransitionPeriod: time.Minute * 5,
		Thresholds: []evictionapi.Threshold{
			{
				Signal:   evictionapi.SignalMemoryAvailable,
				Operator: evictionapi.OpLessThan,
				Value: evictionapi.ThresholdValue{
					Quantity: quantityMustParse("1Gi"),
				},
			},
			{
				Signal:   evictionapi.SignalMemoryAvailable,
				Operator: evictionapi.OpLessThan,
				Value: evictionapi.ThresholdValue{
					Quantity: quantityMustParse("2Gi"),
				},
				GracePeriod: time.Minute * 2,
			},
		},
	}
	summaryProvider := &fakeSummaryProvider{result: summaryStatsMaker("2Gi", podStats)}
	manager := &managerImpl{
		clock:                        fakeClock,
		killPodFunc:                  podKiller.killPodNow,
		imageGC:                      diskGC,
		containerGC:                  diskGC,
		config:                       config,
		recorder:                     &record.FakeRecorder{},
		summaryProvider:              summaryProvider,
		nodeRef:                      nodeRef,
		nodeConditionsLastObservedAt: nodeConditionsObservedAt{},
		thresholdsFirstObservedAt:    thresholdsObservedAt{},
	}

	// create a best effort pod to test admission
	bestEffortPodToAdmit, _ := podMaker("best-admit", defaultPriority, newResourceList("", "", ""), newResourceList("", "", ""), "0Gi")
	burstablePodToAdmit, _ := podMaker("burst-admit", defaultPriority, newResourceList("100m", "100Mi", ""), newResourceList("200m", "200Mi", ""), "0Gi")

	// synchronize
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should not have memory pressure
	if manager.IsUnderMemoryPressure() {
		t.Errorf("Manager should not report memory pressure")
	}

	// try to admit our pods (they should succeed)
	expected := []bool{true, true}
	for i, pod := range []*v1.Pod{bestEffortPodToAdmit, burstablePodToAdmit} {
		if result := manager.Admit(&lifecycle.PodAdmitAttributes{Pod: pod}); expected[i] != result.Admit {
			t.Errorf("Admit pod: %v, expected: %v, actual: %v", pod, expected[i], result.Admit)
		}
	}

	// induce soft threshold
	fakeClock.Step(1 * time.Minute)
	summaryProvider.result = summaryStatsMaker("1500Mi", podStats)
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should have memory pressure
	if !manager.IsUnderMemoryPressure() {
		t.Errorf("Manager should report memory pressure since soft threshold was met")
	}

	// verify no pod was yet killed because there has not yet been enough time passed.
	if podKiller.pod != nil {
		t.Errorf("Manager should not have killed a pod yet, but killed: %v", podKiller.pod.Name)
	}

	// step forward in time pass the grace period
	fakeClock.Step(3 * time.Minute)
	summaryProvider.result = summaryStatsMaker("1500Mi", podStats)
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should have memory pressure
	if !manager.IsUnderMemoryPressure() {
		t.Errorf("Manager should report memory pressure since soft threshold was met")
	}

	// verify the right pod was killed with the right grace period.
	if podKiller.pod != podToEvict {
		t.Errorf("Manager chose to kill pod: %v, but should have chosen %v", podKiller.pod.Name, podToEvict.Name)
	}
	if podKiller.gracePeriodOverride == nil {
		t.Errorf("Manager chose to kill pod but should have had a grace period override.")
	}
	observedGracePeriod := *podKiller.gracePeriodOverride
	if observedGracePeriod != manager.config.MaxPodGracePeriodSeconds {
		t.Errorf("Manager chose to kill pod with incorrect grace period.  Expected: %d, actual: %d", manager.config.MaxPodGracePeriodSeconds, observedGracePeriod)
	}
	// reset state
	podKiller.pod = nil
	podKiller.gracePeriodOverride = nil

	// remove memory pressure
	fakeClock.Step(20 * time.Minute)
	summaryProvider.result = summaryStatsMaker("3Gi", podStats)
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should not have memory pressure
	if manager.IsUnderMemoryPressure() {
		t.Errorf("Manager should not report memory pressure")
	}

	// induce memory pressure!
	fakeClock.Step(1 * time.Minute)
	summaryProvider.result = summaryStatsMaker("500Mi", podStats)
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should have memory pressure
	if !manager.IsUnderMemoryPressure() {
		t.Errorf("Manager should report memory pressure")
	}

	// check the right pod was killed
	if podKiller.pod != podToEvict {
		t.Errorf("Manager chose to kill pod: %v, but should have chosen %v", podKiller.pod.Name, podToEvict.Name)
	}
	observedGracePeriod = *podKiller.gracePeriodOverride
	if observedGracePeriod != int64(0) {
		t.Errorf("Manager chose to kill pod with incorrect grace period.  Expected: %d, actual: %d", 0, observedGracePeriod)
	}

	// the best-effort pod should not admit, burstable should
	expected = []bool{false, true}
	for i, pod := range []*v1.Pod{bestEffortPodToAdmit, burstablePodToAdmit} {
		if result := manager.Admit(&lifecycle.PodAdmitAttributes{Pod: pod}); expected[i] != result.Admit {
			t.Errorf("Admit pod: %v, expected: %v, actual: %v", pod, expected[i], result.Admit)
		}
	}

	// reduce memory pressure
	fakeClock.Step(1 * time.Minute)
	summaryProvider.result = summaryStatsMaker("2Gi", podStats)
	podKiller.pod = nil // reset state
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should have memory pressure (because transition period not yet met)
	if !manager.IsUnderMemoryPressure() {
		t.Errorf("Manager should report memory pressure")
	}

	// no pod should have been killed
	if podKiller.pod != nil {
		t.Errorf("Manager chose to kill pod: %v when no pod should have been killed", podKiller.pod.Name)
	}

	// the best-effort pod should not admit, burstable should
	expected = []bool{false, true}
	for i, pod := range []*v1.Pod{bestEffortPodToAdmit, burstablePodToAdmit} {
		if result := manager.Admit(&lifecycle.PodAdmitAttributes{Pod: pod}); expected[i] != result.Admit {
			t.Errorf("Admit pod: %v, expected: %v, actual: %v", pod, expected[i], result.Admit)
		}
	}

	// move the clock past transition period to ensure that we stop reporting pressure
	fakeClock.Step(5 * time.Minute)
	summaryProvider.result = summaryStatsMaker("2Gi", podStats)
	podKiller.pod = nil // reset state
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should not have memory pressure (because transition period met)
	if manager.IsUnderMemoryPressure() {
		t.Errorf("Manager should not report memory pressure")
	}

	// no pod should have been killed
	if podKiller.pod != nil {
		t.Errorf("Manager chose to kill pod: %v when no pod should have been killed", podKiller.pod.Name)
	}

	// all pods should admit now
	expected = []bool{true, true}
	for i, pod := range []*v1.Pod{bestEffortPodToAdmit, burstablePodToAdmit} {
		if result := manager.Admit(&lifecycle.PodAdmitAttributes{Pod: pod}); expected[i] != result.Admit {
			t.Errorf("Admit pod: %v, expected: %v, actual: %v", pod, expected[i], result.Admit)
		}
	}
}

func makeContainersByQOS(class v1.PodQOSClass) []v1.Container {
	resource := newResourceList("100m", "1Gi", "")
	switch class {
	case v1.PodQOSGuaranteed:
		return []v1.Container{newContainer("guaranteed-container", resource, resource)}
	case v1.PodQOSBurstable:
		return []v1.Container{newContainer("burtable-container", resource, nil)}
	case v1.PodQOSBestEffort:
		fallthrough
	default:
		return []v1.Container{newContainer("best-effort-container", nil, nil)}
	}
}

func TestAdmitUnderNodeConditions(t *testing.T) {
	manager := &managerImpl{}
	pods := []*v1.Pod{
		newPod("guaranteed-pod", scheduling.DefaultPriorityWhenNoDefaultClassExists, makeContainersByQOS(v1.PodQOSGuaranteed), nil),
		newPod("burstable-pod", scheduling.DefaultPriorityWhenNoDefaultClassExists, makeContainersByQOS(v1.PodQOSBurstable), nil),
		newPod("best-effort-pod", scheduling.DefaultPriorityWhenNoDefaultClassExists, makeContainersByQOS(v1.PodQOSBestEffort), nil),
	}

	expected := []bool{true, true, true}
	for i, pod := range pods {
		if result := manager.Admit(&lifecycle.PodAdmitAttributes{Pod: pod}); expected[i] != result.Admit {
			t.Errorf("Admit pod: %v, expected: %v, actual: %v", pod, expected[i], result.Admit)
		}
	}

	manager.nodeConditions = []v1.NodeConditionType{v1.NodeMemoryPressure}
	expected = []bool{true, true, false}
	for i, pod := range pods {
		if result := manager.Admit(&lifecycle.PodAdmitAttributes{Pod: pod}); expected[i] != result.Admit {
			t.Errorf("Admit pod: %v, expected: %v, actual: %v", pod, expected[i], result.Admit)
		}
	}

	manager.nodeConditions = []v1.NodeConditionType{v1.NodeMemoryPressure, v1.NodeDiskPressure}
	expected = []bool{false, false, false}
	for i, pod := range pods {
		if result := manager.Admit(&lifecycle.PodAdmitAttributes{Pod: pod}); expected[i] != result.Admit {
			t.Errorf("Admit pod: %v, expected: %v, actual: %v", pod, expected[i], result.Admit)
		}
	}
}

// parseQuantity parses the specified value (if provided) otherwise returns 0 value
func parseQuantity(value string) resource.Quantity {
	if len(value) == 0 {
		return resource.MustParse("0")
	}
	return resource.MustParse(value)
}

func TestDiskPressureNodeFs(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.LocalStorageCapacityIsolation, true)()

	podMaker := makePodWithDiskStats
	summaryStatsMaker := makeDiskStats
	podsToMake := []podToMake{
		{name: "low-priority-high-usage", priority: lowPriority, requests: newResourceList("100m", "1Gi", ""), limits: newResourceList("100m", "1Gi", ""), rootFsUsed: "900Mi"},
		{name: "below-requests", priority: defaultPriority, requests: newResourceList("100m", "100Mi", ""), limits: newResourceList("200m", "1Gi", ""), logsFsUsed: "50Mi"},
		{name: "above-requests", priority: defaultPriority, requests: newResourceList("100m", "100Mi", ""), limits: newResourceList("200m", "1Gi", ""), rootFsUsed: "400Mi"},
		{name: "high-priority-high-usage", priority: highPriority, requests: newResourceList("", "", ""), limits: newResourceList("", "", ""), perLocalVolumeUsed: "400Mi"},
		{name: "low-priority-low-usage", priority: lowPriority, requests: newResourceList("", "", ""), limits: newResourceList("", "", ""), rootFsUsed: "100Mi"},
	}
	pods := []*v1.Pod{}
	podStats := map[*v1.Pod]statsapi.PodStats{}
	for _, podToMake := range podsToMake {
		pod, podStat := podMaker(podToMake.name, podToMake.priority, podToMake.requests, podToMake.limits, podToMake.rootFsUsed, podToMake.logsFsUsed, podToMake.perLocalVolumeUsed)
		pods = append(pods, pod)
		podStats[pod] = podStat
	}
	podToEvict := pods[0]
	activePodsFunc := func() []*v1.Pod {
		return pods
	}

	fakeClock := clock.NewFakeClock(time.Now())
	podKiller := &mockPodKiller{}
	diskInfoProvider := &mockDiskInfoProvider{dedicatedImageFs: false}
	diskGC := &mockDiskGC{err: nil}
	nodeRef := &v1.ObjectReference{Kind: "Node", Name: "test", UID: types.UID("test"), Namespace: ""}

	config := Config{
		MaxPodGracePeriodSeconds: 5,
		PressureTransitionPeriod: time.Minute * 5,
		Thresholds: []evictionapi.Threshold{
			{
				Signal:   evictionapi.SignalNodeFsAvailable,
				Operator: evictionapi.OpLessThan,
				Value: evictionapi.ThresholdValue{
					Quantity: quantityMustParse("1Gi"),
				},
			},
			{
				Signal:   evictionapi.SignalNodeFsAvailable,
				Operator: evictionapi.OpLessThan,
				Value: evictionapi.ThresholdValue{
					Quantity: quantityMustParse("2Gi"),
				},
				GracePeriod: time.Minute * 2,
			},
		},
	}
	summaryProvider := &fakeSummaryProvider{result: summaryStatsMaker("16Gi", "200Gi", podStats)}
	manager := &managerImpl{
		clock:                        fakeClock,
		killPodFunc:                  podKiller.killPodNow,
		imageGC:                      diskGC,
		containerGC:                  diskGC,
		config:                       config,
		recorder:                     &record.FakeRecorder{},
		summaryProvider:              summaryProvider,
		nodeRef:                      nodeRef,
		nodeConditionsLastObservedAt: nodeConditionsObservedAt{},
		thresholdsFirstObservedAt:    thresholdsObservedAt{},
	}

	// create a best effort pod to test admission
	podToAdmit, _ := podMaker("pod-to-admit", defaultPriority, newResourceList("", "", ""), newResourceList("", "", ""), "0Gi", "0Gi", "0Gi")

	// synchronize
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should not have disk pressure
	if manager.IsUnderDiskPressure() {
		t.Errorf("Manager should not report disk pressure")
	}

	// try to admit our pod (should succeed)
	if result := manager.Admit(&lifecycle.PodAdmitAttributes{Pod: podToAdmit}); !result.Admit {
		t.Errorf("Admit pod: %v, expected: %v, actual: %v", podToAdmit, true, result.Admit)
	}

	// induce soft threshold
	fakeClock.Step(1 * time.Minute)
	summaryProvider.result = summaryStatsMaker("1.5Gi", "200Gi", podStats)
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should have disk pressure
	if !manager.IsUnderDiskPressure() {
		t.Errorf("Manager should report disk pressure since soft threshold was met")
	}

	// verify no pod was yet killed because there has not yet been enough time passed.
	if podKiller.pod != nil {
		t.Errorf("Manager should not have killed a pod yet, but killed: %v", podKiller.pod.Name)
	}

	// step forward in time pass the grace period
	fakeClock.Step(3 * time.Minute)
	summaryProvider.result = summaryStatsMaker("1.5Gi", "200Gi", podStats)
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should have disk pressure
	if !manager.IsUnderDiskPressure() {
		t.Errorf("Manager should report disk pressure since soft threshold was met")
	}

	// verify the right pod was killed with the right grace period.
	if podKiller.pod != podToEvict {
		t.Errorf("Manager chose to kill pod: %v, but should have chosen %v", podKiller.pod.Name, podToEvict.Name)
	}
	if podKiller.gracePeriodOverride == nil {
		t.Errorf("Manager chose to kill pod but should have had a grace period override.")
	}
	observedGracePeriod := *podKiller.gracePeriodOverride
	if observedGracePeriod != manager.config.MaxPodGracePeriodSeconds {
		t.Errorf("Manager chose to kill pod with incorrect grace period.  Expected: %d, actual: %d", manager.config.MaxPodGracePeriodSeconds, observedGracePeriod)
	}
	// reset state
	podKiller.pod = nil
	podKiller.gracePeriodOverride = nil

	// remove disk pressure
	fakeClock.Step(20 * time.Minute)
	summaryProvider.result = summaryStatsMaker("16Gi", "200Gi", podStats)
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should not have disk pressure
	if manager.IsUnderDiskPressure() {
		t.Errorf("Manager should not report disk pressure")
	}

	// induce disk pressure!
	fakeClock.Step(1 * time.Minute)
	summaryProvider.result = summaryStatsMaker("500Mi", "200Gi", podStats)
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should have disk pressure
	if !manager.IsUnderDiskPressure() {
		t.Errorf("Manager should report disk pressure")
	}

	// check the right pod was killed
	if podKiller.pod != podToEvict {
		t.Errorf("Manager chose to kill pod: %v, but should have chosen %v", podKiller.pod.Name, podToEvict.Name)
	}
	observedGracePeriod = *podKiller.gracePeriodOverride
	if observedGracePeriod != int64(0) {
		t.Errorf("Manager chose to kill pod with incorrect grace period.  Expected: %d, actual: %d", 0, observedGracePeriod)
	}

	// try to admit our pod (should fail)
	if result := manager.Admit(&lifecycle.PodAdmitAttributes{Pod: podToAdmit}); result.Admit {
		t.Errorf("Admit pod: %v, expected: %v, actual: %v", podToAdmit, false, result.Admit)
	}

	// reduce disk pressure
	fakeClock.Step(1 * time.Minute)
	summaryProvider.result = summaryStatsMaker("16Gi", "200Gi", podStats)
	podKiller.pod = nil // reset state
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should have disk pressure (because transition period not yet met)
	if !manager.IsUnderDiskPressure() {
		t.Errorf("Manager should report disk pressure")
	}

	// no pod should have been killed
	if podKiller.pod != nil {
		t.Errorf("Manager chose to kill pod: %v when no pod should have been killed", podKiller.pod.Name)
	}

	// try to admit our pod (should fail)
	if result := manager.Admit(&lifecycle.PodAdmitAttributes{Pod: podToAdmit}); result.Admit {
		t.Errorf("Admit pod: %v, expected: %v, actual: %v", podToAdmit, false, result.Admit)
	}

	// move the clock past transition period to ensure that we stop reporting pressure
	fakeClock.Step(5 * time.Minute)
	summaryProvider.result = summaryStatsMaker("16Gi", "200Gi", podStats)
	podKiller.pod = nil // reset state
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should not have disk pressure (because transition period met)
	if manager.IsUnderDiskPressure() {
		t.Errorf("Manager should not report disk pressure")
	}

	// no pod should have been killed
	if podKiller.pod != nil {
		t.Errorf("Manager chose to kill pod: %v when no pod should have been killed", podKiller.pod.Name)
	}

	// try to admit our pod (should succeed)
	if result := manager.Admit(&lifecycle.PodAdmitAttributes{Pod: podToAdmit}); !result.Admit {
		t.Errorf("Admit pod: %v, expected: %v, actual: %v", podToAdmit, true, result.Admit)
	}
}

// TestMinReclaim verifies that min-reclaim works as desired.
func TestMinReclaim(t *testing.T) {
	podMaker := makePodWithMemoryStats
	summaryStatsMaker := makeMemoryStats
	podsToMake := []podToMake{
		{name: "guaranteed-low-priority-high-usage", priority: lowPriority, requests: newResourceList("100m", "1Gi", ""), limits: newResourceList("100m", "1Gi", ""), memoryWorkingSet: "900Mi"},
		{name: "burstable-below-requests", priority: defaultPriority, requests: newResourceList("100m", "100Mi", ""), limits: newResourceList("200m", "1Gi", ""), memoryWorkingSet: "50Mi"},
		{name: "burstable-above-requests", priority: defaultPriority, requests: newResourceList("100m", "100Mi", ""), limits: newResourceList("200m", "1Gi", ""), memoryWorkingSet: "400Mi"},
		{name: "best-effort-high-priority-high-usage", priority: highPriority, requests: newResourceList("", "", ""), limits: newResourceList("", "", ""), memoryWorkingSet: "400Mi"},
		{name: "best-effort-low-priority-low-usage", priority: lowPriority, requests: newResourceList("", "", ""), limits: newResourceList("", "", ""), memoryWorkingSet: "100Mi"},
	}
	pods := []*v1.Pod{}
	podStats := map[*v1.Pod]statsapi.PodStats{}
	for _, podToMake := range podsToMake {
		pod, podStat := podMaker(podToMake.name, podToMake.priority, podToMake.requests, podToMake.limits, podToMake.memoryWorkingSet)
		pods = append(pods, pod)
		podStats[pod] = podStat
	}
	podToEvict := pods[4]
	activePodsFunc := func() []*v1.Pod {
		return pods
	}

	fakeClock := clock.NewFakeClock(time.Now())
	podKiller := &mockPodKiller{}
	diskInfoProvider := &mockDiskInfoProvider{dedicatedImageFs: false}
	diskGC := &mockDiskGC{err: nil}
	nodeRef := &v1.ObjectReference{Kind: "Node", Name: "test", UID: types.UID("test"), Namespace: ""}

	config := Config{
		MaxPodGracePeriodSeconds: 5,
		PressureTransitionPeriod: time.Minute * 5,
		Thresholds: []evictionapi.Threshold{
			{
				Signal:   evictionapi.SignalMemoryAvailable,
				Operator: evictionapi.OpLessThan,
				Value: evictionapi.ThresholdValue{
					Quantity: quantityMustParse("1Gi"),
				},
				MinReclaim: &evictionapi.ThresholdValue{
					Quantity: quantityMustParse("500Mi"),
				},
			},
		},
	}
	summaryProvider := &fakeSummaryProvider{result: summaryStatsMaker("2Gi", podStats)}
	manager := &managerImpl{
		clock:                        fakeClock,
		killPodFunc:                  podKiller.killPodNow,
		imageGC:                      diskGC,
		containerGC:                  diskGC,
		config:                       config,
		recorder:                     &record.FakeRecorder{},
		summaryProvider:              summaryProvider,
		nodeRef:                      nodeRef,
		nodeConditionsLastObservedAt: nodeConditionsObservedAt{},
		thresholdsFirstObservedAt:    thresholdsObservedAt{},
	}

	// synchronize
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should not have memory pressure
	if manager.IsUnderMemoryPressure() {
		t.Errorf("Manager should not report memory pressure")
	}

	// induce memory pressure!
	fakeClock.Step(1 * time.Minute)
	summaryProvider.result = summaryStatsMaker("500Mi", podStats)
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should have memory pressure
	if !manager.IsUnderMemoryPressure() {
		t.Errorf("Manager should report memory pressure")
	}

	// check the right pod was killed
	if podKiller.pod != podToEvict {
		t.Errorf("Manager chose to kill pod: %v, but should have chosen %v", podKiller.pod.Name, podToEvict.Name)
	}
	observedGracePeriod := *podKiller.gracePeriodOverride
	if observedGracePeriod != int64(0) {
		t.Errorf("Manager chose to kill pod with incorrect grace period.  Expected: %d, actual: %d", 0, observedGracePeriod)
	}

	// reduce memory pressure, but not below the min-reclaim amount
	fakeClock.Step(1 * time.Minute)
	summaryProvider.result = summaryStatsMaker("1.2Gi", podStats)
	podKiller.pod = nil // reset state
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should have memory pressure (because transition period not yet met)
	if !manager.IsUnderMemoryPressure() {
		t.Errorf("Manager should report memory pressure")
	}

	// check the right pod was killed
	if podKiller.pod != podToEvict {
		t.Errorf("Manager chose to kill pod: %v, but should have chosen %v", podKiller.pod.Name, podToEvict.Name)
	}
	observedGracePeriod = *podKiller.gracePeriodOverride
	if observedGracePeriod != int64(0) {
		t.Errorf("Manager chose to kill pod with incorrect grace period.  Expected: %d, actual: %d", 0, observedGracePeriod)
	}

	// reduce memory pressure and ensure the min-reclaim amount
	fakeClock.Step(1 * time.Minute)
	summaryProvider.result = summaryStatsMaker("2Gi", podStats)
	podKiller.pod = nil // reset state
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should have memory pressure (because transition period not yet met)
	if !manager.IsUnderMemoryPressure() {
		t.Errorf("Manager should report memory pressure")
	}

	// no pod should have been killed
	if podKiller.pod != nil {
		t.Errorf("Manager chose to kill pod: %v when no pod should have been killed", podKiller.pod.Name)
	}

	// move the clock past transition period to ensure that we stop reporting pressure
	fakeClock.Step(5 * time.Minute)
	summaryProvider.result = summaryStatsMaker("2Gi", podStats)
	podKiller.pod = nil // reset state
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should not have memory pressure (because transition period met)
	if manager.IsUnderMemoryPressure() {
		t.Errorf("Manager should not report memory pressure")
	}

	// no pod should have been killed
	if podKiller.pod != nil {
		t.Errorf("Manager chose to kill pod: %v when no pod should have been killed", podKiller.pod.Name)
	}
}

func TestNodeReclaimFuncs(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.LocalStorageCapacityIsolation, true)()

	podMaker := makePodWithDiskStats
	summaryStatsMaker := makeDiskStats
	podsToMake := []podToMake{
		{name: "low-priority-high-usage", priority: lowPriority, requests: newResourceList("100m", "1Gi", ""), limits: newResourceList("100m", "1Gi", ""), rootFsUsed: "900Mi"},
		{name: "below-requests", priority: defaultPriority, requests: newResourceList("100m", "100Mi", ""), limits: newResourceList("200m", "1Gi", ""), logsFsUsed: "50Mi"},
		{name: "above-requests", priority: defaultPriority, requests: newResourceList("100m", "100Mi", ""), limits: newResourceList("200m", "1Gi", ""), rootFsUsed: "400Mi"},
		{name: "high-priority-high-usage", priority: highPriority, requests: newResourceList("", "", ""), limits: newResourceList("", "", ""), perLocalVolumeUsed: "400Mi"},
		{name: "low-priority-low-usage", priority: lowPriority, requests: newResourceList("", "", ""), limits: newResourceList("", "", ""), rootFsUsed: "100Mi"},
	}
	pods := []*v1.Pod{}
	podStats := map[*v1.Pod]statsapi.PodStats{}
	for _, podToMake := range podsToMake {
		pod, podStat := podMaker(podToMake.name, podToMake.priority, podToMake.requests, podToMake.limits, podToMake.rootFsUsed, podToMake.logsFsUsed, podToMake.perLocalVolumeUsed)
		pods = append(pods, pod)
		podStats[pod] = podStat
	}
	podToEvict := pods[0]
	activePodsFunc := func() []*v1.Pod {
		return pods
	}

	fakeClock := clock.NewFakeClock(time.Now())
	podKiller := &mockPodKiller{}
	diskInfoProvider := &mockDiskInfoProvider{dedicatedImageFs: false}
	nodeRef := &v1.ObjectReference{Kind: "Node", Name: "test", UID: types.UID("test"), Namespace: ""}

	config := Config{
		MaxPodGracePeriodSeconds: 5,
		PressureTransitionPeriod: time.Minute * 5,
		Thresholds: []evictionapi.Threshold{
			{
				Signal:   evictionapi.SignalNodeFsAvailable,
				Operator: evictionapi.OpLessThan,
				Value: evictionapi.ThresholdValue{
					Quantity: quantityMustParse("1Gi"),
				},
				MinReclaim: &evictionapi.ThresholdValue{
					Quantity: quantityMustParse("500Mi"),
				},
			},
		},
	}
	summaryProvider := &fakeSummaryProvider{result: summaryStatsMaker("16Gi", "200Gi", podStats)}
	diskGC := &mockDiskGC{fakeSummaryProvider: summaryProvider, err: nil}
	manager := &managerImpl{
		clock:                        fakeClock,
		killPodFunc:                  podKiller.killPodNow,
		imageGC:                      diskGC,
		containerGC:                  diskGC,
		config:                       config,
		recorder:                     &record.FakeRecorder{},
		summaryProvider:              summaryProvider,
		nodeRef:                      nodeRef,
		nodeConditionsLastObservedAt: nodeConditionsObservedAt{},
		thresholdsFirstObservedAt:    thresholdsObservedAt{},
	}

	// synchronize
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should not have disk pressure
	if manager.IsUnderDiskPressure() {
		t.Errorf("Manager should not report disk pressure")
	}

	// induce hard threshold
	fakeClock.Step(1 * time.Minute)
	summaryProvider.result = summaryStatsMaker(".9Gi", "200Gi", podStats)
	// make GC successfully return disk usage to previous levels
	diskGC.summaryAfterGC = summaryStatsMaker("16Gi", "200Gi", podStats)
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should have disk pressure
	if !manager.IsUnderDiskPressure() {
		t.Errorf("Manager should report disk pressure since soft threshold was met")
	}

	// verify image gc was invoked
	if !diskGC.imageGCInvoked || !diskGC.containerGCInvoked {
		t.Errorf("Manager should have invoked image gc")
	}

	// verify no pod was killed because image gc was sufficient
	if podKiller.pod != nil {
		t.Errorf("Manager should not have killed a pod, but killed: %v", podKiller.pod.Name)
	}

	// reset state
	diskGC.imageGCInvoked = false
	diskGC.containerGCInvoked = false

	// remove disk pressure
	fakeClock.Step(20 * time.Minute)
	summaryProvider.result = summaryStatsMaker("16Gi", "200Gi", podStats)
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should not have disk pressure
	if manager.IsUnderDiskPressure() {
		t.Errorf("Manager should not report disk pressure")
	}

	// induce disk pressure!
	fakeClock.Step(1 * time.Minute)
	summaryProvider.result = summaryStatsMaker("400Mi", "200Gi", podStats)
	// Dont reclaim any disk
	diskGC.summaryAfterGC = summaryStatsMaker("400Mi", "200Gi", podStats)
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should have disk pressure
	if !manager.IsUnderDiskPressure() {
		t.Errorf("Manager should report disk pressure")
	}

	// ensure disk gc was invoked
	if !diskGC.imageGCInvoked || !diskGC.containerGCInvoked {
		t.Errorf("Manager should have invoked image gc")
	}

	// check the right pod was killed
	if podKiller.pod != podToEvict {
		t.Errorf("Manager chose to kill pod: %v, but should have chosen %v", podKiller.pod.Name, podToEvict.Name)
	}
	observedGracePeriod := *podKiller.gracePeriodOverride
	if observedGracePeriod != int64(0) {
		t.Errorf("Manager chose to kill pod with incorrect grace period.  Expected: %d, actual: %d", 0, observedGracePeriod)
	}

	// reduce disk pressure
	fakeClock.Step(1 * time.Minute)
	summaryProvider.result = summaryStatsMaker("16Gi", "200Gi", podStats)
	diskGC.imageGCInvoked = false     // reset state
	diskGC.containerGCInvoked = false // reset state
	podKiller.pod = nil               // reset state
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should have disk pressure (because transition period not yet met)
	if !manager.IsUnderDiskPressure() {
		t.Errorf("Manager should report disk pressure")
	}

	// no image gc should have occurred
	if diskGC.imageGCInvoked || diskGC.containerGCInvoked {
		t.Errorf("Manager chose to perform image gc when it was not needed")
	}

	// no pod should have been killed
	if podKiller.pod != nil {
		t.Errorf("Manager chose to kill pod: %v when no pod should have been killed", podKiller.pod.Name)
	}

	// move the clock past transition period to ensure that we stop reporting pressure
	fakeClock.Step(5 * time.Minute)
	summaryProvider.result = summaryStatsMaker("16Gi", "200Gi", podStats)
	diskGC.imageGCInvoked = false     // reset state
	diskGC.containerGCInvoked = false // reset state
	podKiller.pod = nil               // reset state
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should not have disk pressure (because transition period met)
	if manager.IsUnderDiskPressure() {
		t.Errorf("Manager should not report disk pressure")
	}

	// no image gc should have occurred
	if diskGC.imageGCInvoked || diskGC.containerGCInvoked {
		t.Errorf("Manager chose to perform image gc when it was not needed")
	}

	// no pod should have been killed
	if podKiller.pod != nil {
		t.Errorf("Manager chose to kill pod: %v when no pod should have been killed", podKiller.pod.Name)
	}
}

func TestInodePressureNodeFsInodes(t *testing.T) {
	podMaker := func(name string, priority int32, requests v1.ResourceList, limits v1.ResourceList, rootInodes, logInodes, volumeInodes string) (*v1.Pod, statsapi.PodStats) {
		pod := newPod(name, priority, []v1.Container{
			newContainer(name, requests, limits),
		}, nil)
		podStats := newPodInodeStats(pod, parseQuantity(rootInodes), parseQuantity(logInodes), parseQuantity(volumeInodes))
		return pod, podStats
	}
	summaryStatsMaker := func(rootFsInodesFree, rootFsInodes string, podStats map[*v1.Pod]statsapi.PodStats) *statsapi.Summary {
		rootFsInodesFreeVal := resource.MustParse(rootFsInodesFree)
		internalRootFsInodesFree := uint64(rootFsInodesFreeVal.Value())
		rootFsInodesVal := resource.MustParse(rootFsInodes)
		internalRootFsInodes := uint64(rootFsInodesVal.Value())
		result := &statsapi.Summary{
			Node: statsapi.NodeStats{
				Fs: &statsapi.FsStats{
					InodesFree: &internalRootFsInodesFree,
					Inodes:     &internalRootFsInodes,
				},
			},
			Pods: []statsapi.PodStats{},
		}
		for _, podStat := range podStats {
			result.Pods = append(result.Pods, podStat)
		}
		return result
	}
	podsToMake := []podToMake{
		{name: "low-priority-high-usage", priority: lowPriority, requests: newResourceList("100m", "1Gi", ""), limits: newResourceList("100m", "1Gi", ""), rootFsInodesUsed: "900Mi"},
		{name: "below-requests", priority: defaultPriority, requests: newResourceList("100m", "100Mi", ""), limits: newResourceList("200m", "1Gi", ""), rootFsInodesUsed: "50Mi"},
		{name: "above-requests", priority: defaultPriority, requests: newResourceList("100m", "100Mi", ""), limits: newResourceList("200m", "1Gi", ""), rootFsInodesUsed: "400Mi"},
		{name: "high-priority-high-usage", priority: highPriority, requests: newResourceList("", "", ""), limits: newResourceList("", "", ""), rootFsInodesUsed: "400Mi"},
		{name: "low-priority-low-usage", priority: lowPriority, requests: newResourceList("", "", ""), limits: newResourceList("", "", ""), rootFsInodesUsed: "100Mi"},
	}
	pods := []*v1.Pod{}
	podStats := map[*v1.Pod]statsapi.PodStats{}
	for _, podToMake := range podsToMake {
		pod, podStat := podMaker(podToMake.name, podToMake.priority, podToMake.requests, podToMake.limits, podToMake.rootFsInodesUsed, podToMake.logsFsInodesUsed, podToMake.perLocalVolumeInodesUsed)
		pods = append(pods, pod)
		podStats[pod] = podStat
	}
	podToEvict := pods[0]
	activePodsFunc := func() []*v1.Pod {
		return pods
	}

	fakeClock := clock.NewFakeClock(time.Now())
	podKiller := &mockPodKiller{}
	diskInfoProvider := &mockDiskInfoProvider{dedicatedImageFs: false}
	diskGC := &mockDiskGC{err: nil}
	nodeRef := &v1.ObjectReference{Kind: "Node", Name: "test", UID: types.UID("test"), Namespace: ""}

	config := Config{
		MaxPodGracePeriodSeconds: 5,
		PressureTransitionPeriod: time.Minute * 5,
		Thresholds: []evictionapi.Threshold{
			{
				Signal:   evictionapi.SignalNodeFsInodesFree,
				Operator: evictionapi.OpLessThan,
				Value: evictionapi.ThresholdValue{
					Quantity: quantityMustParse("1Mi"),
				},
			},
			{
				Signal:   evictionapi.SignalNodeFsInodesFree,
				Operator: evictionapi.OpLessThan,
				Value: evictionapi.ThresholdValue{
					Quantity: quantityMustParse("2Mi"),
				},
				GracePeriod: time.Minute * 2,
			},
		},
	}
	summaryProvider := &fakeSummaryProvider{result: summaryStatsMaker("3Mi", "4Mi", podStats)}
	manager := &managerImpl{
		clock:                        fakeClock,
		killPodFunc:                  podKiller.killPodNow,
		imageGC:                      diskGC,
		containerGC:                  diskGC,
		config:                       config,
		recorder:                     &record.FakeRecorder{},
		summaryProvider:              summaryProvider,
		nodeRef:                      nodeRef,
		nodeConditionsLastObservedAt: nodeConditionsObservedAt{},
		thresholdsFirstObservedAt:    thresholdsObservedAt{},
	}

	// create a best effort pod to test admission
	podToAdmit, _ := podMaker("pod-to-admit", defaultPriority, newResourceList("", "", ""), newResourceList("", "", ""), "0", "0", "0")

	// synchronize
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should not have disk pressure
	if manager.IsUnderDiskPressure() {
		t.Errorf("Manager should not report inode pressure")
	}

	// try to admit our pod (should succeed)
	if result := manager.Admit(&lifecycle.PodAdmitAttributes{Pod: podToAdmit}); !result.Admit {
		t.Errorf("Admit pod: %v, expected: %v, actual: %v", podToAdmit, true, result.Admit)
	}

	// induce soft threshold
	fakeClock.Step(1 * time.Minute)
	summaryProvider.result = summaryStatsMaker("1.5Mi", "4Mi", podStats)
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should have disk pressure
	if !manager.IsUnderDiskPressure() {
		t.Errorf("Manager should report inode pressure since soft threshold was met")
	}

	// verify no pod was yet killed because there has not yet been enough time passed.
	if podKiller.pod != nil {
		t.Errorf("Manager should not have killed a pod yet, but killed: %v", podKiller.pod.Name)
	}

	// step forward in time pass the grace period
	fakeClock.Step(3 * time.Minute)
	summaryProvider.result = summaryStatsMaker("1.5Mi", "4Mi", podStats)
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should have disk pressure
	if !manager.IsUnderDiskPressure() {
		t.Errorf("Manager should report inode pressure since soft threshold was met")
	}

	// verify the right pod was killed with the right grace period.
	if podKiller.pod != podToEvict {
		t.Errorf("Manager chose to kill pod: %v, but should have chosen %v", podKiller.pod.Name, podToEvict.Name)
	}
	if podKiller.gracePeriodOverride == nil {
		t.Errorf("Manager chose to kill pod but should have had a grace period override.")
	}
	observedGracePeriod := *podKiller.gracePeriodOverride
	if observedGracePeriod != manager.config.MaxPodGracePeriodSeconds {
		t.Errorf("Manager chose to kill pod with incorrect grace period.  Expected: %d, actual: %d", manager.config.MaxPodGracePeriodSeconds, observedGracePeriod)
	}
	// reset state
	podKiller.pod = nil
	podKiller.gracePeriodOverride = nil

	// remove inode pressure
	fakeClock.Step(20 * time.Minute)
	summaryProvider.result = summaryStatsMaker("3Mi", "4Mi", podStats)
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should not have disk pressure
	if manager.IsUnderDiskPressure() {
		t.Errorf("Manager should not report inode pressure")
	}

	// induce inode pressure!
	fakeClock.Step(1 * time.Minute)
	summaryProvider.result = summaryStatsMaker("0.5Mi", "4Mi", podStats)
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should have disk pressure
	if !manager.IsUnderDiskPressure() {
		t.Errorf("Manager should report inode pressure")
	}

	// check the right pod was killed
	if podKiller.pod != podToEvict {
		t.Errorf("Manager chose to kill pod: %v, but should have chosen %v", podKiller.pod.Name, podToEvict.Name)
	}
	observedGracePeriod = *podKiller.gracePeriodOverride
	if observedGracePeriod != int64(0) {
		t.Errorf("Manager chose to kill pod with incorrect grace period.  Expected: %d, actual: %d", 0, observedGracePeriod)
	}

	// try to admit our pod (should fail)
	if result := manager.Admit(&lifecycle.PodAdmitAttributes{Pod: podToAdmit}); result.Admit {
		t.Errorf("Admit pod: %v, expected: %v, actual: %v", podToAdmit, false, result.Admit)
	}

	// reduce inode pressure
	fakeClock.Step(1 * time.Minute)
	summaryProvider.result = summaryStatsMaker("3Mi", "4Mi", podStats)
	podKiller.pod = nil // reset state
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should have disk pressure (because transition period not yet met)
	if !manager.IsUnderDiskPressure() {
		t.Errorf("Manager should report inode pressure")
	}

	// no pod should have been killed
	if podKiller.pod != nil {
		t.Errorf("Manager chose to kill pod: %v when no pod should have been killed", podKiller.pod.Name)
	}

	// try to admit our pod (should fail)
	if result := manager.Admit(&lifecycle.PodAdmitAttributes{Pod: podToAdmit}); result.Admit {
		t.Errorf("Admit pod: %v, expected: %v, actual: %v", podToAdmit, false, result.Admit)
	}

	// move the clock past transition period to ensure that we stop reporting pressure
	fakeClock.Step(5 * time.Minute)
	summaryProvider.result = summaryStatsMaker("3Mi", "4Mi", podStats)
	podKiller.pod = nil // reset state
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should not have disk pressure (because transition period met)
	if manager.IsUnderDiskPressure() {
		t.Errorf("Manager should not report inode pressure")
	}

	// no pod should have been killed
	if podKiller.pod != nil {
		t.Errorf("Manager chose to kill pod: %v when no pod should have been killed", podKiller.pod.Name)
	}

	// try to admit our pod (should succeed)
	if result := manager.Admit(&lifecycle.PodAdmitAttributes{Pod: podToAdmit}); !result.Admit {
		t.Errorf("Admit pod: %v, expected: %v, actual: %v", podToAdmit, true, result.Admit)
	}
}

// TestStaticCriticalPodsAreNotEvicted
func TestStaticCriticalPodsAreNotEvicted(t *testing.T) {
	podMaker := makePodWithMemoryStats
	summaryStatsMaker := makeMemoryStats
	podsToMake := []podToMake{
		{name: "critical", priority: scheduling.SystemCriticalPriority, requests: newResourceList("100m", "1Gi", ""), limits: newResourceList("100m", "1Gi", ""), memoryWorkingSet: "800Mi"},
	}
	pods := []*v1.Pod{}
	podStats := map[*v1.Pod]statsapi.PodStats{}
	for _, podToMake := range podsToMake {
		pod, podStat := podMaker(podToMake.name, podToMake.priority, podToMake.requests, podToMake.limits, podToMake.memoryWorkingSet)
		pods = append(pods, pod)
		podStats[pod] = podStat
	}

	pods[0].Annotations = map[string]string{
		kubelettypes.ConfigSourceAnnotationKey: kubelettypes.FileSource,
	}
	// Mark the pod as critical
	podPriority := scheduling.SystemCriticalPriority
	pods[0].Spec.Priority = &podPriority
	pods[0].Namespace = kubeapi.NamespaceSystem

	podToEvict := pods[0]
	activePodsFunc := func() []*v1.Pod {
		return pods
	}
	mirrorPodFunc := func(staticPod *v1.Pod) (*v1.Pod, bool) {
		mirrorPod := staticPod.DeepCopy()
		mirrorPod.Annotations[kubelettypes.ConfigSourceAnnotationKey] = kubelettypes.ApiserverSource
		return mirrorPod, true
	}

	fakeClock := clock.NewFakeClock(time.Now())
	podKiller := &mockPodKiller{}
	diskInfoProvider := &mockDiskInfoProvider{dedicatedImageFs: false}
	diskGC := &mockDiskGC{err: nil}
	nodeRef := &v1.ObjectReference{
		Kind: "Node", Name: "test", UID: types.UID("test"), Namespace: "",
	}

	config := Config{
		MaxPodGracePeriodSeconds: 5,
		PressureTransitionPeriod: time.Minute * 5,
		Thresholds: []evictionapi.Threshold{
			{
				Signal:   evictionapi.SignalMemoryAvailable,
				Operator: evictionapi.OpLessThan,
				Value: evictionapi.ThresholdValue{
					Quantity: quantityMustParse("1Gi"),
				},
			},
			{
				Signal:   evictionapi.SignalMemoryAvailable,
				Operator: evictionapi.OpLessThan,
				Value: evictionapi.ThresholdValue{
					Quantity: quantityMustParse("2Gi"),
				},
				GracePeriod: time.Minute * 2,
			},
		},
	}
	summaryProvider := &fakeSummaryProvider{result: summaryStatsMaker("2Gi", podStats)}
	manager := &managerImpl{
		clock:                        fakeClock,
		killPodFunc:                  podKiller.killPodNow,
		mirrorPodFunc:                mirrorPodFunc,
		imageGC:                      diskGC,
		containerGC:                  diskGC,
		config:                       config,
		recorder:                     &record.FakeRecorder{},
		summaryProvider:              summaryProvider,
		nodeRef:                      nodeRef,
		nodeConditionsLastObservedAt: nodeConditionsObservedAt{},
		thresholdsFirstObservedAt:    thresholdsObservedAt{},
	}

	fakeClock.Step(1 * time.Minute)
	summaryProvider.result = summaryStatsMaker("1500Mi", podStats)
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should have memory pressure
	if !manager.IsUnderMemoryPressure() {
		t.Errorf("Manager should report memory pressure since soft threshold was met")
	}

	// verify no pod was yet killed because there has not yet been enough time passed.
	if podKiller.pod != nil {
		t.Errorf("Manager should not have killed a pod yet, but killed: %v", podKiller.pod.Name)
	}

	// step forward in time pass the grace period
	fakeClock.Step(3 * time.Minute)
	summaryProvider.result = summaryStatsMaker("1500Mi", podStats)
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should have memory pressure
	if !manager.IsUnderMemoryPressure() {
		t.Errorf("Manager should report memory pressure since soft threshold was met")
	}

	// verify the right pod was killed with the right grace period.
	if podKiller.pod == podToEvict {
		t.Errorf("Manager chose to kill critical pod: %v, but should have ignored it", podKiller.pod.Name)
	}
	// reset state
	podKiller.pod = nil
	podKiller.gracePeriodOverride = nil

	// remove memory pressure
	fakeClock.Step(20 * time.Minute)
	summaryProvider.result = summaryStatsMaker("3Gi", podStats)
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should not have memory pressure
	if manager.IsUnderMemoryPressure() {
		t.Errorf("Manager should not report memory pressure")
	}

	pods[0].Annotations = map[string]string{
		kubelettypes.ConfigSourceAnnotationKey: kubelettypes.FileSource,
	}
	pods[0].Spec.Priority = nil
	pods[0].Namespace = kubeapi.NamespaceSystem

	// induce memory pressure!
	fakeClock.Step(1 * time.Minute)
	summaryProvider.result = summaryStatsMaker("500Mi", podStats)
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should have memory pressure
	if !manager.IsUnderMemoryPressure() {
		t.Errorf("Manager should report memory pressure")
	}
}

// TestAllocatableMemoryPressure
func TestAllocatableMemoryPressure(t *testing.T) {
	podMaker := makePodWithMemoryStats
	summaryStatsMaker := makeMemoryStats
	podsToMake := []podToMake{
		{name: "guaranteed-low-priority-high-usage", priority: lowPriority, requests: newResourceList("100m", "1Gi", ""), limits: newResourceList("100m", "1Gi", ""), memoryWorkingSet: "900Mi"},
		{name: "burstable-below-requests", priority: defaultPriority, requests: newResourceList("100m", "100Mi", ""), limits: newResourceList("200m", "1Gi", ""), memoryWorkingSet: "50Mi"},
		{name: "burstable-above-requests", priority: defaultPriority, requests: newResourceList("100m", "100Mi", ""), limits: newResourceList("200m", "1Gi", ""), memoryWorkingSet: "400Mi"},
		{name: "best-effort-high-priority-high-usage", priority: highPriority, requests: newResourceList("", "", ""), limits: newResourceList("", "", ""), memoryWorkingSet: "400Mi"},
		{name: "best-effort-low-priority-low-usage", priority: lowPriority, requests: newResourceList("", "", ""), limits: newResourceList("", "", ""), memoryWorkingSet: "100Mi"},
	}
	pods := []*v1.Pod{}
	podStats := map[*v1.Pod]statsapi.PodStats{}
	for _, podToMake := range podsToMake {
		pod, podStat := podMaker(podToMake.name, podToMake.priority, podToMake.requests, podToMake.limits, podToMake.memoryWorkingSet)
		pods = append(pods, pod)
		podStats[pod] = podStat
	}
	podToEvict := pods[4]
	activePodsFunc := func() []*v1.Pod {
		return pods
	}

	fakeClock := clock.NewFakeClock(time.Now())
	podKiller := &mockPodKiller{}
	diskInfoProvider := &mockDiskInfoProvider{dedicatedImageFs: false}
	diskGC := &mockDiskGC{err: nil}
	nodeRef := &v1.ObjectReference{Kind: "Node", Name: "test", UID: types.UID("test"), Namespace: ""}

	config := Config{
		MaxPodGracePeriodSeconds: 5,
		PressureTransitionPeriod: time.Minute * 5,
		Thresholds: []evictionapi.Threshold{
			{
				Signal:   evictionapi.SignalAllocatableMemoryAvailable,
				Operator: evictionapi.OpLessThan,
				Value: evictionapi.ThresholdValue{
					Quantity: quantityMustParse("1Gi"),
				},
			},
		},
	}
	summaryProvider := &fakeSummaryProvider{result: summaryStatsMaker("4Gi", podStats)}
	manager := &managerImpl{
		clock:                        fakeClock,
		killPodFunc:                  podKiller.killPodNow,
		imageGC:                      diskGC,
		containerGC:                  diskGC,
		config:                       config,
		recorder:                     &record.FakeRecorder{},
		summaryProvider:              summaryProvider,
		nodeRef:                      nodeRef,
		nodeConditionsLastObservedAt: nodeConditionsObservedAt{},
		thresholdsFirstObservedAt:    thresholdsObservedAt{},
	}

	// create a best effort pod to test admission
	bestEffortPodToAdmit, _ := podMaker("best-admit", defaultPriority, newResourceList("", "", ""), newResourceList("", "", ""), "0Gi")
	burstablePodToAdmit, _ := podMaker("burst-admit", defaultPriority, newResourceList("100m", "100Mi", ""), newResourceList("200m", "200Mi", ""), "0Gi")

	// synchronize
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should not have memory pressure
	if manager.IsUnderMemoryPressure() {
		t.Errorf("Manager should not report memory pressure")
	}

	// try to admit our pods (they should succeed)
	expected := []bool{true, true}
	for i, pod := range []*v1.Pod{bestEffortPodToAdmit, burstablePodToAdmit} {
		if result := manager.Admit(&lifecycle.PodAdmitAttributes{Pod: pod}); expected[i] != result.Admit {
			t.Errorf("Admit pod: %v, expected: %v, actual: %v", pod, expected[i], result.Admit)
		}
	}

	// induce memory pressure!
	fakeClock.Step(1 * time.Minute)
	pod, podStat := podMaker("guaranteed-high-2", defaultPriority, newResourceList("100m", "1Gi", ""), newResourceList("100m", "1Gi", ""), "1Gi")
	podStats[pod] = podStat
	summaryProvider.result = summaryStatsMaker("500Mi", podStats)
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should have memory pressure
	if !manager.IsUnderMemoryPressure() {
		t.Errorf("Manager should report memory pressure")
	}

	// check the right pod was killed
	if podKiller.pod != podToEvict {
		t.Errorf("Manager chose to kill pod: %v, but should have chosen %v", podKiller.pod.Name, podToEvict.Name)
	}
	observedGracePeriod := *podKiller.gracePeriodOverride
	if observedGracePeriod != int64(0) {
		t.Errorf("Manager chose to kill pod with incorrect grace period.  Expected: %d, actual: %d", 0, observedGracePeriod)
	}
	// reset state
	podKiller.pod = nil
	podKiller.gracePeriodOverride = nil

	// the best-effort pod should not admit, burstable should
	expected = []bool{false, true}
	for i, pod := range []*v1.Pod{bestEffortPodToAdmit, burstablePodToAdmit} {
		if result := manager.Admit(&lifecycle.PodAdmitAttributes{Pod: pod}); expected[i] != result.Admit {
			t.Errorf("Admit pod: %v, expected: %v, actual: %v", pod, expected[i], result.Admit)
		}
	}

	// reduce memory pressure
	fakeClock.Step(1 * time.Minute)
	for pod := range podStats {
		if pod.Name == "guaranteed-high-2" {
			delete(podStats, pod)
		}
	}
	summaryProvider.result = summaryStatsMaker("2Gi", podStats)
	podKiller.pod = nil // reset state
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should have memory pressure (because transition period not yet met)
	if !manager.IsUnderMemoryPressure() {
		t.Errorf("Manager should report memory pressure")
	}

	// no pod should have been killed
	if podKiller.pod != nil {
		t.Errorf("Manager chose to kill pod: %v when no pod should have been killed", podKiller.pod.Name)
	}

	// the best-effort pod should not admit, burstable should
	expected = []bool{false, true}
	for i, pod := range []*v1.Pod{bestEffortPodToAdmit, burstablePodToAdmit} {
		if result := manager.Admit(&lifecycle.PodAdmitAttributes{Pod: pod}); expected[i] != result.Admit {
			t.Errorf("Admit pod: %v, expected: %v, actual: %v", pod, expected[i], result.Admit)
		}
	}

	// move the clock past transition period to ensure that we stop reporting pressure
	fakeClock.Step(5 * time.Minute)
	summaryProvider.result = summaryStatsMaker("2Gi", podStats)
	podKiller.pod = nil // reset state
	manager.synchronize(diskInfoProvider, activePodsFunc)

	// we should not have memory pressure (because transition period met)
	if manager.IsUnderMemoryPressure() {
		t.Errorf("Manager should not report memory pressure")
	}

	// no pod should have been killed
	if podKiller.pod != nil {
		t.Errorf("Manager chose to kill pod: %v when no pod should have been killed", podKiller.pod.Name)
	}

	// all pods should admit now
	expected = []bool{true, true}
	for i, pod := range []*v1.Pod{bestEffortPodToAdmit, burstablePodToAdmit} {
		if result := manager.Admit(&lifecycle.PodAdmitAttributes{Pod: pod}); expected[i] != result.Admit {
			t.Errorf("Admit pod: %v, expected: %v, actual: %v", pod, expected[i], result.Admit)
		}
	}
}

func TestUpdateMemcgThreshold(t *testing.T) {
	activePodsFunc := func() []*v1.Pod {
		return []*v1.Pod{}
	}

	fakeClock := clock.NewFakeClock(time.Now())
	podKiller := &mockPodKiller{}
	diskInfoProvider := &mockDiskInfoProvider{dedicatedImageFs: false}
	diskGC := &mockDiskGC{err: nil}
	nodeRef := &v1.ObjectReference{Kind: "Node", Name: "test", UID: types.UID("test"), Namespace: ""}

	config := Config{
		MaxPodGracePeriodSeconds: 5,
		PressureTransitionPeriod: time.Minute * 5,
		Thresholds: []evictionapi.Threshold{
			{
				Signal:   evictionapi.SignalMemoryAvailable,
				Operator: evictionapi.OpLessThan,
				Value: evictionapi.ThresholdValue{
					Quantity: quantityMustParse("1Gi"),
				},
			},
		},
		PodCgroupRoot: "kubepods",
	}
	summaryProvider := &fakeSummaryProvider{result: makeMemoryStats("2Gi", map[*v1.Pod]statsapi.PodStats{})}

	thresholdNotifier := &MockThresholdNotifier{}
	thresholdNotifier.On("UpdateThreshold", summaryProvider.result).Return(nil).Twice()

	manager := &managerImpl{
		clock:                        fakeClock,
		killPodFunc:                  podKiller.killPodNow,
		imageGC:                      diskGC,
		containerGC:                  diskGC,
		config:                       config,
		recorder:                     &record.FakeRecorder{},
		summaryProvider:              summaryProvider,
		nodeRef:                      nodeRef,
		nodeConditionsLastObservedAt: nodeConditionsObservedAt{},
		thresholdsFirstObservedAt:    thresholdsObservedAt{},
		thresholdNotifiers:           []ThresholdNotifier{thresholdNotifier},
	}

	manager.synchronize(diskInfoProvider, activePodsFunc)
	// The UpdateThreshold method should have been called once, since this is the first run.
	thresholdNotifier.AssertNumberOfCalls(t, "UpdateThreshold", 1)

	manager.synchronize(diskInfoProvider, activePodsFunc)
	// The UpdateThreshold method should not have been called again, since not enough time has passed
	thresholdNotifier.AssertNumberOfCalls(t, "UpdateThreshold", 1)

	fakeClock.Step(2 * notifierRefreshInterval)
	manager.synchronize(diskInfoProvider, activePodsFunc)
	// The UpdateThreshold method should be called again since enough time has passed
	thresholdNotifier.AssertNumberOfCalls(t, "UpdateThreshold", 2)

	// new memory threshold notifier that returns an error
	thresholdNotifier = &MockThresholdNotifier{}
	thresholdNotifier.On("UpdateThreshold", summaryProvider.result).Return(fmt.Errorf("error updating threshold"))
	thresholdNotifier.On("Description").Return("mock thresholdNotifier").Once()
	manager.thresholdNotifiers = []ThresholdNotifier{thresholdNotifier}

	fakeClock.Step(2 * notifierRefreshInterval)
	manager.synchronize(diskInfoProvider, activePodsFunc)
	// The UpdateThreshold method should be called because at least notifierRefreshInterval time has passed.
	// The Description method should be called because UpdateThreshold returned an error
	thresholdNotifier.AssertNumberOfCalls(t, "UpdateThreshold", 1)
	thresholdNotifier.AssertNumberOfCalls(t, "Description", 1)
}

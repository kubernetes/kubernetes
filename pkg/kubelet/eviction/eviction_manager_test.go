/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/client/record"
	statsapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/stats"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
)

// mockPodKiller is used to testing which pod is killed
type mockPodKiller struct {
	pod                 *api.Pod
	status              api.PodStatus
	gracePeriodOverride *int64
}

// killPodNow records the pod that was killed
func (m *mockPodKiller) killPodNow(pod *api.Pod, status api.PodStatus, gracePeriodOverride *int64) error {
	m.pod = pod
	m.status = status
	m.gracePeriodOverride = gracePeriodOverride
	return nil
}

// TestMemoryPressure
func TestMemoryPressure(t *testing.T) {
	podMaker := func(name string, requests api.ResourceList, limits api.ResourceList, memoryWorkingSet string) (*api.Pod, statsapi.PodStats) {
		pod := newPod(name, []api.Container{
			newContainer(name, requests, api.ResourceList{}),
		})
		podStats := newPodMemoryStats(pod, resource.MustParse(memoryWorkingSet))
		return pod, podStats
	}
	summaryStatsMaker := func(nodeAvailableBytes string, podStats map[*api.Pod]statsapi.PodStats) *statsapi.Summary {
		val := resource.MustParse(nodeAvailableBytes)
		availableBytes := uint64(val.Value())
		result := &statsapi.Summary{
			Node: statsapi.NodeStats{
				Memory: &statsapi.MemoryStats{
					AvailableBytes: &availableBytes,
				},
			},
			Pods: []statsapi.PodStats{},
		}
		for _, podStat := range podStats {
			result.Pods = append(result.Pods, podStat)
		}
		return result
	}
	podsToMake := []struct {
		name             string
		requests         api.ResourceList
		limits           api.ResourceList
		memoryWorkingSet string
	}{
		{name: "best-effort-high", requests: newResourceList("", ""), limits: newResourceList("", ""), memoryWorkingSet: "500Mi"},
		{name: "best-effort-low", requests: newResourceList("", ""), limits: newResourceList("", ""), memoryWorkingSet: "300Mi"},
		{name: "burstable-high", requests: newResourceList("100m", "100Mi"), limits: newResourceList("200m", "1Gi"), memoryWorkingSet: "800Mi"},
		{name: "burstable-low", requests: newResourceList("100m", "100Mi"), limits: newResourceList("200m", "1Gi"), memoryWorkingSet: "300Mi"},
		{name: "guaranteed-high", requests: newResourceList("100m", "1Gi"), limits: newResourceList("100m", "1Gi"), memoryWorkingSet: "800Mi"},
		{name: "guaranteed-low", requests: newResourceList("100m", "1Gi"), limits: newResourceList("100m", "1Gi"), memoryWorkingSet: "200Mi"},
	}
	pods := []*api.Pod{}
	podStats := map[*api.Pod]statsapi.PodStats{}
	for _, podToMake := range podsToMake {
		pod, podStat := podMaker(podToMake.name, podToMake.requests, podToMake.limits, podToMake.memoryWorkingSet)
		pods = append(pods, pod)
		podStats[pod] = podStat
	}
	activePodsFunc := func() []*api.Pod {
		return pods
	}

	fakeClock := util.NewFakeClock(time.Now())
	podKiller := &mockPodKiller{}
	nodeRef := &api.ObjectReference{Kind: "Node", Name: "test", UID: types.UID("test"), Namespace: ""}

	config := Config{
		MaxPodGracePeriodSeconds: 5,
		PressureTransitionPeriod: time.Minute * 5,
		Thresholds: []Threshold{
			{
				Signal:   SignalMemoryAvailable,
				Operator: OpLessThan,
				Value:    quantityMustParse("1Gi"),
			},
			{
				Signal:      SignalMemoryAvailable,
				Operator:    OpLessThan,
				Value:       quantityMustParse("2Gi"),
				GracePeriod: time.Minute * 2,
			},
		},
	}
	summaryProvider := &fakeSummaryProvider{result: summaryStatsMaker("2Gi", podStats)}
	manager := &managerImpl{
		clock:           fakeClock,
		killPodFunc:     podKiller.killPodNow,
		config:          config,
		recorder:        &record.FakeRecorder{},
		summaryProvider: summaryProvider,
		nodeRef:         nodeRef,
		nodeConditionsLastObservedAt: nodeConditionsObservedAt{},
		thresholdsFirstObservedAt:    thresholdsObservedAt{},
	}

	// create a best effort pod to test admission
	bestEffortPodToAdmit, _ := podMaker("best-admit", newResourceList("", ""), newResourceList("", ""), "0Gi")
	burstablePodToAdmit, _ := podMaker("burst-admit", newResourceList("100m", "100Mi"), newResourceList("200m", "200Mi"), "0Gi")

	// synchronize
	manager.synchronize(activePodsFunc)

	// we should not have memory pressure
	if manager.IsUnderMemoryPressure() {
		t.Errorf("Manager should not report memory pressure")
	}

	// try to admit our pods (they should succeed)
	expected := []bool{true, true}
	for i, pod := range []*api.Pod{bestEffortPodToAdmit, burstablePodToAdmit} {
		if result := manager.Admit(&lifecycle.PodAdmitAttributes{Pod: pod}); expected[i] != result.Admit {
			t.Errorf("Admit pod: %v, expected: %v, actual: %v", pod, expected[i], result.Admit)
		}
	}

	// induce soft threshold
	fakeClock.Step(1 * time.Minute)
	summaryProvider.result = summaryStatsMaker("1500Mi", podStats)
	manager.synchronize(activePodsFunc)

	// we should have memory pressure
	if !manager.IsUnderMemoryPressure() {
		t.Errorf("Manager should report memory pressure since soft threshold was met")
	}

	// verify no pod was yet killed because there has not yet been enough time passed.
	if podKiller.pod != nil {
		t.Errorf("Manager should not have killed a pod yet, but killed: %v", podKiller.pod)
	}

	// step forward in time pass the grace period
	fakeClock.Step(3 * time.Minute)
	summaryProvider.result = summaryStatsMaker("1500Mi", podStats)
	manager.synchronize(activePodsFunc)

	// we should have memory pressure
	if !manager.IsUnderMemoryPressure() {
		t.Errorf("Manager should report memory pressure since soft threshold was met")
	}

	// verify the right pod was killed with the right grace period.
	if podKiller.pod != pods[0] {
		t.Errorf("Manager chose to kill pod: %v, but should have chosen %v", podKiller.pod, pods[0])
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
	manager.synchronize(activePodsFunc)

	// we should not have memory pressure
	if manager.IsUnderMemoryPressure() {
		t.Errorf("Manager should not report memory pressure")
	}

	// induce memory pressure!
	fakeClock.Step(1 * time.Minute)
	summaryProvider.result = summaryStatsMaker("500Mi", podStats)
	manager.synchronize(activePodsFunc)

	// we should have memory pressure
	if !manager.IsUnderMemoryPressure() {
		t.Errorf("Manager should report memory pressure")
	}

	// check the right pod was killed
	if podKiller.pod != pods[0] {
		t.Errorf("Manager chose to kill pod: %v, but should have chosen %v", podKiller.pod, pods[0])
	}
	observedGracePeriod = *podKiller.gracePeriodOverride
	if observedGracePeriod != int64(0) {
		t.Errorf("Manager chose to kill pod with incorrect grace period.  Expected: %d, actual: %d", 0, observedGracePeriod)
	}

	// the best-effort pod should not admit, burstable should
	expected = []bool{false, true}
	for i, pod := range []*api.Pod{bestEffortPodToAdmit, burstablePodToAdmit} {
		if result := manager.Admit(&lifecycle.PodAdmitAttributes{Pod: pod}); expected[i] != result.Admit {
			t.Errorf("Admit pod: %v, expected: %v, actual: %v", pod, expected[i], result.Admit)
		}
	}

	// reduce memory pressure
	fakeClock.Step(1 * time.Minute)
	summaryProvider.result = summaryStatsMaker("2Gi", podStats)
	podKiller.pod = nil // reset state
	manager.synchronize(activePodsFunc)

	// we should have memory pressure (because transition period not yet met)
	if !manager.IsUnderMemoryPressure() {
		t.Errorf("Manager should report memory pressure")
	}

	// no pod should have been killed
	if podKiller.pod != nil {
		t.Errorf("Manager chose to kill pod: %v when no pod should have been killed", podKiller.pod)
	}

	// the best-effort pod should not admit, burstable should
	expected = []bool{false, true}
	for i, pod := range []*api.Pod{bestEffortPodToAdmit, burstablePodToAdmit} {
		if result := manager.Admit(&lifecycle.PodAdmitAttributes{Pod: pod}); expected[i] != result.Admit {
			t.Errorf("Admit pod: %v, expected: %v, actual: %v", pod, expected[i], result.Admit)
		}
	}

	// move the clock past transition period to ensure that we stop reporting pressure
	fakeClock.Step(5 * time.Minute)
	summaryProvider.result = summaryStatsMaker("2Gi", podStats)
	podKiller.pod = nil // reset state
	manager.synchronize(activePodsFunc)

	// we should not have memory pressure (because transition period met)
	if manager.IsUnderMemoryPressure() {
		t.Errorf("Manager should not report memory pressure")
	}

	// no pod should have been killed
	if podKiller.pod != nil {
		t.Errorf("Manager chose to kill pod: %v when no pod should have been killed", podKiller.pod)
	}

	// all pods should admit now
	expected = []bool{true, true}
	for i, pod := range []*api.Pod{bestEffortPodToAdmit, burstablePodToAdmit} {
		if result := manager.Admit(&lifecycle.PodAdmitAttributes{Pod: pod}); expected[i] != result.Admit {
			t.Errorf("Admit pod: %v, expected: %v, actual: %v", pod, expected[i], result.Admit)
		}
	}
}

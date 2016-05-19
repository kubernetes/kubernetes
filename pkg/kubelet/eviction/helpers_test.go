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
	"fmt"
	"reflect"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	statsapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/stats"
	"k8s.io/kubernetes/pkg/quota"
	"k8s.io/kubernetes/pkg/types"
)

func quantityMustParse(value string) *resource.Quantity {
	q := resource.MustParse(value)
	return &q
}

func TestParseThresholdConfig(t *testing.T) {
	gracePeriod, _ := time.ParseDuration("30s")
	testCases := map[string]struct {
		evictionHard            string
		evictionSoft            string
		evictionSoftGracePeriod string
		expectErr               bool
		expectThresholds        []Threshold
	}{
		"no values": {
			evictionHard:            "",
			evictionSoft:            "",
			evictionSoftGracePeriod: "",
			expectErr:               false,
			expectThresholds:        []Threshold{},
		},
		"all flag values": {
			evictionHard:            "memory.available<150Mi",
			evictionSoft:            "memory.available<300Mi",
			evictionSoftGracePeriod: "memory.available=30s",
			expectErr:               false,
			expectThresholds: []Threshold{
				{
					Signal:   SignalMemoryAvailable,
					Operator: OpLessThan,
					Value:    quantityMustParse("150Mi"),
				},
				{
					Signal:      SignalMemoryAvailable,
					Operator:    OpLessThan,
					Value:       quantityMustParse("300Mi"),
					GracePeriod: gracePeriod,
				},
			},
		},
		"invalid-signal": {
			evictionHard:            "mem.available<150Mi",
			evictionSoft:            "",
			evictionSoftGracePeriod: "",
			expectErr:               true,
			expectThresholds:        []Threshold{},
		},
		"duplicate-signal": {
			evictionHard:            "memory.available<150Mi,memory.available<100Mi",
			evictionSoft:            "",
			evictionSoftGracePeriod: "",
			expectErr:               true,
			expectThresholds:        []Threshold{},
		},
		"valid-and-invalid-signal": {
			evictionHard:            "memory.available<150Mi,invalid.foo<150Mi",
			evictionSoft:            "",
			evictionSoftGracePeriod: "",
			expectErr:               true,
			expectThresholds:        []Threshold{},
		},
		"soft-no-grace-period": {
			evictionHard:            "",
			evictionSoft:            "memory.available<150Mi",
			evictionSoftGracePeriod: "",
			expectErr:               true,
			expectThresholds:        []Threshold{},
		},
		"soft-neg-grace-period": {
			evictionHard:            "",
			evictionSoft:            "memory.available<150Mi",
			evictionSoftGracePeriod: "memory.available=-30s",
			expectErr:               true,
			expectThresholds:        []Threshold{},
		},
	}
	for testName, testCase := range testCases {
		thresholds, err := ParseThresholdConfig(testCase.evictionHard, testCase.evictionSoft, testCase.evictionSoftGracePeriod)
		if testCase.expectErr != (err != nil) {
			t.Errorf("Err not as expected, test: %v, error expected: %v, actual: %v", testName, testCase.expectErr, err)
		}
		if !thresholdsEqual(testCase.expectThresholds, thresholds) {
			t.Errorf("thresholds not as expected, test: %v, expected: %v, actual: %v", testName, testCase.expectThresholds, thresholds)
		}
	}
}

func thresholdsEqual(expected []Threshold, actual []Threshold) bool {
	if len(expected) != len(actual) {
		return false
	}
	for _, aThreshold := range expected {
		equal := false
		for _, bThreshold := range actual {
			if thresholdEqual(aThreshold, bThreshold) {
				equal = true
			}
		}
		if !equal {
			return false
		}
	}
	for _, aThreshold := range actual {
		equal := false
		for _, bThreshold := range expected {
			if thresholdEqual(aThreshold, bThreshold) {
				equal = true
			}
		}
		if !equal {
			return false
		}
	}
	return true
}

func thresholdEqual(a Threshold, b Threshold) bool {
	return a.GracePeriod == b.GracePeriod &&
		a.Operator == b.Operator &&
		a.Signal == b.Signal &&
		a.Value.Cmp(*b.Value) == 0
}

// TestOrderedByQoS ensures we order BestEffort < Burstable < Guaranteed
func TestOrderedByQoS(t *testing.T) {
	bestEffort := newPod("best-effort", []api.Container{
		newContainer("best-effort", newResourceList("", ""), newResourceList("", "")),
	})
	burstable := newPod("burstable", []api.Container{
		newContainer("burstable", newResourceList("100m", "100Mi"), newResourceList("200m", "200Mi")),
	})
	guaranteed := newPod("guaranteed", []api.Container{
		newContainer("guaranteed", newResourceList("200m", "200Mi"), newResourceList("200m", "200Mi")),
	})

	pods := []*api.Pod{guaranteed, burstable, bestEffort}
	orderedBy(qos).Sort(pods)

	expected := []*api.Pod{bestEffort, burstable, guaranteed}
	for i := range expected {
		if pods[i] != expected[i] {
			t.Errorf("Expected pod: %s, but got: %s", expected[i].Name, pods[i].Name)
		}
	}
}

// TestOrderedByMemory ensures we order pods by greediest memory consumer relative to request.
func TestOrderedByMemory(t *testing.T) {
	pod1 := newPod("best-effort-high", []api.Container{
		newContainer("best-effort-high", newResourceList("", ""), newResourceList("", "")),
	})
	pod2 := newPod("best-effort-low", []api.Container{
		newContainer("best-effort-low", newResourceList("", ""), newResourceList("", "")),
	})
	pod3 := newPod("burstable-high", []api.Container{
		newContainer("burstable-high", newResourceList("100m", "100Mi"), newResourceList("200m", "1Gi")),
	})
	pod4 := newPod("burstable-low", []api.Container{
		newContainer("burstable-low", newResourceList("100m", "100Mi"), newResourceList("200m", "1Gi")),
	})
	pod5 := newPod("guaranteed-high", []api.Container{
		newContainer("guaranteed-high", newResourceList("100m", "1Gi"), newResourceList("100m", "1Gi")),
	})
	pod6 := newPod("guaranteed-low", []api.Container{
		newContainer("guaranteed-low", newResourceList("100m", "1Gi"), newResourceList("100m", "1Gi")),
	})
	stats := map[*api.Pod]statsapi.PodStats{
		pod1: newPodMemoryStats(pod1, resource.MustParse("500Mi")), // 500 relative to request
		pod2: newPodMemoryStats(pod2, resource.MustParse("300Mi")), // 300 relative to request
		pod3: newPodMemoryStats(pod3, resource.MustParse("800Mi")), // 700 relative to request
		pod4: newPodMemoryStats(pod4, resource.MustParse("300Mi")), // 200 relative to request
		pod5: newPodMemoryStats(pod5, resource.MustParse("800Mi")), // -200 relative to request
		pod6: newPodMemoryStats(pod6, resource.MustParse("200Mi")), // -800 relative to request
	}
	statsFn := func(pod *api.Pod) (statsapi.PodStats, bool) {
		result, found := stats[pod]
		return result, found
	}
	pods := []*api.Pod{pod1, pod2, pod3, pod4, pod5, pod6}
	orderedBy(memory(statsFn)).Sort(pods)
	expected := []*api.Pod{pod3, pod1, pod2, pod4, pod5, pod6}
	for i := range expected {
		if pods[i] != expected[i] {
			t.Errorf("Expected pod[%d]: %s, but got: %s", i, expected[i].Name, pods[i].Name)
		}
	}
}

// TestOrderedByQoSMemory ensures we order by qos and then memory consumption relative to request.
func TestOrderedByQoSMemory(t *testing.T) {
	pod1 := newPod("best-effort-high", []api.Container{
		newContainer("best-effort-high", newResourceList("", ""), newResourceList("", "")),
	})
	pod2 := newPod("best-effort-low", []api.Container{
		newContainer("best-effort-low", newResourceList("", ""), newResourceList("", "")),
	})
	pod3 := newPod("burstable-high", []api.Container{
		newContainer("burstable-high", newResourceList("100m", "100Mi"), newResourceList("200m", "1Gi")),
	})
	pod4 := newPod("burstable-low", []api.Container{
		newContainer("burstable-low", newResourceList("100m", "100Mi"), newResourceList("200m", "1Gi")),
	})
	pod5 := newPod("guaranteed-high", []api.Container{
		newContainer("guaranteed-high", newResourceList("100m", "1Gi"), newResourceList("100m", "1Gi")),
	})
	pod6 := newPod("guaranteed-low", []api.Container{
		newContainer("guaranteed-low", newResourceList("100m", "1Gi"), newResourceList("100m", "1Gi")),
	})
	stats := map[*api.Pod]statsapi.PodStats{
		pod1: newPodMemoryStats(pod1, resource.MustParse("500Mi")), // 500 relative to request
		pod2: newPodMemoryStats(pod2, resource.MustParse("50Mi")),  // 50 relative to request
		pod3: newPodMemoryStats(pod3, resource.MustParse("50Mi")),  // -50 relative to request
		pod4: newPodMemoryStats(pod4, resource.MustParse("300Mi")), // 200 relative to request
		pod5: newPodMemoryStats(pod5, resource.MustParse("800Mi")), // -200 relative to request
		pod6: newPodMemoryStats(pod6, resource.MustParse("200Mi")), // -800 relative to request
	}
	statsFn := func(pod *api.Pod) (statsapi.PodStats, bool) {
		result, found := stats[pod]
		return result, found
	}
	pods := []*api.Pod{pod1, pod2, pod3, pod4, pod5, pod6}
	expected := []*api.Pod{pod1, pod2, pod4, pod3, pod5, pod6}
	orderedBy(qos, memory(statsFn)).Sort(pods)
	for i := range expected {
		if pods[i] != expected[i] {
			t.Errorf("Expected pod[%d]: %s, but got: %s", i, expected[i].Name, pods[i].Name)
		}
	}
}

type fakeSummaryProvider struct {
	result *statsapi.Summary
}

func (f *fakeSummaryProvider) Get() (*statsapi.Summary, error) {
	return f.result, nil
}

// newPodStats returns a pod stat where each container is using the specified working set
// each pod must have a Name, UID, Namespace
func newPodStats(pod *api.Pod, containerWorkingSetBytes int64) statsapi.PodStats {
	result := statsapi.PodStats{
		PodRef: statsapi.PodReference{
			Name:      pod.Name,
			Namespace: pod.Namespace,
			UID:       string(pod.UID),
		},
	}
	val := uint64(containerWorkingSetBytes)
	for range pod.Spec.Containers {
		result.Containers = append(result.Containers, statsapi.ContainerStats{
			Memory: &statsapi.MemoryStats{
				WorkingSetBytes: &val,
			},
		})
	}
	return result
}

func TestMakeSignalObservations(t *testing.T) {
	podMaker := func(name, namespace, uid string, numContainers int) *api.Pod {
		pod := &api.Pod{}
		pod.Name = name
		pod.Namespace = namespace
		pod.UID = types.UID(uid)
		pod.Spec = api.PodSpec{}
		for i := 0; i < numContainers; i++ {
			pod.Spec.Containers = append(pod.Spec.Containers, api.Container{
				Name: fmt.Sprintf("ctr%v", i),
			})
		}
		return pod
	}
	nodeAvailableBytes := uint64(1024 * 1024 * 1024)
	fakeStats := &statsapi.Summary{
		Node: statsapi.NodeStats{
			Memory: &statsapi.MemoryStats{
				AvailableBytes: &nodeAvailableBytes,
			},
		},
		Pods: []statsapi.PodStats{},
	}
	provider := &fakeSummaryProvider{
		result: fakeStats,
	}
	pods := []*api.Pod{
		podMaker("pod1", "ns1", "uuid1", 1),
		podMaker("pod1", "ns2", "uuid2", 1),
		podMaker("pod3", "ns3", "uuid3", 1),
	}
	containerWorkingSetBytes := int64(1024 * 1024)
	for _, pod := range pods {
		fakeStats.Pods = append(fakeStats.Pods, newPodStats(pod, containerWorkingSetBytes))
	}
	actualObservations, statsFunc, err := makeSignalObservations(provider)
	if err != nil {
		t.Errorf("Unexpected err: %v", err)
	}
	quantity, found := actualObservations[SignalMemoryAvailable]
	if !found {
		t.Errorf("Expected available memory observation: %v", err)
	}
	if expectedBytes := int64(nodeAvailableBytes); quantity.Value() != expectedBytes {
		t.Errorf("Expected %v, actual: %v", expectedBytes, quantity.Value())
	}
	for _, pod := range pods {
		podStats, found := statsFunc(pod)
		if !found {
			t.Errorf("Pod stats were not found for pod %v", pod.UID)
		}
		for _, container := range podStats.Containers {
			actual := int64(*container.Memory.WorkingSetBytes)
			if containerWorkingSetBytes != actual {
				t.Errorf("Container working set expected %v, actual: %v", containerWorkingSetBytes, actual)
			}
		}
	}
}

func TestThresholdsMet(t *testing.T) {
	hardThreshold := Threshold{
		Signal:   SignalMemoryAvailable,
		Operator: OpLessThan,
		Value:    quantityMustParse("1Gi"),
	}
	testCases := map[string]struct {
		thresholds   []Threshold
		observations signalObservations
		result       []Threshold
	}{
		"empty": {
			thresholds:   []Threshold{},
			observations: signalObservations{},
			result:       []Threshold{},
		},
		"threshold-met": {
			thresholds: []Threshold{hardThreshold},
			observations: signalObservations{
				SignalMemoryAvailable: quantityMustParse("500Mi"),
			},
			result: []Threshold{hardThreshold},
		},
		"threshold-not-met": {
			thresholds: []Threshold{hardThreshold},
			observations: signalObservations{
				SignalMemoryAvailable: quantityMustParse("2Gi"),
			},
			result: []Threshold{},
		},
	}
	for testName, testCase := range testCases {
		actual := thresholdsMet(testCase.thresholds, testCase.observations)
		if !thresholdList(actual).Equal(thresholdList(testCase.result)) {
			t.Errorf("Test case: %s, expected: %v, actual: %v", testName, testCase.result, actual)
		}
	}
}

func TestThresholdsFirstObservedAt(t *testing.T) {
	hardThreshold := Threshold{
		Signal:   SignalMemoryAvailable,
		Operator: OpLessThan,
		Value:    quantityMustParse("1Gi"),
	}
	now := unversioned.Now()
	oldTime := unversioned.NewTime(now.Time.Add(-1 * time.Minute))
	testCases := map[string]struct {
		thresholds     []Threshold
		lastObservedAt thresholdsObservedAt
		now            time.Time
		result         thresholdsObservedAt
	}{
		"empty": {
			thresholds:     []Threshold{},
			lastObservedAt: thresholdsObservedAt{},
			now:            now.Time,
			result:         thresholdsObservedAt{},
		},
		"no-previous-observation": {
			thresholds:     []Threshold{hardThreshold},
			lastObservedAt: thresholdsObservedAt{},
			now:            now.Time,
			result: thresholdsObservedAt{
				hardThreshold: now.Time,
			},
		},
		"previous-observation": {
			thresholds: []Threshold{hardThreshold},
			lastObservedAt: thresholdsObservedAt{
				hardThreshold: oldTime.Time,
			},
			now: now.Time,
			result: thresholdsObservedAt{
				hardThreshold: oldTime.Time,
			},
		},
	}
	for testName, testCase := range testCases {
		actual := thresholdsFirstObservedAt(testCase.thresholds, testCase.lastObservedAt, testCase.now)
		if !reflect.DeepEqual(actual, testCase.result) {
			t.Errorf("Test case: %s, expected: %v, actual: %v", testName, testCase.result, actual)
		}
	}
}

func TestThresholdsMetGracePeriod(t *testing.T) {
	now := unversioned.Now()
	hardThreshold := Threshold{
		Signal:   SignalMemoryAvailable,
		Operator: OpLessThan,
		Value:    quantityMustParse("1Gi"),
	}
	softThreshold := Threshold{
		Signal:      SignalMemoryAvailable,
		Operator:    OpLessThan,
		Value:       quantityMustParse("2Gi"),
		GracePeriod: 1 * time.Minute,
	}
	oldTime := unversioned.NewTime(now.Time.Add(-2 * time.Minute))
	testCases := map[string]struct {
		observedAt thresholdsObservedAt
		now        time.Time
		result     []Threshold
	}{
		"empty": {
			observedAt: thresholdsObservedAt{},
			now:        now.Time,
			result:     []Threshold{},
		},
		"hard-threshold-met": {
			observedAt: thresholdsObservedAt{
				hardThreshold: now.Time,
			},
			now:    now.Time,
			result: []Threshold{hardThreshold},
		},
		"soft-threshold-not-met": {
			observedAt: thresholdsObservedAt{
				softThreshold: now.Time,
			},
			now:    now.Time,
			result: []Threshold{},
		},
		"soft-threshold-met": {
			observedAt: thresholdsObservedAt{
				softThreshold: oldTime.Time,
			},
			now:    now.Time,
			result: []Threshold{softThreshold},
		},
	}
	for testName, testCase := range testCases {
		actual := thresholdsMetGracePeriod(testCase.observedAt, now.Time)
		if !thresholdList(actual).Equal(thresholdList(testCase.result)) {
			t.Errorf("Test case: %s, expected: %v, actual: %v", testName, testCase.result, actual)
		}
	}
}

func TestNodeConditions(t *testing.T) {
	testCases := map[string]struct {
		inputs []Threshold
		result []api.NodeConditionType
	}{
		"empty-list": {
			inputs: []Threshold{},
			result: []api.NodeConditionType{},
		},
		"memory.available": {
			inputs: []Threshold{
				{Signal: SignalMemoryAvailable},
			},
			result: []api.NodeConditionType{api.NodeMemoryPressure},
		},
	}
	for testName, testCase := range testCases {
		actual := nodeConditions(testCase.inputs)
		if !nodeConditionList(actual).Equal(nodeConditionList(testCase.result)) {
			t.Errorf("Test case: %s, expected: %v, actual: %v", testName, testCase.result, actual)
		}
	}
}

func TestNodeConditionsLastObservedAt(t *testing.T) {
	now := unversioned.Now()
	oldTime := unversioned.NewTime(now.Time.Add(-1 * time.Minute))
	testCases := map[string]struct {
		nodeConditions []api.NodeConditionType
		lastObservedAt nodeConditionsObservedAt
		now            time.Time
		result         nodeConditionsObservedAt
	}{
		"no-previous-observation": {
			nodeConditions: []api.NodeConditionType{api.NodeMemoryPressure},
			lastObservedAt: nodeConditionsObservedAt{},
			now:            now.Time,
			result: nodeConditionsObservedAt{
				api.NodeMemoryPressure: now.Time,
			},
		},
		"previous-observation": {
			nodeConditions: []api.NodeConditionType{api.NodeMemoryPressure},
			lastObservedAt: nodeConditionsObservedAt{
				api.NodeMemoryPressure: oldTime.Time,
			},
			now: now.Time,
			result: nodeConditionsObservedAt{
				api.NodeMemoryPressure: now.Time,
			},
		},
		"old-observation": {
			nodeConditions: []api.NodeConditionType{},
			lastObservedAt: nodeConditionsObservedAt{
				api.NodeMemoryPressure: oldTime.Time,
			},
			now: now.Time,
			result: nodeConditionsObservedAt{
				api.NodeMemoryPressure: oldTime.Time,
			},
		},
	}
	for testName, testCase := range testCases {
		actual := nodeConditionsLastObservedAt(testCase.nodeConditions, testCase.lastObservedAt, testCase.now)
		if !reflect.DeepEqual(actual, testCase.result) {
			t.Errorf("Test case: %s, expected: %v, actual: %v", testName, testCase.result, actual)
		}
	}
}

func TestNodeConditionsObservedSince(t *testing.T) {
	now := unversioned.Now()
	observedTime := unversioned.NewTime(now.Time.Add(-1 * time.Minute))
	testCases := map[string]struct {
		observedAt nodeConditionsObservedAt
		period     time.Duration
		now        time.Time
		result     []api.NodeConditionType
	}{
		"in-period": {
			observedAt: nodeConditionsObservedAt{
				api.NodeMemoryPressure: observedTime.Time,
			},
			period: 2 * time.Minute,
			now:    now.Time,
			result: []api.NodeConditionType{api.NodeMemoryPressure},
		},
		"out-of-period": {
			observedAt: nodeConditionsObservedAt{
				api.NodeMemoryPressure: observedTime.Time,
			},
			period: 30 * time.Second,
			now:    now.Time,
			result: []api.NodeConditionType{},
		},
	}
	for testName, testCase := range testCases {
		actual := nodeConditionsObservedSince(testCase.observedAt, testCase.period, testCase.now)
		if !nodeConditionList(actual).Equal(nodeConditionList(testCase.result)) {
			t.Errorf("Test case: %s, expected: %v, actual: %v", testName, testCase.result, actual)
		}
	}
}

func TestHasNodeConditions(t *testing.T) {
	testCases := map[string]struct {
		inputs []api.NodeConditionType
		item   api.NodeConditionType
		result bool
	}{
		"has-condition": {
			inputs: []api.NodeConditionType{api.NodeReady, api.NodeOutOfDisk, api.NodeMemoryPressure},
			item:   api.NodeMemoryPressure,
			result: true,
		},
		"does-not-have-condition": {
			inputs: []api.NodeConditionType{api.NodeReady, api.NodeOutOfDisk},
			item:   api.NodeMemoryPressure,
			result: false,
		},
	}
	for testName, testCase := range testCases {
		if actual := hasNodeCondition(testCase.inputs, testCase.item); actual != testCase.result {
			t.Errorf("Test case: %s, expected: %v, actual: %v", testName, testCase.result, actual)
		}
	}
}

func TestReclaimResources(t *testing.T) {
	testCases := map[string]struct {
		inputs []Threshold
		result []api.ResourceName
	}{
		"memory.available": {
			inputs: []Threshold{
				{Signal: SignalMemoryAvailable},
			},
			result: []api.ResourceName{api.ResourceMemory},
		},
	}
	for testName, testCase := range testCases {
		actual := reclaimResources(testCase.inputs)
		actualSet := quota.ToSet(actual)
		expectedSet := quota.ToSet(testCase.result)
		if !actualSet.Equal(expectedSet) {
			t.Errorf("Test case: %s, expected: %v, actual: %v", testName, expectedSet, actualSet)
		}
	}
}

func newPodMemoryStats(pod *api.Pod, workingSet resource.Quantity) statsapi.PodStats {
	result := statsapi.PodStats{
		PodRef: statsapi.PodReference{
			Name: pod.Name, Namespace: pod.Namespace, UID: string(pod.UID),
		},
	}
	for range pod.Spec.Containers {
		workingSetBytes := uint64(workingSet.Value())
		result.Containers = append(result.Containers, statsapi.ContainerStats{
			Memory: &statsapi.MemoryStats{
				WorkingSetBytes: &workingSetBytes,
			},
		})
	}
	return result
}

func newResourceList(cpu, memory string) api.ResourceList {
	res := api.ResourceList{}
	if cpu != "" {
		res[api.ResourceCPU] = resource.MustParse(cpu)
	}
	if memory != "" {
		res[api.ResourceMemory] = resource.MustParse(memory)
	}
	return res
}

func newResourceRequirements(requests, limits api.ResourceList) api.ResourceRequirements {
	res := api.ResourceRequirements{}
	res.Requests = requests
	res.Limits = limits
	return res
}

func newContainer(name string, requests api.ResourceList, limits api.ResourceList) api.Container {
	return api.Container{
		Name:      name,
		Resources: newResourceRequirements(requests, limits),
	}
}

func newPod(name string, containers []api.Container) *api.Pod {
	return &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: api.PodSpec{
			Containers: containers,
		},
	}
}

// nodeConditionList is a simple alias to support equality checking independent of order
type nodeConditionList []api.NodeConditionType

// Equal adds the ability to check equality between two lists of node conditions.
func (s1 nodeConditionList) Equal(s2 nodeConditionList) bool {
	if len(s1) != len(s2) {
		return false
	}
	for _, item := range s1 {
		if !hasNodeCondition(s2, item) {
			return false
		}
	}
	return true
}

// thresholdList is a simple alias to support equality checking independent of order
type thresholdList []Threshold

// Equal adds the ability to check equality between two lists of node conditions.
func (s1 thresholdList) Equal(s2 thresholdList) bool {
	if len(s1) != len(s2) {
		return false
	}
	for _, item := range s1 {
		if !hasThreshold(s2, item) {
			return false
		}
	}
	return true
}

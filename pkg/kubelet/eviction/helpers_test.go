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
	statsapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/stats"
)

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
					Value:    resource.MustParse("150Mi"),
				},
				{
					Signal:      SignalMemoryAvailable,
					Operator:    OpLessThan,
					Value:       resource.MustParse("300Mi"),
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
		a.Value.Cmp(b.Value) == 0
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

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
	"reflect"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	statsapi "k8s.io/kubernetes/pkg/kubelet/apis/stats/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	evictionapi "k8s.io/kubernetes/pkg/kubelet/eviction/api"
	"k8s.io/kubernetes/pkg/quota"
)

func quantityMustParse(value string) *resource.Quantity {
	q := resource.MustParse(value)
	return &q
}

func TestParseThresholdConfig(t *testing.T) {
	gracePeriod, _ := time.ParseDuration("30s")
	testCases := map[string]struct {
		allocatableConfig       []string
		evictionHard            string
		evictionSoft            string
		evictionSoftGracePeriod string
		evictionMinReclaim      string
		expectErr               bool
		expectThresholds        []evictionapi.Threshold
	}{
		"no values": {
			allocatableConfig:       []string{},
			evictionHard:            "",
			evictionSoft:            "",
			evictionSoftGracePeriod: "",
			evictionMinReclaim:      "",
			expectErr:               false,
			expectThresholds:        []evictionapi.Threshold{},
		},
		"all flag values": {
			allocatableConfig:       []string{cm.NodeAllocatableEnforcementKey},
			evictionHard:            "memory.available<150Mi",
			evictionSoft:            "memory.available<300Mi",
			evictionSoftGracePeriod: "memory.available=30s",
			evictionMinReclaim:      "memory.available=0",
			expectErr:               false,
			expectThresholds: []evictionapi.Threshold{
				{
					Signal:   evictionapi.SignalAllocatableMemoryAvailable,
					Operator: evictionapi.OpLessThan,
					Value: evictionapi.ThresholdValue{
						Quantity: quantityMustParse("0"),
					},
					MinReclaim: &evictionapi.ThresholdValue{
						Quantity: quantityMustParse("0"),
					},
				},
				{
					Signal:   evictionapi.SignalAllocatableNodeFsAvailable,
					Operator: evictionapi.OpLessThan,
					Value: evictionapi.ThresholdValue{
						Quantity: quantityMustParse("0"),
					},
					MinReclaim: &evictionapi.ThresholdValue{
						Quantity: quantityMustParse("0"),
					},
				},
				{
					Signal:   evictionapi.SignalMemoryAvailable,
					Operator: evictionapi.OpLessThan,
					Value: evictionapi.ThresholdValue{
						Quantity: quantityMustParse("150Mi"),
					},
					MinReclaim: &evictionapi.ThresholdValue{
						Quantity: quantityMustParse("0"),
					},
				},
				{
					Signal:   evictionapi.SignalMemoryAvailable,
					Operator: evictionapi.OpLessThan,
					Value: evictionapi.ThresholdValue{
						Quantity: quantityMustParse("300Mi"),
					},
					GracePeriod: gracePeriod,
					MinReclaim: &evictionapi.ThresholdValue{
						Quantity: quantityMustParse("0"),
					},
				},
			},
		},
		"all flag values in percentages": {
			allocatableConfig:       []string{},
			evictionHard:            "memory.available<10%",
			evictionSoft:            "memory.available<30%",
			evictionSoftGracePeriod: "memory.available=30s",
			evictionMinReclaim:      "memory.available=5%",
			expectErr:               false,
			expectThresholds: []evictionapi.Threshold{
				{
					Signal:   evictionapi.SignalMemoryAvailable,
					Operator: evictionapi.OpLessThan,
					Value: evictionapi.ThresholdValue{
						Percentage: 0.1,
					},
					MinReclaim: &evictionapi.ThresholdValue{
						Percentage: 0.05,
					},
				},
				{
					Signal:   evictionapi.SignalMemoryAvailable,
					Operator: evictionapi.OpLessThan,
					Value: evictionapi.ThresholdValue{
						Percentage: 0.3,
					},
					GracePeriod: gracePeriod,
					MinReclaim: &evictionapi.ThresholdValue{
						Percentage: 0.05,
					},
				},
			},
		},
		"disk flag values": {
			allocatableConfig:       []string{},
			evictionHard:            "imagefs.available<150Mi,nodefs.available<100Mi",
			evictionSoft:            "imagefs.available<300Mi,nodefs.available<200Mi",
			evictionSoftGracePeriod: "imagefs.available=30s,nodefs.available=30s",
			evictionMinReclaim:      "imagefs.available=2Gi,nodefs.available=1Gi",
			expectErr:               false,
			expectThresholds: []evictionapi.Threshold{
				{
					Signal:   evictionapi.SignalImageFsAvailable,
					Operator: evictionapi.OpLessThan,
					Value: evictionapi.ThresholdValue{
						Quantity: quantityMustParse("150Mi"),
					},
					MinReclaim: &evictionapi.ThresholdValue{
						Quantity: quantityMustParse("2Gi"),
					},
				},
				{
					Signal:   evictionapi.SignalNodeFsAvailable,
					Operator: evictionapi.OpLessThan,
					Value: evictionapi.ThresholdValue{
						Quantity: quantityMustParse("100Mi"),
					},
					MinReclaim: &evictionapi.ThresholdValue{
						Quantity: quantityMustParse("1Gi"),
					},
				},
				{
					Signal:   evictionapi.SignalImageFsAvailable,
					Operator: evictionapi.OpLessThan,
					Value: evictionapi.ThresholdValue{
						Quantity: quantityMustParse("300Mi"),
					},
					GracePeriod: gracePeriod,
					MinReclaim: &evictionapi.ThresholdValue{
						Quantity: quantityMustParse("2Gi"),
					},
				},
				{
					Signal:   evictionapi.SignalNodeFsAvailable,
					Operator: evictionapi.OpLessThan,
					Value: evictionapi.ThresholdValue{
						Quantity: quantityMustParse("200Mi"),
					},
					GracePeriod: gracePeriod,
					MinReclaim: &evictionapi.ThresholdValue{
						Quantity: quantityMustParse("1Gi"),
					},
				},
			},
		},
		"disk flag values in percentages": {
			allocatableConfig:       []string{},
			evictionHard:            "imagefs.available<15%,nodefs.available<10.5%",
			evictionSoft:            "imagefs.available<30%,nodefs.available<20.5%",
			evictionSoftGracePeriod: "imagefs.available=30s,nodefs.available=30s",
			evictionMinReclaim:      "imagefs.available=10%,nodefs.available=5%",
			expectErr:               false,
			expectThresholds: []evictionapi.Threshold{
				{
					Signal:   evictionapi.SignalImageFsAvailable,
					Operator: evictionapi.OpLessThan,
					Value: evictionapi.ThresholdValue{
						Percentage: 0.15,
					},
					MinReclaim: &evictionapi.ThresholdValue{
						Percentage: 0.1,
					},
				},
				{
					Signal:   evictionapi.SignalNodeFsAvailable,
					Operator: evictionapi.OpLessThan,
					Value: evictionapi.ThresholdValue{
						Percentage: 0.105,
					},
					MinReclaim: &evictionapi.ThresholdValue{
						Percentage: 0.05,
					},
				},
				{
					Signal:   evictionapi.SignalImageFsAvailable,
					Operator: evictionapi.OpLessThan,
					Value: evictionapi.ThresholdValue{
						Percentage: 0.3,
					},
					GracePeriod: gracePeriod,
					MinReclaim: &evictionapi.ThresholdValue{
						Percentage: 0.1,
					},
				},
				{
					Signal:   evictionapi.SignalNodeFsAvailable,
					Operator: evictionapi.OpLessThan,
					Value: evictionapi.ThresholdValue{
						Percentage: 0.205,
					},
					GracePeriod: gracePeriod,
					MinReclaim: &evictionapi.ThresholdValue{
						Percentage: 0.05,
					},
				},
			},
		},
		"inode flag values": {
			allocatableConfig:       []string{},
			evictionHard:            "imagefs.inodesFree<150Mi,nodefs.inodesFree<100Mi",
			evictionSoft:            "imagefs.inodesFree<300Mi,nodefs.inodesFree<200Mi",
			evictionSoftGracePeriod: "imagefs.inodesFree=30s,nodefs.inodesFree=30s",
			evictionMinReclaim:      "imagefs.inodesFree=2Gi,nodefs.inodesFree=1Gi",
			expectErr:               false,
			expectThresholds: []evictionapi.Threshold{
				{
					Signal:   evictionapi.SignalImageFsInodesFree,
					Operator: evictionapi.OpLessThan,
					Value: evictionapi.ThresholdValue{
						Quantity: quantityMustParse("150Mi"),
					},
					MinReclaim: &evictionapi.ThresholdValue{
						Quantity: quantityMustParse("2Gi"),
					},
				},
				{
					Signal:   evictionapi.SignalNodeFsInodesFree,
					Operator: evictionapi.OpLessThan,
					Value: evictionapi.ThresholdValue{
						Quantity: quantityMustParse("100Mi"),
					},
					MinReclaim: &evictionapi.ThresholdValue{
						Quantity: quantityMustParse("1Gi"),
					},
				},
				{
					Signal:   evictionapi.SignalImageFsInodesFree,
					Operator: evictionapi.OpLessThan,
					Value: evictionapi.ThresholdValue{
						Quantity: quantityMustParse("300Mi"),
					},
					GracePeriod: gracePeriod,
					MinReclaim: &evictionapi.ThresholdValue{
						Quantity: quantityMustParse("2Gi"),
					},
				},
				{
					Signal:   evictionapi.SignalNodeFsInodesFree,
					Operator: evictionapi.OpLessThan,
					Value: evictionapi.ThresholdValue{
						Quantity: quantityMustParse("200Mi"),
					},
					GracePeriod: gracePeriod,
					MinReclaim: &evictionapi.ThresholdValue{
						Quantity: quantityMustParse("1Gi"),
					},
				},
			},
		},
		"invalid-signal": {
			allocatableConfig:       []string{},
			evictionHard:            "mem.available<150Mi",
			evictionSoft:            "",
			evictionSoftGracePeriod: "",
			evictionMinReclaim:      "",
			expectErr:               true,
			expectThresholds:        []evictionapi.Threshold{},
		},
		"hard-signal-negative": {
			allocatableConfig:       []string{},
			evictionHard:            "memory.available<-150Mi",
			evictionSoft:            "",
			evictionSoftGracePeriod: "",
			evictionMinReclaim:      "",
			expectErr:               true,
			expectThresholds:        []evictionapi.Threshold{},
		},
		"hard-signal-negative-percentage": {
			allocatableConfig:       []string{},
			evictionHard:            "memory.available<-15%",
			evictionSoft:            "",
			evictionSoftGracePeriod: "",
			evictionMinReclaim:      "",
			expectErr:               true,
			expectThresholds:        []evictionapi.Threshold{},
		},
		"soft-signal-negative": {
			allocatableConfig:       []string{},
			evictionHard:            "",
			evictionSoft:            "memory.available<-150Mi",
			evictionSoftGracePeriod: "",
			evictionMinReclaim:      "",
			expectErr:               true,
			expectThresholds:        []evictionapi.Threshold{},
		},
		"duplicate-signal": {
			allocatableConfig:       []string{},
			evictionHard:            "memory.available<150Mi,memory.available<100Mi",
			evictionSoft:            "",
			evictionSoftGracePeriod: "",
			evictionMinReclaim:      "",
			expectErr:               true,
			expectThresholds:        []evictionapi.Threshold{},
		},
		"valid-and-invalid-signal": {
			allocatableConfig:       []string{},
			evictionHard:            "memory.available<150Mi,invalid.foo<150Mi",
			evictionSoft:            "",
			evictionSoftGracePeriod: "",
			evictionMinReclaim:      "",
			expectErr:               true,
			expectThresholds:        []evictionapi.Threshold{},
		},
		"soft-no-grace-period": {
			allocatableConfig:       []string{},
			evictionHard:            "",
			evictionSoft:            "memory.available<150Mi",
			evictionSoftGracePeriod: "",
			evictionMinReclaim:      "",
			expectErr:               true,
			expectThresholds:        []evictionapi.Threshold{},
		},
		"soft-neg-grace-period": {
			allocatableConfig:       []string{},
			evictionHard:            "",
			evictionSoft:            "memory.available<150Mi",
			evictionSoftGracePeriod: "memory.available=-30s",
			evictionMinReclaim:      "",
			expectErr:               true,
			expectThresholds:        []evictionapi.Threshold{},
		},
		"neg-reclaim": {
			allocatableConfig:       []string{},
			evictionHard:            "",
			evictionSoft:            "",
			evictionSoftGracePeriod: "",
			evictionMinReclaim:      "memory.available=-300Mi",
			expectErr:               true,
			expectThresholds:        []evictionapi.Threshold{},
		},
		"duplicate-reclaim": {
			allocatableConfig:       []string{},
			evictionHard:            "",
			evictionSoft:            "",
			evictionSoftGracePeriod: "",
			evictionMinReclaim:      "memory.available=-300Mi,memory.available=-100Mi",
			expectErr:               true,
			expectThresholds:        []evictionapi.Threshold{},
		},
	}
	for testName, testCase := range testCases {
		thresholds, err := ParseThresholdConfig(testCase.allocatableConfig, testCase.evictionHard, testCase.evictionSoft, testCase.evictionSoftGracePeriod, testCase.evictionMinReclaim)
		if testCase.expectErr != (err != nil) {
			t.Errorf("Err not as expected, test: %v, error expected: %v, actual: %v", testName, testCase.expectErr, err)
		}
		if !thresholdsEqual(testCase.expectThresholds, thresholds) {
			t.Errorf("thresholds not as expected, test: %v, expected: %v, actual: %v", testName, testCase.expectThresholds, thresholds)
		}
	}
}

func thresholdsEqual(expected []evictionapi.Threshold, actual []evictionapi.Threshold) bool {
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

func thresholdEqual(a evictionapi.Threshold, b evictionapi.Threshold) bool {
	return a.GracePeriod == b.GracePeriod &&
		a.Operator == b.Operator &&
		a.Signal == b.Signal &&
		compareThresholdValue(*a.MinReclaim, *b.MinReclaim) &&
		compareThresholdValue(a.Value, b.Value)
}

// TestOrderedByQoS ensures we order BestEffort < Burstable < Guaranteed
func TestOrderedByQoS(t *testing.T) {
	bestEffort := newPod("best-effort", []v1.Container{
		newContainer("best-effort", newResourceList("", ""), newResourceList("", "")),
	}, nil)
	burstable := newPod("burstable", []v1.Container{
		newContainer("burstable", newResourceList("100m", "100Mi"), newResourceList("200m", "200Mi")),
	}, nil)
	guaranteed := newPod("guaranteed", []v1.Container{
		newContainer("guaranteed", newResourceList("200m", "200Mi"), newResourceList("200m", "200Mi")),
	}, nil)

	pods := []*v1.Pod{guaranteed, burstable, bestEffort}
	orderedBy(qosComparator).Sort(pods)

	expected := []*v1.Pod{bestEffort, burstable, guaranteed}
	for i := range expected {
		if pods[i] != expected[i] {
			t.Errorf("Expected pod: %s, but got: %s", expected[i].Name, pods[i].Name)
		}
	}
}

func TestOrderedbyDisk(t *testing.T) {
	testOrderedByResource(t, resourceDisk, newPodDiskStats)
}

func TestOrderedbyInodes(t *testing.T) {
	testOrderedByResource(t, resourceInodes, newPodInodeStats)
}

// testOrderedByDisk ensures we order pods by greediest resource consumer
func testOrderedByResource(t *testing.T, orderedByResource v1.ResourceName,
	newPodStatsFunc func(pod *v1.Pod, rootFsUsed, logsUsed, perLocalVolumeUsed resource.Quantity) statsapi.PodStats) {
	pod1 := newPod("best-effort-high", []v1.Container{
		newContainer("best-effort-high", newResourceList("", ""), newResourceList("", "")),
	}, []v1.Volume{
		newVolume("local-volume", v1.VolumeSource{
			EmptyDir: &v1.EmptyDirVolumeSource{},
		}),
	})
	pod2 := newPod("best-effort-low", []v1.Container{
		newContainer("best-effort-low", newResourceList("", ""), newResourceList("", "")),
	}, []v1.Volume{
		newVolume("local-volume", v1.VolumeSource{
			EmptyDir: &v1.EmptyDirVolumeSource{},
		}),
	})
	pod3 := newPod("burstable-high", []v1.Container{
		newContainer("burstable-high", newResourceList("100m", "100Mi"), newResourceList("200m", "1Gi")),
	}, []v1.Volume{
		newVolume("local-volume", v1.VolumeSource{
			EmptyDir: &v1.EmptyDirVolumeSource{},
		}),
	})
	pod4 := newPod("burstable-low", []v1.Container{
		newContainer("burstable-low", newResourceList("100m", "100Mi"), newResourceList("200m", "1Gi")),
	}, []v1.Volume{
		newVolume("local-volume", v1.VolumeSource{
			EmptyDir: &v1.EmptyDirVolumeSource{},
		}),
	})
	pod5 := newPod("guaranteed-high", []v1.Container{
		newContainer("guaranteed-high", newResourceList("100m", "1Gi"), newResourceList("100m", "1Gi")),
	}, []v1.Volume{
		newVolume("local-volume", v1.VolumeSource{
			EmptyDir: &v1.EmptyDirVolumeSource{},
		}),
	})
	pod6 := newPod("guaranteed-low", []v1.Container{
		newContainer("guaranteed-low", newResourceList("100m", "1Gi"), newResourceList("100m", "1Gi")),
	}, []v1.Volume{
		newVolume("local-volume", v1.VolumeSource{
			EmptyDir: &v1.EmptyDirVolumeSource{},
		}),
	})
	stats := map[*v1.Pod]statsapi.PodStats{
		pod1: newPodStatsFunc(pod1, resource.MustParse("50Mi"), resource.MustParse("100Mi"), resource.MustParse("50Mi")),  // 200Mi
		pod2: newPodStatsFunc(pod2, resource.MustParse("100Mi"), resource.MustParse("150Mi"), resource.MustParse("50Mi")), // 300Mi
		pod3: newPodStatsFunc(pod3, resource.MustParse("200Mi"), resource.MustParse("150Mi"), resource.MustParse("50Mi")), // 400Mi
		pod4: newPodStatsFunc(pod4, resource.MustParse("300Mi"), resource.MustParse("100Mi"), resource.MustParse("50Mi")), // 450Mi
		pod5: newPodStatsFunc(pod5, resource.MustParse("400Mi"), resource.MustParse("100Mi"), resource.MustParse("50Mi")), // 550Mi
		pod6: newPodStatsFunc(pod6, resource.MustParse("500Mi"), resource.MustParse("100Mi"), resource.MustParse("50Mi")), // 650Mi
	}
	statsFn := func(pod *v1.Pod) (statsapi.PodStats, bool) {
		result, found := stats[pod]
		return result, found
	}
	pods := []*v1.Pod{pod1, pod2, pod3, pod4, pod5, pod6}
	orderedBy(disk(statsFn, []fsStatsType{fsStatsRoot, fsStatsLogs, fsStatsLocalVolumeSource}, orderedByResource)).Sort(pods)
	expected := []*v1.Pod{pod6, pod5, pod4, pod3, pod2, pod1}
	for i := range expected {
		if pods[i] != expected[i] {
			t.Errorf("Expected pod[%d]: %s, but got: %s", i, expected[i].Name, pods[i].Name)
		}
	}
}

func TestOrderedbyQoSDisk(t *testing.T) {
	testOrderedByQoSResource(t, resourceDisk, newPodDiskStats)
}

func TestOrderedbyQoSInodes(t *testing.T) {
	testOrderedByQoSResource(t, resourceInodes, newPodInodeStats)
}

// testOrderedByQoSDisk ensures we order pods by qos and then greediest resource consumer
func testOrderedByQoSResource(t *testing.T, orderedByResource v1.ResourceName,
	newPodStatsFunc func(pod *v1.Pod, rootFsUsed, logsUsed, perLocalVolumeUsed resource.Quantity) statsapi.PodStats) {
	pod1 := newPod("best-effort-high", []v1.Container{
		newContainer("best-effort-high", newResourceList("", ""), newResourceList("", "")),
	}, []v1.Volume{
		newVolume("local-volume", v1.VolumeSource{
			EmptyDir: &v1.EmptyDirVolumeSource{},
		}),
	})
	pod2 := newPod("best-effort-low", []v1.Container{
		newContainer("best-effort-low", newResourceList("", ""), newResourceList("", "")),
	}, []v1.Volume{
		newVolume("local-volume", v1.VolumeSource{
			EmptyDir: &v1.EmptyDirVolumeSource{},
		}),
	})
	pod3 := newPod("burstable-high", []v1.Container{
		newContainer("burstable-high", newResourceList("100m", "100Mi"), newResourceList("200m", "1Gi")),
	}, []v1.Volume{
		newVolume("local-volume", v1.VolumeSource{
			EmptyDir: &v1.EmptyDirVolumeSource{},
		}),
	})
	pod4 := newPod("burstable-low", []v1.Container{
		newContainer("burstable-low", newResourceList("100m", "100Mi"), newResourceList("200m", "1Gi")),
	}, []v1.Volume{
		newVolume("local-volume", v1.VolumeSource{
			EmptyDir: &v1.EmptyDirVolumeSource{},
		}),
	})
	pod5 := newPod("guaranteed-high", []v1.Container{
		newContainer("guaranteed-high", newResourceList("100m", "1Gi"), newResourceList("100m", "1Gi")),
	}, []v1.Volume{
		newVolume("local-volume", v1.VolumeSource{
			EmptyDir: &v1.EmptyDirVolumeSource{},
		}),
	})
	pod6 := newPod("guaranteed-low", []v1.Container{
		newContainer("guaranteed-low", newResourceList("100m", "1Gi"), newResourceList("100m", "1Gi")),
	}, []v1.Volume{
		newVolume("local-volume", v1.VolumeSource{
			EmptyDir: &v1.EmptyDirVolumeSource{},
		}),
	})
	stats := map[*v1.Pod]statsapi.PodStats{
		pod1: newPodStatsFunc(pod1, resource.MustParse("50Mi"), resource.MustParse("100Mi"), resource.MustParse("50Mi")),  // 200Mi
		pod2: newPodStatsFunc(pod2, resource.MustParse("100Mi"), resource.MustParse("150Mi"), resource.MustParse("50Mi")), // 300Mi
		pod3: newPodStatsFunc(pod3, resource.MustParse("200Mi"), resource.MustParse("150Mi"), resource.MustParse("50Mi")), // 400Mi
		pod4: newPodStatsFunc(pod4, resource.MustParse("300Mi"), resource.MustParse("100Mi"), resource.MustParse("50Mi")), // 450Mi
		pod5: newPodStatsFunc(pod5, resource.MustParse("400Mi"), resource.MustParse("100Mi"), resource.MustParse("50Mi")), // 550Mi
		pod6: newPodStatsFunc(pod6, resource.MustParse("500Mi"), resource.MustParse("100Mi"), resource.MustParse("50Mi")), // 650Mi
	}
	statsFn := func(pod *v1.Pod) (statsapi.PodStats, bool) {
		result, found := stats[pod]
		return result, found
	}
	pods := []*v1.Pod{pod1, pod2, pod3, pod4, pod5, pod6}
	orderedBy(qosComparator, disk(statsFn, []fsStatsType{fsStatsRoot, fsStatsLogs, fsStatsLocalVolumeSource}, orderedByResource)).Sort(pods)
	expected := []*v1.Pod{pod2, pod1, pod4, pod3, pod6, pod5}
	for i := range expected {
		if pods[i] != expected[i] {
			t.Errorf("Expected pod[%d]: %s, but got: %s", i, expected[i].Name, pods[i].Name)
		}
	}
}

// TestOrderedByMemory ensures we order pods by greediest memory consumer relative to request.
func TestOrderedByMemory(t *testing.T) {
	pod1 := newPod("best-effort-high", []v1.Container{
		newContainer("best-effort-high", newResourceList("", ""), newResourceList("", "")),
	}, nil)
	pod2 := newPod("best-effort-low", []v1.Container{
		newContainer("best-effort-low", newResourceList("", ""), newResourceList("", "")),
	}, nil)
	pod3 := newPod("burstable-high", []v1.Container{
		newContainer("burstable-high", newResourceList("100m", "100Mi"), newResourceList("200m", "1Gi")),
	}, nil)
	pod4 := newPod("burstable-low", []v1.Container{
		newContainer("burstable-low", newResourceList("100m", "100Mi"), newResourceList("200m", "1Gi")),
	}, nil)
	pod5 := newPod("guaranteed-high", []v1.Container{
		newContainer("guaranteed-high", newResourceList("100m", "1Gi"), newResourceList("100m", "1Gi")),
	}, nil)
	pod6 := newPod("guaranteed-low", []v1.Container{
		newContainer("guaranteed-low", newResourceList("100m", "1Gi"), newResourceList("100m", "1Gi")),
	}, nil)
	stats := map[*v1.Pod]statsapi.PodStats{
		pod1: newPodMemoryStats(pod1, resource.MustParse("500Mi")), // 500 relative to request
		pod2: newPodMemoryStats(pod2, resource.MustParse("300Mi")), // 300 relative to request
		pod3: newPodMemoryStats(pod3, resource.MustParse("800Mi")), // 700 relative to request
		pod4: newPodMemoryStats(pod4, resource.MustParse("300Mi")), // 200 relative to request
		pod5: newPodMemoryStats(pod5, resource.MustParse("800Mi")), // -200 relative to request
		pod6: newPodMemoryStats(pod6, resource.MustParse("200Mi")), // -800 relative to request
	}
	statsFn := func(pod *v1.Pod) (statsapi.PodStats, bool) {
		result, found := stats[pod]
		return result, found
	}
	pods := []*v1.Pod{pod1, pod2, pod3, pod4, pod5, pod6}
	orderedBy(memory(statsFn)).Sort(pods)
	expected := []*v1.Pod{pod3, pod1, pod2, pod4, pod5, pod6}
	for i := range expected {
		if pods[i] != expected[i] {
			t.Errorf("Expected pod[%d]: %s, but got: %s", i, expected[i].Name, pods[i].Name)
		}
	}
}

// TestOrderedByQoSMemory ensures we order by qosComparator and then memory consumption relative to request.
func TestOrderedByQoSMemory(t *testing.T) {
	pod1 := newPod("best-effort-high", []v1.Container{
		newContainer("best-effort-high", newResourceList("", ""), newResourceList("", "")),
	}, nil)
	pod2 := newPod("best-effort-low", []v1.Container{
		newContainer("best-effort-low", newResourceList("", ""), newResourceList("", "")),
	}, nil)
	pod3 := newPod("burstable-high", []v1.Container{
		newContainer("burstable-high", newResourceList("100m", "100Mi"), newResourceList("200m", "1Gi")),
	}, nil)
	pod4 := newPod("burstable-low", []v1.Container{
		newContainer("burstable-low", newResourceList("100m", "100Mi"), newResourceList("200m", "1Gi")),
	}, nil)
	pod5 := newPod("guaranteed-high", []v1.Container{
		newContainer("guaranteed-high", newResourceList("100m", "1Gi"), newResourceList("100m", "1Gi")),
	}, nil)
	pod6 := newPod("guaranteed-low", []v1.Container{
		newContainer("guaranteed-low", newResourceList("100m", "1Gi"), newResourceList("100m", "1Gi")),
	}, nil)
	stats := map[*v1.Pod]statsapi.PodStats{
		pod1: newPodMemoryStats(pod1, resource.MustParse("500Mi")), // 500 relative to request
		pod2: newPodMemoryStats(pod2, resource.MustParse("50Mi")),  // 50 relative to request
		pod3: newPodMemoryStats(pod3, resource.MustParse("50Mi")),  // -50 relative to request
		pod4: newPodMemoryStats(pod4, resource.MustParse("300Mi")), // 200 relative to request
		pod5: newPodMemoryStats(pod5, resource.MustParse("800Mi")), // -200 relative to request
		pod6: newPodMemoryStats(pod6, resource.MustParse("200Mi")), // -800 relative to request
	}
	statsFn := func(pod *v1.Pod) (statsapi.PodStats, bool) {
		result, found := stats[pod]
		return result, found
	}
	pods := []*v1.Pod{pod1, pod2, pod3, pod4, pod5, pod6}
	expected := []*v1.Pod{pod1, pod2, pod4, pod3, pod5, pod6}
	orderedBy(qosComparator, memory(statsFn)).Sort(pods)
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
func newPodStats(pod *v1.Pod, containerWorkingSetBytes int64) statsapi.PodStats {
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
	podMaker := func(name, namespace, uid string, numContainers int) *v1.Pod {
		pod := &v1.Pod{}
		pod.Name = name
		pod.Namespace = namespace
		pod.UID = types.UID(uid)
		pod.Spec = v1.PodSpec{}
		for i := 0; i < numContainers; i++ {
			pod.Spec.Containers = append(pod.Spec.Containers, v1.Container{
				Name: fmt.Sprintf("ctr%v", i),
			})
		}
		return pod
	}
	nodeAvailableBytes := uint64(1024 * 1024 * 1024)
	nodeWorkingSetBytes := uint64(1024 * 1024 * 1024)
	allocatableMemoryCapacity := uint64(5 * 1024 * 1024 * 1024)
	imageFsAvailableBytes := uint64(1024 * 1024)
	imageFsCapacityBytes := uint64(1024 * 1024 * 2)
	nodeFsAvailableBytes := uint64(1024)
	nodeFsCapacityBytes := uint64(1024 * 2)
	imageFsInodesFree := uint64(1024)
	imageFsInodes := uint64(1024 * 1024)
	nodeFsInodesFree := uint64(1024)
	nodeFsInodes := uint64(1024 * 1024)
	fakeStats := &statsapi.Summary{
		Node: statsapi.NodeStats{
			Memory: &statsapi.MemoryStats{
				AvailableBytes:  &nodeAvailableBytes,
				WorkingSetBytes: &nodeWorkingSetBytes,
			},
			Runtime: &statsapi.RuntimeStats{
				ImageFs: &statsapi.FsStats{
					AvailableBytes: &imageFsAvailableBytes,
					CapacityBytes:  &imageFsCapacityBytes,
					InodesFree:     &imageFsInodesFree,
					Inodes:         &imageFsInodes,
				},
			},
			Fs: &statsapi.FsStats{
				AvailableBytes: &nodeFsAvailableBytes,
				CapacityBytes:  &nodeFsCapacityBytes,
				InodesFree:     &nodeFsInodesFree,
				Inodes:         &nodeFsInodes,
			},
		},
		Pods: []statsapi.PodStats{},
	}
	provider := &fakeSummaryProvider{
		result: fakeStats,
	}
	pods := []*v1.Pod{
		podMaker("pod1", "ns1", "uuid1", 1),
		podMaker("pod1", "ns2", "uuid2", 1),
		podMaker("pod3", "ns3", "uuid3", 1),
	}
	containerWorkingSetBytes := int64(1024 * 1024 * 1024)
	for _, pod := range pods {
		fakeStats.Pods = append(fakeStats.Pods, newPodStats(pod, containerWorkingSetBytes))
	}
	res := quantityMustParse("5Gi")
	nodeProvider := newMockNodeProvider(v1.ResourceList{v1.ResourceMemory: *res})
	// Allocatable thresholds are always 100%.  Verify that Threshold == Capacity.
	if res.CmpInt64(int64(allocatableMemoryCapacity)) != 0 {
		t.Errorf("Expected Threshold %v to be equal to value %v", res.Value(), allocatableMemoryCapacity)
	}
	actualObservations, statsFunc, err := makeSignalObservations(provider, nodeProvider, pods, false)
	if err != nil {
		t.Errorf("Unexpected err: %v", err)
	}
	allocatableMemQuantity, found := actualObservations[evictionapi.SignalAllocatableMemoryAvailable]
	if !found {
		t.Errorf("Expected allocatable memory observation, but didnt find one")
	}
	if allocatableMemQuantity.available.Value() != 2*containerWorkingSetBytes {
		t.Errorf("Expected %v, actual: %v", containerWorkingSetBytes, allocatableMemQuantity.available.Value())
	}
	if expectedBytes := int64(allocatableMemoryCapacity); allocatableMemQuantity.capacity.Value() != expectedBytes {
		t.Errorf("Expected %v, actual: %v", expectedBytes, allocatableMemQuantity.capacity.Value())
	}
	memQuantity, found := actualObservations[evictionapi.SignalMemoryAvailable]
	if !found {
		t.Errorf("Expected available memory observation: %v", err)
	}
	if expectedBytes := int64(nodeAvailableBytes); memQuantity.available.Value() != expectedBytes {
		t.Errorf("Expected %v, actual: %v", expectedBytes, memQuantity.available.Value())
	}
	if expectedBytes := int64(nodeWorkingSetBytes + nodeAvailableBytes); memQuantity.capacity.Value() != expectedBytes {
		t.Errorf("Expected %v, actual: %v", expectedBytes, memQuantity.capacity.Value())
	}
	nodeFsQuantity, found := actualObservations[evictionapi.SignalNodeFsAvailable]
	if !found {
		t.Errorf("Expected available nodefs observation: %v", err)
	}
	if expectedBytes := int64(nodeFsAvailableBytes); nodeFsQuantity.available.Value() != expectedBytes {
		t.Errorf("Expected %v, actual: %v", expectedBytes, nodeFsQuantity.available.Value())
	}
	if expectedBytes := int64(nodeFsCapacityBytes); nodeFsQuantity.capacity.Value() != expectedBytes {
		t.Errorf("Expected %v, actual: %v", expectedBytes, nodeFsQuantity.capacity.Value())
	}
	nodeFsInodesQuantity, found := actualObservations[evictionapi.SignalNodeFsInodesFree]
	if !found {
		t.Errorf("Expected inodes free nodefs observation: %v", err)
	}
	if expected := int64(nodeFsInodesFree); nodeFsInodesQuantity.available.Value() != expected {
		t.Errorf("Expected %v, actual: %v", expected, nodeFsInodesQuantity.available.Value())
	}
	if expected := int64(nodeFsInodes); nodeFsInodesQuantity.capacity.Value() != expected {
		t.Errorf("Expected %v, actual: %v", expected, nodeFsInodesQuantity.capacity.Value())
	}
	imageFsQuantity, found := actualObservations[evictionapi.SignalImageFsAvailable]
	if !found {
		t.Errorf("Expected available imagefs observation: %v", err)
	}
	if expectedBytes := int64(imageFsAvailableBytes); imageFsQuantity.available.Value() != expectedBytes {
		t.Errorf("Expected %v, actual: %v", expectedBytes, imageFsQuantity.available.Value())
	}
	if expectedBytes := int64(imageFsCapacityBytes); imageFsQuantity.capacity.Value() != expectedBytes {
		t.Errorf("Expected %v, actual: %v", expectedBytes, imageFsQuantity.capacity.Value())
	}
	imageFsInodesQuantity, found := actualObservations[evictionapi.SignalImageFsInodesFree]
	if !found {
		t.Errorf("Expected inodes free imagefs observation: %v", err)
	}
	if expected := int64(imageFsInodesFree); imageFsInodesQuantity.available.Value() != expected {
		t.Errorf("Expected %v, actual: %v", expected, imageFsInodesQuantity.available.Value())
	}
	if expected := int64(imageFsInodes); imageFsInodesQuantity.capacity.Value() != expected {
		t.Errorf("Expected %v, actual: %v", expected, imageFsInodesQuantity.capacity.Value())
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
	hardThreshold := evictionapi.Threshold{
		Signal:   evictionapi.SignalMemoryAvailable,
		Operator: evictionapi.OpLessThan,
		Value: evictionapi.ThresholdValue{
			Quantity: quantityMustParse("1Gi"),
		},
		MinReclaim: &evictionapi.ThresholdValue{
			Quantity: quantityMustParse("500Mi"),
		},
	}
	testCases := map[string]struct {
		enforceMinReclaim bool
		thresholds        []evictionapi.Threshold
		observations      signalObservations
		result            []evictionapi.Threshold
	}{
		"empty": {
			enforceMinReclaim: false,
			thresholds:        []evictionapi.Threshold{},
			observations:      signalObservations{},
			result:            []evictionapi.Threshold{},
		},
		"threshold-met-memory": {
			enforceMinReclaim: false,
			thresholds:        []evictionapi.Threshold{hardThreshold},
			observations: signalObservations{
				evictionapi.SignalMemoryAvailable: signalObservation{
					available: quantityMustParse("500Mi"),
				},
			},
			result: []evictionapi.Threshold{hardThreshold},
		},
		"threshold-not-met": {
			enforceMinReclaim: false,
			thresholds:        []evictionapi.Threshold{hardThreshold},
			observations: signalObservations{
				evictionapi.SignalMemoryAvailable: signalObservation{
					available: quantityMustParse("2Gi"),
				},
			},
			result: []evictionapi.Threshold{},
		},
		"threshold-met-with-min-reclaim": {
			enforceMinReclaim: true,
			thresholds:        []evictionapi.Threshold{hardThreshold},
			observations: signalObservations{
				evictionapi.SignalMemoryAvailable: signalObservation{
					available: quantityMustParse("1.05Gi"),
				},
			},
			result: []evictionapi.Threshold{hardThreshold},
		},
		"threshold-not-met-with-min-reclaim": {
			enforceMinReclaim: true,
			thresholds:        []evictionapi.Threshold{hardThreshold},
			observations: signalObservations{
				evictionapi.SignalMemoryAvailable: signalObservation{
					available: quantityMustParse("2Gi"),
				},
			},
			result: []evictionapi.Threshold{},
		},
	}
	for testName, testCase := range testCases {
		actual := thresholdsMet(testCase.thresholds, testCase.observations, testCase.enforceMinReclaim)
		if !thresholdList(actual).Equal(thresholdList(testCase.result)) {
			t.Errorf("Test case: %s, expected: %v, actual: %v", testName, testCase.result, actual)
		}
	}
}

func TestThresholdsUpdatedStats(t *testing.T) {
	updatedThreshold := evictionapi.Threshold{
		Signal: evictionapi.SignalMemoryAvailable,
	}
	locationUTC, err := time.LoadLocation("UTC")
	if err != nil {
		t.Error(err)
		return
	}
	testCases := map[string]struct {
		thresholds   []evictionapi.Threshold
		observations signalObservations
		last         signalObservations
		result       []evictionapi.Threshold
	}{
		"empty": {
			thresholds:   []evictionapi.Threshold{},
			observations: signalObservations{},
			last:         signalObservations{},
			result:       []evictionapi.Threshold{},
		},
		"no-time": {
			thresholds: []evictionapi.Threshold{updatedThreshold},
			observations: signalObservations{
				evictionapi.SignalMemoryAvailable: signalObservation{},
			},
			last:   signalObservations{},
			result: []evictionapi.Threshold{updatedThreshold},
		},
		"no-last-observation": {
			thresholds: []evictionapi.Threshold{updatedThreshold},
			observations: signalObservations{
				evictionapi.SignalMemoryAvailable: signalObservation{
					time: metav1.Date(2016, 1, 1, 0, 0, 0, 0, locationUTC),
				},
			},
			last:   signalObservations{},
			result: []evictionapi.Threshold{updatedThreshold},
		},
		"time-machine": {
			thresholds: []evictionapi.Threshold{updatedThreshold},
			observations: signalObservations{
				evictionapi.SignalMemoryAvailable: signalObservation{
					time: metav1.Date(2016, 1, 1, 0, 0, 0, 0, locationUTC),
				},
			},
			last: signalObservations{
				evictionapi.SignalMemoryAvailable: signalObservation{
					time: metav1.Date(2016, 1, 1, 0, 1, 0, 0, locationUTC),
				},
			},
			result: []evictionapi.Threshold{},
		},
		"same-observation": {
			thresholds: []evictionapi.Threshold{updatedThreshold},
			observations: signalObservations{
				evictionapi.SignalMemoryAvailable: signalObservation{
					time: metav1.Date(2016, 1, 1, 0, 0, 0, 0, locationUTC),
				},
			},
			last: signalObservations{
				evictionapi.SignalMemoryAvailable: signalObservation{
					time: metav1.Date(2016, 1, 1, 0, 0, 0, 0, locationUTC),
				},
			},
			result: []evictionapi.Threshold{},
		},
		"new-observation": {
			thresholds: []evictionapi.Threshold{updatedThreshold},
			observations: signalObservations{
				evictionapi.SignalMemoryAvailable: signalObservation{
					time: metav1.Date(2016, 1, 1, 0, 1, 0, 0, locationUTC),
				},
			},
			last: signalObservations{
				evictionapi.SignalMemoryAvailable: signalObservation{
					time: metav1.Date(2016, 1, 1, 0, 0, 0, 0, locationUTC),
				},
			},
			result: []evictionapi.Threshold{updatedThreshold},
		},
	}
	for testName, testCase := range testCases {
		actual := thresholdsUpdatedStats(testCase.thresholds, testCase.observations, testCase.last)
		if !thresholdList(actual).Equal(thresholdList(testCase.result)) {
			t.Errorf("Test case: %s, expected: %v, actual: %v", testName, testCase.result, actual)
		}
	}
}

func TestPercentageThresholdsMet(t *testing.T) {
	specificThresholds := []evictionapi.Threshold{
		{
			Signal:   evictionapi.SignalMemoryAvailable,
			Operator: evictionapi.OpLessThan,
			Value: evictionapi.ThresholdValue{
				Percentage: 0.2,
			},
			MinReclaim: &evictionapi.ThresholdValue{
				Percentage: 0.05,
			},
		},
		{
			Signal:   evictionapi.SignalNodeFsAvailable,
			Operator: evictionapi.OpLessThan,
			Value: evictionapi.ThresholdValue{
				Percentage: 0.3,
			},
		},
	}

	testCases := map[string]struct {
		enforceMinRelaim bool
		thresholds       []evictionapi.Threshold
		observations     signalObservations
		result           []evictionapi.Threshold
	}{
		"BothMet": {
			enforceMinRelaim: false,
			thresholds:       specificThresholds,
			observations: signalObservations{
				evictionapi.SignalMemoryAvailable: signalObservation{
					available: quantityMustParse("100Mi"),
					capacity:  quantityMustParse("1000Mi"),
				},
				evictionapi.SignalNodeFsAvailable: signalObservation{
					available: quantityMustParse("100Gi"),
					capacity:  quantityMustParse("1000Gi"),
				},
			},
			result: specificThresholds,
		},
		"NoneMet": {
			enforceMinRelaim: false,
			thresholds:       specificThresholds,
			observations: signalObservations{
				evictionapi.SignalMemoryAvailable: signalObservation{
					available: quantityMustParse("300Mi"),
					capacity:  quantityMustParse("1000Mi"),
				},
				evictionapi.SignalNodeFsAvailable: signalObservation{
					available: quantityMustParse("400Gi"),
					capacity:  quantityMustParse("1000Gi"),
				},
			},
			result: []evictionapi.Threshold{},
		},
		"DiskMet": {
			enforceMinRelaim: false,
			thresholds:       specificThresholds,
			observations: signalObservations{
				evictionapi.SignalMemoryAvailable: signalObservation{
					available: quantityMustParse("300Mi"),
					capacity:  quantityMustParse("1000Mi"),
				},
				evictionapi.SignalNodeFsAvailable: signalObservation{
					available: quantityMustParse("100Gi"),
					capacity:  quantityMustParse("1000Gi"),
				},
			},
			result: []evictionapi.Threshold{specificThresholds[1]},
		},
		"MemoryMet": {
			enforceMinRelaim: false,
			thresholds:       specificThresholds,
			observations: signalObservations{
				evictionapi.SignalMemoryAvailable: signalObservation{
					available: quantityMustParse("100Mi"),
					capacity:  quantityMustParse("1000Mi"),
				},
				evictionapi.SignalNodeFsAvailable: signalObservation{
					available: quantityMustParse("400Gi"),
					capacity:  quantityMustParse("1000Gi"),
				},
			},
			result: []evictionapi.Threshold{specificThresholds[0]},
		},
		"MemoryMetWithMinReclaim": {
			enforceMinRelaim: true,
			thresholds:       specificThresholds,
			observations: signalObservations{
				evictionapi.SignalMemoryAvailable: signalObservation{
					available: quantityMustParse("225Mi"),
					capacity:  quantityMustParse("1000Mi"),
				},
			},
			result: []evictionapi.Threshold{specificThresholds[0]},
		},
		"MemoryNotMetWithMinReclaim": {
			enforceMinRelaim: true,
			thresholds:       specificThresholds,
			observations: signalObservations{
				evictionapi.SignalMemoryAvailable: signalObservation{
					available: quantityMustParse("300Mi"),
					capacity:  quantityMustParse("1000Mi"),
				},
			},
			result: []evictionapi.Threshold{},
		},
	}
	for testName, testCase := range testCases {
		actual := thresholdsMet(testCase.thresholds, testCase.observations, testCase.enforceMinRelaim)
		if !thresholdList(actual).Equal(thresholdList(testCase.result)) {
			t.Errorf("Test case: %s, expected: %v, actual: %v", testName, testCase.result, actual)
		}
	}
}

func TestThresholdsFirstObservedAt(t *testing.T) {
	hardThreshold := evictionapi.Threshold{
		Signal:   evictionapi.SignalMemoryAvailable,
		Operator: evictionapi.OpLessThan,
		Value: evictionapi.ThresholdValue{
			Quantity: quantityMustParse("1Gi"),
		},
	}
	now := metav1.Now()
	oldTime := metav1.NewTime(now.Time.Add(-1 * time.Minute))
	testCases := map[string]struct {
		thresholds     []evictionapi.Threshold
		lastObservedAt thresholdsObservedAt
		now            time.Time
		result         thresholdsObservedAt
	}{
		"empty": {
			thresholds:     []evictionapi.Threshold{},
			lastObservedAt: thresholdsObservedAt{},
			now:            now.Time,
			result:         thresholdsObservedAt{},
		},
		"no-previous-observation": {
			thresholds:     []evictionapi.Threshold{hardThreshold},
			lastObservedAt: thresholdsObservedAt{},
			now:            now.Time,
			result: thresholdsObservedAt{
				hardThreshold: now.Time,
			},
		},
		"previous-observation": {
			thresholds: []evictionapi.Threshold{hardThreshold},
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
	now := metav1.Now()
	hardThreshold := evictionapi.Threshold{
		Signal:   evictionapi.SignalMemoryAvailable,
		Operator: evictionapi.OpLessThan,
		Value: evictionapi.ThresholdValue{
			Quantity: quantityMustParse("1Gi"),
		},
	}
	softThreshold := evictionapi.Threshold{
		Signal:   evictionapi.SignalMemoryAvailable,
		Operator: evictionapi.OpLessThan,
		Value: evictionapi.ThresholdValue{
			Quantity: quantityMustParse("2Gi"),
		},
		GracePeriod: 1 * time.Minute,
	}
	oldTime := metav1.NewTime(now.Time.Add(-2 * time.Minute))
	testCases := map[string]struct {
		observedAt thresholdsObservedAt
		now        time.Time
		result     []evictionapi.Threshold
	}{
		"empty": {
			observedAt: thresholdsObservedAt{},
			now:        now.Time,
			result:     []evictionapi.Threshold{},
		},
		"hard-threshold-met": {
			observedAt: thresholdsObservedAt{
				hardThreshold: now.Time,
			},
			now:    now.Time,
			result: []evictionapi.Threshold{hardThreshold},
		},
		"soft-threshold-not-met": {
			observedAt: thresholdsObservedAt{
				softThreshold: now.Time,
			},
			now:    now.Time,
			result: []evictionapi.Threshold{},
		},
		"soft-threshold-met": {
			observedAt: thresholdsObservedAt{
				softThreshold: oldTime.Time,
			},
			now:    now.Time,
			result: []evictionapi.Threshold{softThreshold},
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
		inputs []evictionapi.Threshold
		result []v1.NodeConditionType
	}{
		"empty-list": {
			inputs: []evictionapi.Threshold{},
			result: []v1.NodeConditionType{},
		},
		"memory.available": {
			inputs: []evictionapi.Threshold{
				{Signal: evictionapi.SignalMemoryAvailable},
			},
			result: []v1.NodeConditionType{v1.NodeMemoryPressure},
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
	now := metav1.Now()
	oldTime := metav1.NewTime(now.Time.Add(-1 * time.Minute))
	testCases := map[string]struct {
		nodeConditions []v1.NodeConditionType
		lastObservedAt nodeConditionsObservedAt
		now            time.Time
		result         nodeConditionsObservedAt
	}{
		"no-previous-observation": {
			nodeConditions: []v1.NodeConditionType{v1.NodeMemoryPressure},
			lastObservedAt: nodeConditionsObservedAt{},
			now:            now.Time,
			result: nodeConditionsObservedAt{
				v1.NodeMemoryPressure: now.Time,
			},
		},
		"previous-observation": {
			nodeConditions: []v1.NodeConditionType{v1.NodeMemoryPressure},
			lastObservedAt: nodeConditionsObservedAt{
				v1.NodeMemoryPressure: oldTime.Time,
			},
			now: now.Time,
			result: nodeConditionsObservedAt{
				v1.NodeMemoryPressure: now.Time,
			},
		},
		"old-observation": {
			nodeConditions: []v1.NodeConditionType{},
			lastObservedAt: nodeConditionsObservedAt{
				v1.NodeMemoryPressure: oldTime.Time,
			},
			now: now.Time,
			result: nodeConditionsObservedAt{
				v1.NodeMemoryPressure: oldTime.Time,
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
	now := metav1.Now()
	observedTime := metav1.NewTime(now.Time.Add(-1 * time.Minute))
	testCases := map[string]struct {
		observedAt nodeConditionsObservedAt
		period     time.Duration
		now        time.Time
		result     []v1.NodeConditionType
	}{
		"in-period": {
			observedAt: nodeConditionsObservedAt{
				v1.NodeMemoryPressure: observedTime.Time,
			},
			period: 2 * time.Minute,
			now:    now.Time,
			result: []v1.NodeConditionType{v1.NodeMemoryPressure},
		},
		"out-of-period": {
			observedAt: nodeConditionsObservedAt{
				v1.NodeMemoryPressure: observedTime.Time,
			},
			period: 30 * time.Second,
			now:    now.Time,
			result: []v1.NodeConditionType{},
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
		inputs []v1.NodeConditionType
		item   v1.NodeConditionType
		result bool
	}{
		"has-condition": {
			inputs: []v1.NodeConditionType{v1.NodeReady, v1.NodeOutOfDisk, v1.NodeMemoryPressure},
			item:   v1.NodeMemoryPressure,
			result: true,
		},
		"does-not-have-condition": {
			inputs: []v1.NodeConditionType{v1.NodeReady, v1.NodeOutOfDisk},
			item:   v1.NodeMemoryPressure,
			result: false,
		},
	}
	for testName, testCase := range testCases {
		if actual := hasNodeCondition(testCase.inputs, testCase.item); actual != testCase.result {
			t.Errorf("Test case: %s, expected: %v, actual: %v", testName, testCase.result, actual)
		}
	}
}

func TestGetStarvedResources(t *testing.T) {
	testCases := map[string]struct {
		inputs []evictionapi.Threshold
		result []v1.ResourceName
	}{
		"memory.available": {
			inputs: []evictionapi.Threshold{
				{Signal: evictionapi.SignalMemoryAvailable},
			},
			result: []v1.ResourceName{v1.ResourceMemory},
		},
		"imagefs.available": {
			inputs: []evictionapi.Threshold{
				{Signal: evictionapi.SignalImageFsAvailable},
			},
			result: []v1.ResourceName{resourceImageFs},
		},
		"nodefs.available": {
			inputs: []evictionapi.Threshold{
				{Signal: evictionapi.SignalNodeFsAvailable},
			},
			result: []v1.ResourceName{resourceNodeFs},
		},
	}
	var internalResourceNames = func(in []v1.ResourceName) []api.ResourceName {
		var out []api.ResourceName
		for _, name := range in {
			out = append(out, api.ResourceName(name))
		}
		return out
	}
	for testName, testCase := range testCases {
		actual := getStarvedResources(testCase.inputs)
		actualSet := quota.ToSet(internalResourceNames(actual))
		expectedSet := quota.ToSet(internalResourceNames(testCase.result))
		if !actualSet.Equal(expectedSet) {
			t.Errorf("Test case: %s, expected: %v, actual: %v", testName, expectedSet, actualSet)
		}
	}
}

func testParsePercentage(t *testing.T) {
	testCases := map[string]struct {
		hasError bool
		value    float32
	}{
		"blah": {
			hasError: true,
		},
		"25.5%": {
			value: 0.255,
		},
		"foo%": {
			hasError: true,
		},
		"12%345": {
			hasError: true,
		},
	}
	for input, expected := range testCases {
		value, err := parsePercentage(input)
		if (err != nil) != expected.hasError {
			t.Errorf("Test case: %s, expected: %v, actual: %v", input, expected.hasError, err != nil)
		}
		if value != expected.value {
			t.Errorf("Test case: %s, expected: %v, actual: %v", input, expected.value, value)
		}
	}
}

func testCompareThresholdValue(t *testing.T) {
	testCases := []struct {
		a, b  evictionapi.ThresholdValue
		equal bool
	}{
		{
			a: evictionapi.ThresholdValue{
				Quantity: resource.NewQuantity(123, resource.BinarySI),
			},
			b: evictionapi.ThresholdValue{
				Quantity: resource.NewQuantity(123, resource.BinarySI),
			},
			equal: true,
		},
		{
			a: evictionapi.ThresholdValue{
				Quantity: resource.NewQuantity(123, resource.BinarySI),
			},
			b: evictionapi.ThresholdValue{
				Quantity: resource.NewQuantity(456, resource.BinarySI),
			},
			equal: false,
		},
		{
			a: evictionapi.ThresholdValue{
				Quantity: resource.NewQuantity(123, resource.BinarySI),
			},
			b: evictionapi.ThresholdValue{
				Percentage: 0.1,
			},
			equal: false,
		},
		{
			a: evictionapi.ThresholdValue{
				Percentage: 0.1,
			},
			b: evictionapi.ThresholdValue{
				Percentage: 0.1,
			},
			equal: true,
		},
		{
			a: evictionapi.ThresholdValue{
				Percentage: 0.2,
			},
			b: evictionapi.ThresholdValue{
				Percentage: 0.1,
			},
			equal: false,
		},
	}

	for i, testCase := range testCases {
		if compareThresholdValue(testCase.a, testCase.b) != testCase.equal ||
			compareThresholdValue(testCase.b, testCase.a) != testCase.equal {
			t.Errorf("Test case: %v failed", i)
		}
	}
}

// newPodInodeStats returns stats with specified usage amounts.
func newPodInodeStats(pod *v1.Pod, rootFsInodesUsed, logsInodesUsed, perLocalVolumeInodesUsed resource.Quantity) statsapi.PodStats {
	result := statsapi.PodStats{
		PodRef: statsapi.PodReference{
			Name: pod.Name, Namespace: pod.Namespace, UID: string(pod.UID),
		},
	}
	rootFsUsed := uint64(rootFsInodesUsed.Value())
	logsUsed := uint64(logsInodesUsed.Value())
	for range pod.Spec.Containers {
		result.Containers = append(result.Containers, statsapi.ContainerStats{
			Rootfs: &statsapi.FsStats{
				InodesUsed: &rootFsUsed,
			},
			Logs: &statsapi.FsStats{
				InodesUsed: &logsUsed,
			},
		})
	}

	perLocalVolumeUsed := uint64(perLocalVolumeInodesUsed.Value())
	for _, volumeName := range localVolumeNames(pod) {
		result.VolumeStats = append(result.VolumeStats, statsapi.VolumeStats{
			Name: volumeName,
			FsStats: statsapi.FsStats{
				InodesUsed: &perLocalVolumeUsed,
			},
		})
	}
	return result
}

// newPodDiskStats returns stats with specified usage amounts.
func newPodDiskStats(pod *v1.Pod, rootFsUsed, logsUsed, perLocalVolumeUsed resource.Quantity) statsapi.PodStats {
	result := statsapi.PodStats{
		PodRef: statsapi.PodReference{
			Name: pod.Name, Namespace: pod.Namespace, UID: string(pod.UID),
		},
	}

	rootFsUsedBytes := uint64(rootFsUsed.Value())
	logsUsedBytes := uint64(logsUsed.Value())
	for range pod.Spec.Containers {
		result.Containers = append(result.Containers, statsapi.ContainerStats{
			Rootfs: &statsapi.FsStats{
				UsedBytes: &rootFsUsedBytes,
			},
			Logs: &statsapi.FsStats{
				UsedBytes: &logsUsedBytes,
			},
		})
	}

	perLocalVolumeUsedBytes := uint64(perLocalVolumeUsed.Value())
	for _, volumeName := range localVolumeNames(pod) {
		result.VolumeStats = append(result.VolumeStats, statsapi.VolumeStats{
			Name: volumeName,
			FsStats: statsapi.FsStats{
				UsedBytes: &perLocalVolumeUsedBytes,
			},
		})
	}

	return result
}

func newPodMemoryStats(pod *v1.Pod, workingSet resource.Quantity) statsapi.PodStats {
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

func newResourceList(cpu, memory string) v1.ResourceList {
	res := v1.ResourceList{}
	if cpu != "" {
		res[v1.ResourceCPU] = resource.MustParse(cpu)
	}
	if memory != "" {
		res[v1.ResourceMemory] = resource.MustParse(memory)
	}
	return res
}

func newResourceRequirements(requests, limits v1.ResourceList) v1.ResourceRequirements {
	res := v1.ResourceRequirements{}
	res.Requests = requests
	res.Limits = limits
	return res
}

func newContainer(name string, requests v1.ResourceList, limits v1.ResourceList) v1.Container {
	return v1.Container{
		Name:      name,
		Resources: newResourceRequirements(requests, limits),
	}
}

func newVolume(name string, volumeSource v1.VolumeSource) v1.Volume {
	return v1.Volume{
		Name:         name,
		VolumeSource: volumeSource,
	}
}

// newPod uses the name as the uid.  Make names unique for testing.
func newPod(name string, containers []v1.Container, volumes []v1.Volume) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			UID:  types.UID(name),
		},
		Spec: v1.PodSpec{
			Containers: containers,
			Volumes:    volumes,
		},
	}
}

// nodeConditionList is a simple alias to support equality checking independent of order
type nodeConditionList []v1.NodeConditionType

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
type thresholdList []evictionapi.Threshold

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

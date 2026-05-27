/*
Copyright 2025 The Kubernetes Authors.

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

package queue

import (
	"sync"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

var _ metrics.MetricRecorder = &mockMetricRecorder{}

type mockMetricRecorder struct {
	val atomic.Int64
}

func (m *mockMetricRecorder) Inc() {
	m.val.Add(1)
}

func (m *mockMetricRecorder) Dec() {
	m.val.Add(-1)
}

func (m *mockMetricRecorder) Clear() {
	m.val.Store(0)
}

func (m *mockMetricRecorder) value() int64 {
	return m.val.Load()
}

var _ metrics.GatedPodsByGateRecorder = &mockGatedPodsByGateRecorder{}

type mockGatedPodsByGateRecorder struct {
	mu     sync.Mutex
	counts map[string]int64
}

func newMockGatedPodsByGateRecorder() *mockGatedPodsByGateRecorder {
	return &mockGatedPodsByGateRecorder{counts: map[string]int64{}}
}

func (m *mockGatedPodsByGateRecorder) Inc(gate string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.counts[gate]++
}

func (m *mockGatedPodsByGateRecorder) Dec(gate string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.counts[gate]--
}

func (m *mockGatedPodsByGateRecorder) Clear() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.counts = map[string]int64{}
}

func (m *mockGatedPodsByGateRecorder) snapshot() map[string]int64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make(map[string]int64, len(m.counts))
	for k, v := range m.counts {
		if v != 0 {
			out[k] = v
		}
	}
	return out
}

func TestUnschedulablePods(t *testing.T) {
	type action string

	const (
		add    action = "adding"
		update action = "updating"
		delete action = "deleting"
		clear  action = "clearing"
	)

	type step struct {
		action       action
		pods         []*framework.QueuedPodInfo
		expectedPods []*framework.QueuedPodInfo
	}

	var actionToOperation = map[action]func(upm *unschedulablePods, pInfo *framework.QueuedPodInfo, gatedBefore bool){
		add: func(upm *unschedulablePods, pInfo *framework.QueuedPodInfo, _ bool) {
			upm.addOrUpdate(pInfo, false, framework.EventUnscheduledPodAdd.Label())
		},
		update: func(upm *unschedulablePods, pInfo *framework.QueuedPodInfo, gatedBefore bool) {
			upm.addOrUpdate(pInfo, gatedBefore, framework.EventUnscheduledPodUpdate.Label())
		},
		delete: func(upm *unschedulablePods, pInfo *framework.QueuedPodInfo, gatedBefore bool) {
			upm.delete(pInfo.Pod, gatedBefore)
		},
		clear: func(upm *unschedulablePods, _ *framework.QueuedPodInfo, _ bool) {
			upm.clear()
		},
	}

	var pods = []*v1.Pod{
		st.MakePod().Name("p0").Namespace("ns1").Annotation("annot1", "val1").NominatedNodeName("node1").Obj(),
		st.MakePod().Name("p1").Namespace("ns1").Annotation("annot", "val").Obj(),
		st.MakePod().Name("p2").Namespace("ns2").Annotation("annot2", "val2").Annotation("annot3", "val3").NominatedNodeName("node3").Obj(),
		st.MakePod().Name("p3").Namespace("ns1").Annotation("annot4", "val4").Obj(),
		st.MakePod().Name("p4").Namespace("ns2").Annotation("annot5", "val5").NominatedNodeName("node4").Obj(),
		st.MakePod().Name("p5").Namespace("ns1").Annotation("annot6", "val6").Obj(),
	}

	gated := func(p *v1.Pod, upm *unschedulablePods) bool {
		pInfo := upm.get(p)
		return pInfo != nil && pInfo.Gated()
	}

	makePodInfo := func(p *v1.Pod, gated bool) *framework.QueuedPodInfo {
		info := &framework.QueuedPodInfo{
			PodInfo:              mustNewTestPodInfo(t, p),
			UnschedulablePlugins: sets.New[string](),
		}
		if gated {
			info.GatingPlugin = "test"
			info.UnschedulablePlugins.Insert("test")
		}
		return info
	}

	makePodInfoMap := func(pods []*framework.QueuedPodInfo) map[string]*framework.QueuedPodInfo {
		podInfoMap := make(map[string]*framework.QueuedPodInfo)
		for _, p := range pods {
			podInfoMap[util.GetPodFullName(p.Pod)] = p
		}
		return podInfoMap
	}

	tests := []struct {
		name  string
		steps []step
	}{
		{
			name: "create, update, delete subset of pods",
			steps: []step{
				{
					action: add,
					pods: []*framework.QueuedPodInfo{
						makePodInfo(pods[0], false),
						makePodInfo(pods[1], false),
						makePodInfo(pods[2], false),
						makePodInfo(pods[2], false),
						makePodInfo(pods[3], false),
					},
					expectedPods: []*framework.QueuedPodInfo{
						makePodInfo(pods[0], false),
						makePodInfo(pods[1], false),
						makePodInfo(pods[2], false),
						makePodInfo(pods[3], false),
					},
				},
				{
					action: update,
					pods:   []*framework.QueuedPodInfo{makePodInfo(pods[0], false)},
					expectedPods: []*framework.QueuedPodInfo{
						makePodInfo(pods[0], false),
						makePodInfo(pods[1], false),
						makePodInfo(pods[2], false),
						makePodInfo(pods[3], false),
					},
				},
				{
					action: delete,
					pods:   []*framework.QueuedPodInfo{makePodInfo(pods[0], false), makePodInfo(pods[1], false)},
					expectedPods: []*framework.QueuedPodInfo{
						makePodInfo(pods[2], false),
						makePodInfo(pods[3], false),
					},
				},
			},
		},
		{
			name: "create, update, delete all",
			steps: []step{
				{
					action: add,
					pods:   []*framework.QueuedPodInfo{makePodInfo(pods[0], false), makePodInfo(pods[3], false)},
					expectedPods: []*framework.QueuedPodInfo{
						makePodInfo(pods[0], false),
						makePodInfo(pods[3], false),
					},
				},
				{
					action: update,
					pods:   []*framework.QueuedPodInfo{makePodInfo(pods[3], false)},
					expectedPods: []*framework.QueuedPodInfo{
						makePodInfo(pods[0], false),
						makePodInfo(pods[3], false),
					},
				},
				{
					action:       delete,
					pods:         []*framework.QueuedPodInfo{makePodInfo(pods[0], false), makePodInfo(pods[3], false)},
					expectedPods: []*framework.QueuedPodInfo{},
				},
			},
		},
		{
			name: "delete non-existing and existing pods",
			steps: []step{
				{
					action: add,
					pods: []*framework.QueuedPodInfo{
						makePodInfo(pods[1], false),
						makePodInfo(pods[2], false),
					},
					expectedPods: []*framework.QueuedPodInfo{
						makePodInfo(pods[1], false),
						makePodInfo(pods[2], false),
					},
				},
				{
					action: update,
					pods: []*framework.QueuedPodInfo{
						makePodInfo(pods[1], false),
					},
					expectedPods: []*framework.QueuedPodInfo{
						makePodInfo(pods[1], false),
						makePodInfo(pods[2], false),
					},
				},
				{
					action: delete,
					pods: []*framework.QueuedPodInfo{
						makePodInfo(pods[2], false),
						makePodInfo(pods[3], false),
					},
					expectedPods: []*framework.QueuedPodInfo{
						makePodInfo(pods[1], false),
					},
				},
			},
		},
		{
			name: "add/delete gated pods",
			steps: []step{
				{
					action: add,
					pods: []*framework.QueuedPodInfo{
						makePodInfo(pods[0], true),
						makePodInfo(pods[1], true),
					},
					expectedPods: []*framework.QueuedPodInfo{
						makePodInfo(pods[0], true),
						makePodInfo(pods[1], true),
					},
				},
				{
					action: delete,
					pods: []*framework.QueuedPodInfo{
						makePodInfo(pods[0], true),
					},
					expectedPods: []*framework.QueuedPodInfo{
						makePodInfo(pods[1], true),
					},
				},
			},
		},
		{
			name: "add gated and non-gated pods, then delete",
			steps: []step{
				{
					action: add,
					pods: []*framework.QueuedPodInfo{
						makePodInfo(pods[0], false),
						makePodInfo(pods[1], true),
					},
					expectedPods: []*framework.QueuedPodInfo{
						makePodInfo(pods[0], false),
						makePodInfo(pods[1], true),
					},
				},
				{
					action: delete,
					pods: []*framework.QueuedPodInfo{
						makePodInfo(pods[0], false),
						makePodInfo(pods[1], true),
					},
					expectedPods: []*framework.QueuedPodInfo{},
				},
			},
		},
		{
			name: "add gated pod, update it to non-gated",
			steps: []step{
				{
					action: add,
					pods: []*framework.QueuedPodInfo{
						makePodInfo(pods[0], true),
					},
					expectedPods: []*framework.QueuedPodInfo{
						makePodInfo(pods[0], true),
					},
				},
				{
					action: update,
					pods: []*framework.QueuedPodInfo{
						makePodInfo(pods[0], false),
					},
					expectedPods: []*framework.QueuedPodInfo{
						makePodInfo(pods[0], false),
					},
				},
				{
					action: delete,
					pods: []*framework.QueuedPodInfo{
						makePodInfo(pods[0], false),
					},
					expectedPods: []*framework.QueuedPodInfo{},
				},
			},
		},
		{
			name: "add non-gated pod, update it to gated",
			steps: []step{
				{
					action: add,
					pods: []*framework.QueuedPodInfo{
						makePodInfo(pods[0], false),
					},
					expectedPods: []*framework.QueuedPodInfo{
						makePodInfo(pods[0], false),
					},
				},
				{
					action: update,
					pods: []*framework.QueuedPodInfo{
						makePodInfo(pods[0], true),
					},
					expectedPods: []*framework.QueuedPodInfo{
						makePodInfo(pods[0], true),
					},
				},
			},
		},
		{
			name: "add 4 ungated and 2 gated, update 2 to gated and 1 to ungated",
			steps: []step{
				{
					action: add,
					pods: []*framework.QueuedPodInfo{
						makePodInfo(pods[0], false),
						makePodInfo(pods[1], false),
						makePodInfo(pods[2], false),
						makePodInfo(pods[3], false),
						makePodInfo(pods[4], true),
						makePodInfo(pods[5], true),
					},
					expectedPods: []*framework.QueuedPodInfo{
						makePodInfo(pods[0], false),
						makePodInfo(pods[1], false),
						makePodInfo(pods[2], false),
						makePodInfo(pods[3], false),
						makePodInfo(pods[4], true),
						makePodInfo(pods[5], true),
					},
				},
				{
					action: update,
					pods: []*framework.QueuedPodInfo{
						makePodInfo(pods[0], true),
						makePodInfo(pods[1], true),
						makePodInfo(pods[4], false),
					},
					expectedPods: []*framework.QueuedPodInfo{
						makePodInfo(pods[0], true),
						makePodInfo(pods[1], true),
						makePodInfo(pods[2], false),
						makePodInfo(pods[3], false),
						makePodInfo(pods[4], false),
						makePodInfo(pods[5], true),
					},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			unschedulableRecorder := &mockMetricRecorder{}
			gatedRecorder := &mockMetricRecorder{}
			gatedByGateRecorder := newMockGatedPodsByGateRecorder()
			upm := newUnschedulablePods(unschedulableRecorder, gatedRecorder, gatedByGateRecorder)
			assertMetrics := func(expectedPods []*framework.QueuedPodInfo, action string) {
				t.Helper()

				expectedUnschedulableMetric := 0
				expectedGatedMetric := 0
				for _, p := range expectedPods {
					if p.Gated() {
						expectedGatedMetric++
					} else {
						expectedUnschedulableMetric++
					}
				}
				if unschedulableRecorder.value() != int64(expectedUnschedulableMetric) {
					t.Errorf("Expected unschedulable metric to be %d, but got %d after %s", expectedUnschedulableMetric, unschedulableRecorder.value(), action)
				}
				if gatedRecorder.value() != int64(expectedGatedMetric) {
					t.Errorf("Expected gated metric to be %d, but got %d after %s", expectedGatedMetric, gatedRecorder.value(), action)
				}
			}

			for _, step := range test.steps {
				op := actionToOperation[step.action]
				for _, p := range step.pods {
					op(upm, p, gated(p.Pod, upm))
				}
				if diff := cmp.Diff(makePodInfoMap(step.expectedPods), upm.podInfoMap, cmpopts.IgnoreUnexported(framework.PodInfo{})); diff != "" {
					t.Errorf("Unexpected map after %s pods(-want, +got):\n%s", step.action, diff)
				}

				assertMetrics(step.expectedPods, string(step.action))
			}

			upm.clear()
			if len(upm.podInfoMap) != 0 {
				t.Errorf("Expected the map to be empty, but has %v elements.", len(upm.podInfoMap))
			}
			assertMetrics([]*framework.QueuedPodInfo{}, string(clear))
		})
	}
}

func TestUnschedulablePods_GatedByGateRecorder(t *testing.T) {
	const (
		gateFoo = "example.com/foo"
		gateBar = "example.com/bar"
		gateBaz = "example.com/baz"
	)

	makeGatedPodInfo := func(name string, gates ...string) *framework.QueuedPodInfo {
		pod := st.MakePod().Name(name).Namespace("ns").SchedulingGates(gates).Obj()
		info := &framework.QueuedPodInfo{
			PodInfo:              mustNewTestPodInfo(t, pod),
			GatingPlugin:         "test",
			UnschedulablePlugins: sets.New[string]("test"),
		}
		return info
	}

	ungated := func(info *framework.QueuedPodInfo) *framework.QueuedPodInfo {
		clone := *info
		clone.GatingPlugin = ""
		clone.UnschedulablePlugins = sets.New[string]()
		clone.PodInfo = mustNewTestPodInfo(t, info.Pod.DeepCopy())
		clone.Pod.Spec.SchedulingGates = nil
		return &clone
	}

	dropGate := func(info *framework.QueuedPodInfo, gate string) *framework.QueuedPodInfo {
		pod := info.Pod.DeepCopy()
		filtered := pod.Spec.SchedulingGates[:0]
		for _, g := range pod.Spec.SchedulingGates {
			if g.Name != gate {
				filtered = append(filtered, g)
			}
		}
		pod.Spec.SchedulingGates = filtered
		clone := *info
		clone.PodInfo = mustNewTestPodInfo(t, pod)
		return &clone
	}

	assertCounts := func(t *testing.T, rec *mockGatedPodsByGateRecorder, want map[string]int64, msg string) {
		t.Helper()
		if diff := cmp.Diff(want, rec.snapshot()); diff != "" {
			t.Errorf("Unexpected gatedByGateRecorder counts after %s (-want, +got):\n%s", msg, diff)
		}
	}

	t.Run("adds and decrements per-gate counters across gated pods", func(t *testing.T) {
		rec := newMockGatedPodsByGateRecorder()
		upm := newUnschedulablePods(&mockMetricRecorder{}, &mockMetricRecorder{}, rec)

		p1 := makeGatedPodInfo("p1", gateFoo, gateBar)
		p2 := makeGatedPodInfo("p2", gateFoo)

		upm.addOrUpdate(p1, false, framework.EventUnscheduledPodAdd.Label())
		assertCounts(t, rec, map[string]int64{gateFoo: 1, gateBar: 1}, "adding p1")

		upm.addOrUpdate(p2, false, framework.EventUnscheduledPodAdd.Label())
		assertCounts(t, rec, map[string]int64{gateFoo: 2, gateBar: 1}, "adding p2")

		upm.delete(p1.Pod, true)
		assertCounts(t, rec, map[string]int64{gateFoo: 1}, "deleting p1")

		upm.delete(p2.Pod, true)
		assertCounts(t, rec, map[string]int64{}, "deleting p2")
	})

	t.Run("reconciles when a gate is removed while pod stays gated", func(t *testing.T) {
		rec := newMockGatedPodsByGateRecorder()
		upm := newUnschedulablePods(&mockMetricRecorder{}, &mockMetricRecorder{}, rec)

		p := makeGatedPodInfo("p", gateFoo, gateBar)
		upm.addOrUpdate(p, false, framework.EventUnscheduledPodAdd.Label())
		assertCounts(t, rec, map[string]int64{gateFoo: 1, gateBar: 1}, "initial add")

		updated := dropGate(p, gateBar)
		upm.addOrUpdate(updated, true, framework.EventUnscheduledPodUpdate.Label())
		assertCounts(t, rec, map[string]int64{gateFoo: 1}, "dropping gateBar")
	})

	t.Run("releases per-gate counters when pod transitions to ungated", func(t *testing.T) {
		rec := newMockGatedPodsByGateRecorder()
		upm := newUnschedulablePods(&mockMetricRecorder{}, &mockMetricRecorder{}, rec)

		p := makeGatedPodInfo("p", gateFoo, gateBaz)
		upm.addOrUpdate(p, false, framework.EventUnscheduledPodAdd.Label())
		assertCounts(t, rec, map[string]int64{gateFoo: 1, gateBaz: 1}, "initial add")

		upm.addOrUpdate(ungated(p), true, framework.EventUnscheduledPodUpdate.Label())
		assertCounts(t, rec, map[string]int64{}, "transition to ungated")
	})

	t.Run("ignores pods that are gated without spec.schedulingGates", func(t *testing.T) {
		rec := newMockGatedPodsByGateRecorder()
		upm := newUnschedulablePods(&mockMetricRecorder{}, &mockMetricRecorder{}, rec)

		// Pod is Gated() because a plugin set GatingPlugin, but has no spec gates.
		p := makeGatedPodInfo("p")
		upm.addOrUpdate(p, false, framework.EventUnscheduledPodAdd.Label())
		assertCounts(t, rec, map[string]int64{}, "plugin-only gating")

		upm.delete(p.Pod, true)
		assertCounts(t, rec, map[string]int64{}, "delete plugin-only gated pod")
	})

	t.Run("clear() resets the recorder", func(t *testing.T) {
		rec := newMockGatedPodsByGateRecorder()
		upm := newUnschedulablePods(&mockMetricRecorder{}, &mockMetricRecorder{}, rec)

		upm.addOrUpdate(makeGatedPodInfo("p1", gateFoo), false, framework.EventUnscheduledPodAdd.Label())
		upm.addOrUpdate(makeGatedPodInfo("p2", gateBar), false, framework.EventUnscheduledPodAdd.Label())
		assertCounts(t, rec, map[string]int64{gateFoo: 1, gateBar: 1}, "two gated pods")

		upm.clear()
		assertCounts(t, rec, map[string]int64{}, "clear")
		if len(upm.countedGates) != 0 {
			t.Errorf("expected countedGates to be empty after clear, got %d entries", len(upm.countedGates))
		}
	})
}

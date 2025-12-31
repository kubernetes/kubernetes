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
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

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

func (m *mockMetricRecorder) Value() int64 {
	return m.val.Load()
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
		action      action
		pods        []*framework.QueuedPodInfo
		expectedMap map[string]*framework.QueuedPodInfo
	}

	var actionToOperation = map[action]func(pInfo *framework.QueuedPodInfo, upm *unschedulablePods, gatedBefore bool){
		add: func(pInfo *framework.QueuedPodInfo, upm *unschedulablePods, _ bool) {
			upm.addOrUpdate(pInfo, false, framework.EventUnscheduledPodAdd.Label())
		},
		update: func(pInfo *framework.QueuedPodInfo, upm *unschedulablePods, gatedBefore bool) {
			upm.addOrUpdate(pInfo, gatedBefore, framework.EventUnscheduledPodUpdate.Label())
		},
		delete: func(pInfo *framework.QueuedPodInfo, upm *unschedulablePods, gatedBefore bool) {
			upm.delete(pInfo.Pod, gatedBefore)
		},
		clear: func(_ *framework.QueuedPodInfo, upm *unschedulablePods, _ bool) {
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

	makPodInfo := func(p *v1.Pod, gated bool) *framework.QueuedPodInfo {
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

	tests := []struct {
		name  string
		steps []step
	}{
		{
			name: "create, update, delete subset of pods",
			steps: []step{
				{
					action: add,
					pods:   []*framework.QueuedPodInfo{makPodInfo(pods[0], false), makPodInfo(pods[1], false), makPodInfo(pods[2], false), makPodInfo(pods[2], false), makPodInfo(pods[3], false)},
					expectedMap: map[string]*framework.QueuedPodInfo{
						util.GetPodFullName(pods[0]): makPodInfo(pods[0], false),
						util.GetPodFullName(pods[1]): makPodInfo(pods[1], false),
						util.GetPodFullName(pods[2]): makPodInfo(pods[2], false),
						util.GetPodFullName(pods[3]): makPodInfo(pods[3], false),
					},
				},
				{
					action: update,
					pods:   []*framework.QueuedPodInfo{makPodInfo(pods[0], false)},
					expectedMap: map[string]*framework.QueuedPodInfo{
						util.GetPodFullName(pods[0]): makPodInfo(pods[0], false),
						util.GetPodFullName(pods[1]): makPodInfo(pods[1], false),
						util.GetPodFullName(pods[2]): makPodInfo(pods[2], false),
						util.GetPodFullName(pods[3]): makPodInfo(pods[3], false),
					},
				},
				{
					action: delete,
					pods:   []*framework.QueuedPodInfo{makPodInfo(pods[0], false), makPodInfo(pods[1], false)},
					expectedMap: map[string]*framework.QueuedPodInfo{
						util.GetPodFullName(pods[2]): makPodInfo(pods[2], false),
						util.GetPodFullName(pods[3]): makPodInfo(pods[3], false),
					},
				},
			},
		},
		{
			name: "create, update, delete all",
			steps: []step{
				{
					action: add,
					pods:   []*framework.QueuedPodInfo{makPodInfo(pods[0], false), makPodInfo(pods[3], false)},
					expectedMap: map[string]*framework.QueuedPodInfo{
						util.GetPodFullName(pods[0]): makPodInfo(pods[0], false),
						util.GetPodFullName(pods[3]): makPodInfo(pods[3], false),
					},
				},
				{
					action: update,
					pods:   []*framework.QueuedPodInfo{makPodInfo(pods[3], false)},
					expectedMap: map[string]*framework.QueuedPodInfo{
						util.GetPodFullName(pods[0]): makPodInfo(pods[0], false),
						util.GetPodFullName(pods[3]): makPodInfo(pods[3], false),
					},
				},
				{
					action:      delete,
					pods:        []*framework.QueuedPodInfo{makPodInfo(pods[0], false), makPodInfo(pods[3], false)},
					expectedMap: map[string]*framework.QueuedPodInfo{},
				},
			},
		},
		{
			name: "delete non-existing and existing pods",
			steps: []step{
				{
					action: add,
					pods:   []*framework.QueuedPodInfo{makPodInfo(pods[1], false), makPodInfo(pods[2], false)},
					expectedMap: map[string]*framework.QueuedPodInfo{
						util.GetPodFullName(pods[1]): makPodInfo(pods[1], false),
						util.GetPodFullName(pods[2]): makPodInfo(pods[2], false),
					},
				},
				{
					action: update,
					pods:   []*framework.QueuedPodInfo{makPodInfo(pods[1], false)},
					expectedMap: map[string]*framework.QueuedPodInfo{
						util.GetPodFullName(pods[1]): makPodInfo(pods[1], false),
						util.GetPodFullName(pods[2]): makPodInfo(pods[2], false),
					},
				},
				{
					action: delete,
					pods:   []*framework.QueuedPodInfo{makPodInfo(pods[2], false), makPodInfo(pods[3], false)},
					expectedMap: map[string]*framework.QueuedPodInfo{
						util.GetPodFullName(pods[1]): makPodInfo(pods[1], false),
					},
				},
			},
		},
		{
			name: "add/delete gated pods",
			steps: []step{
				{
					action: add,
					pods:   []*framework.QueuedPodInfo{makPodInfo(pods[0], true), makPodInfo(pods[1], true)},
					expectedMap: map[string]*framework.QueuedPodInfo{
						util.GetPodFullName(pods[0]): makPodInfo(pods[0], true),
						util.GetPodFullName(pods[1]): makPodInfo(pods[1], true),
					},
				},
				{
					action: delete,
					pods:   []*framework.QueuedPodInfo{makPodInfo(pods[0], true)},
					expectedMap: map[string]*framework.QueuedPodInfo{
						util.GetPodFullName(pods[1]): makPodInfo(pods[1], true),
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
						makPodInfo(pods[0], false),
						makPodInfo(pods[1], true),
					},
					expectedMap: map[string]*framework.QueuedPodInfo{
						util.GetPodFullName(pods[0]): makPodInfo(pods[0], false),
						util.GetPodFullName(pods[1]): makPodInfo(pods[1], true),
					},
				},
				{
					action:      delete,
					pods:        []*framework.QueuedPodInfo{makPodInfo(pods[0], false), makPodInfo(pods[1], true)},
					expectedMap: map[string]*framework.QueuedPodInfo{},
				},
			},
		},
		{
			name: "add gated pod, update it to non-gated",
			steps: []step{
				{
					action: add,
					pods:   []*framework.QueuedPodInfo{makPodInfo(pods[0], true)},
					expectedMap: map[string]*framework.QueuedPodInfo{
						util.GetPodFullName(pods[0]): makPodInfo(pods[0], true),
					},
				},
				{
					action: update,
					pods:   []*framework.QueuedPodInfo{makPodInfo(pods[0], false)},
					expectedMap: map[string]*framework.QueuedPodInfo{
						util.GetPodFullName(pods[0]): makPodInfo(pods[0], false),
					},
				},
				{
					action: delete,
					pods: []*framework.QueuedPodInfo{
						makPodInfo(pods[0], false),
					},
					expectedMap: map[string]*framework.QueuedPodInfo{},
				},
			},
		},
		{
			name: "add non-gated pod, update it to gated",
			steps: []step{
				{
					action: add,
					pods:   []*framework.QueuedPodInfo{makPodInfo(pods[0], false)},
					expectedMap: map[string]*framework.QueuedPodInfo{
						util.GetPodFullName(pods[0]): makPodInfo(pods[0], false),
					},
				},
				{
					action: update,
					pods: []*framework.QueuedPodInfo{
						makPodInfo(pods[0], true),
					},
					expectedMap: map[string]*framework.QueuedPodInfo{
						util.GetPodFullName(pods[0]): makPodInfo(pods[0], true),
					},
				},
			},
		},
		{
			name: "add 4 ungated and 2 gate, update 2 to gated and 1 to ungated",
			steps: []step{
				{
					action: add,
					pods: []*framework.QueuedPodInfo{
						makPodInfo(pods[0], false),
						makPodInfo(pods[1], false),
						makPodInfo(pods[2], false),
						makPodInfo(pods[3], false),
						makPodInfo(pods[4], true),
						makPodInfo(pods[5], true),
					},
					expectedMap: map[string]*framework.QueuedPodInfo{
						util.GetPodFullName(pods[0]): makPodInfo(pods[0], false),
						util.GetPodFullName(pods[1]): makPodInfo(pods[1], false),
						util.GetPodFullName(pods[2]): makPodInfo(pods[2], false),
						util.GetPodFullName(pods[3]): makPodInfo(pods[3], false),
						util.GetPodFullName(pods[4]): makPodInfo(pods[4], true),
						util.GetPodFullName(pods[5]): makPodInfo(pods[5], true),
					},
				},
				{
					action: update,
					pods: []*framework.QueuedPodInfo{
						makPodInfo(pods[0], true),
						makPodInfo(pods[1], true),
						makPodInfo(pods[4], false),
					},
					expectedMap: map[string]*framework.QueuedPodInfo{
						util.GetPodFullName(pods[0]): makPodInfo(pods[0], true),
						util.GetPodFullName(pods[1]): makPodInfo(pods[1], true),
						util.GetPodFullName(pods[2]): makPodInfo(pods[2], false),
						util.GetPodFullName(pods[3]): makPodInfo(pods[3], false),
						util.GetPodFullName(pods[4]): makPodInfo(pods[4], false),
						util.GetPodFullName(pods[5]): makPodInfo(pods[5], true),
					},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			unschedulableRecorder := &mockMetricRecorder{}
			gatedRecorder := &mockMetricRecorder{}
			upm := newUnschedulablePods(unschedulableRecorder, gatedRecorder)
			assertMetrics := func(expectedMapAfterAction map[string]*framework.QueuedPodInfo, action string) {
				t.Helper()

				expectedUnschedulableMetric := 0
				expectedGatedMetric := 0
				for _, p := range expectedMapAfterAction {
					if p.Gated() {
						expectedGatedMetric++
					} else {
						expectedUnschedulableMetric++
					}
				}
				if unschedulableRecorder.Value() != int64(expectedUnschedulableMetric) {
					t.Errorf("Expected unschedulable metric to be %d, but got %d after %s", expectedUnschedulableMetric, unschedulableRecorder.Value(), action)
				}
				if gatedRecorder.Value() != int64(expectedGatedMetric) {
					t.Errorf("Expected gated metric to be %d, but got %d after %s", expectedGatedMetric, gatedRecorder.Value(), action)
				}
			}

			for _, step := range test.steps {
				op := actionToOperation[step.action]
				for _, p := range step.pods {
					op(p, upm, gated(p.Pod, upm))
				}
				if diff := cmp.Diff(step.expectedMap, upm.podInfoMap, cmpopts.IgnoreUnexported(framework.PodInfo{})); diff != "" {
					t.Errorf("Unexpected map after %s pods(-want, +got):\n%s", step.action, diff)
				}

				assertMetrics(step.expectedMap, string(step.action))
			}

			upm.clear()
			if len(upm.podInfoMap) != 0 {
				t.Errorf("Expected the map to be empty, but has %v elements.", len(upm.podInfoMap))
			}
			assertMetrics(map[string]*framework.QueuedPodInfo{}, string(clear))
		})
	}
}

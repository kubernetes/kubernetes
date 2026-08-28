/*
Copyright 2024 The Kubernetes Authors.

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
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/scheduler/backend/heap"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestClose(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	rr := metrics.NewMetricsAsyncRecorder(10, time.Second, ctx.Done())
	aq := newActiveQueue(heap.NewWithRecorder(queuedEntityKeyFunc, heap.LessFunc[framework.QueuedEntityInfo](convertLessFn(newDefaultQueueSort())), metrics.NewActiveEntitiesRecorder()), rr, nil)

	aq.add(logger, &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: st.MakePod().Namespace("foo").Name("p1").UID("p1").Obj()}}, framework.EventUnscheduledPodAdd.Label(), nil)
	aq.add(logger, &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: st.MakePod().Namespace("bar").Name("p2").UID("p2").Obj()}}, framework.EventUnscheduledPodAdd.Label(), nil)

	_, err := aq.pop(logger)
	if err != nil {
		t.Fatalf("unexpected error while pop(): %v", err)
	}
	_, err = aq.pop(logger)
	if err != nil {
		t.Fatalf("unexpected error while pop(): %v", err)
	}
	aq.addEventIfAnyInFlight(nil, nil, nodeAdd)
	aq.addEventIfAnyInFlight(nil, nil, csiNodeUpdate)

	if len(aq.listInFlightEvents()) != 4 {
		t.Fatalf("unexpected number of in-flight events: %v", len(aq.listInFlightEvents()))
	}
	if len(aq.listInFlightPods()) != 2 {
		t.Fatalf("unexpected number of in-flight pods: %v", len(aq.listInFlightPods()))
	}

	aq.close()

	// make sure the in-flight events and pods are cleaned up by close()

	if len(aq.listInFlightEvents()) != 0 {
		t.Fatalf("in-flight events should be cleaned up, but %v item(s) is remaining", len(aq.listInFlightEvents()))
	}

	if len(aq.listInFlightPods()) != 0 {
		t.Fatalf("in-flight pods should be cleaned up, but %v pod(s) is remaining", len(aq.listInFlightPods()))
	}
}

func TestActiveQueue_InFlightPods(t *testing.T) {
	tests := []struct {
		name                 string
		initialPods          []*v1.Pod
		popCount             int
		updatePod            *v1.Pod
		callDoneSchedCycle   types.UID
		callDone             types.UID
		callDelete           types.UID
		wantInFlightPodLabel map[string]string
		wantInFlightCount    int
		wantInFlightEvents   int
	}{
		{
			name: "update in-flight pod updates pod content",
			initialPods: []*v1.Pod{
				st.MakePod().Namespace("ns").Name("p1").UID("p1").Label("version", "v1").Obj(),
			},
			popCount:             1,
			updatePod:            st.MakePod().Namespace("ns").Name("p1").UID("p1").Label("version", "v2").Obj(),
			wantInFlightPodLabel: map[string]string{"version": "v2"},
			wantInFlightCount:    1,
		},
		{
			name: "doneSchedulingCycle removes events marker but keeps pod in-flight",
			initialPods: []*v1.Pod{
				st.MakePod().Namespace("ns").Name("p1").UID("p1").Label("version", "v1").Obj(),
			},
			popCount:             1,
			callDoneSchedCycle:   "p1",
			wantInFlightPodLabel: map[string]string{"version": "v1"},
			wantInFlightCount:    1,
			wantInFlightEvents:   0,
		},
		{
			name: "delete in-flight pod removes it completely",
			initialPods: []*v1.Pod{
				st.MakePod().Namespace("ns").Name("p1").UID("p1").Label("version", "v1").Obj(),
			},
			popCount:          1,
			callDelete:        "p1",
			wantInFlightCount: 0,
		},
		{
			name: "done on in-flight pod removes it completely",
			initialPods: []*v1.Pod{
				st.MakePod().Namespace("ns").Name("p1").UID("p1").Label("version", "v1").Obj(),
			},
			popCount:          1,
			callDone:          "p1",
			wantInFlightCount: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			rr := metrics.NewMetricsAsyncRecorder(10, time.Second, ctx.Done())
			aq := newActiveQueue(heap.NewWithRecorder(queuedEntityKeyFunc, heap.LessFunc[framework.QueuedEntityInfo](convertLessFn(newDefaultQueueSort())), metrics.NewActiveEntitiesRecorder()), rr, nil)

			for _, pod := range tt.initialPods {
				aq.add(logger, &framework.QueuedPodInfo{PodInfo: &framework.PodInfo{Pod: pod}}, framework.EventUnscheduledPodAdd.Label(), nil)
			}
			for i := 0; i < tt.popCount; i++ {
				if _, err := aq.pop(logger); err != nil {
					t.Fatalf("pop failed: %v", err)
				}
			}

			if tt.updatePod != nil {
				aq.underLock(func(unlockedActiveQ unlockedActiveQueuer) {
					unlockedActiveQ.updateInFlightPod(tt.updatePod)
				})
			}
			if tt.callDoneSchedCycle != "" {
				aq.doneSchedulingCycle(tt.callDoneSchedCycle)
			}
			if tt.callDone != "" {
				aq.done(tt.callDone)
			}
			if tt.callDelete != "" {
				aq.deleteInFlight(tt.callDelete)
			}

			if len(aq.listInFlightPods()) != tt.wantInFlightCount {
				t.Fatalf("expected %d in-flight pods, got %d", tt.wantInFlightCount, len(aq.listInFlightPods()))
			}
			if tt.wantInFlightPodLabel != nil {
				pod := aq.inFlightPod(tt.initialPods[0].UID)
				if pod == nil {
					t.Fatalf("expected inFlightPod to find pod %s", tt.initialPods[0].UID)
				}
				for k, v := range tt.wantInFlightPodLabel {
					if pod.Labels[k] != v {
						t.Fatalf("expected label %s=%s, got %s", k, v, pod.Labels[k])
					}
				}
			}
			if tt.callDoneSchedCycle != "" && len(aq.listInFlightEvents()) != tt.wantInFlightEvents {
				t.Fatalf("expected %d in-flight events, got %d", tt.wantInFlightEvents, len(aq.listInFlightEvents()))
			}
		})
	}
}

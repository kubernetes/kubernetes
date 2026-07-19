/*
Copyright 2019 The Kubernetes Authors.

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

package metrics

import (
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	"k8s.io/component-base/metrics/testutil"
	fwk "k8s.io/kube-scheduler/framework"
)

var _ MetricRecorder = &fakePodsRecorder{}

type testEntity struct {
	size int
	t    fwk.EntityKeyType
}

func (te *testEntity) Size() int {
	return te.size
}

func (te *testEntity) Type() fwk.EntityKeyType {
	return te.t
}

type fakePodsRecorder struct {
	counter int64
}

func (r *fakePodsRecorder) Add(entity Entity) {
	atomic.AddInt64(&r.counter, int64(entity.Size()))
}

func (r *fakePodsRecorder) Remove(entity Entity) {
	atomic.AddInt64(&r.counter, -int64(entity.Size()))
}

func (r *fakePodsRecorder) Update(oldEntity, newEntity Entity) {
	atomic.AddInt64(&r.counter, int64(newEntity.Size()-oldEntity.Size()))
}

func (r *fakePodsRecorder) Clear() {
	atomic.StoreInt64(&r.counter, 0)
}

func TestAdd(t *testing.T) {
	fakeRecorder := fakePodsRecorder{}
	var wg sync.WaitGroup
	loops := 100
	wg.Add(loops)
	for i := range loops {
		go func() {
			fakeRecorder.Add(&testEntity{size: i, t: "pod"})
			wg.Done()
		}()
	}
	wg.Wait()
	if fakeRecorder.counter != int64(loops*(loops-1)/2) {
		t.Errorf("Expected %v, got %v", loops, fakeRecorder.counter)
	}
}

func TestRemove(t *testing.T) {
	fakeRecorder := fakePodsRecorder{counter: 100}
	var wg sync.WaitGroup
	loops := 50
	wg.Add(loops)
	for range loops {
		go func() {
			fakeRecorder.Remove(&testEntity{size: 2, t: "podgroup"})
			wg.Done()
		}()
	}
	wg.Wait()
	if fakeRecorder.counter != int64(0) {
		t.Errorf("Expected %v, got %v", 0, fakeRecorder.counter)
	}
}

func TestClear(t *testing.T) {
	fakeRecorder := fakePodsRecorder{}
	var wg sync.WaitGroup
	incLoops, decLoops := 100, 80
	wg.Add(incLoops + decLoops)
	for range incLoops {
		go func() {
			fakeRecorder.Add(&testEntity{size: 1, t: "pod"})
			fakeRecorder.Add(&testEntity{size: 2, t: "podgroup"})
			wg.Done()
		}()
	}
	for range decLoops {
		go func() {
			fakeRecorder.Remove(&testEntity{size: 1, t: "pod"})
			wg.Done()
		}()
	}
	wg.Wait()
	expected := int64(incLoops*3 - decLoops)
	if fakeRecorder.counter != expected {
		t.Errorf("Expected %v, got %v", expected, fakeRecorder.counter)
	}
	// verify Clear() works
	fakeRecorder.Clear()
	if fakeRecorder.counter != int64(0) {
		t.Errorf("Expected %v, got %v", 0, fakeRecorder.counter)
	}
}

func TestInFlightEventAsync(t *testing.T) {
	Register()
	r := &MetricAsyncRecorder{
		aggregatedInflightEventMetric:              map[gaugeVecMetricKey]int{},
		aggregatedInflightEventMetricLastFlushTime: time.Now(),
		aggregatedInflightEventMetricBufferCh:      make(chan *gaugeVecMetric, 100),
		interval:                                   time.Hour,
	}

	podAddLabel := "Pod/Add"
	r.ObserveInFlightEventsAsync(podAddLabel, 10, false)
	r.ObserveInFlightEventsAsync(podAddLabel, -1, false)
	r.ObserveInFlightEventsAsync(PodPoppedInFlightEvent, 1, false)

	if d := cmp.Diff(r.aggregatedInflightEventMetric, map[gaugeVecMetricKey]int{
		{metricName: InFlightEvents.Name, labelValue: podAddLabel}:            9,
		{metricName: InFlightEvents.Name, labelValue: PodPoppedInFlightEvent}: 1,
	}, cmp.AllowUnexported(gaugeVecMetric{})); d != "" {
		t.Errorf("unexpected aggregatedInflightEventMetric: %s", d)
	}

	r.aggregatedInflightEventMetricLastFlushTime = time.Now().Add(-time.Hour) // to test flush

	// It adds -4 and flushes the metric to the channel.
	r.ObserveInFlightEventsAsync(podAddLabel, -4, false)
	if len(r.aggregatedInflightEventMetric) != 0 {
		t.Errorf("aggregatedInflightEventMetric should be cleared, but got: %v", r.aggregatedInflightEventMetric)
	}

	got := []gaugeVecMetric{}
	for {
		select {
		case m := <-r.aggregatedInflightEventMetricBufferCh:
			got = append(got, *m)
			continue
		default:
		}
		// got all
		break
	}

	// sort got to avoid the flaky test
	sort.Slice(got, func(i, j int) bool {
		return got[i].labelValues[0] < got[j].labelValues[0]
	})

	if d := cmp.Diff(got, []gaugeVecMetric{
		{
			labelValues: []string{podAddLabel},
			valueToAdd:  5,
		},
		{
			labelValues: []string{PodPoppedInFlightEvent},
			valueToAdd:  1,
		},
	}, cmp.AllowUnexported(gaugeVecMetric{}), cmpopts.IgnoreFields(gaugeVecMetric{}, "metric")); d != "" {
		t.Errorf("unexpected metrics are sent to aggregatedInflightEventMetricBufferCh: %s", d)
	}

	// Test force flush
	r.ObserveInFlightEventsAsync(podAddLabel, 1, true)
	if len(r.aggregatedInflightEventMetric) != 0 {
		t.Errorf("aggregatedInflightEventMetric should be force-flushed, but got: %v", r.aggregatedInflightEventMetric)
	}
}

func TestEntityToLabel(t *testing.T) {
	tests := []struct {
		name      string
		entity    Entity
		wantLabel string
		wantOK    bool
	}{
		{
			name:   "nil entity",
			entity: nil,
			wantOK: false,
		},
		{
			name:      "pod entity",
			entity:    &testEntity{t: "pod"},
			wantLabel: Pod,
			wantOK:    true,
		},
		{
			name:      "podgroup entity",
			entity:    &testEntity{t: "podgroup"},
			wantLabel: PodGroup,
			wantOK:    true,
		},
		{
			name:   "unknown entity type",
			entity: &testEntity{t: "unknown"},
			wantOK: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotLabel, gotOK := EntityToLabel(tt.entity)
			if gotLabel != tt.wantLabel || gotOK != tt.wantOK {
				t.Errorf("EntityToLabel() = (%v, %v), want (%v, %v)", gotLabel, gotOK, tt.wantLabel, tt.wantOK)
			}
		})
	}
}

func TestQueuedEntitiesRecorder(t *testing.T) {
	Register()

	recorder := NewActiveEntitiesRecorder()
	recorder.Clear()

	pInfo1 := &testEntity{size: 1, t: "pod"}
	pInfo2 := &testEntity{size: 5, t: "podgroup"}
	pInfo3 := &testEntity{size: 1, t: "pod"}
	updatedPInfo2 := &testEntity{size: 8, t: "podgroup"}

	tests := []struct {
		name string
		op   func()
		want string
	}{
		{
			name: "Add entity",
			op: func() {
				recorder.Add(pInfo1)
				recorder.Add(pInfo2)
				recorder.Add(pInfo3)
			},
			want: `
				# HELP scheduler_pending_pods [STABLE] Number of pending pods, by the queue type. 'active' means number of pods in activeQ; 'backoff' means number of pods in backoffQ; 'unschedulable' means number of pods in unschedulableEntities that the scheduler attempted to schedule and failed; 'gated' is the number of unschedulable pods that the scheduler never attempted to schedule because they are gated; 'incomplete' means number of pods in incompletePodGroupPods; 'pending' means number of pods in pendingPodGroupPods.
				# TYPE scheduler_pending_pods gauge
				scheduler_pending_pods{queue="active"} 7
				# HELP scheduler_queued_entities [ALPHA] Number of queued scheduling entities ('pod' or 'podgroup'; 'pod' stands for individual pods that are not members of any podgroup) by the queue type. 'active' means number of entities in activeQ; 'backoff' means number of entities in backoffQ; 'unschedulable' means number of entities in unschedulableEntities that the scheduler attempted to schedule and failed; 'gated' is the number of unschedulable entities that the scheduler never attempted to schedule because they are gated.
				# TYPE scheduler_queued_entities gauge
				scheduler_queued_entities{queue="active",type="pod"} 2
				scheduler_queued_entities{queue="active",type="podgroup"} 1
			`,
		},
		{
			name: "Remove entity",
			op: func() {
				recorder.Remove(pInfo1)
			},
			want: `
				# HELP scheduler_pending_pods [STABLE] Number of pending pods, by the queue type. 'active' means number of pods in activeQ; 'backoff' means number of pods in backoffQ; 'unschedulable' means number of pods in unschedulableEntities that the scheduler attempted to schedule and failed; 'gated' is the number of unschedulable pods that the scheduler never attempted to schedule because they are gated; 'incomplete' means number of pods in incompletePodGroupPods; 'pending' means number of pods in pendingPodGroupPods.
				# TYPE scheduler_pending_pods gauge
				scheduler_pending_pods{queue="active"} 6
				# HELP scheduler_queued_entities [ALPHA] Number of queued scheduling entities ('pod' or 'podgroup'; 'pod' stands for individual pods that are not members of any podgroup) by the queue type. 'active' means number of entities in activeQ; 'backoff' means number of entities in backoffQ; 'unschedulable' means number of entities in unschedulableEntities that the scheduler attempted to schedule and failed; 'gated' is the number of unschedulable entities that the scheduler never attempted to schedule because they are gated.
				# TYPE scheduler_queued_entities gauge
				scheduler_queued_entities{queue="active",type="pod"} 1
				scheduler_queued_entities{queue="active",type="podgroup"} 1
			`,
		},
		{
			name: "Update entity",
			op: func() {
				recorder.Update(pInfo2, updatedPInfo2)
			},
			want: `
				# HELP scheduler_pending_pods [STABLE] Number of pending pods, by the queue type. 'active' means number of pods in activeQ; 'backoff' means number of pods in backoffQ; 'unschedulable' means number of pods in unschedulableEntities that the scheduler attempted to schedule and failed; 'gated' is the number of unschedulable pods that the scheduler never attempted to schedule because they are gated; 'incomplete' means number of pods in incompletePodGroupPods; 'pending' means number of pods in pendingPodGroupPods.
				# TYPE scheduler_pending_pods gauge
				scheduler_pending_pods{queue="active"} 9
				# HELP scheduler_queued_entities [ALPHA] Number of queued scheduling entities ('pod' or 'podgroup'; 'pod' stands for individual pods that are not members of any podgroup) by the queue type. 'active' means number of entities in activeQ; 'backoff' means number of entities in backoffQ; 'unschedulable' means number of entities in unschedulableEntities that the scheduler attempted to schedule and failed; 'gated' is the number of unschedulable entities that the scheduler never attempted to schedule because they are gated.
				# TYPE scheduler_queued_entities gauge
				scheduler_queued_entities{queue="active",type="pod"} 1
				scheduler_queued_entities{queue="active",type="podgroup"} 1
			`,
		},
		{
			name: "Clear",
			op: func() {
				recorder.Clear()
			},
			want: `
				# HELP scheduler_pending_pods [STABLE] Number of pending pods, by the queue type. 'active' means number of pods in activeQ; 'backoff' means number of pods in backoffQ; 'unschedulable' means number of pods in unschedulableEntities that the scheduler attempted to schedule and failed; 'gated' is the number of unschedulable pods that the scheduler never attempted to schedule because they are gated; 'incomplete' means number of pods in incompletePodGroupPods; 'pending' means number of pods in pendingPodGroupPods.
				# TYPE scheduler_pending_pods gauge
				scheduler_pending_pods{queue="active"} 0
				# HELP scheduler_queued_entities [ALPHA] Number of queued scheduling entities ('pod' or 'podgroup'; 'pod' stands for individual pods that are not members of any podgroup) by the queue type. 'active' means number of entities in activeQ; 'backoff' means number of entities in backoffQ; 'unschedulable' means number of entities in unschedulableEntities that the scheduler attempted to schedule and failed; 'gated' is the number of unschedulable entities that the scheduler never attempted to schedule because they are gated.
				# TYPE scheduler_queued_entities gauge
				scheduler_queued_entities{queue="active",type="pod"} 0
				scheduler_queued_entities{queue="active",type="podgroup"} 0
			`,
		},
	}

	for _, step := range tests {
		t.Run(step.name, func(t *testing.T) {
			step.op()
			if err := testutil.GatherAndCompare(GetGather(), strings.NewReader(step.want), "scheduler_pending_pods", "scheduler_queued_entities"); err != nil {
				t.Errorf("unexpected metrics %s: %v", step.name, err)
			}
		})
	}
}

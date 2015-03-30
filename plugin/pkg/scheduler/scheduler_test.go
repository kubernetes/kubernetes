/*
Copyright 2014 Google Inc. All rights reserved.

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

package scheduler

import (
	"errors"
	"math/rand"
	"reflect"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/record"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/scheduler"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

type fakeBinder struct {
	b func(binding *api.Binding) error
}

func (fb fakeBinder) Bind(binding *api.Binding) error { return fb.b(binding) }

func podWithID(id, desiredHost string) *api.Pod {
	return &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: id, SelfLink: testapi.SelfLink("pods", id)},
		Spec: api.PodSpec{
			Host: desiredHost,
		},
		Status: api.PodStatus{
			Host: desiredHost,
		},
	}
}

func podWithPort(id, desiredHost string, port int) *api.Pod {
	pod := podWithID(id, desiredHost)
	pod.Spec.Containers = []api.Container{
		{Name: "ctr", Ports: []api.ContainerPort{{HostPort: port}}},
	}
	return pod
}

type mockScheduler struct {
	machine string
	err     error
}

func (es mockScheduler) Schedule(pod api.Pod, ml scheduler.MinionLister) (string, error) {
	return es.machine, es.err
}

func TestScheduler(t *testing.T) {
	defer record.StartLogging(t.Logf).Stop()
	errS := errors.New("scheduler")
	errB := errors.New("binder")

	table := []struct {
		injectBindError  error
		sendPod          *api.Pod
		algo             scheduler.Scheduler
		expectErrorPod   *api.Pod
		expectAssumedPod *api.Pod
		expectError      error
		expectBind       *api.Binding
		eventReason      string
	}{
		{
			sendPod:          podWithID("foo", ""),
			algo:             mockScheduler{"machine1", nil},
			expectBind:       &api.Binding{ObjectMeta: api.ObjectMeta{Name: "foo"}, Target: api.ObjectReference{Kind: "Node", Name: "machine1"}},
			expectAssumedPod: podWithID("foo", "machine1"),
			eventReason:      "scheduled",
		}, {
			sendPod:        podWithID("foo", ""),
			algo:           mockScheduler{"machine1", errS},
			expectError:    errS,
			expectErrorPod: podWithID("foo", ""),
			eventReason:    "failedScheduling",
		}, {
			sendPod:         podWithID("foo", ""),
			algo:            mockScheduler{"machine1", nil},
			expectBind:      &api.Binding{ObjectMeta: api.ObjectMeta{Name: "foo"}, Target: api.ObjectReference{Kind: "Node", Name: "machine1"}},
			injectBindError: errB,
			expectError:     errB,
			expectErrorPod:  podWithID("foo", ""),
			eventReason:     "failedScheduling",
		},
	}

	for i, item := range table {
		var gotError error
		var gotPod *api.Pod
		var gotAssumedPod *api.Pod
		var gotBinding *api.Binding
		c := &Config{
			Modeler: &FakeModeler{
				AssumePodFunc: func(pod *api.Pod) {
					gotAssumedPod = pod
				},
			},
			MinionLister: scheduler.FakeMinionLister(
				api.NodeList{Items: []api.Node{{ObjectMeta: api.ObjectMeta{Name: "machine1"}}}},
			),
			Algorithm: item.algo,
			Binder: fakeBinder{func(b *api.Binding) error {
				gotBinding = b
				return item.injectBindError
			}},
			Error: func(p *api.Pod, err error) {
				gotPod = p
				gotError = err
			},
			NextPod: func() *api.Pod {
				return item.sendPod
			},
			Recorder: record.FromSource(api.EventSource{Component: "scheduler"}),
		}
		s := New(c)
		called := make(chan struct{})
		events := record.GetEvents(func(e *api.Event) {
			if e, a := item.eventReason, e.Reason; e != a {
				t.Errorf("%v: expected %v, got %v", i, e, a)
			}
			close(called)
		})
		s.scheduleOne()
		if e, a := item.expectAssumedPod, gotAssumedPod; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: assumed pod: wanted %v, got %v", i, e, a)
		}
		if e, a := item.expectErrorPod, gotPod; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: error pod: wanted %v, got %v", i, e, a)
		}
		if e, a := item.expectError, gotError; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: error: wanted %v, got %v", i, e, a)
		}
		if e, a := item.expectBind, gotBinding; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: error: %s", i, util.ObjectDiff(e, a))
		}
		<-called
		events.Stop()
	}
}

func TestSchedulerForgetAssumedPodAfterDelete(t *testing.T) {
	defer record.StartLogging(t.Logf).Stop()

	scheduledPodStore := cache.NewStore(cache.MetaNamespaceKeyFunc)
	scheduledPodLister := &cache.StoreToPodLister{scheduledPodStore}

	queuedPodStore := cache.NewFIFO(cache.MetaNamespaceKeyFunc)
	queuedPodLister := &cache.StoreToPodLister{queuedPodStore}

	modeler := NewSimpleModeler(queuedPodLister, scheduledPodLister)
	assumedPodsStore := cache.NewStore(cache.MetaNamespaceKeyFunc)
	modeler.assumedPods = &cache.StoreToPodLister{assumedPodsStore}

	podPort := 8080
	newPod := podWithPort("foo", "", podPort)

	algo := scheduler.NewGenericScheduler(
		map[string]scheduler.FitPredicate{"PodFitsPorts": scheduler.PodFitsPorts},
		[]scheduler.PriorityConfig{},
		modeler.PodLister(),
		rand.New(rand.NewSource(time.Now().UnixNano())))

	var gotBinding *api.Binding
	expectBind := &api.Binding{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Target:     api.ObjectReference{Kind: "Node", Name: "machine1"},
	}

	c := &Config{
		Modeler: modeler,
		MinionLister: scheduler.FakeMinionLister(
			api.NodeList{Items: []api.Node{{ObjectMeta: api.ObjectMeta{Name: "machine1"}}}},
		),
		Algorithm: algo,
		Binder: fakeBinder{func(b *api.Binding) error {
			scheduledPodStore.Add(podWithPort(b.Name, b.Target.Name, podPort))
			gotBinding = b
			return nil
		}},
		NextPod: func() *api.Pod {
			return queuedPodStore.Pop().(*api.Pod)
		},
		Error: func(p *api.Pod, err error) {
			t.Errorf("Unexpected error when scheduling pod %+v: %v", p, err)
		},
		Recorder: record.FromSource(api.EventSource{Component: "scheduler"}),
	}

	s := New(c)
	called := make(chan struct{})
	events := record.GetEvents(func(e *api.Event) {
		if e, a := "scheduled", e.Reason; e != a {
			t.Errorf("expected %v, got %v", e, a)
		}
		close(called)
	})

	queuedPodStore.Add(newPod)
	// queuedPodStore: [foo]
	// scheduledPodStore: []
	// assumedPods: []

	s.scheduleOne()
	// queuedPodStore: []
	// scheduledPodStore: [foo]
	// assumedPods: [foo]

	pod, exists, _ := scheduledPodStore.GetByKey("foo")
	if !exists {
		t.Errorf("Expected scheduled pod store to contain pod")
	}
	pod, exists, _ = queuedPodStore.GetByKey("foo")
	if exists {
		t.Errorf("Did not expect a queued pod, found %+v", pod)
	}
	pod, exists, _ = assumedPodsStore.GetByKey("foo")
	if !exists {
		t.Errorf("Assumed pod store should contain stale pod")
	}
	if e, a := expectBind, gotBinding; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected exact match on binding: %s", util.ObjectDiff(e, a))
	}

	<-called
	events.Stop()

	scheduledPodStore.Delete(pod)
	queuedPodStore.Add(newPod)
	// queuedPodStore: [foo]
	// scheduledPodStore: []
	// assumedPods: [foo]

	called = make(chan struct{})
	events = record.GetEvents(func(e *api.Event) {
		if e, a := "scheduled", e.Reason; e != a {
			t.Errorf("expected %v, got %v", e, a)
		}
		close(called)
	})

	gotBinding = nil
	s.scheduleOne()
	if e, a := expectBind, gotBinding; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected exact match on binding: %s", util.ObjectDiff(e, a))
	}

	<-called
	events.Stop()
}

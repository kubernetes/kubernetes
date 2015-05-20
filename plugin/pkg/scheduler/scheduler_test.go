/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler/algorithm"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler/algorithm/predicates"
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

func (es mockScheduler) Schedule(pod *api.Pod, ml algorithm.MinionLister) (string, error) {
	return es.machine, es.err
}

func TestScheduler(t *testing.T) {
	eventBroadcaster := record.NewBroadcaster()
	defer eventBroadcaster.StartLogging(t.Logf).Stop()
	errS := errors.New("scheduler")
	errB := errors.New("binder")

	table := []struct {
		injectBindError  error
		sendPod          *api.Pod
		algo             algorithm.ScheduleAlgorithm
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
			MinionLister: algorithm.FakeMinionLister(
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
			Recorder: eventBroadcaster.NewRecorder(api.EventSource{Component: "scheduler"}),
		}
		s := New(c)
		called := make(chan struct{})
		events := eventBroadcaster.StartEventWatcher(func(e *api.Event) {
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
	eventBroadcaster := record.NewBroadcaster()
	defer eventBroadcaster.StartLogging(t.Logf).Stop()

	// Setup modeler so we control the contents of all 3 stores: assumed,
	// scheduled and queued
	scheduledPodStore := cache.NewStore(cache.MetaNamespaceKeyFunc)
	scheduledPodLister := &cache.StoreToPodLister{scheduledPodStore}

	queuedPodStore := cache.NewFIFO(cache.MetaNamespaceKeyFunc)
	queuedPodLister := &cache.StoreToPodLister{queuedPodStore}

	modeler := NewSimpleModeler(queuedPodLister, scheduledPodLister)

	// Create a fake clock used to timestamp entries and calculate ttl. Nothing
	// will expire till we flip to something older than the ttl, at which point
	// all entries inserted with fakeTime will expire.
	ttl := 30 * time.Second
	fakeTime := time.Date(2009, time.November, 10, 23, 0, 0, 0, time.UTC)
	fakeClock := &util.FakeClock{fakeTime}
	ttlPolicy := &cache.TTLPolicy{ttl, fakeClock}
	assumedPodsStore := cache.NewFakeExpirationStore(
		cache.MetaNamespaceKeyFunc, nil, ttlPolicy, fakeClock)
	modeler.assumedPods = &cache.StoreToPodLister{assumedPodsStore}

	// Port is the easiest way to cause a fit predicate failure
	podPort := 8080
	firstPod := podWithPort("foo", "", podPort)

	// Create the scheduler config
	algo := NewGenericScheduler(
		map[string]algorithm.FitPredicate{"PodFitsPorts": predicates.PodFitsPorts},
		[]algorithm.PriorityConfig{},
		modeler.PodLister(),
		rand.New(rand.NewSource(time.Now().UnixNano())))

	var gotBinding *api.Binding
	c := &Config{
		Modeler: modeler,
		MinionLister: algorithm.FakeMinionLister(
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
		Recorder: eventBroadcaster.NewRecorder(api.EventSource{Component: "scheduler"}),
	}

	// First scheduling pass should schedule the pod
	s := New(c)
	called := make(chan struct{})
	events := eventBroadcaster.StartEventWatcher(func(e *api.Event) {
		if e, a := "scheduled", e.Reason; e != a {
			t.Errorf("expected %v, got %v", e, a)
		}
		close(called)
	})

	queuedPodStore.Add(firstPod)
	// queuedPodStore: [foo:8080]
	// scheduledPodStore: []
	// assumedPods: []

	s.scheduleOne()
	// queuedPodStore: []
	// scheduledPodStore: [foo:8080]
	// assumedPods: [foo:8080]

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

	expectBind := &api.Binding{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Target:     api.ObjectReference{Kind: "Node", Name: "machine1"},
	}
	if ex, ac := expectBind, gotBinding; !reflect.DeepEqual(ex, ac) {
		t.Errorf("Expected exact match on binding: %s", util.ObjectDiff(ex, ac))
	}

	<-called
	events.Stop()

	scheduledPodStore.Delete(pod)
	_, exists, _ = assumedPodsStore.Get(pod)
	if !exists {
		t.Errorf("Expected pod %#v in assumed pod store", pod)
	}

	secondPod := podWithPort("bar", "", podPort)
	queuedPodStore.Add(secondPod)
	// queuedPodStore: [bar:8080]
	// scheduledPodStore: []
	// assumedPods: [foo:8080]

	// Second scheduling pass will fail to schedule if the store hasn't expired
	// the deleted pod. This would normally happen with a timeout.
	//expirationPolicy.NeverExpire = util.NewStringSet()
	fakeClock.Time = fakeClock.Time.Add(ttl + 1)

	called = make(chan struct{})
	events = eventBroadcaster.StartEventWatcher(func(e *api.Event) {
		if e, a := "scheduled", e.Reason; e != a {
			t.Errorf("expected %v, got %v", e, a)
		}
		close(called)
	})

	s.scheduleOne()

	expectBind = &api.Binding{
		ObjectMeta: api.ObjectMeta{Name: "bar"},
		Target:     api.ObjectReference{Kind: "Node", Name: "machine1"},
	}
	if ex, ac := expectBind, gotBinding; !reflect.DeepEqual(ex, ac) {
		t.Errorf("Expected exact match on binding: %s", util.ObjectDiff(ex, ac))
	}
	<-called
	events.Stop()
}

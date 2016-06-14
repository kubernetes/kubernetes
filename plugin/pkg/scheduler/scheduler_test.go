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
	"fmt"
	"reflect"
	"sync"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	clientcache "k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/diff"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/predicates"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
	schedulertesting "k8s.io/kubernetes/plugin/pkg/scheduler/testing"
)

type fakeBinder struct {
	b func(binding *api.Binding) error
}

func (fb fakeBinder) Bind(binding *api.Binding) error { return fb.b(binding) }

type fakePodConditionUpdater struct{}

func (fc fakePodConditionUpdater) Update(pod *api.Pod, podCondition *api.PodCondition) error {
	return nil
}

func podWithID(id, desiredHost string) *api.Pod {
	return &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: id, SelfLink: testapi.Default.SelfLink("pods", id)},
		Spec: api.PodSpec{
			NodeName: desiredHost,
		},
	}
}

func podWithPort(id, desiredHost string, port int) *api.Pod {
	pod := podWithID(id, desiredHost)
	pod.Spec.Containers = []api.Container{
		{Name: "ctr", Ports: []api.ContainerPort{{HostPort: int32(port)}}},
	}
	return pod
}

type mockScheduler struct {
	machine string
	err     error
}

func (es mockScheduler) Schedule(pod *api.Pod, ml algorithm.NodeLister) (string, error) {
	return es.machine, es.err
}

func TestScheduler(t *testing.T) {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(t.Logf).Stop()
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
			eventReason:      "Scheduled",
		}, {
			sendPod:        podWithID("foo", ""),
			algo:           mockScheduler{"machine1", errS},
			expectError:    errS,
			expectErrorPod: podWithID("foo", ""),
			eventReason:    "FailedScheduling",
		}, {
			sendPod:          podWithID("foo", ""),
			algo:             mockScheduler{"machine1", nil},
			expectBind:       &api.Binding{ObjectMeta: api.ObjectMeta{Name: "foo"}, Target: api.ObjectReference{Kind: "Node", Name: "machine1"}},
			expectAssumedPod: podWithID("foo", "machine1"),
			injectBindError:  errB,
			expectError:      errB,
			expectErrorPod:   podWithID("foo", ""),
			eventReason:      "FailedScheduling",
		},
	}

	for i, item := range table {
		var gotError error
		var gotPod *api.Pod
		var gotAssumedPod *api.Pod
		var gotBinding *api.Binding
		c := &Config{
			SchedulerCache: &schedulertesting.FakeCache{
				AssumeFunc: func(pod *api.Pod) {
					gotAssumedPod = pod
				},
			},
			NodeLister: algorithm.FakeNodeLister(
				api.NodeList{Items: []api.Node{{ObjectMeta: api.ObjectMeta{Name: "machine1"}}}},
			),
			Algorithm: item.algo,
			Binder: fakeBinder{func(b *api.Binding) error {
				gotBinding = b
				return item.injectBindError
			}},
			PodConditionUpdater: fakePodConditionUpdater{},
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
		<-called
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
			t.Errorf("%v: error: %s", i, diff.ObjectDiff(e, a))
		}
		events.Stop()
	}
}

func TestSchedulerForgetAssumedPodAfterDelete(t *testing.T) {
	// Set up a channel through which we'll funnel log messages from the watcher.
	// This way, we can guarantee that when the test ends no thread will still be
	// trying to write to t.Logf (which it would if we handed t.Logf directly to
	// StartLogging).
	ch := make(chan string)
	done := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case msg := <-ch:
				t.Log(msg)
			case <-done:
				return
			}
		}
	}()
	eventBroadcaster := record.NewBroadcaster()
	watcher := eventBroadcaster.StartLogging(func(format string, args ...interface{}) {
		ch <- fmt.Sprintf(format, args...)
	})
	defer func() {
		watcher.Stop()
		close(done)
		wg.Wait()
	}()

	// Setup stores to test pod's workflow:
	// - queuedPodStore: pods queued before processing
	// - scheduledPodStore: pods that has a scheduling decision
	scheduledPodStore := clientcache.NewStore(clientcache.MetaNamespaceKeyFunc)
	queuedPodStore := clientcache.NewFIFO(clientcache.MetaNamespaceKeyFunc)

	// Port is the easiest way to cause a fit predicate failure
	podPort := 8080
	firstPod := podWithPort("foo", "", podPort)

	stop := make(chan struct{})
	defer close(stop)
	cache := schedulercache.New(1*time.Second, stop)
	// Create the scheduler config
	algo := NewGenericScheduler(
		cache,
		map[string]algorithm.FitPredicate{"PodFitsHostPorts": predicates.PodFitsHostPorts},
		[]algorithm.PriorityConfig{},
		[]algorithm.SchedulerExtender{})

	var gotBinding *api.Binding
	c := &Config{
		SchedulerCache: cache,
		NodeLister: algorithm.FakeNodeLister(
			api.NodeList{Items: []api.Node{{ObjectMeta: api.ObjectMeta{Name: "machine1"}}}},
		),
		Algorithm: algo,
		Binder: fakeBinder{func(b *api.Binding) error {
			scheduledPodStore.Add(podWithPort(b.Name, b.Target.Name, podPort))
			gotBinding = b
			return nil
		}},
		NextPod: func() *api.Pod {
			return clientcache.Pop(queuedPodStore).(*api.Pod)
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
		if e, a := "Scheduled", e.Reason; e != a {
			t.Errorf("expected %v, got %v", e, a)
		}
		close(called)
	})

	queuedPodStore.Add(firstPod)
	// queuedPodStore: [foo:8080]
	// scheduledPodStore: []
	// assumedPods: []

	s.scheduleOne()
	<-called
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

	expectBind := &api.Binding{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Target:     api.ObjectReference{Kind: "Node", Name: "machine1"},
	}
	if ex, ac := expectBind, gotBinding; !reflect.DeepEqual(ex, ac) {
		t.Errorf("Expected exact match on binding: %s", diff.ObjectDiff(ex, ac))
	}

	events.Stop()

	scheduledPodStore.Delete(pod)

	secondPod := podWithPort("bar", "", podPort)
	queuedPodStore.Add(secondPod)
	// queuedPodStore: [bar:8080]
	// scheduledPodStore: []
	// assumedPods: [foo:8080]

	var waitUntilExpired sync.WaitGroup
	waitUntilExpired.Add(1)
	// waiting for the assumed pod to expire
	go func() {
		for {
			pods, err := cache.List(labels.Everything())
			if err != nil {
				t.Fatalf("cache.List failed: %v", err)
			}
			if len(pods) == 0 {
				waitUntilExpired.Done()
				return
			}
			time.Sleep(1 * time.Second)
		}
	}()
	waitUntilExpired.Wait()

	// Second scheduling pass will fail to schedule if the store hasn't expired
	// the deleted pod. This would normally happen with a timeout.

	called = make(chan struct{})
	events = eventBroadcaster.StartEventWatcher(func(e *api.Event) {
		if e, a := "Scheduled", e.Reason; e != a {
			t.Errorf("expected %v, got %v", e, a)
		}
		close(called)
	})

	s.scheduleOne()
	<-called

	expectBind = &api.Binding{
		ObjectMeta: api.ObjectMeta{Name: "bar"},
		Target:     api.ObjectReference{Kind: "Node", Name: "machine1"},
	}
	if ex, ac := expectBind, gotBinding; !reflect.DeepEqual(ex, ac) {
		t.Errorf("Expected exact match on binding: %s", diff.ObjectDiff(ex, ac))
	}
	events.Stop()
}

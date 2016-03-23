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

	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/plugin/pkg/ube-scheduler/algorithm"
	schedulertesting "k8s.io/kubernetes/plugin/pkg/ube-scheduler/testing"
	"k8s.io/kubernetes/pkg/apis/controlplane"
)

type fakeBinder struct {
	b func(binding *api.Binding) error
}

func (fb fakeBinder) Bind(binding *api.Binding) error { return fb.b(binding) }

func podWithID(id, desiredHost string) *api.ReplicationController {
	return &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{Name: id, SelfLink: testapi.Default.SelfLink("pods", id)},
		Spec: api.PodSpec{
			NodeName: desiredHost,
		},
	}
}

func podWithPort(id, desiredHost string, port int) *api.ReplicationController {
	pod := podWithID(id, desiredHost)
	pod.Spec.Template.Spec.Containers = []api.Container{
		{Name: "ctr", Ports: []api.ContainerPort{{HostPort: port}}},
	}
	return pod
}

type mockScheduler struct {
	machine string
	err     error
}

func (es mockScheduler) Schedule(pod *api.ReplicationController, ml algorithm.ClusterLister) (string, error) {
	return es.machine, es.err
}

func TestScheduler(t *testing.T) {
	eventBroadcaster := record.NewBroadcaster()
	defer eventBroadcaster.StartLogging(t.Logf).Stop()
	errS := errors.New("scheduler")
	errB := errors.New("binder")

	table := []struct {
		injectBindError  error
		sendFederationRC          *api.ReplicationController
		algo             algorithm.ScheduleAlgorithm
		expectErrorPod   *api.ReplicationController
		expectAssumedPod *api.ReplicationController
		expectError      error
		expectBind       *api.Binding
		eventReason      string
	}{
		{
			sendFederationRC: podWithID("foo", ""),
			algo:             mockScheduler{"machine1", nil},
			expectBind:       &api.Binding{ObjectMeta: api.ObjectMeta{Name: "foo"}, Target: api.ObjectReference{Kind: "Node", Name: "machine1"}},
			expectAssumedPod: podWithID("foo", "machine1"),
			eventReason:      "Scheduled",
		}, {
			sendFederationRC:        podWithID("foo", ""),
			algo:           mockScheduler{"machine1", errS},
			expectError:    errS,
			expectErrorPod: podWithID("foo", ""),
			eventReason:    "FailedScheduling",
		}, {
			sendFederationRC:         podWithID("foo", ""),
			algo:            mockScheduler{"machine1", nil},
			expectBind:      &api.Binding{ObjectMeta: api.ObjectMeta{Name: "foo"}, Target: api.ObjectReference{Kind: "Node", Name: "machine1"}},
			injectBindError: errB,
			expectError:     errB,
			expectErrorPod:  podWithID("foo", ""),
			eventReason:     "FailedScheduling",
		},
	}

	for i, item := range table {
		var gotError error
		var gotPod *api.ReplicationController
		var gotAssumedPod *api.ReplicationController
		var gotBinding *api.Binding
		c := &Config{
			SchedulerCache: &schedulertesting.FakeCache{
				AssumeFunc: func(pod *api.ReplicationController) {
					gotAssumedPod = pod
				},
			},
			ClusterLister: algorithm.FakeClusterLister(
				api.ClusterList{Items: []controlplane.Cluster{{ObjectMeta: api.ObjectMeta{Name: "cluster1"}}}},
			),
			Algorithm: item.algo,
			Binder: fakeBinder{func(b *api.Binding) error {
				gotBinding = b
				return item.injectBindError
			}},
			Error: func(p *api.ReplicationController, err error) {
				gotPod = p
				gotError = err
			},
			NextFederationRC: func() *api.ReplicationController {
				return item.sendFederationRC
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


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
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/util/diff"
	"k8s.io/kubernetes/federation/plugin/pkg/scheduler/algorithm"
	schedulertesting "k8s.io/kubernetes/federation/plugin/pkg/scheduler/testing"

	"k8s.io/kubernetes/federation/apis/federation"
)

type fakeBinder struct {
	b func(binding *api.Binding) error
}

func (fb fakeBinder) Bind(binding *api.Binding) error { return fb.b(binding) }

func replicaSetWithID(id, desiredHost string) *extensions.ReplicaSet {
	annotations := map[string]string{}
	annotations[federation.TargetClusterKey] = "cluster1"
	return &extensions.ReplicaSet{
		ObjectMeta: api.ObjectMeta{
			Name: id,
			SelfLink: testapi.Default.SelfLink("replicaSets", id),
			Annotations:annotations,
		},
	}
}

func replicaSetWithPort(id, desiredHost string, port int) *extensions.ReplicaSet {
	replicaSet := replicaSetWithID(id, desiredHost)
	replicaSet.Spec.Template.Spec.Containers = []api.Container{
		{Name: "ctr", Ports: []api.ContainerPort{{HostPort: port}}},
	}
	return replicaSet
}

type mockScheduler struct {
	cluster string
	err     error
}

func (es mockScheduler) Schedule(replicaSet *extensions.ReplicaSet, ml algorithm.ClusterLister) (string, error) {
	return es.cluster, es.err
}

func TestScheduler(t *testing.T) {
	eventBroadcaster := record.NewBroadcaster()
	defer eventBroadcaster.StartLogging(t.Logf).Stop()
	errS := errors.New("scheduler")
	errB := errors.New("binder")

	table := []struct {
		injectBindError         error
		sendReplicaSet          *extensions.ReplicaSet
		algo                    algorithm.ScheduleAlgorithm
		expectErrorReplicaSet   *extensions.ReplicaSet
		expectAssumedReplicaSet *extensions.ReplicaSet
		expectError             error
		expectBind              *api.Binding
		eventReason             string
	}{
		{
			sendReplicaSet: replicaSetWithID("foo", ""),
			algo:             mockScheduler{"cluster1", nil},
			expectBind:       &api.Binding{ObjectMeta: api.ObjectMeta{Name: "foo"}, Target: api.ObjectReference{Kind: "Node", Name: "cluster1"}},
			expectAssumedReplicaSet: replicaSetWithID("foo", "cluster1"),
			eventReason:      "Scheduled",
		}, {
			sendReplicaSet:        replicaSetWithID("foo", ""),
			algo:           mockScheduler{"cluster1", errS},
			expectError:    errS,
			expectErrorReplicaSet: replicaSetWithID("foo", ""),
			eventReason:    "FailedScheduling",
		}, {
			sendReplicaSet:         replicaSetWithID("foo", ""),
			algo:            mockScheduler{"cluster1", nil},
			expectBind:      &api.Binding{ObjectMeta: api.ObjectMeta{Name: "foo"}, Target: api.ObjectReference{Kind: "Node", Name: "cluster1"}},
			injectBindError: errB,
			expectError:     errB,
			expectErrorReplicaSet:  replicaSetWithID("foo", ""),
			eventReason:     "FailedScheduling",
		},
	}

	for i, item := range table {
		var gotError error
		var gotReplicaSet *extensions.ReplicaSet
		var gotAssumedReplicaSet *extensions.ReplicaSet
		var gotBinding *api.Binding
		c := &Config{
			SchedulerCache: &schedulertesting.FakeCache{
				AssumeFunc: func(replicaSet *extensions.ReplicaSet) {
					gotAssumedReplicaSet = replicaSet
				},
			},
			ClusterLister: algorithm.FakeClusterLister(
				federation.ClusterList{Items: []federation.Cluster{{ObjectMeta: api.ObjectMeta{Name: "cluster1"}}}},
			),
			Algorithm: item.algo,
			Binder: fakeBinder{func(b *api.Binding) error {
				gotBinding = b
				return item.injectBindError
			}},
			Error: func(p *extensions.ReplicaSet, err error) {
				gotReplicaSet = p
				gotError = err
			},
			NextReplicaSet: func() *extensions.ReplicaSet {
				return item.sendReplicaSet
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
		if e, a := item.expectAssumedReplicaSet, gotAssumedReplicaSet; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: assumed replicaSet: wanted %v, got %v", i, e, a)
		}
		if e, a := item.expectErrorReplicaSet, gotReplicaSet; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: error replicaSet: wanted %v, got %v", i, e, a)
		}
		if e, a := item.expectError, gotError; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: error: wanted %v, got %v", i, e, a)
		}
		if e, a := item.expectBind, gotBinding; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: error: %s", i, diff.ObjectDiff(e, a))
		}
		<-called
		events.Stop()
	}
}


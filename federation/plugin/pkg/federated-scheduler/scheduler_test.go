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
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/util/diff"
	"k8s.io/kubernetes/federation/apis/federation/unversioned"
	federation "k8s.io/kubernetes/federation/apis/federation/v1alpha1"
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/algorithm"
	schedulertesting "k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/testing"

)

type fakeBinder struct {
	b func(subRS *federation.SubReplicaSet) error
}

func (fb fakeBinder) Bind(subRS *federation.SubReplicaSet) error { return fb.b(subRS) }

func replicaSetWithID(id, desiredHost string) *extensions.ReplicaSet {
	annotations := map[string]string{}
	annotations[unversioned.FederationReplicaSetKey] = "cluster1,cluster2"
	return &extensions.ReplicaSet{
		ObjectMeta: v1.ObjectMeta{
			Name: id,
			SelfLink: testapi.Default.SelfLink("replicasets", id),
			Annotations:annotations,
		},
	}
}
func getExpectSubRS(rs *extensions.ReplicaSet, desiredHost string) *federation.SubReplicaSet {
	subRS, _ := splitSubReplicaSet(rs,"cluster1")
	return subRS
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
	errS := errors.New("federated-scheduler")
	errB := errors.New("binder")
	table := []struct {
		injectBindError       error
		sendReplicaSet        *extensions.ReplicaSet
		algo                  algorithm.ScheduleAlgorithm
		expectErrorReplicaSet *extensions.ReplicaSet
		expectAssumedSubRS    *federation.SubReplicaSet
		expectError           error
		expectBind            *federation.SubReplicaSet
		eventReason           string
	}{
		{
			sendReplicaSet: replicaSetWithID("foo", ""),
			algo:             mockScheduler{"cluster1", nil},
			expectBind:       getExpectSubRS(replicaSetWithID("foo", ""),"cluster1"),
			expectAssumedSubRS: getExpectSubRS(replicaSetWithID("foo", ""),"cluster1"),
			eventReason:      "Scheduled",
		}, {
			sendReplicaSet:        replicaSetWithID("foo1", ""),
			algo:           mockScheduler{"cluster1", errS},
			expectError:    errS,
			expectErrorReplicaSet: replicaSetWithID("foo1", ""),
			eventReason:    "FailedScheduling",
		}, {
			sendReplicaSet:         replicaSetWithID("foo2", ""),
			algo:            mockScheduler{"cluster1", nil},
			expectBind:      &federation.SubReplicaSet{ObjectMeta: v1.ObjectMeta{Name: "foo2"}, },
			injectBindError: errB,
			expectError:     errB,
			expectErrorReplicaSet:  replicaSetWithID("foo2", ""),
			eventReason:     "FailedBinding",
		},
	}

	for i, item := range table {
		var gotError error
		var gotReplicaSet *extensions.ReplicaSet
		var gotAssumedSubRS *federation.SubReplicaSet
		var gotBinding *federation.SubReplicaSet
		c := &Config{
			SchedulerCache: &schedulertesting.FakeCache{
				AssumeFunc: func(subRS *federation.SubReplicaSet) {
					gotAssumedSubRS = subRS
				},
			},
			ClusterLister: algorithm.FakeClusterLister(
				federation.ClusterList{Items: []federation.Cluster{{ObjectMeta: v1.ObjectMeta{Name: "cluster1"}}}},
			),
			Algorithm: item.algo,
			Binder: fakeBinder{func(subRS *federation.SubReplicaSet) error {
				gotBinding = subRS
				return item.injectBindError
			}},
			Error: func(p *extensions.ReplicaSet, err error) {
				gotReplicaSet = p
				gotError = err
			},
			NextReplicaSet: func() *extensions.ReplicaSet {
				return item.sendReplicaSet
			},
			Recorder: eventBroadcaster.NewRecorder(api.EventSource{Component: "federated-scheduler"}),
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
		//error go first
		if e, a := item.expectErrorReplicaSet, gotReplicaSet; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: error replicaSet: wanted %v, got %v", i, e, a)
		}
		if e, a := item.expectError, gotError; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: error: wanted %v, got %v", i, e, a)
		}
		//happy path: case 1
		e := item.expectAssumedSubRS
		a := gotAssumedSubRS
		if a != nil {
			if a.Name == a.GenerateName || !strings.HasPrefix(a.Name, a.GenerateName) {
				t.Errorf("unexpected name: %#v", a)
			}
			//reset name before compare as it has random generated suffix
			e.Name = e.GenerateName
			a.Name = a.GenerateName
			if !reflect.DeepEqual(e, a) {
				t.Errorf("%v: assumed replicaSet: wanted %v, got %v", i, e, a)
			}

			b := item.expectBind
			b1 := gotBinding
			//reset name before compare as it has random generated suffix
			b.Name = e.GenerateName
			b1.Name = a.GenerateName
			if !reflect.DeepEqual(b, b1) {
				t.Errorf("%v: error: wanted %v, got %v", i, b, b1)
				//reset name before compare as it has random generated suffix
				b.Name = b.GenerateName
				b1.Name = b1.GenerateName
				t.Errorf("%v: error: %s", i, diff.ObjectDiff(b, b1))
			}
		}
		<-called
		events.Stop()
	}
}


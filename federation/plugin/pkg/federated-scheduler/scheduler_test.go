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
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/conversion"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/util/diff"
	"k8s.io/kubernetes/federation/apis/federation/unversioned"
	apiunversioned "k8s.io/kubernetes/pkg/api/unversioned"
	federation "k8s.io/kubernetes/federation/apis/federation/v1alpha1"
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/algorithm"
	schedulertesting "k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/testing"
)

type fakeBinder struct {
	b func(subRS *extensions.ReplicaSet) error
}

func (fb fakeBinder) Bind(rs *extensions.ReplicaSet) error { return fb.b(rs) }

func replicaSetWithID(id, desiredCluster string) *extensions.ReplicaSet {
	return &extensions.ReplicaSet{
		TypeMeta: apiunversioned.TypeMeta{
			Kind: "ReplicaSet",
			APIVersion: "extensions/v1beta1",
		},
		ObjectMeta: v1.ObjectMeta{
			Name: id,
			GenerateName: id,
			SelfLink: testapi.Default.SelfLink("replicasets", id),
			Annotations: map[string]string{unversioned.ClusterSelectorKey: desiredCluster},
		},
	}
}
func getExpectReplicaSet(rs *extensions.ReplicaSet, desiredCluster string) *extensions.ReplicaSet {
	annotations := map[string]string{}
	annotations[unversioned.TargetClusterKey] = desiredCluster
	rs.Annotations = map[string]string{}
	rs.Annotations[unversioned.TargetClusterKey] = desiredCluster
	return rs
}

//create subrs and assign assign properties of rs to be scheduled, generate and assign random name
func generateSubRS(replicaSet *extensions.ReplicaSet) (*federation.SubReplicaSet, error) {
	clone, err := conversion.NewCloner().DeepCopy(replicaSet)
	if err != nil {
		return nil, err
	}
	rsTemp, ok := clone.(*extensions.ReplicaSet)
	if !ok {
		return nil, fmt.Errorf("Unexpected replicaset cast error : %v\n", rsTemp)
	}
	result := &federation.SubReplicaSet{
		TypeMeta: apiunversioned.TypeMeta{
			Kind: "SubReplicaSet",
			APIVersion: "federation/v1alpha1",
		},
		Spec : rsTemp.Spec,
		Status: rsTemp.Status,
		ObjectMeta: rsTemp.ObjectMeta,
	}

	//to generate subrs name, we need a api.ObjectMeta instead of v1
	meta := &api.ObjectMeta{}
	meta.GenerateName = result.ObjectMeta.Name + "-"

	api.GenerateName(api.SimpleNameGenerator, meta)
	result.Name = meta.Name
	result.GenerateName = rsTemp.Name

	//unset resourceVersion before create the actual resource
	result.ResourceVersion = ""

	return result, nil
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
		expectAssumedReplicaSet *extensions.ReplicaSet
		expectError           error
		expectBind            *extensions.ReplicaSet
		eventReason           string
	}{
		{
			sendReplicaSet: replicaSetWithID("foo", "cluster1, cluster2"),
			algo:             mockScheduler{"cluster1", nil},
			expectBind:       getExpectReplicaSet(replicaSetWithID("foo", ""),"cluster1"),
			expectAssumedReplicaSet: getExpectReplicaSet(replicaSetWithID("foo", ""),"cluster1"),
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
			expectBind:       getExpectReplicaSet(replicaSetWithID("foo2", ""),"cluster1"),
			expectAssumedReplicaSet: getExpectReplicaSet(replicaSetWithID("foo2", ""),"cluster1"),
			injectBindError: errB,
			expectError:     errB,
			expectErrorReplicaSet:  getExpectReplicaSet(replicaSetWithID("foo2", ""),"cluster1"),
			eventReason:     "FailedBinding",
		},
	}

	for i, item := range table {
		var gotError error
		var gotReplicaSet *extensions.ReplicaSet
		var gotAssumedReplicaSet *extensions.ReplicaSet
		var gotBinding *extensions.ReplicaSet
		c := &Config{
			SchedulerCache: &schedulertesting.FakeCache{
				AssumeFunc: func(rs *extensions.ReplicaSet) {
					gotAssumedReplicaSet = rs
				},
			},
			ClusterLister: algorithm.FakeClusterLister(
				federation.ClusterList{Items: []federation.Cluster{{ObjectMeta: v1.ObjectMeta{Name: "cluster1"}}}},
			),
			Algorithm: item.algo,
			Binder: fakeBinder{func(rs *extensions.ReplicaSet) error {
				gotBinding = rs
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
		<-called
		//error go first
		if e, a := item.expectErrorReplicaSet, gotReplicaSet; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: error replicaSet: wanted %v, got %v", i, e, a)
		}
		if e, a := item.expectError, gotError; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: error: wanted %v, got %v", i, e, a)
		}
		//happy path: case 1
		e := item.expectAssumedReplicaSet
		a := gotAssumedReplicaSet
		if a != nil {
			if !reflect.DeepEqual(e, a) {
				t.Errorf("%v: assumed replicaSet: wanted %v, got %v", i, e, a)
			}
			b := item.expectBind
			b1 := gotBinding
			if !reflect.DeepEqual(b, b1) {
				t.Errorf("%v: error: wanted %v, got %v", i, b, b1)
				t.Errorf("%v: error: %s", i, diff.ObjectDiff(b, b1))
			}
		}
		events.Stop()
	}
}


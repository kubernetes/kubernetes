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

package defaultpodtopologyspread

import (
	"context"
	"fmt"
	"reflect"
	"sort"
	"testing"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/internal/cache"
)

func controllerRef(kind, name, uid string) []metav1.OwnerReference {
	// TODO: When ControllerRef will be implemented uncomment code below.
	return nil
	//trueVar := true
	//return []metav1.OwnerReference{
	//	{Kind: kind, Name: name, UID: types.UID(uid), Controller: &trueVar},
	//}
}

func TestDefaultPodTopologySpreadScore(t *testing.T) {
	labels1 := map[string]string{
		"foo": "bar",
		"baz": "blah",
	}
	labels2 := map[string]string{
		"bar": "foo",
		"baz": "blah",
	}
	zone1Spec := v1.PodSpec{
		NodeName: "machine1",
	}
	zone2Spec := v1.PodSpec{
		NodeName: "machine2",
	}
	tests := []struct {
		pod          *v1.Pod
		pods         []*v1.Pod
		nodes        []string
		rcs          []*v1.ReplicationController
		rss          []*apps.ReplicaSet
		services     []*v1.Service
		sss          []*apps.StatefulSet
		expectedList framework.NodeScoreList
		name         string
	}{
		{
			pod:          new(v1.Pod),
			nodes:        []string{"machine1", "machine2"},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: framework.MaxNodeScore}, {Name: "machine2", Score: framework.MaxNodeScore}},
			name:         "nothing scheduled",
		},
		{
			pod:          &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			pods:         []*v1.Pod{{Spec: zone1Spec}},
			nodes:        []string{"machine1", "machine2"},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: framework.MaxNodeScore}, {Name: "machine2", Score: framework.MaxNodeScore}},
			name:         "no services",
		},
		{
			pod:          &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			pods:         []*v1.Pod{{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2}}},
			nodes:        []string{"machine1", "machine2"},
			services:     []*v1.Service{{Spec: v1.ServiceSpec{Selector: map[string]string{"key": "value"}}}},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: framework.MaxNodeScore}, {Name: "machine2", Score: framework.MaxNodeScore}},
			name:         "different services",
		},
		{
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			pods: []*v1.Pod{
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			},
			nodes:        []string{"machine1", "machine2"},
			services:     []*v1.Service{{Spec: v1.ServiceSpec{Selector: labels1}}},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: framework.MaxNodeScore}, {Name: "machine2", Score: 0}},
			name:         "two pods, one service pod",
		},
		{
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			pods: []*v1.Pod{
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, Namespace: metav1.NamespaceDefault}},
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, Namespace: "ns1"}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2}},
			},
			nodes:        []string{"machine1", "machine2"},
			services:     []*v1.Service{{Spec: v1.ServiceSpec{Selector: labels1}}},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: framework.MaxNodeScore}, {Name: "machine2", Score: 0}},
			name:         "five pods, one service pod in no namespace",
		},
		{
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: labels1, Namespace: metav1.NamespaceDefault}},
			pods: []*v1.Pod{
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, Namespace: "ns1"}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, Namespace: metav1.NamespaceDefault}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2}},
			},
			nodes:        []string{"machine1", "machine2"},
			services:     []*v1.Service{{Spec: v1.ServiceSpec{Selector: labels1}, ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault}}},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: framework.MaxNodeScore}, {Name: "machine2", Score: 0}},
			name:         "four pods, one service pod in default namespace",
		},
		{
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: labels1, Namespace: "ns1"}},
			pods: []*v1.Pod{
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, Namespace: metav1.NamespaceDefault}},
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, Namespace: "ns2"}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, Namespace: "ns1"}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2}},
			},
			nodes:        []string{"machine1", "machine2"},
			services:     []*v1.Service{{Spec: v1.ServiceSpec{Selector: labels1}, ObjectMeta: metav1.ObjectMeta{Namespace: "ns1"}}},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: framework.MaxNodeScore}, {Name: "machine2", Score: 0}},
			name:         "five pods, one service pod in specific namespace",
		},
		{
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			pods: []*v1.Pod{
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			},
			nodes:        []string{"machine1", "machine2"},
			services:     []*v1.Service{{Spec: v1.ServiceSpec{Selector: labels1}}},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 0}, {Name: "machine2", Score: 0}},
			name:         "three pods, two service pods on different machines",
		},
		{
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			pods: []*v1.Pod{
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			},
			nodes:        []string{"machine1", "machine2"},
			services:     []*v1.Service{{Spec: v1.ServiceSpec{Selector: labels1}}},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 50}, {Name: "machine2", Score: 0}},
			name:         "four pods, three service pods",
		},
		{
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			pods: []*v1.Pod{
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			},
			nodes:        []string{"machine1", "machine2"},
			services:     []*v1.Service{{Spec: v1.ServiceSpec{Selector: map[string]string{"baz": "blah"}}}},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 0}, {Name: "machine2", Score: 50}},
			name:         "service with partial pod label matches",
		},
		{
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicationController", "name", "abc123")}},
			pods: []*v1.Pod{
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicationController", "name", "abc123")}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicationController", "name", "abc123")}},
			},
			nodes:    []string{"machine1", "machine2"},
			rcs:      []*v1.ReplicationController{{Spec: v1.ReplicationControllerSpec{Selector: map[string]string{"foo": "bar"}}}},
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: map[string]string{"baz": "blah"}}}},
			// "baz=blah" matches both labels1 and labels2, and "foo=bar" matches only labels 1. This means that we assume that we want to
			// do spreading pod2 and pod3 and not pod1.
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 0}, {Name: "machine2", Score: 0}},
			name:         "service with partial pod label matches with service and replication controller",
		},
		{
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicaSet", "name", "abc123")}},
			pods: []*v1.Pod{
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicaSet", "name", "abc123")}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicaSet", "name", "abc123")}},
			},
			nodes:    []string{"machine1", "machine2"},
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: map[string]string{"baz": "blah"}}}},
			rss:      []*apps.ReplicaSet{{Spec: apps.ReplicaSetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}}}}},
			// We use ReplicaSet, instead of ReplicationController. The result should be exactly as above.
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 0}, {Name: "machine2", Score: 0}},
			name:         "service with partial pod label matches with service and replica set",
		},
		{
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("StatefulSet", "name", "abc123")}},
			pods: []*v1.Pod{
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("StatefulSet", "name", "abc123")}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("StatefulSet", "name", "abc123")}},
			},
			nodes:        []string{"machine1", "machine2"},
			services:     []*v1.Service{{Spec: v1.ServiceSpec{Selector: map[string]string{"baz": "blah"}}}},
			sss:          []*apps.StatefulSet{{Spec: apps.StatefulSetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}}}}},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 0}, {Name: "machine2", Score: 0}},
			name:         "service with partial pod label matches with service and stateful set",
		},
		{
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"foo": "bar", "bar": "foo"}, OwnerReferences: controllerRef("ReplicationController", "name", "abc123")}},
			pods: []*v1.Pod{
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicationController", "name", "abc123")}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicationController", "name", "abc123")}},
			},
			nodes:    []string{"machine1", "machine2"},
			rcs:      []*v1.ReplicationController{{Spec: v1.ReplicationControllerSpec{Selector: map[string]string{"foo": "bar"}}}},
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: map[string]string{"bar": "foo"}}}},
			// Taken together Service and Replication Controller should match no pods.
			expectedList: []framework.NodeScore{{Name: "machine1", Score: framework.MaxNodeScore}, {Name: "machine2", Score: framework.MaxNodeScore}},
			name:         "disjoined service and replication controller matches no pods",
		},
		{
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"foo": "bar", "bar": "foo"}, OwnerReferences: controllerRef("ReplicaSet", "name", "abc123")}},
			pods: []*v1.Pod{
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicaSet", "name", "abc123")}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicaSet", "name", "abc123")}},
			},
			nodes:    []string{"machine1", "machine2"},
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: map[string]string{"bar": "foo"}}}},
			rss:      []*apps.ReplicaSet{{Spec: apps.ReplicaSetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}}}}},
			// We use ReplicaSet, instead of ReplicationController. The result should be exactly as above.
			expectedList: []framework.NodeScore{{Name: "machine1", Score: framework.MaxNodeScore}, {Name: "machine2", Score: framework.MaxNodeScore}},
			name:         "disjoined service and replica set matches no pods",
		},
		{
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"foo": "bar", "bar": "foo"}, OwnerReferences: controllerRef("StatefulSet", "name", "abc123")}},
			pods: []*v1.Pod{
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("StatefulSet", "name", "abc123")}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("StatefulSet", "name", "abc123")}},
			},
			nodes:        []string{"machine1", "machine2"},
			services:     []*v1.Service{{Spec: v1.ServiceSpec{Selector: map[string]string{"bar": "foo"}}}},
			sss:          []*apps.StatefulSet{{Spec: apps.StatefulSetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}}}}},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: framework.MaxNodeScore}, {Name: "machine2", Score: framework.MaxNodeScore}},
			name:         "disjoined service and stateful set matches no pods",
		},
		{
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicationController", "name", "abc123")}},
			pods: []*v1.Pod{
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicationController", "name", "abc123")}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicationController", "name", "abc123")}},
			},
			nodes: []string{"machine1", "machine2"},
			rcs:   []*v1.ReplicationController{{Spec: v1.ReplicationControllerSpec{Selector: map[string]string{"foo": "bar"}}}},
			// Both Nodes have one pod from the given RC, hence both get 0 score.
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 0}, {Name: "machine2", Score: 0}},
			name:         "Replication controller with partial pod label matches",
		},
		{
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicaSet", "name", "abc123")}},
			pods: []*v1.Pod{
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicaSet", "name", "abc123")}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicaSet", "name", "abc123")}},
			},
			nodes: []string{"machine1", "machine2"},
			rss:   []*apps.ReplicaSet{{Spec: apps.ReplicaSetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}}}}},
			// We use ReplicaSet, instead of ReplicationController. The result should be exactly as above.
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 0}, {Name: "machine2", Score: 0}},
			name:         "Replica set with partial pod label matches",
		},
		{
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("StatefulSet", "name", "abc123")}},
			pods: []*v1.Pod{
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("StatefulSet", "name", "abc123")}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("StatefulSet", "name", "abc123")}},
			},
			nodes: []string{"machine1", "machine2"},
			sss:   []*apps.StatefulSet{{Spec: apps.StatefulSetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}}}}},
			// We use StatefulSet, instead of ReplicationController. The result should be exactly as above.
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 0}, {Name: "machine2", Score: 0}},
			name:         "StatefulSet with partial pod label matches",
		},
		{
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicationController", "name", "abc123")}},
			pods: []*v1.Pod{
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2, OwnerReferences: controllerRef("ReplicationController", "name", "abc123")}},
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicationController", "name", "abc123")}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicationController", "name", "abc123")}},
			},
			nodes:        []string{"machine1", "machine2"},
			rcs:          []*v1.ReplicationController{{Spec: v1.ReplicationControllerSpec{Selector: map[string]string{"baz": "blah"}}}},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 0}, {Name: "machine2", Score: 50}},
			name:         "Another replication controller with partial pod label matches",
		},
		{
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicaSet", "name", "abc123")}},
			pods: []*v1.Pod{
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2, OwnerReferences: controllerRef("ReplicaSet", "name", "abc123")}},
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicaSet", "name", "abc123")}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicaSet", "name", "abc123")}},
			},
			nodes: []string{"machine1", "machine2"},
			rss:   []*apps.ReplicaSet{{Spec: apps.ReplicaSetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"baz": "blah"}}}}},
			// We use ReplicaSet, instead of ReplicationController. The result should be exactly as above.
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 0}, {Name: "machine2", Score: 50}},
			name:         "Another replication set with partial pod label matches",
		},
		{
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("StatefulSet", "name", "abc123")}},
			pods: []*v1.Pod{
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2, OwnerReferences: controllerRef("StatefulSet", "name", "abc123")}},
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("StatefulSet", "name", "abc123")}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("StatefulSet", "name", "abc123")}},
			},
			nodes: []string{"machine1", "machine2"},
			sss:   []*apps.StatefulSet{{Spec: apps.StatefulSetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"baz": "blah"}}}}},
			// We use StatefulSet, instead of ReplicationController. The result should be exactly as above.
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 0}, {Name: "machine2", Score: 50}},
			name:         "Another stateful set with partial pod label matches",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nodes := makeNodeList(test.nodes)
			snapshot := cache.NewSnapshot(test.pods, nodes)
			ctx := context.Background()
			informerFactory, err := populateAndStartInformers(ctx, test.rcs, test.rss, test.services, test.sss)
			if err != nil {
				t.Errorf("error creating informerFactory: %+v", err)
			}
			fh, err := framework.NewFramework(nil, nil, nil, framework.WithSnapshotSharedLister(snapshot), framework.WithInformerFactory(informerFactory))
			if err != nil {
				t.Errorf("error creating new framework handle: %+v", err)
			}

			state := framework.NewCycleState()

			plugin := &DefaultPodTopologySpread{
				handle: fh,
			}

			status := plugin.PostFilter(ctx, state, test.pod, nodes, nil)
			if !status.IsSuccess() {
				t.Fatalf("unexpected error: %v", status)
			}

			var gotList framework.NodeScoreList
			for _, nodeName := range test.nodes {
				score, status := plugin.Score(ctx, state, test.pod, nodeName)
				if !status.IsSuccess() {
					t.Errorf("unexpected error: %v", status)
				}
				gotList = append(gotList, framework.NodeScore{Name: nodeName, Score: score})
			}

			status = plugin.ScoreExtensions().NormalizeScore(ctx, state, test.pod, gotList)
			if !status.IsSuccess() {
				t.Errorf("unexpected error: %v", status)
			}

			if !reflect.DeepEqual(test.expectedList, gotList) {
				t.Errorf("expected:\n\t%+v,\ngot:\n\t%+v", test.expectedList, gotList)
			}
		})
	}
}

func buildPod(nodeName string, labels map[string]string, ownerRefs []metav1.OwnerReference) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Labels: labels, OwnerReferences: ownerRefs},
		Spec:       v1.PodSpec{NodeName: nodeName},
	}
}

func TestZoneSelectorSpreadPriority(t *testing.T) {
	labels1 := map[string]string{
		"label1": "l1",
		"baz":    "blah",
	}
	labels2 := map[string]string{
		"label2": "l2",
		"baz":    "blah",
	}

	const nodeMachine1Zone1 = "machine1.zone1"
	const nodeMachine1Zone2 = "machine1.zone2"
	const nodeMachine2Zone2 = "machine2.zone2"
	const nodeMachine1Zone3 = "machine1.zone3"
	const nodeMachine2Zone3 = "machine2.zone3"
	const nodeMachine3Zone3 = "machine3.zone3"

	buildNodeLabels := func(failureDomain string) map[string]string {
		labels := map[string]string{
			v1.LabelZoneFailureDomain: failureDomain,
		}
		return labels
	}
	labeledNodes := map[string]map[string]string{
		nodeMachine1Zone1: buildNodeLabels("zone1"),
		nodeMachine1Zone2: buildNodeLabels("zone2"),
		nodeMachine2Zone2: buildNodeLabels("zone2"),
		nodeMachine1Zone3: buildNodeLabels("zone3"),
		nodeMachine2Zone3: buildNodeLabels("zone3"),
		nodeMachine3Zone3: buildNodeLabels("zone3"),
	}

	tests := []struct {
		pod          *v1.Pod
		pods         []*v1.Pod
		rcs          []*v1.ReplicationController
		rss          []*apps.ReplicaSet
		services     []*v1.Service
		sss          []*apps.StatefulSet
		expectedList framework.NodeScoreList
		name         string
	}{
		{
			pod: new(v1.Pod),
			expectedList: []framework.NodeScore{
				{Name: nodeMachine1Zone1, Score: framework.MaxNodeScore},
				{Name: nodeMachine1Zone2, Score: framework.MaxNodeScore},
				{Name: nodeMachine2Zone2, Score: framework.MaxNodeScore},
				{Name: nodeMachine1Zone3, Score: framework.MaxNodeScore},
				{Name: nodeMachine2Zone3, Score: framework.MaxNodeScore},
				{Name: nodeMachine3Zone3, Score: framework.MaxNodeScore},
			},
			name: "nothing scheduled",
		},
		{
			pod:  buildPod("", labels1, nil),
			pods: []*v1.Pod{buildPod(nodeMachine1Zone1, nil, nil)},
			expectedList: []framework.NodeScore{
				{Name: nodeMachine1Zone1, Score: framework.MaxNodeScore},
				{Name: nodeMachine1Zone2, Score: framework.MaxNodeScore},
				{Name: nodeMachine2Zone2, Score: framework.MaxNodeScore},
				{Name: nodeMachine1Zone3, Score: framework.MaxNodeScore},
				{Name: nodeMachine2Zone3, Score: framework.MaxNodeScore},
				{Name: nodeMachine3Zone3, Score: framework.MaxNodeScore},
			},
			name: "no services",
		},
		{
			pod:      buildPod("", labels1, nil),
			pods:     []*v1.Pod{buildPod(nodeMachine1Zone1, labels2, nil)},
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: map[string]string{"key": "value"}}}},
			expectedList: []framework.NodeScore{
				{Name: nodeMachine1Zone1, Score: framework.MaxNodeScore},
				{Name: nodeMachine1Zone2, Score: framework.MaxNodeScore},
				{Name: nodeMachine2Zone2, Score: framework.MaxNodeScore},
				{Name: nodeMachine1Zone3, Score: framework.MaxNodeScore},
				{Name: nodeMachine2Zone3, Score: framework.MaxNodeScore},
				{Name: nodeMachine3Zone3, Score: framework.MaxNodeScore},
			},
			name: "different services",
		},
		{
			pod: buildPod("", labels1, nil),
			pods: []*v1.Pod{
				buildPod(nodeMachine1Zone1, labels2, nil),
				buildPod(nodeMachine1Zone2, labels2, nil),
			},
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: labels1}}},
			expectedList: []framework.NodeScore{
				{Name: nodeMachine1Zone1, Score: framework.MaxNodeScore},
				{Name: nodeMachine1Zone2, Score: framework.MaxNodeScore},
				{Name: nodeMachine2Zone2, Score: framework.MaxNodeScore},
				{Name: nodeMachine1Zone3, Score: framework.MaxNodeScore},
				{Name: nodeMachine2Zone3, Score: framework.MaxNodeScore},
				{Name: nodeMachine3Zone3, Score: framework.MaxNodeScore},
			},
			name: "two pods, 0 matching",
		},
		{
			pod: buildPod("", labels1, nil),
			pods: []*v1.Pod{
				buildPod(nodeMachine1Zone1, labels2, nil),
				buildPod(nodeMachine1Zone2, labels1, nil),
			},
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: labels1}}},
			expectedList: []framework.NodeScore{
				{Name: nodeMachine1Zone1, Score: framework.MaxNodeScore},
				{Name: nodeMachine1Zone2, Score: 0},  // Already have pod on machine
				{Name: nodeMachine2Zone2, Score: 33}, // Already have pod in zone
				{Name: nodeMachine1Zone3, Score: framework.MaxNodeScore},
				{Name: nodeMachine2Zone3, Score: framework.MaxNodeScore},
				{Name: nodeMachine3Zone3, Score: framework.MaxNodeScore},
			},
			name: "two pods, 1 matching (in z2)",
		},
		{
			pod: buildPod("", labels1, nil),
			pods: []*v1.Pod{
				buildPod(nodeMachine1Zone1, labels2, nil),
				buildPod(nodeMachine1Zone2, labels1, nil),
				buildPod(nodeMachine2Zone2, labels1, nil),
				buildPod(nodeMachine1Zone3, labels2, nil),
				buildPod(nodeMachine2Zone3, labels1, nil),
			},
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: labels1}}},
			expectedList: []framework.NodeScore{
				{Name: nodeMachine1Zone1, Score: framework.MaxNodeScore},
				{Name: nodeMachine1Zone2, Score: 0},  // Pod on node
				{Name: nodeMachine2Zone2, Score: 0},  // Pod on node
				{Name: nodeMachine1Zone3, Score: 66}, // Pod in zone
				{Name: nodeMachine2Zone3, Score: 33}, // Pod on node
				{Name: nodeMachine3Zone3, Score: 66}, // Pod in zone
			},
			name: "five pods, 3 matching (z2=2, z3=1)",
		},
		{
			pod: buildPod("", labels1, nil),
			pods: []*v1.Pod{
				buildPod(nodeMachine1Zone1, labels1, nil),
				buildPod(nodeMachine1Zone2, labels1, nil),
				buildPod(nodeMachine2Zone2, labels2, nil),
				buildPod(nodeMachine1Zone3, labels1, nil),
			},
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: labels1}}},
			expectedList: []framework.NodeScore{
				{Name: nodeMachine1Zone1, Score: 0},  // Pod on node
				{Name: nodeMachine1Zone2, Score: 0},  // Pod on node
				{Name: nodeMachine2Zone2, Score: 33}, // Pod in zone
				{Name: nodeMachine1Zone3, Score: 0},  // Pod on node
				{Name: nodeMachine2Zone3, Score: 33}, // Pod in zone
				{Name: nodeMachine3Zone3, Score: 33}, // Pod in zone
			},
			name: "four pods, 3 matching (z1=1, z2=1, z3=1)",
		},
		{
			pod: buildPod("", labels1, nil),
			pods: []*v1.Pod{
				buildPod(nodeMachine1Zone1, labels1, nil),
				buildPod(nodeMachine1Zone2, labels1, nil),
				buildPod(nodeMachine1Zone3, labels1, nil),
				buildPod(nodeMachine2Zone2, labels2, nil),
			},
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: labels1}}},
			expectedList: []framework.NodeScore{
				{Name: nodeMachine1Zone1, Score: 0},  // Pod on node
				{Name: nodeMachine1Zone2, Score: 0},  // Pod on node
				{Name: nodeMachine2Zone2, Score: 33}, // Pod in zone
				{Name: nodeMachine1Zone3, Score: 0},  // Pod on node
				{Name: nodeMachine2Zone3, Score: 33}, // Pod in zone
				{Name: nodeMachine3Zone3, Score: 33}, // Pod in zone
			},
			name: "four pods, 3 matching (z1=1, z2=1, z3=1)",
		},
		{
			pod: buildPod("", labels1, controllerRef("ReplicationController", "name", "abc123")),
			pods: []*v1.Pod{
				buildPod(nodeMachine1Zone3, labels1, controllerRef("ReplicationController", "name", "abc123")),
				buildPod(nodeMachine1Zone2, labels1, controllerRef("ReplicationController", "name", "abc123")),
				buildPod(nodeMachine1Zone3, labels1, controllerRef("ReplicationController", "name", "abc123")),
			},
			rcs: []*v1.ReplicationController{{Spec: v1.ReplicationControllerSpec{Selector: labels1}}},
			expectedList: []framework.NodeScore{
				// Note that because we put two pods on the same node (nodeMachine1Zone3),
				// the values here are questionable for zone2, in particular for nodeMachine1Zone2.
				// However they kind of make sense; zone1 is still most-highly favored.
				// zone3 is in general least favored, and m1.z3 particularly low priority.
				// We would probably prefer to see a bigger gap between putting a second
				// pod on m1.z2 and putting a pod on m2.z2, but the ordering is correct.
				// This is also consistent with what we have already.
				{Name: nodeMachine1Zone1, Score: framework.MaxNodeScore}, // No pods in zone
				{Name: nodeMachine1Zone2, Score: 50},                     // Pod on node
				{Name: nodeMachine2Zone2, Score: 66},                     // Pod in zone
				{Name: nodeMachine1Zone3, Score: 0},                      // Two pods on node
				{Name: nodeMachine2Zone3, Score: 33},                     // Pod in zone
				{Name: nodeMachine3Zone3, Score: 33},                     // Pod in zone
			},
			name: "Replication controller spreading (z1=0, z2=1, z3=2)",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nodes := makeLabeledNodeList(labeledNodes)
			snapshot := cache.NewSnapshot(test.pods, nodes)
			ctx := context.Background()
			informerFactory, err := populateAndStartInformers(ctx, test.rcs, test.rss, test.services, test.sss)
			if err != nil {
				t.Errorf("error creating informerFactory: %+v", err)
			}
			fh, err := framework.NewFramework(nil, nil, nil, framework.WithSnapshotSharedLister(snapshot), framework.WithInformerFactory(informerFactory))
			if err != nil {
				t.Errorf("error creating new framework handle: %+v", err)
			}

			plugin := &DefaultPodTopologySpread{
				handle: fh,
			}

			state := framework.NewCycleState()
			status := plugin.PostFilter(ctx, state, test.pod, nodes, nil)
			if !status.IsSuccess() {
				t.Fatalf("unexpected error: %v", status)
			}

			var gotList framework.NodeScoreList
			for _, n := range nodes {
				nodeName := n.ObjectMeta.Name
				score, status := plugin.Score(ctx, state, test.pod, nodeName)
				if !status.IsSuccess() {
					t.Errorf("unexpected error: %v", status)
				}
				gotList = append(gotList, framework.NodeScore{Name: nodeName, Score: score})
			}

			status = plugin.ScoreExtensions().NormalizeScore(ctx, state, test.pod, gotList)
			if !status.IsSuccess() {
				t.Errorf("unexpected error: %v", status)
			}

			sortNodeScoreList(test.expectedList)
			sortNodeScoreList(gotList)
			if !reflect.DeepEqual(test.expectedList, gotList) {
				t.Errorf("expected:\n\t%+v,\ngot:\n\t%+v", test.expectedList, gotList)
			}
		})
	}
}

func populateAndStartInformers(ctx context.Context, rcs []*v1.ReplicationController, rss []*apps.ReplicaSet, services []*v1.Service, sss []*apps.StatefulSet) (informers.SharedInformerFactory, error) {
	objects := make([]runtime.Object, 0, len(rcs)+len(rss)+len(services)+len(sss))
	for _, rc := range rcs {
		objects = append(objects, rc.DeepCopyObject())
	}
	for _, rs := range rss {
		objects = append(objects, rs.DeepCopyObject())
	}
	for _, service := range services {
		objects = append(objects, service.DeepCopyObject())
	}
	for _, ss := range sss {
		objects = append(objects, ss.DeepCopyObject())
	}
	client := clientsetfake.NewSimpleClientset(objects...)
	informerFactory := informers.NewSharedInformerFactory(client, 0)

	// Because we use an informer factory, we need to make requests for the specific informers we want before calling Start()
	_ = informerFactory.Core().V1().Services().Lister()
	_ = informerFactory.Core().V1().ReplicationControllers().Lister()
	_ = informerFactory.Apps().V1().ReplicaSets().Lister()
	_ = informerFactory.Apps().V1().StatefulSets().Lister()
	informerFactory.Start(ctx.Done())
	caches := informerFactory.WaitForCacheSync(ctx.Done())
	for _, synced := range caches {
		if !synced {
			return nil, fmt.Errorf("error waiting for informer cache sync")
		}
	}
	return informerFactory, nil
}

func makeLabeledNodeList(nodeMap map[string]map[string]string) []*v1.Node {
	nodes := make([]*v1.Node, 0, len(nodeMap))
	for nodeName, labels := range nodeMap {
		nodes = append(nodes, &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName, Labels: labels}})
	}
	return nodes
}

func makeNodeList(nodeNames []string) []*v1.Node {
	nodes := make([]*v1.Node, 0, len(nodeNames))
	for _, nodeName := range nodeNames {
		nodes = append(nodes, &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName}})
	}
	return nodes
}

func sortNodeScoreList(out framework.NodeScoreList) {
	sort.Slice(out, func(i, j int) bool {
		if out[i].Score == out[j].Score {
			return out[i].Name < out[j].Name
		}
		return out[i].Score < out[j].Score
	})
}

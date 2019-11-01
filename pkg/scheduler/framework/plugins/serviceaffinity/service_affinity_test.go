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

package serviceaffinity

import (
	"context"
	"reflect"
	"sort"
	"testing"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/priorities"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/migration"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	fakelisters "k8s.io/kubernetes/pkg/scheduler/listers/fake"
	nodeinfosnapshot "k8s.io/kubernetes/pkg/scheduler/nodeinfo/snapshot"
)

func TestServiceAffinityScore(t *testing.T) {
	labels1 := map[string]string{
		"foo": "bar",
		"baz": "blah",
	}
	labels2 := map[string]string{
		"bar": "foo",
		"baz": "blah",
	}
	zone1 := map[string]string{
		"zone": "zone1",
	}
	zone2 := map[string]string{
		"zone": "zone2",
	}
	nozone := map[string]string{
		"name": "value",
	}
	zone0Spec := v1.PodSpec{
		NodeName: "machine01",
	}
	zone1Spec := v1.PodSpec{
		NodeName: "machine11",
	}
	zone2Spec := v1.PodSpec{
		NodeName: "machine21",
	}
	labeledNodes := map[string]map[string]string{
		"machine01": nozone, "machine02": nozone,
		"machine11": zone1, "machine12": zone1,
		"machine21": zone2, "machine22": zone2,
	}
	tests := []struct {
		pod          *v1.Pod
		pods         []*v1.Pod
		nodes        map[string]map[string]string
		services     []*v1.Service
		expectedList framework.NodeScoreList
		name         string
	}{
		{
			pod:   new(v1.Pod),
			nodes: labeledNodes,
			expectedList: []framework.NodeScore{{Name: "machine11", Score: framework.MaxNodeScore}, {Name: "machine12", Score: framework.MaxNodeScore},
				{Name: "machine21", Score: framework.MaxNodeScore}, {Name: "machine22", Score: framework.MaxNodeScore},
				{Name: "machine01", Score: 0}, {Name: "machine02", Score: 0}},
			name: "nothing scheduled",
		},
		{
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			pods:  []*v1.Pod{{Spec: zone1Spec}},
			nodes: labeledNodes,
			expectedList: []framework.NodeScore{{Name: "machine11", Score: framework.MaxNodeScore}, {Name: "machine12", Score: framework.MaxNodeScore},
				{Name: "machine21", Score: framework.MaxNodeScore}, {Name: "machine22", Score: framework.MaxNodeScore},
				{Name: "machine01", Score: 0}, {Name: "machine02", Score: 0}},
			name: "no services",
		},
		{
			pod:      &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			pods:     []*v1.Pod{{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2}}},
			nodes:    labeledNodes,
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: map[string]string{"key": "value"}}}},
			expectedList: []framework.NodeScore{{Name: "machine11", Score: framework.MaxNodeScore}, {Name: "machine12", Score: framework.MaxNodeScore},
				{Name: "machine21", Score: framework.MaxNodeScore}, {Name: "machine22", Score: framework.MaxNodeScore},
				{Name: "machine01", Score: 0}, {Name: "machine02", Score: 0}},
			name: "different services",
		},
		{
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			pods: []*v1.Pod{
				{Spec: zone0Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			},
			nodes:    labeledNodes,
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: labels1}}},
			expectedList: []framework.NodeScore{{Name: "machine11", Score: framework.MaxNodeScore}, {Name: "machine12", Score: framework.MaxNodeScore},
				{Name: "machine21", Score: 0}, {Name: "machine22", Score: 0},
				{Name: "machine01", Score: 0}, {Name: "machine02", Score: 0}},
			name: "three pods, one service pod",
		},
		{
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			pods: []*v1.Pod{
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			},
			nodes:    labeledNodes,
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: labels1}}},
			expectedList: []framework.NodeScore{{Name: "machine11", Score: 50}, {Name: "machine12", Score: 50},
				{Name: "machine21", Score: 50}, {Name: "machine22", Score: 50},
				{Name: "machine01", Score: 0}, {Name: "machine02", Score: 0}},
			name: "three pods, two service pods on different machines",
		},
		{
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: labels1, Namespace: metav1.NamespaceDefault}},
			pods: []*v1.Pod{
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, Namespace: metav1.NamespaceDefault}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1, Namespace: "ns1"}},
			},
			nodes:    labeledNodes,
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: labels1}, ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceDefault}}},
			expectedList: []framework.NodeScore{{Name: "machine11", Score: 0}, {Name: "machine12", Score: 0},
				{Name: "machine21", Score: framework.MaxNodeScore}, {Name: "machine22", Score: framework.MaxNodeScore},
				{Name: "machine01", Score: 0}, {Name: "machine02", Score: 0}},
			name: "three service label match pods in different namespaces",
		},
		{
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			pods: []*v1.Pod{
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			},
			nodes:    labeledNodes,
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: labels1}}},
			expectedList: []framework.NodeScore{{Name: "machine11", Score: 66}, {Name: "machine12", Score: 66},
				{Name: "machine21", Score: 33}, {Name: "machine22", Score: 33},
				{Name: "machine01", Score: 0}, {Name: "machine02", Score: 0}},
			name: "four pods, three service pods",
		},
		{
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			pods: []*v1.Pod{
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			},
			nodes:    labeledNodes,
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: map[string]string{"baz": "blah"}}}},
			expectedList: []framework.NodeScore{{Name: "machine11", Score: 33}, {Name: "machine12", Score: 33},
				{Name: "machine21", Score: 66}, {Name: "machine22", Score: 66},
				{Name: "machine01", Score: 0}, {Name: "machine02", Score: 0}},
			name: "service with partial pod label matches",
		},
		{
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			pods: []*v1.Pod{
				{Spec: zone0Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			},
			nodes:    labeledNodes,
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: labels1}}},
			expectedList: []framework.NodeScore{{Name: "machine11", Score: 75}, {Name: "machine12", Score: 75},
				{Name: "machine21", Score: 50}, {Name: "machine22", Score: 50},
				{Name: "machine01", Score: 0}, {Name: "machine02", Score: 0}},
			name: "service pod on non-zoned node",
		},
	}
	// these local variables just make sure controllerLister\replicaSetLister\statefulSetLister not nil
	// when construct metaDataProducer
	sss := []*apps.StatefulSet{{Spec: apps.StatefulSetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}}}}}
	rcs := []*v1.ReplicationController{{Spec: v1.ReplicationControllerSpec{Selector: map[string]string{"foo": "bar"}}}}
	rss := []*apps.ReplicaSet{{Spec: apps.ReplicaSetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}}}}}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nodes := makeLabeledNodeList(labeledNodes)
			snapshot := nodeinfosnapshot.NewSnapshot(test.pods, nodes)
			fh, _ := framework.NewFramework(nil, nil, nil, framework.WithNodeInfoSnapshot(snapshot))
			serviceLister := fakelisters.ServiceLister(test.services)
			priorityMapFunction, priorityReduceFunction := priorities.NewServiceAntiAffinityPriority(snapshot.Pods(), serviceLister, "zone")

			p := &ServiceAffinity{
				handle:                 fh,
				priorityMapFunction:    priorityMapFunction,
				priorityReduceFunction: priorityReduceFunction,
			}
			metaDataProducer := priorities.NewPriorityMetadataFactory(
				fakelisters.ServiceLister(test.services),
				fakelisters.ControllerLister(rcs),
				fakelisters.ReplicaSetLister(rss),
				fakelisters.StatefulSetLister(sss))
			metaData := metaDataProducer(test.pod, nodes, snapshot)
			state := framework.NewCycleState()
			state.Write(migration.PrioritiesStateKey, &migration.PrioritiesStateData{Reference: metaData})

			var gotList framework.NodeScoreList
			for _, n := range makeLabeledNodeList(test.nodes) {
				score, status := p.Score(context.Background(), state, test.pod, n.Name)
				if !status.IsSuccess() {
					t.Errorf("unexpected error: %v", status)
				}
				gotList = append(gotList, framework.NodeScore{Name: n.Name, Score: score})
			}

			status := p.ScoreExtensions().NormalizeScore(context.Background(), state, test.pod, gotList)
			if !status.IsSuccess() {
				t.Errorf("unexpected error: %v", status)
			}

			// sort the two lists to avoid failures on account of different ordering
			sortNodeScoreList(test.expectedList)
			sortNodeScoreList(gotList)
			if !reflect.DeepEqual(test.expectedList, gotList) {
				t.Errorf("expected %#v, got %#v", test.expectedList, gotList)
			}
		})
	}
}

func makeLabeledNodeList(nodeMap map[string]map[string]string) []*v1.Node {
	nodes := make([]*v1.Node, 0, len(nodeMap))
	for nodeName, labels := range nodeMap {
		nodes = append(nodes, &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName, Labels: labels}})
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

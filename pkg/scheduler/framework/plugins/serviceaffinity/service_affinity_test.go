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
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/migration"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	fakelisters "k8s.io/kubernetes/pkg/scheduler/listers/fake"
	nodeinfosnapshot "k8s.io/kubernetes/pkg/scheduler/nodeinfo/snapshot"
)

func TestServiceAffinity(t *testing.T) {
	selector := map[string]string{"foo": "bar"}
	labels1 := map[string]string{
		"region": "r1",
		"zone":   "z11",
	}
	labels2 := map[string]string{
		"region": "r1",
		"zone":   "z12",
	}
	labels3 := map[string]string{
		"region": "r2",
		"zone":   "z21",
	}
	labels4 := map[string]string{
		"region": "r2",
		"zone":   "z22",
	}
	node1 := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "machine1", Labels: labels1}}
	node2 := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "machine2", Labels: labels2}}
	node3 := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "machine3", Labels: labels3}}
	node4 := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "machine4", Labels: labels4}}
	node5 := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "machine5", Labels: labels4}}
	tests := []struct {
		name     string
		pod      *v1.Pod
		pods     []*v1.Pod
		services []*v1.Service
		node     *v1.Node
		labels   []string
		res      framework.Code
	}{
		{
			name:   "nothing scheduled",
			pod:    new(v1.Pod),
			node:   &node1,
			labels: []string{"region"},
			res:    framework.Success,
		},
		{
			name:   "pod with region label match",
			pod:    &v1.Pod{Spec: v1.PodSpec{NodeSelector: map[string]string{"region": "r1"}}},
			node:   &node1,
			labels: []string{"region"},
			res:    framework.Success,
		},
		{
			name:   "pod with region label mismatch",
			pod:    &v1.Pod{Spec: v1.PodSpec{NodeSelector: map[string]string{"region": "r2"}}},
			node:   &node1,
			labels: []string{"region"},
			res:    framework.Unschedulable,
		},
		{
			name:     "service pod on same node",
			pod:      &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: selector}},
			pods:     []*v1.Pod{{Spec: v1.PodSpec{NodeName: "machine1"}, ObjectMeta: metav1.ObjectMeta{Labels: selector}}},
			node:     &node1,
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: selector}}},
			labels:   []string{"region"},
			res:      framework.Success,
		},
		{
			name:     "service pod on different node, region match",
			pod:      &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: selector}},
			pods:     []*v1.Pod{{Spec: v1.PodSpec{NodeName: "machine2"}, ObjectMeta: metav1.ObjectMeta{Labels: selector}}},
			node:     &node1,
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: selector}}},
			labels:   []string{"region"},
			res:      framework.Success,
		},
		{
			name:     "service pod on different node, region mismatch",
			pod:      &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: selector}},
			pods:     []*v1.Pod{{Spec: v1.PodSpec{NodeName: "machine3"}, ObjectMeta: metav1.ObjectMeta{Labels: selector}}},
			node:     &node1,
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: selector}}},
			labels:   []string{"region"},
			res:      framework.Unschedulable,
		},
		{
			name:     "service in different namespace, region mismatch",
			pod:      &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: selector, Namespace: "ns1"}},
			pods:     []*v1.Pod{{Spec: v1.PodSpec{NodeName: "machine3"}, ObjectMeta: metav1.ObjectMeta{Labels: selector, Namespace: "ns1"}}},
			node:     &node1,
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: selector}, ObjectMeta: metav1.ObjectMeta{Namespace: "ns2"}}},
			labels:   []string{"region"},
			res:      framework.Success,
		},
		{
			name:     "pod in different namespace, region mismatch",
			pod:      &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: selector, Namespace: "ns1"}},
			pods:     []*v1.Pod{{Spec: v1.PodSpec{NodeName: "machine3"}, ObjectMeta: metav1.ObjectMeta{Labels: selector, Namespace: "ns2"}}},
			node:     &node1,
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: selector}, ObjectMeta: metav1.ObjectMeta{Namespace: "ns1"}}},
			labels:   []string{"region"},
			res:      framework.Success,
		},
		{
			name:     "service and pod in same namespace, region mismatch",
			pod:      &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: selector, Namespace: "ns1"}},
			pods:     []*v1.Pod{{Spec: v1.PodSpec{NodeName: "machine3"}, ObjectMeta: metav1.ObjectMeta{Labels: selector, Namespace: "ns1"}}},
			node:     &node1,
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: selector}, ObjectMeta: metav1.ObjectMeta{Namespace: "ns1"}}},
			labels:   []string{"region"},
			res:      framework.Unschedulable,
		},
		{
			name:     "service pod on different node, multiple labels, not all match",
			pod:      &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: selector}},
			pods:     []*v1.Pod{{Spec: v1.PodSpec{NodeName: "machine2"}, ObjectMeta: metav1.ObjectMeta{Labels: selector}}},
			node:     &node1,
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: selector}}},
			labels:   []string{"region", "zone"},
			res:      framework.Unschedulable,
		},
		{
			name:     "service pod on different node, multiple labels, all match",
			pod:      &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: selector}},
			pods:     []*v1.Pod{{Spec: v1.PodSpec{NodeName: "machine5"}, ObjectMeta: metav1.ObjectMeta{Labels: selector}}},
			node:     &node4,
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: selector}}},
			labels:   []string{"region", "zone"},
			res:      framework.Success,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nodes := []*v1.Node{&node1, &node2, &node3, &node4, &node5}
			snapshot := nodeinfosnapshot.NewSnapshot(test.pods, nodes)

			predicate, precompute := predicates.NewServiceAffinityPredicate(snapshot.NodeInfos(), snapshot.Pods(), fakelisters.ServiceLister(test.services), test.labels)
			predicates.RegisterPredicateMetadataProducer("ServiceAffinityMetaProducer", precompute)

			p := &ServiceAffinity{
				predicate: predicate,
			}

			meta := predicates.GetPredicateMetadata(test.pod, snapshot)
			state := framework.NewCycleState()
			state.Write(migration.PredicatesStateKey, &migration.PredicatesStateData{Reference: meta})

			status := p.Filter(context.Background(), state, test.pod, snapshot.NodeInfoMap[test.node.Name])
			if status.Code() != test.res {
				t.Errorf("Status mismatch. got: %v, want: %v", status.Code(), test.res)
			}
		})
	}
}

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
	"fmt"
	"reflect"
	"sort"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	fakeframework "k8s.io/kubernetes/pkg/scheduler/framework/fake"
	"k8s.io/kubernetes/pkg/scheduler/internal/cache"
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
			snapshot := cache.NewSnapshot(test.pods, nodes)

			p := &ServiceAffinity{
				sharedLister:  snapshot,
				serviceLister: fakeframework.ServiceLister(test.services),
				args: config.ServiceAffinityArgs{
					AffinityLabels: test.labels,
				},
			}

			state := framework.NewCycleState()
			if s := p.PreFilter(context.Background(), state, test.pod); !s.IsSuccess() {
				t.Errorf("PreFilter failed: %v", s.Message())
			}
			nodeInfo := mustGetNodeInfo(t, snapshot, test.node.Name)
			status := p.Filter(context.Background(), state, test.pod, nodeInfo)
			if status.Code() != test.res {
				t.Errorf("Status mismatch. got: %v, want: %v", status.Code(), test.res)
			}
		})
	}
}
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
	zone1Rack1 := map[string]string{
		"zone": "zone1",
		"rack": "rack1",
	}
	zone1Rack2 := map[string]string{
		"zone": "zone1",
		"rack": "rack2",
	}
	zone2 := map[string]string{
		"zone": "zone2",
	}
	zone2Rack1 := map[string]string{
		"zone": "zone2",
		"rack": "rack1",
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
	nodesWithZoneAndRackLabels := map[string]map[string]string{
		"machine01": nozone, "machine02": nozone,
		"machine11": zone1Rack1, "machine12": zone1Rack2,
		"machine21": zone2Rack1, "machine22": zone2Rack1,
	}
	tests := []struct {
		pod          *v1.Pod
		pods         []*v1.Pod
		nodes        map[string]map[string]string
		services     []*v1.Service
		labels       []string
		expectedList framework.NodeScoreList
		name         string
	}{
		{
			pod:    new(v1.Pod),
			nodes:  labeledNodes,
			labels: []string{"zone"},
			expectedList: []framework.NodeScore{{Name: "machine11", Score: framework.MaxNodeScore}, {Name: "machine12", Score: framework.MaxNodeScore},
				{Name: "machine21", Score: framework.MaxNodeScore}, {Name: "machine22", Score: framework.MaxNodeScore},
				{Name: "machine01", Score: 0}, {Name: "machine02", Score: 0}},
			name: "nothing scheduled",
		},
		{
			pod:    &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			pods:   []*v1.Pod{{Spec: zone1Spec}},
			nodes:  labeledNodes,
			labels: []string{"zone"},
			expectedList: []framework.NodeScore{{Name: "machine11", Score: framework.MaxNodeScore}, {Name: "machine12", Score: framework.MaxNodeScore},
				{Name: "machine21", Score: framework.MaxNodeScore}, {Name: "machine22", Score: framework.MaxNodeScore},
				{Name: "machine01", Score: 0}, {Name: "machine02", Score: 0}},
			name: "no services",
		},
		{
			pod:      &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			pods:     []*v1.Pod{{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2}}},
			nodes:    labeledNodes,
			labels:   []string{"zone"},
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
			labels:   []string{"zone"},
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
			labels:   []string{"zone"},
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
			labels:   []string{"zone"},
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
			labels:   []string{"zone"},
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
			labels:   []string{"zone"},
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
			labels:   []string{"zone"},
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: labels1}}},
			expectedList: []framework.NodeScore{{Name: "machine11", Score: 75}, {Name: "machine12", Score: 75},
				{Name: "machine21", Score: 50}, {Name: "machine22", Score: 50},
				{Name: "machine01", Score: 0}, {Name: "machine02", Score: 0}},
			name: "service pod on non-zoned node",
		},
		{
			pod: &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			pods: []*v1.Pod{
				{Spec: zone0Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: zone2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			},
			nodes:    nodesWithZoneAndRackLabels,
			labels:   []string{"zone", "rack"},
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: labels1}}},
			expectedList: []framework.NodeScore{{Name: "machine11", Score: 25}, {Name: "machine12", Score: 75},
				{Name: "machine21", Score: 25}, {Name: "machine22", Score: 25},
				{Name: "machine01", Score: 0}, {Name: "machine02", Score: 0}},
			name: "three pods, two service pods, with rack label",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nodes := makeLabeledNodeList(test.nodes)
			snapshot := cache.NewSnapshot(test.pods, nodes)
			serviceLister := fakeframework.ServiceLister(test.services)

			p := &ServiceAffinity{
				sharedLister:  snapshot,
				serviceLister: serviceLister,
				args: config.ServiceAffinityArgs{
					AntiAffinityLabelsPreference: test.labels,
				},
			}
			state := framework.NewCycleState()

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

func TestPreFilterStateAddRemovePod(t *testing.T) {
	var label1 = map[string]string{
		"region": "r1",
		"zone":   "z11",
	}
	var label2 = map[string]string{
		"region": "r1",
		"zone":   "z12",
	}
	var label3 = map[string]string{
		"region": "r2",
		"zone":   "z21",
	}
	selector1 := map[string]string{"foo": "bar"}

	tests := []struct {
		name         string
		pendingPod   *v1.Pod
		addedPod     *v1.Pod
		existingPods []*v1.Pod
		nodes        []*v1.Node
		services     []*v1.Service
	}{
		{
			name: "no anti-affinity or service affinity exist",
			pendingPod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pending", Labels: selector1},
			},
			existingPods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "p1", Labels: selector1},
					Spec: v1.PodSpec{NodeName: "nodeA"},
				},
				{ObjectMeta: metav1.ObjectMeta{Name: "p2"},
					Spec: v1.PodSpec{NodeName: "nodeC"},
				},
			},
			addedPod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "addedPod", Labels: selector1},
				Spec:       v1.PodSpec{NodeName: "nodeB"},
			},
			nodes: []*v1.Node{
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeA", Labels: label1}},
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeB", Labels: label2}},
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeC", Labels: label3}},
			},
		},
		{
			name: "metadata service-affinity data are updated correctly after adding and removing a pod",
			pendingPod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pending", Labels: selector1},
			},
			existingPods: []*v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "p1", Labels: selector1},
					Spec: v1.PodSpec{NodeName: "nodeA"},
				},
				{ObjectMeta: metav1.ObjectMeta{Name: "p2"},
					Spec: v1.PodSpec{NodeName: "nodeC"},
				},
			},
			addedPod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "addedPod", Labels: selector1},
				Spec:       v1.PodSpec{NodeName: "nodeB"},
			},
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: selector1}}},
			nodes: []*v1.Node{
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeA", Labels: label1}},
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeB", Labels: label2}},
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeC", Labels: label3}},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// getMeta creates predicate meta data given the list of pods.
			getState := func(pods []*v1.Pod) (*ServiceAffinity, *framework.CycleState, *preFilterState, *cache.Snapshot) {
				snapshot := cache.NewSnapshot(pods, test.nodes)

				p := &ServiceAffinity{
					sharedLister:  snapshot,
					serviceLister: fakeframework.ServiceLister(test.services),
					args: config.ServiceAffinityArgs{
						AffinityLabels: []string{"region", "zone"},
					},
				}
				cycleState := framework.NewCycleState()
				preFilterStatus := p.PreFilter(context.Background(), cycleState, test.pendingPod)
				if !preFilterStatus.IsSuccess() {
					t.Errorf("prefilter failed with status: %v", preFilterStatus)
				}

				plState, err := getPreFilterState(cycleState)
				if err != nil {
					t.Errorf("failed to get metadata from cycleState: %v", err)
				}

				return p, cycleState, plState, snapshot
			}

			sortState := func(plState *preFilterState) *preFilterState {
				sort.SliceStable(plState.matchingPodList, func(i, j int) bool {
					return plState.matchingPodList[i].Name < plState.matchingPodList[j].Name
				})
				sort.SliceStable(plState.matchingPodServices, func(i, j int) bool {
					return plState.matchingPodServices[i].Name < plState.matchingPodServices[j].Name
				})
				return plState
			}

			// allPodsState is the state produced when all pods, including test.addedPod are given to prefilter.
			_, _, plStateAllPods, _ := getState(append(test.existingPods, test.addedPod))

			// state is produced for test.existingPods (without test.addedPod).
			ipa, state, plState, snapshot := getState(test.existingPods)
			// clone the state so that we can compare it later when performing Remove.
			plStateOriginal, _ := plState.Clone().(*preFilterState)

			// Add test.addedPod to state1 and verify it is equal to allPodsState.
			nodeInfo := mustGetNodeInfo(t, snapshot, test.addedPod.Spec.NodeName)
			if err := ipa.AddPod(context.Background(), state, test.pendingPod, test.addedPod, nodeInfo); err != nil {
				t.Errorf("error adding pod to preFilterState: %v", err)
			}

			if !reflect.DeepEqual(sortState(plStateAllPods), sortState(plState)) {
				t.Errorf("State is not equal, got: %v, want: %v", plState, plStateAllPods)
			}

			// Remove the added pod pod and make sure it is equal to the original state.
			if err := ipa.RemovePod(context.Background(), state, test.pendingPod, test.addedPod, nodeInfo); err != nil {
				t.Errorf("error removing pod from preFilterState: %v", err)
			}
			if !reflect.DeepEqual(sortState(plStateOriginal), sortState(plState)) {
				t.Errorf("State is not equal, got: %v, want: %v", plState, plStateOriginal)
			}
		})
	}
}

func TestPreFilterStateClone(t *testing.T) {
	source := &preFilterState{
		matchingPodList: []*v1.Pod{
			{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}},
			{ObjectMeta: metav1.ObjectMeta{Name: "pod2"}},
		},
		matchingPodServices: []*v1.Service{
			{ObjectMeta: metav1.ObjectMeta{Name: "service1"}},
		},
	}

	clone := source.Clone()
	if clone == source {
		t.Errorf("Clone returned the exact same object!")
	}
	if !reflect.DeepEqual(clone, source) {
		t.Errorf("Copy is not equal to source!")
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

func mustGetNodeInfo(t *testing.T, snapshot *cache.Snapshot, name string) *framework.NodeInfo {
	t.Helper()
	nodeInfo, err := snapshot.NodeInfos().Get(name)
	if err != nil {
		t.Fatal(err)
	}
	return nodeInfo
}

func TestPreFilterDisabled(t *testing.T) {
	pod := &v1.Pod{}
	nodeInfo := framework.NewNodeInfo()
	node := v1.Node{}
	nodeInfo.SetNode(&node)
	p := &ServiceAffinity{
		args: config.ServiceAffinityArgs{
			AffinityLabels: []string{"region"},
		},
	}
	cycleState := framework.NewCycleState()
	gotStatus := p.Filter(context.Background(), cycleState, pod, nodeInfo)
	wantStatus := framework.AsStatus(fmt.Errorf(`error reading "PreFilterServiceAffinity" from cycleState: not found`))
	if !reflect.DeepEqual(gotStatus, wantStatus) {
		t.Errorf("status does not match: %v, want: %v", gotStatus, wantStatus)
	}
}

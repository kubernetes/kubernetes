/*
Copyright 2026 The Kubernetes Authors.

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

package preemption

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2/ktesting"
)

func TestCanRescheduleElsewhere(t *testing.T) {
	node1 := makeNode("node1", map[string]string{"hostname": "node1"})
	node2 := makeNode("node2", map[string]string{"hostname": "node2"})
	taintedNode := makeNode("node3", map[string]string{"hostname": "node3"})
	taintedNode.Spec.Taints = []v1.Taint{{Key: "dedicated", Value: "foo", Effect: v1.TaintEffectNoSchedule}}

	testCases := map[string]struct {
		pod         *v1.Pod
		current     string
		nodes       []*v1.Node
		wantResched bool
	}{
		"unconstrained pod can move to another node": {
			pod:         makePod("p", nil, nil, nil),
			current:     "node1",
			nodes:       []*v1.Node{node1, node2},
			wantResched: true,
		},
		"no other node: unconstrained pod is treated as pinned": {
			pod:         makePod("p", nil, nil, nil),
			current:     "node1",
			nodes:       []*v1.Node{node1},
			wantResched: false,
		},
		"nodeSelector matching another node allows rescheduling": {
			pod:         makePod("p", map[string]string{"hostname": "node2"}, nil, nil),
			current:     "node1",
			nodes:       []*v1.Node{node1, node2},
			wantResched: true,
		},
		"nodeSelector matching only the current node pins the pod": {
			pod:         makePod("p", map[string]string{"hostname": "node1"}, nil, nil),
			current:     "node1",
			nodes:       []*v1.Node{node1, node2},
			wantResched: false,
		},
		"required nodeAffinity matching another node allows rescheduling": {
			pod:         makePod("p", nil, affinityFor("node2"), nil),
			current:     "node1",
			nodes:       []*v1.Node{node1, node2},
			wantResched: true,
		},
		"required nodeAffinity matching only the current node pins the pod": {
			pod:         makePod("p", nil, affinityFor("node1"), nil),
			current:     "node1",
			nodes:       []*v1.Node{node1, node2},
			wantResched: false,
		},
		"untolerated NoSchedule taint on every other node pins the pod": {
			pod:         makePod("p", nil, nil, nil),
			current:     "node1",
			nodes:       []*v1.Node{node1, taintedNode},
			wantResched: false,
		},
		"tolerated taint on another node allows rescheduling": {
			pod: makePod("p", nil, nil, []v1.Toleration{
				{Key: "dedicated", Value: "foo", Operator: v1.TolerationOpEqual, Effect: v1.TaintEffectNoSchedule},
			}),
			current:     "node1",
			nodes:       []*v1.Node{node1, taintedNode},
			wantResched: true,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			got := CanRescheduleElsewhere(logger, tc.pod, tc.current, tc.nodes, false)
			if got != tc.wantResched {
				t.Errorf("CanRescheduleElsewhere() = %v, want %v", got, tc.wantResched)
			}
		})
	}
}

func makeNode(name string, labels map[string]string) *v1.Node {
	return &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: name, Labels: labels},
	}
}

func makePod(name string, nodeSelector map[string]string, affinity *v1.Affinity, tolerations []v1.Toleration) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: v1.PodSpec{
			NodeSelector: nodeSelector,
			Affinity:     affinity,
			Tolerations:  tolerations,
		},
	}
}

func affinityFor(nodeName string) *v1.Affinity {
	return &v1.Affinity{
		NodeAffinity: &v1.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
				NodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchExpressions: []v1.NodeSelectorRequirement{
							{Key: "hostname", Operator: v1.NodeSelectorOpIn, Values: []string{nodeName}},
						},
					},
				},
			},
		},
	}
}

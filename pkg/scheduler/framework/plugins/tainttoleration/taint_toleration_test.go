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

package tainttoleration

import (
	"context"
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/internal/cache"
	tf "k8s.io/kubernetes/pkg/scheduler/testing/framework"
)

func nodeWithTaints(nodeName string, taints []v1.Taint) *v1.Node {
	return &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: nodeName,
		},
		Spec: v1.NodeSpec{
			Taints: taints,
		},
	}
}

func podWithTolerations(podName string, tolerations []v1.Toleration) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			Tolerations: tolerations,
		},
	}
}

func TestTaintTolerationScore(t *testing.T) {
	tests := []struct {
		name         string
		pod          *v1.Pod
		nodes        []*v1.Node
		expectedList framework.NodeScoreList
	}{
		// basic test case
		{
			name: "node with taints tolerated by the pod, gets a higher score than those node with intolerable taints",
			pod: podWithTolerations("pod1", []v1.Toleration{{
				Key:      "foo",
				Operator: v1.TolerationOpEqual,
				Value:    "bar",
				Effect:   v1.TaintEffectPreferNoSchedule,
			}}),
			nodes: []*v1.Node{
				nodeWithTaints("nodeA", []v1.Taint{{
					Key:    "foo",
					Value:  "bar",
					Effect: v1.TaintEffectPreferNoSchedule,
				}}),
				nodeWithTaints("nodeB", []v1.Taint{{
					Key:    "foo",
					Value:  "blah",
					Effect: v1.TaintEffectPreferNoSchedule,
				}}),
			},
			expectedList: []framework.NodeScore{
				{Name: "nodeA", Score: framework.MaxNodeScore},
				{Name: "nodeB", Score: 0},
			},
		},
		// the count of taints that are tolerated by pod, does not matter.
		{
			name: "the nodes that all of their taints are tolerated by the pod, get the same score, no matter how many tolerable taints a node has",
			pod: podWithTolerations("pod1", []v1.Toleration{
				{
					Key:      "cpu-type",
					Operator: v1.TolerationOpEqual,
					Value:    "arm64",
					Effect:   v1.TaintEffectPreferNoSchedule,
				}, {
					Key:      "disk-type",
					Operator: v1.TolerationOpEqual,
					Value:    "ssd",
					Effect:   v1.TaintEffectPreferNoSchedule,
				},
			}),
			nodes: []*v1.Node{
				nodeWithTaints("nodeA", []v1.Taint{}),
				nodeWithTaints("nodeB", []v1.Taint{
					{
						Key:    "cpu-type",
						Value:  "arm64",
						Effect: v1.TaintEffectPreferNoSchedule,
					},
				}),
				nodeWithTaints("nodeC", []v1.Taint{
					{
						Key:    "cpu-type",
						Value:  "arm64",
						Effect: v1.TaintEffectPreferNoSchedule,
					}, {
						Key:    "disk-type",
						Value:  "ssd",
						Effect: v1.TaintEffectPreferNoSchedule,
					},
				}),
			},
			expectedList: []framework.NodeScore{
				{Name: "nodeA", Score: framework.MaxNodeScore},
				{Name: "nodeB", Score: framework.MaxNodeScore},
				{Name: "nodeC", Score: framework.MaxNodeScore},
			},
		},
		// the count of taints on a node that are not tolerated by pod, matters.
		{
			name: "the more intolerable taints a node has, the lower score it gets.",
			pod: podWithTolerations("pod1", []v1.Toleration{{
				Key:      "foo",
				Operator: v1.TolerationOpEqual,
				Value:    "bar",
				Effect:   v1.TaintEffectPreferNoSchedule,
			}}),
			nodes: []*v1.Node{
				nodeWithTaints("nodeA", []v1.Taint{}),
				nodeWithTaints("nodeB", []v1.Taint{
					{
						Key:    "cpu-type",
						Value:  "arm64",
						Effect: v1.TaintEffectPreferNoSchedule,
					},
				}),
				nodeWithTaints("nodeC", []v1.Taint{
					{
						Key:    "cpu-type",
						Value:  "arm64",
						Effect: v1.TaintEffectPreferNoSchedule,
					}, {
						Key:    "disk-type",
						Value:  "ssd",
						Effect: v1.TaintEffectPreferNoSchedule,
					},
				}),
			},
			expectedList: []framework.NodeScore{
				{Name: "nodeA", Score: framework.MaxNodeScore},
				{Name: "nodeB", Score: 50},
				{Name: "nodeC", Score: 0},
			},
		},
		// taints-tolerations priority only takes care about the taints and tolerations that have effect PreferNoSchedule
		{
			name: "only taints and tolerations that have effect PreferNoSchedule are checked by taints-tolerations priority function",
			pod: podWithTolerations("pod1", []v1.Toleration{
				{
					Key:      "cpu-type",
					Operator: v1.TolerationOpEqual,
					Value:    "arm64",
					Effect:   v1.TaintEffectNoSchedule,
				}, {
					Key:      "disk-type",
					Operator: v1.TolerationOpEqual,
					Value:    "ssd",
					Effect:   v1.TaintEffectNoSchedule,
				},
			}),
			nodes: []*v1.Node{
				nodeWithTaints("nodeA", []v1.Taint{}),
				nodeWithTaints("nodeB", []v1.Taint{
					{
						Key:    "cpu-type",
						Value:  "arm64",
						Effect: v1.TaintEffectNoSchedule,
					},
				}),
				nodeWithTaints("nodeC", []v1.Taint{
					{
						Key:    "cpu-type",
						Value:  "arm64",
						Effect: v1.TaintEffectPreferNoSchedule,
					}, {
						Key:    "disk-type",
						Value:  "ssd",
						Effect: v1.TaintEffectPreferNoSchedule,
					},
				}),
			},
			expectedList: []framework.NodeScore{
				{Name: "nodeA", Score: framework.MaxNodeScore},
				{Name: "nodeB", Score: framework.MaxNodeScore},
				{Name: "nodeC", Score: 0},
			},
		},
		{
			name: "Default behaviour No taints and tolerations, lands on node with no taints",
			//pod without tolerations
			pod: podWithTolerations("pod1", []v1.Toleration{}),
			nodes: []*v1.Node{
				//Node without taints
				nodeWithTaints("nodeA", []v1.Taint{}),
				nodeWithTaints("nodeB", []v1.Taint{
					{
						Key:    "cpu-type",
						Value:  "arm64",
						Effect: v1.TaintEffectPreferNoSchedule,
					},
				}),
			},
			expectedList: []framework.NodeScore{
				{Name: "nodeA", Score: framework.MaxNodeScore},
				{Name: "nodeB", Score: 0},
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			state := framework.NewCycleState()
			snapshot := cache.NewSnapshot(nil, test.nodes)
			fh, _ := runtime.NewFramework(ctx, nil, nil, runtime.WithSnapshotSharedLister(snapshot))

			p, err := New(ctx, nil, fh)
			if err != nil {
				t.Fatalf("creating plugin: %v", err)
			}
			status := p.(framework.PreScorePlugin).PreScore(ctx, state, test.pod, tf.BuildNodeInfos(test.nodes))
			if !status.IsSuccess() {
				t.Errorf("unexpected error: %v", status)
			}
			var gotList framework.NodeScoreList
			for _, n := range test.nodes {
				nodeName := n.ObjectMeta.Name
				score, status := p.(framework.ScorePlugin).Score(ctx, state, test.pod, nodeName)
				if !status.IsSuccess() {
					t.Errorf("unexpected error: %v", status)
				}
				gotList = append(gotList, framework.NodeScore{Name: nodeName, Score: score})
			}

			status = p.(framework.ScorePlugin).ScoreExtensions().NormalizeScore(ctx, state, test.pod, gotList)
			if !status.IsSuccess() {
				t.Errorf("unexpected error: %v", status)
			}

			if !reflect.DeepEqual(test.expectedList, gotList) {
				t.Errorf("expected:\n\t%+v,\ngot:\n\t%+v", test.expectedList, gotList)
			}
		})
	}
}

func TestTaintTolerationFilter(t *testing.T) {
	tests := []struct {
		name       string
		pod        *v1.Pod
		node       *v1.Node
		wantStatus *framework.Status
	}{
		{
			name: "A pod having no tolerations can't be scheduled onto a node with nonempty taints",
			pod:  podWithTolerations("pod1", []v1.Toleration{}),
			node: nodeWithTaints("nodeA", []v1.Taint{{Key: "dedicated", Value: "user1", Effect: "NoSchedule"}}),
			wantStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable,
				"node(s) had untolerated taint {dedicated: user1}"),
		},
		{
			name: "A pod which can be scheduled on a dedicated node assigned to user1 with effect NoSchedule",
			pod:  podWithTolerations("pod1", []v1.Toleration{{Key: "dedicated", Value: "user1", Effect: "NoSchedule"}}),
			node: nodeWithTaints("nodeA", []v1.Taint{{Key: "dedicated", Value: "user1", Effect: "NoSchedule"}}),
		},
		{
			name: "A pod which can't be scheduled on a dedicated node assigned to user2 with effect NoSchedule",
			pod:  podWithTolerations("pod1", []v1.Toleration{{Key: "dedicated", Operator: "Equal", Value: "user2", Effect: "NoSchedule"}}),
			node: nodeWithTaints("nodeA", []v1.Taint{{Key: "dedicated", Value: "user1", Effect: "NoSchedule"}}),
			wantStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable,
				"node(s) had untolerated taint {dedicated: user1}"),
		},
		{
			name: "A pod can be scheduled onto the node, with a toleration uses operator Exists that tolerates the taints on the node",
			pod:  podWithTolerations("pod1", []v1.Toleration{{Key: "foo", Operator: "Exists", Effect: "NoSchedule"}}),
			node: nodeWithTaints("nodeA", []v1.Taint{{Key: "foo", Value: "bar", Effect: "NoSchedule"}}),
		},
		{
			name: "A pod has multiple tolerations, node has multiple taints, all the taints are tolerated, pod can be scheduled onto the node",
			pod: podWithTolerations("pod1", []v1.Toleration{
				{Key: "dedicated", Operator: "Equal", Value: "user2", Effect: "NoSchedule"},
				{Key: "foo", Operator: "Exists", Effect: "NoSchedule"},
			}),
			node: nodeWithTaints("nodeA", []v1.Taint{
				{Key: "dedicated", Value: "user2", Effect: "NoSchedule"},
				{Key: "foo", Value: "bar", Effect: "NoSchedule"},
			}),
		},
		{
			name: "A pod has a toleration that keys and values match the taint on the node, but (non-empty) effect doesn't match, " +
				"can't be scheduled onto the node",
			pod:  podWithTolerations("pod1", []v1.Toleration{{Key: "foo", Operator: "Equal", Value: "bar", Effect: "PreferNoSchedule"}}),
			node: nodeWithTaints("nodeA", []v1.Taint{{Key: "foo", Value: "bar", Effect: "NoSchedule"}}),
			wantStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable,
				"node(s) had untolerated taint {foo: bar}"),
		},
		{
			name: "The pod has a toleration that keys and values match the taint on the node, the effect of toleration is empty, " +
				"and the effect of taint is NoSchedule. Pod can be scheduled onto the node",
			pod:  podWithTolerations("pod1", []v1.Toleration{{Key: "foo", Operator: "Equal", Value: "bar"}}),
			node: nodeWithTaints("nodeA", []v1.Taint{{Key: "foo", Value: "bar", Effect: "NoSchedule"}}),
		},
		{
			name: "The pod has a toleration that key and value don't match the taint on the node, " +
				"but the effect of taint on node is PreferNoSchedule. Pod can be scheduled onto the node",
			pod:  podWithTolerations("pod1", []v1.Toleration{{Key: "dedicated", Operator: "Equal", Value: "user2", Effect: "NoSchedule"}}),
			node: nodeWithTaints("nodeA", []v1.Taint{{Key: "dedicated", Value: "user1", Effect: "PreferNoSchedule"}}),
		},
		{
			name: "The pod has no toleration, " +
				"but the effect of taint on node is PreferNoSchedule. Pod can be scheduled onto the node",
			pod:  podWithTolerations("pod1", []v1.Toleration{}),
			node: nodeWithTaints("nodeA", []v1.Taint{{Key: "dedicated", Value: "user1", Effect: "PreferNoSchedule"}}),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			nodeInfo := framework.NewNodeInfo()
			nodeInfo.SetNode(test.node)
			p, err := New(ctx, nil, nil)
			if err != nil {
				t.Fatalf("creating plugin: %v", err)
			}
			gotStatus := p.(framework.FilterPlugin).Filter(ctx, nil, test.pod, nodeInfo)
			if !reflect.DeepEqual(gotStatus, test.wantStatus) {
				t.Errorf("status does not match: %v, want: %v", gotStatus, test.wantStatus)
			}
		})
	}
}

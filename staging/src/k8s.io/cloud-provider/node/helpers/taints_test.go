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

package helpers

import (
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/cloud-provider/node/testutil"

	"github.com/stretchr/testify/assert"
)

func TestRemoveTaintOffNode(t *testing.T) {
	tests := []struct {
		name           string
		nodeHandler    *testutil.FakeNodeHandler
		nodeName       string
		taintsToRemove []*v1.Taint
		expectedTaints []v1.Taint
		requestCount   int
	}{
		{
			name: "remove one taint from node",
			nodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node1",
						},
						Spec: v1.NodeSpec{
							Taints: []v1.Taint{
								{Key: "key1", Value: "value1", Effect: "NoSchedule"},
								{Key: "key2", Value: "value2", Effect: "NoExecute"},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			nodeName: "node1",
			taintsToRemove: []*v1.Taint{
				{Key: "key2", Value: "value2", Effect: "NoExecute"},
			},
			expectedTaints: []v1.Taint{
				{Key: "key1", Value: "value1", Effect: "NoSchedule"},
			},
			requestCount: 4,
		},
		{
			name: "remove multiple taints from node",
			nodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node1",
						},
						Spec: v1.NodeSpec{
							Taints: []v1.Taint{
								{Key: "key1", Value: "value1", Effect: "NoSchedule"},
								{Key: "key2", Value: "value2", Effect: "NoExecute"},
								{Key: "key3", Value: "value3", Effect: "NoSchedule"},
								{Key: "key4", Value: "value4", Effect: "NoExecute"},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			nodeName: "node1",
			taintsToRemove: []*v1.Taint{
				{Key: "key2", Value: "value2", Effect: "NoExecute"},
				{Key: "key3", Value: "value3", Effect: "NoSchedule"},
			},
			expectedTaints: []v1.Taint{
				{Key: "key1", Value: "value1", Effect: "NoSchedule"},
				{Key: "key4", Value: "value4", Effect: "NoExecute"},
			},
			requestCount: 4,
		},
		{
			name: "remove no-exist taints from node",
			nodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node1",
						},
						Spec: v1.NodeSpec{
							Taints: []v1.Taint{
								{Key: "key1", Value: "value1", Effect: "NoSchedule"},
								{Key: "key2", Value: "value2", Effect: "NoExecute"},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			nodeName: "node1",
			taintsToRemove: []*v1.Taint{
				{Key: "key3", Value: "value3", Effect: "NoSchedule"},
			},
			expectedTaints: []v1.Taint{
				{Key: "key1", Value: "value1", Effect: "NoSchedule"},
				{Key: "key2", Value: "value2", Effect: "NoExecute"},
			},
			requestCount: 2,
		},
		{
			name: "remove taint from node without taints",
			nodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node1",
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			nodeName: "node1",
			taintsToRemove: []*v1.Taint{
				{Key: "key3", Value: "value3", Effect: "NoSchedule"},
			},
			expectedTaints: nil,
			requestCount:   2,
		},
		{
			name: "remove empty taint list from node without taints",
			nodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node1",
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			nodeName:       "node1",
			taintsToRemove: []*v1.Taint{},
			expectedTaints: nil,
			requestCount:   2,
		},
		{
			name: "remove empty taint list from node",
			nodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node1",
						},
						Spec: v1.NodeSpec{
							Taints: []v1.Taint{
								{Key: "key1", Value: "value1", Effect: "NoSchedule"},
								{Key: "key2", Value: "value2", Effect: "NoExecute"},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			nodeName:       "node1",
			taintsToRemove: []*v1.Taint{},
			expectedTaints: []v1.Taint{
				{Key: "key1", Value: "value1", Effect: "NoSchedule"},
				{Key: "key2", Value: "value2", Effect: "NoExecute"},
			},
			requestCount: 2,
		},
	}
	for _, test := range tests {
		node, _ := test.nodeHandler.Get(test.nodeName, metav1.GetOptions{})
		err := RemoveTaintOffNode(test.nodeHandler, test.nodeName, node, test.taintsToRemove...)
		assert.NoError(t, err, "%s: RemoveTaintOffNode() error = %v", test.name, err)

		node, _ = test.nodeHandler.Get(test.nodeName, metav1.GetOptions{})
		assert.EqualValues(t, test.expectedTaints, node.Spec.Taints,
			"%s: failed to remove taint off node: expected %+v, got %+v",
			test.name, test.expectedTaints, node.Spec.Taints)

		assert.Equal(t, test.requestCount, test.nodeHandler.RequestCount,
			"%s: unexpected request count: expected %+v, got %+v",
			test.name, test.requestCount, test.nodeHandler.RequestCount)
	}
}

func TestAddOrUpdateTaintOnNode(t *testing.T) {
	tests := []struct {
		name           string
		nodeHandler    *testutil.FakeNodeHandler
		nodeName       string
		taintsToAdd    []*v1.Taint
		expectedTaints []v1.Taint
		requestCount   int
	}{
		{
			name: "add one taint on node",
			nodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node1",
						},
						Spec: v1.NodeSpec{
							Taints: []v1.Taint{
								{Key: "key1", Value: "value1", Effect: "NoSchedule"},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			nodeName: "node1",
			taintsToAdd: []*v1.Taint{
				{Key: "key2", Value: "value2", Effect: "NoExecute"},
			},
			expectedTaints: []v1.Taint{
				{Key: "key1", Value: "value1", Effect: "NoSchedule"},
				{Key: "key2", Value: "value2", Effect: "NoExecute"},
			},
			requestCount: 3,
		},
		{
			name: "add multiple taints to node",
			nodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node1",
						},
						Spec: v1.NodeSpec{
							Taints: []v1.Taint{
								{Key: "key1", Value: "value1", Effect: "NoSchedule"},
								{Key: "key2", Value: "value2", Effect: "NoExecute"},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			nodeName: "node1",
			taintsToAdd: []*v1.Taint{
				{Key: "key3", Value: "value3", Effect: "NoSchedule"},
				{Key: "key4", Value: "value4", Effect: "NoExecute"},
			},
			expectedTaints: []v1.Taint{
				{Key: "key1", Value: "value1", Effect: "NoSchedule"},
				{Key: "key2", Value: "value2", Effect: "NoExecute"},
				{Key: "key3", Value: "value3", Effect: "NoSchedule"},
				{Key: "key4", Value: "value4", Effect: "NoExecute"},
			},
			requestCount: 3,
		},
		{
			name: "add exist taints to node",
			nodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node1",
						},
						Spec: v1.NodeSpec{
							Taints: []v1.Taint{
								{Key: "key1", Value: "value1", Effect: "NoSchedule"},
								{Key: "key2", Value: "value2", Effect: "NoExecute"},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			nodeName: "node1",
			taintsToAdd: []*v1.Taint{
				{Key: "key2", Value: "value2", Effect: "NoExecute"},
			},
			expectedTaints: []v1.Taint{
				{Key: "key1", Value: "value1", Effect: "NoSchedule"},
				{Key: "key2", Value: "value2", Effect: "NoExecute"},
			},
			requestCount: 2,
		},
		{
			name: "add taint to node without taints",
			nodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node1",
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			nodeName: "node1",
			taintsToAdd: []*v1.Taint{
				{Key: "key3", Value: "value3", Effect: "NoSchedule"},
			},
			expectedTaints: []v1.Taint{
				{Key: "key3", Value: "value3", Effect: "NoSchedule"},
			},
			requestCount: 3,
		},
		{
			name: "add empty taint list to node without taints",
			nodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node1",
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			nodeName:       "node1",
			taintsToAdd:    []*v1.Taint{},
			expectedTaints: nil,
			requestCount:   1,
		},
		{
			name: "add empty taint list to node",
			nodeHandler: &testutil.FakeNodeHandler{
				Existing: []*v1.Node{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "node1",
						},
						Spec: v1.NodeSpec{
							Taints: []v1.Taint{
								{Key: "key1", Value: "value1", Effect: "NoSchedule"},
								{Key: "key2", Value: "value2", Effect: "NoExecute"},
							},
						},
					},
				},
				Clientset: fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testutil.NewPod("pod0", "node0")}}),
			},
			nodeName:    "node1",
			taintsToAdd: []*v1.Taint{},
			expectedTaints: []v1.Taint{
				{Key: "key1", Value: "value1", Effect: "NoSchedule"},
				{Key: "key2", Value: "value2", Effect: "NoExecute"},
			},
			requestCount: 1,
		},
	}
	for _, test := range tests {
		err := AddOrUpdateTaintOnNode(test.nodeHandler, test.nodeName, test.taintsToAdd...)
		assert.NoError(t, err, "%s: AddOrUpdateTaintOnNode() error = %v", test.name, err)

		node, _ := test.nodeHandler.Get(test.nodeName, metav1.GetOptions{})
		assert.EqualValues(t, test.expectedTaints, node.Spec.Taints,
			"%s: failed to add taint to node: expected %+v, got %+v",
			test.name, test.expectedTaints, node.Spec.Taints)

		assert.Equal(t, test.requestCount, test.nodeHandler.RequestCount,
			"%s: unexpected request count: expected %+v, got %+v",
			test.name, test.requestCount, test.nodeHandler.RequestCount)
	}
}

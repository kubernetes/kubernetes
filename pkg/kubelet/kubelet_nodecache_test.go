/*
Copyright 2025 The Kubernetes Authors.

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

package kubelet

import (
	"context"
	"fmt"
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestGetCachedNode(t *testing.T) {
	nodeStatus := v1.NodeStatus{
		Allocatable: v1.ResourceList{
			v1.ResourcePods: *resource.NewQuantity(110, resource.DecimalSI),
		},
	}

	tests := []struct {
		name               string
		informerNode       *v1.Node
		cachedNode         *v1.Node
		nodeListerErr      error
		expectedNode       *v1.Node
		expectedErr        bool
		expectedCachedNode *v1.Node
	}{
		{
			name:         "informer node is newer",
			informerNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "2"}, Status: nodeStatus},
			cachedNode:   &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}, Status: nodeStatus},

			expectedNode:       &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "2"}, Status: nodeStatus},
			expectedCachedNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "2"}, Status: nodeStatus},
		},
		{
			name:         "cached node is newer",
			informerNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}, Status: nodeStatus},
			cachedNode:   &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "2"}, Status: nodeStatus},

			expectedNode:       &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "2"}, Status: nodeStatus},
			expectedCachedNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "2"}, Status: nodeStatus},
		},
		{
			name:         "resource versions are the same",
			informerNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}, Status: nodeStatus},
			cachedNode:   &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}, Status: nodeStatus},

			expectedNode:       &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}, Status: nodeStatus},
			expectedCachedNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}, Status: nodeStatus},
		},
		{
			name:         "informer node cannot be parsed, default to informer node",
			informerNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "abc"}, Status: nodeStatus},
			cachedNode:   &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}, Status: nodeStatus},

			expectedNode:       &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "abc"}, Status: nodeStatus},
			expectedCachedNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "abc"}, Status: nodeStatus},
		},
		{
			name:         "cached node cannot be parsed, default to informer node",
			informerNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}, Status: nodeStatus},
			cachedNode:   &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "abc"}, Status: nodeStatus},

			expectedNode:       &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}, Status: nodeStatus},
			expectedCachedNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}, Status: nodeStatus},
		},
		{
			name:         "cached node is nil, use informer node",
			informerNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}, Status: nodeStatus},
			cachedNode:   nil,

			expectedNode:       &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}, Status: nodeStatus},
			expectedCachedNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}, Status: nodeStatus},
		},
		{
			name:          "node lister returns error, use cached node",
			informerNode:  nil,
			cachedNode:    &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}, Status: nodeStatus},
			nodeListerErr: fmt.Errorf("test error"),

			expectedNode:       &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}, Status: nodeStatus},
			expectedErr:        false,
			expectedCachedNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}, Status: nodeStatus},
		},
		{
			name:          "node lister returns error, cached node is nil, default to initialNode",
			informerNode:  nil,
			cachedNode:    nil,
			nodeListerErr: fmt.Errorf("test error"),

			expectedNode:       nil, // This will be filled in by the test logic below.
			expectedErr:        false,
			expectedCachedNode: nil,
		},
		{
			name:         "informer node is uninitialized and older/same RV, cached node is initialized",
			informerNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}, Status: v1.NodeStatus{Allocatable: nil}},
			cachedNode:   &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}, Status: nodeStatus},

			expectedNode:       &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}, Status: nodeStatus},
			expectedCachedNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}, Status: nodeStatus},
		},
		{
			name:         "informer node is uninitialized but newer RV, should overlay status onto newer informer node",
			informerNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "2", Labels: map[string]string{"new": "label"}}, Status: v1.NodeStatus{Allocatable: nil}},
			cachedNode:   &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}, Status: nodeStatus},

			expectedNode:       &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "2", Labels: map[string]string{"new": "label"}}, Status: nodeStatus},
			expectedCachedNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "2", Labels: map[string]string{"new": "label"}}, Status: nodeStatus},
		},
		{
			name:         "informer node is uninitialized and cached node is nil, should overlay status and cache",
			informerNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "2"}, Status: v1.NodeStatus{Allocatable: nil}},
			cachedNode:   nil,

			expectedNode:       &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "2"}, Status: nodeStatus},
			expectedCachedNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "2"}, Status: nodeStatus},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			kl := Kubelet{
				cachedNode: test.cachedNode,
				nodeLister: newFakeNodeLister(test.nodeListerErr, test.informerNode),
				kubeClient: &fake.Clientset{},
				setNodeStatusFuncs: []func(context.Context, *v1.Node) error{
					func(ctx context.Context, node *v1.Node) error {
						node.Status = nodeStatus
						return nil
					},
				},
			}
			if test.expectedNode == nil {
				var err error
				test.expectedNode, err = kl.initialNode(tCtx)
				if err != nil {
					test.expectedErr = true
				}
			}

			actualNode, err := kl.GetCachedNode(tCtx, true)

			if (err != nil) != test.expectedErr {
				t.Errorf("GetCachedNode() unexpected error status: %v, expected error: %v", err, test.expectedErr)
			}
			if !reflect.DeepEqual(actualNode, test.expectedNode) {
				t.Errorf("GetCachedNode() = %v, expected %v", actualNode, test.expectedNode)
			}
			if !reflect.DeepEqual(kl.cachedNode, test.expectedCachedNode) {
				t.Errorf("kl.cachedNode after GetCachedNode() = %v, expected %v", kl.cachedNode, test.expectedCachedNode)
			}
		})
	}
}

func TestGetNodeSync(t *testing.T) {
	tCtx := ktesting.Init(t)
	nodeName := "test-node"
	nodeStatus := v1.NodeStatus{
		Allocatable: v1.ResourceList{
			v1.ResourcePods: *resource.NewQuantity(110, resource.DecimalSI),
		},
	}

	// Case 1: API server node is initialized
	initializedNode := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: nodeName, ResourceVersion: "1"},
		Status:     nodeStatus,
	}
	fakeClient1 := fake.NewSimpleClientset(initializedNode)
	kl1 := Kubelet{
		nodeName:   types.NodeName(nodeName),
		kubeClient: fakeClient1,
	}
	actualNode1, err := kl1.GetCachedNode(tCtx, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(actualNode1, initializedNode) {
		t.Errorf("expected node %v, got %v", initializedNode, actualNode1)
	}
	if !reflect.DeepEqual(kl1.cachedNode, initializedNode) {
		t.Errorf("expected cached node %v, got %v", initializedNode, kl1.cachedNode)
	}

	// Case 2: API server node is uninitialized, should overlay status and cache
	uninitializedNode := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: nodeName, ResourceVersion: "2"},
		Status: v1.NodeStatus{
			Allocatable: nil,
		},
	}
	fakeClient2 := fake.NewSimpleClientset(uninitializedNode)
	kl2 := Kubelet{
		nodeName:   types.NodeName(nodeName),
		kubeClient: fakeClient2,
		setNodeStatusFuncs: []func(context.Context, *v1.Node) error{
			func(ctx context.Context, node *v1.Node) error {
				node.Status = nodeStatus
				return nil
			},
		},
	}
	actualNode2, err := kl2.GetCachedNode(tCtx, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expectedNode2 := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: nodeName, ResourceVersion: "2"},
		Status:     nodeStatus,
	}
	if !reflect.DeepEqual(actualNode2, expectedNode2) {
		t.Errorf("expected node %v, got %v", expectedNode2, actualNode2)
	}
	if !reflect.DeepEqual(kl2.cachedNode, expectedNode2) {
		t.Errorf("expected cached node %v, got %v", expectedNode2, kl2.cachedNode)
	}
}

type fakeNodeLister struct {
	node *v1.Node
	err  error
}

func newFakeNodeLister(err error, node *v1.Node) *fakeNodeLister {
	ret := &fakeNodeLister{}
	ret.node = node
	ret.err = err
	return ret
}

func (l *fakeNodeLister) List(selector labels.Selector) (ret []*v1.Node, err error) {
	return []*v1.Node{l.node}, l.err
}

func (l *fakeNodeLister) Get(name string) (*v1.Node, error) {
	return l.node, l.err
}

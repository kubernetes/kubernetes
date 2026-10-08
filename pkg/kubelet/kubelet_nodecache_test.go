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
	"fmt"
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/ktesting"
	"k8s.io/utils/ptr"
)

func TestGetCachedNode(t *testing.T) {
	tests := []struct {
		name          string
		useCache      *bool // nil means true
		informerNode  *v1.Node
		cachedNode    *v1.Node
		nodeListerErr error
		// clientNode and clientErr are returned by the fake clientset's GET
		// when either is set; they only matter on the useCache=false path.
		clientNode *v1.Node
		clientErr  error
		// nilClient leaves kl.kubeClient nil so getNodeSync falls back to initialNode.
		nilClient          bool
		expectedNode       *v1.Node
		expectedErr        bool
		expectedCachedNode *v1.Node
	}{
		{
			name:         "informer node is newer",
			informerNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "2"}},
			cachedNode:   &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}},

			expectedNode:       &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "2"}},
			expectedCachedNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "2"}},
		},
		{
			name:         "cached node is newer",
			informerNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}},
			cachedNode:   &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "2"}},

			expectedNode:       &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "2"}},
			expectedCachedNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "2"}},
		},
		{
			name:         "resource versions are the same",
			informerNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}},
			cachedNode:   &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}},

			expectedNode:       &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}},
			expectedCachedNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}},
		},
		{
			name:         "informer node cannot be parsed, default to informer node",
			informerNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "abc"}},
			cachedNode:   &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}},

			expectedNode:       &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "abc"}},
			expectedCachedNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "abc"}},
		},
		{
			name:         "cached node cannot be parsed, default to informer node",
			informerNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}},
			cachedNode:   &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "abc"}},

			expectedNode:       &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}},
			expectedCachedNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}},
		},
		{
			name:         "cached node is nil, use informer node",
			informerNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}},
			cachedNode:   nil,

			expectedNode:       &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}},
			expectedCachedNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}},
		},
		{
			name:          "node lister returns error, use cached node",
			informerNode:  nil,
			cachedNode:    &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}},
			nodeListerErr: fmt.Errorf("test error"),

			expectedNode:       &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}},
			expectedErr:        false,
			expectedCachedNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}},
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
			name:       "sync GET succeeds, replaces cached node",
			useCache:   new(false),
			cachedNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}},
			clientNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "2"}},

			expectedNode:       &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "2"}},
			expectedErr:        false,
			expectedCachedNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "2"}},
		},
		{
			name:       "sync GET fails, cached node unchanged",
			useCache:   new(false),
			cachedNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}},
			clientErr:  fmt.Errorf("test error"),

			expectedNode:       nil,
			expectedErr:        true,
			expectedCachedNode: &v1.Node{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}},
		},
		{
			name:       "sync with nil client falls back to initialNode",
			useCache:   new(false),
			nilClient:  true,
			cachedNode: nil,

			expectedNode:       nil, // This will be filled in by the test logic below.
			expectedErr:        false,
			expectedCachedNode: nil,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			kl := Kubelet{
				cachedNode: test.cachedNode,
				nodeLister: newFakeNodeLister(test.nodeListerErr, test.informerNode),
			}
			// Assign only when a client is wanted: storing a nil *fake.Clientset in
			// the interface would make the nil check in getNodeSync miss it.
			if !test.nilClient {
				fakeClient := fake.NewClientset()
				if test.clientNode != nil || test.clientErr != nil {
					fakeClient.PrependReactor("get", "nodes", func(action clienttesting.Action) (bool, runtime.Object, error) {
						// Return an untyped nil on error so the reactor matches the
						// usual fake idiom instead of boxing a nil *v1.Node.
						var obj runtime.Object
						if test.clientNode != nil {
							obj = test.clientNode
						}
						return true, obj, test.clientErr
					})
				}
				kl.kubeClient = fakeClient
			}
			// A nil expectedNode with no expected error means "whatever initialNode
			// produces"; with an expected error it really means nil.
			if test.expectedNode == nil && !test.expectedErr {
				var err error
				test.expectedNode, err = kl.initialNode(tCtx)
				if err != nil {
					test.expectedErr = true
				}
			}

			actualNode, err := kl.GetCachedNode(tCtx, ptr.Deref(test.useCache, true))

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

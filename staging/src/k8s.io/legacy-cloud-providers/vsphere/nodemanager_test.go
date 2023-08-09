//go:build !providerless
// +build !providerless

/*
Copyright 2023 The Kubernetes Authors.

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

package vsphere

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/legacy-cloud-providers/vsphere/vclib"
)

// Annotation used to distinguish nodes in node cache / informer / API server
const nodeAnnotation = "test"

func getNode(annotation string) *v1.Node {
	return &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node1",
			Annotations: map[string]string{
				nodeAnnotation: annotation,
			},
		},
	}
}

func TestGetNode(t *testing.T) {
	tests := []struct {
		name           string
		cachedNodes    []*v1.Node
		informerNodes  []*v1.Node // "nil" means that the NodeManager has no nodeLister
		apiServerNodes []*v1.Node // "nil" means that the NodeManager has no nodeGetter

		expectedNodeAnnotation string
		expectNotFound         bool
	}{
		{
			name:           "No cached node anywhere",
			cachedNodes:    []*v1.Node{},
			informerNodes:  []*v1.Node{},
			apiServerNodes: []*v1.Node{},
			expectNotFound: true,
		},
		{
			name:           "No lister & getter",
			cachedNodes:    []*v1.Node{},
			informerNodes:  nil,
			apiServerNodes: nil,
			expectNotFound: true,
		},
		{
			name:                   "cache is used first",
			cachedNodes:            []*v1.Node{getNode("cache")},
			informerNodes:          []*v1.Node{getNode("informer")},
			apiServerNodes:         []*v1.Node{getNode("apiserver")},
			expectedNodeAnnotation: "cache",
		},
		{
			name:                   "informer is used second",
			cachedNodes:            []*v1.Node{},
			informerNodes:          []*v1.Node{getNode("informer")},
			apiServerNodes:         []*v1.Node{getNode("apiserver")},
			expectedNodeAnnotation: "informer",
		},
		{
			name:                   "API server is used last",
			cachedNodes:            []*v1.Node{},
			informerNodes:          []*v1.Node{},
			apiServerNodes:         []*v1.Node{getNode("apiserver")},
			expectedNodeAnnotation: "apiserver",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {

			// local NodeManager cache
			cache := make(map[string]*v1.Node)
			for _, node := range test.cachedNodes {
				cache[node.Name] = node
			}

			// Client with apiServerNodes
			objs := []runtime.Object{}
			for _, node := range test.apiServerNodes {
				objs = append(objs, node)
			}
			client := fake.NewSimpleClientset(objs...)
			nodeGetter := client.CoreV1()

			// Informer + nodeLister. Despite the client already has apiServerNodes, they won't appear in the
			// nodeLister, because the informer is never started.
			factory := informers.NewSharedInformerFactory(client, 0 /* no resync */)
			nodeInformer := factory.Core().V1().Nodes()
			for _, node := range test.informerNodes {
				nodeInformer.Informer().GetStore().Add(node)
			}
			nodeLister := nodeInformer.Lister()

			nodeManager := NodeManager{
				registeredNodes: cache,
			}
			if test.informerNodes != nil {
				nodeManager.SetNodeLister(nodeLister)
			}
			if test.apiServerNodes != nil {
				nodeManager.SetNodeGetter(nodeGetter)
			}

			node, err := nodeManager.GetNode("node1")
			if test.expectNotFound && err != vclib.ErrNoVMFound {
				t.Errorf("Expected NotFound error, got: %v", err)
			}
			if !test.expectNotFound && err != nil {
				t.Errorf("Unexpected error: %s", err)
			}

			if test.expectedNodeAnnotation != "" {
				if node.Annotations == nil {
					t.Errorf("Expected node with annotation %q, got nil", test.expectedNodeAnnotation)
				} else {
					if ann := node.Annotations[nodeAnnotation]; ann != test.expectedNodeAnnotation {
						t.Errorf("Expected node with annotation %q, got %q", test.expectedNodeAnnotation, ann)
					}
				}
			}
		})
	}
}

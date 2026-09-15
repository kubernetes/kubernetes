/*
Copyright The Kubernetes Authors.

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

package hierarchy

import (
	"errors"
	"reflect"
	"testing"

	"k8s.io/kube-scheduler/framework"
)

func TestWalkUp(t *testing.T) {
	k1 := framework.EntityKey{Namespace: "ns", Name: "k1", Type: framework.PodGroupKeyType}
	k2 := framework.EntityKey{Namespace: "ns", Name: "k2", Type: framework.CompositePodGroupKeyType}
	k3 := framework.EntityKey{Namespace: "ns", Name: "k3", Type: framework.CompositePodGroupKeyType}
	k4 := framework.EntityKey{Namespace: "ns", Name: "k4", Type: framework.CompositePodGroupKeyType}

	tests := []struct {
		name        string
		startKey    framework.EntityKey
		parents     map[framework.EntityKey]framework.EntityKey
		visitStopAt framework.EntityKey
		wantRootKey framework.EntityKey
		wantVisited []framework.EntityKey
	}{
		{
			name:        "single node without parent",
			startKey:    k1,
			parents:     map[framework.EntityKey]framework.EntityKey{},
			wantRootKey: k1,
			wantVisited: []framework.EntityKey{k1},
		},
		{
			name:     "two node chain (depth 2)",
			startKey: k1,
			parents: map[framework.EntityKey]framework.EntityKey{
				k1: k2,
			},
			wantRootKey: k2,
			wantVisited: []framework.EntityKey{k1, k2},
		},
		{
			name:     "four node chain (depth 4 max allowed)",
			startKey: k1,
			parents: map[framework.EntityKey]framework.EntityKey{
				k1: k2,
				k2: k3,
				k3: k4,
			},
			wantRootKey: k4,
			wantVisited: []framework.EntityKey{k1, k2, k3, k4},
		},
		{
			name:     "early termination with visitFn",
			startKey: k1,
			parents: map[framework.EntityKey]framework.EntityKey{
				k1: k2,
				k2: k3,
				k3: k4,
			},
			visitStopAt: k2,
			wantRootKey: k2,
			wantVisited: []framework.EntityKey{k1, k2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var visited []framework.EntityKey
			getParent := func(key framework.EntityKey) (*framework.EntityKey, error) {
				p, ok := tt.parents[key]
				if !ok {
					return nil, nil
				}
				return &p, nil
			}
			visitFn := func(key framework.EntityKey, depth int) (bool, error) {
				visited = append(visited, key)
				if tt.visitStopAt != (framework.EntityKey{}) && key == tt.visitStopAt {
					return true, nil
				}
				return false, nil
			}

			rootKey, err := WalkUp(tt.startKey, getParent, visitFn)
			if err != nil {
				t.Fatalf("WalkUp() unexpected error: %v", err)
			}
			if rootKey != tt.wantRootKey {
				t.Errorf("WalkUp() rootKey = %v, want %v", rootKey, tt.wantRootKey)
			}
			if !reflect.DeepEqual(visited, tt.wantVisited) {
				t.Errorf("WalkUp() visited = %v, want %v", visited, tt.wantVisited)
			}
		})
	}
}

func TestWalkUpErrors(t *testing.T) {
	k1 := framework.EntityKey{Namespace: "ns", Name: "k1", Type: framework.PodGroupKeyType}
	k2 := framework.EntityKey{Namespace: "ns", Name: "k2", Type: framework.CompositePodGroupKeyType}
	k3 := framework.EntityKey{Namespace: "ns", Name: "k3", Type: framework.CompositePodGroupKeyType}
	k4 := framework.EntityKey{Namespace: "ns", Name: "k4", Type: framework.CompositePodGroupKeyType}
	k5 := framework.EntityKey{Namespace: "ns", Name: "k5", Type: framework.CompositePodGroupKeyType}

	errCustom := errors.New("custom error")

	tests := []struct {
		name      string
		startKey  framework.EntityKey
		getParent func(key framework.EntityKey) (*framework.EntityKey, error)
		visitFn   func(key framework.EntityKey, depth int) (bool, error)
		wantErr   error
	}{
		{
			name:     "five node chain (depth 5 exceeds max depth 4)",
			startKey: k1,
			getParent: func(key framework.EntityKey) (*framework.EntityKey, error) {
				parents := map[framework.EntityKey]framework.EntityKey{
					k1: k2,
					k2: k3,
					k3: k4,
					k4: k5,
				}
				if p, ok := parents[key]; ok {
					return &p, nil
				}
				return nil, nil
			},
			wantErr: ErrMaxTreeDepthExceeded,
		},
		{
			name:     "self cycle k1 -> k1",
			startKey: k1,
			getParent: func(key framework.EntityKey) (*framework.EntityKey, error) {
				p := k1
				return &p, nil
			},
			wantErr: ErrMaxTreeDepthExceeded,
		},
		{
			name:     "two node cycle k1 -> k2 -> k1",
			startKey: k1,
			getParent: func(key framework.EntityKey) (*framework.EntityKey, error) {
				parents := map[framework.EntityKey]framework.EntityKey{
					k1: k2,
					k2: k1,
				}
				if p, ok := parents[key]; ok {
					return &p, nil
				}
				return nil, nil
			},
			wantErr: ErrMaxTreeDepthExceeded,
		},
		{
			name:     "getParent returns error",
			startKey: k1,
			getParent: func(key framework.EntityKey) (*framework.EntityKey, error) {
				return nil, errCustom
			},
			wantErr: errCustom,
		},
		{
			name:     "visitFn returns error",
			startKey: k1,
			getParent: func(key framework.EntityKey) (*framework.EntityKey, error) {
				return nil, nil
			},
			visitFn: func(key framework.EntityKey, depth int) (bool, error) {
				return false, errCustom
			},
			wantErr: errCustom,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := WalkUp(tt.startKey, tt.getParent, tt.visitFn)
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("WalkUp() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

type testNode struct {
	name     string
	children []*testNode
}

func TestWalkDown(t *testing.T) {
	// n0 (root)
	// ├── n1
	// │   ├── n11
	// │   │   └── n111 (depth 3)
	// │   └── n12
	// └── n2
	n111 := &testNode{name: "n111"}
	n11 := &testNode{name: "n11", children: []*testNode{n111}}
	n12 := &testNode{name: "n12"}
	n1 := &testNode{name: "n1", children: []*testNode{n11, n12}}
	n2 := &testNode{name: "n2"}
	root := &testNode{name: "n0", children: []*testNode{n1, n2}}

	getChildren := func(node *testNode) ([]*testNode, error) {
		return node.children, nil
	}

	tests := []struct {
		name        string
		root        *testNode
		getChildren func(node *testNode) ([]*testNode, error)
		visitFn     func(node *testNode, depth int) (stop bool, skipChildren bool, err error)
		wantVisited []string
	}{
		{
			name:        "full traversal",
			root:        root,
			getChildren: getChildren,
			wantVisited: []string{"n0", "n1", "n11", "n111", "n12", "n2"},
		},
		{
			name:        "skip children of n1",
			root:        root,
			getChildren: getChildren,
			visitFn: func(node *testNode, depth int) (bool, bool, error) {
				return false, node.name == "n1", nil
			},
			wantVisited: []string{"n0", "n1", "n2"},
		},
		{
			name:        "stop at n11",
			root:        root,
			getChildren: getChildren,
			visitFn: func(node *testNode, depth int) (bool, bool, error) {
				return node.name == "n11", false, nil
			},
			wantVisited: []string{"n0", "n1", "n11"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var visited []string
			visit := func(node *testNode, depth int) (bool, bool, error) {
				visited = append(visited, node.name)
				if tt.visitFn != nil {
					return tt.visitFn(node, depth)
				}
				return false, false, nil
			}
			err := WalkDown(tt.root, tt.getChildren, visit)
			if err != nil {
				t.Fatalf("WalkDown() unexpected error: %v", err)
			}
			if !reflect.DeepEqual(visited, tt.wantVisited) {
				t.Errorf("WalkDown() visited = %v, want %v", visited, tt.wantVisited)
			}
		})
	}
}

func TestWalkDownErrors(t *testing.T) {
	n4 := &testNode{name: "n4"}
	n3 := &testNode{name: "n3", children: []*testNode{n4}}
	n2 := &testNode{name: "n2", children: []*testNode{n3}}
	n1 := &testNode{name: "n1", children: []*testNode{n2}}
	deepRoot := &testNode{name: "n0", children: []*testNode{n1}}

	errCustom := errors.New("custom error")

	tests := []struct {
		name        string
		root        *testNode
		getChildren func(node *testNode) ([]*testNode, error)
		visitFn     func(node *testNode, depth int) (bool, bool, error)
		wantErr     error
	}{
		{
			name: "tree depth exceeding max (depth 5)",
			root: deepRoot,
			getChildren: func(node *testNode) ([]*testNode, error) {
				return node.children, nil
			},
			wantErr: ErrMaxTreeDepthExceeded,
		},
		{
			name: "getChildren returns error",
			root: &testNode{name: "n0"},
			getChildren: func(node *testNode) ([]*testNode, error) {
				return nil, errCustom
			},
			wantErr: errCustom,
		},
		{
			name: "visitFn returns error",
			root: &testNode{name: "n0"},
			getChildren: func(node *testNode) ([]*testNode, error) {
				return nil, nil
			},
			visitFn: func(node *testNode, depth int) (bool, bool, error) {
				return false, false, errCustom
			},
			wantErr: errCustom,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := WalkDown(tt.root, tt.getChildren, tt.visitFn)
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("WalkDown() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

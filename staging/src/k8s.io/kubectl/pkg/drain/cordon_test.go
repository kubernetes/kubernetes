/*
Copyright 2024 The Kubernetes Authors.

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

package drain

import (
	"context"
	"errors"
	"testing"

	"github.com/google/go-cmp/cmp"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
)

type newCordonHelperFromRuntimeObjectTestCase struct {
	name        string
	nodeObject  runtime.Object
	expectError bool
	expected    *CordonHelper
}

func TestNewCordonHelperFromRuntimeObject(t *testing.T) {
	tests := []newCordonHelperFromRuntimeObjectTestCase{
		{
			name: "valid node object",
			nodeObject: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-node",
				},
			},
			expectError: false,
			expected: &CordonHelper{
				node: &corev1.Node{
					TypeMeta: metav1.TypeMeta{
						Kind:       "Node",
						APIVersion: "v1",
					},
					ObjectMeta: metav1.ObjectMeta{
						Name: "test-node",
					},
				},
			},
		},
		{
			name: "invalid object type",
			nodeObject: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-pod",
				},
			},
			expectError: true,
			expected:    nil,
		},
	}
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)
	gvk := schema.GroupVersionKind{
		Group:   "",
		Version: "v1",
		Kind:    "Node",
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			helper, err := NewCordonHelperFromRuntimeObject(tt.nodeObject, scheme, gvk)
			if tt.expectError && err == nil {
				t.Error("Expected error but got none")
			}
			if !tt.expectError && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
			if !tt.expectError && helper == nil {
				t.Error("Expected non-nil helper")
			}
			if tt.expected != nil && helper != nil {
				if diff := cmp.Diff(tt.expected.node, helper.node); diff != "" {
					t.Errorf("Node mismatch (-want +got):\n%s", diff)
				}
			}
		})
	}
}

type updateIfRequiredTestCase struct {
	name          string
	currentState  bool
	desiredState  bool
	expectUpdated bool
}

func TestUpdateIfRequired(t *testing.T) {
	tests := []updateIfRequiredTestCase{
		{
			name:          "no change required",
			currentState:  true,
			desiredState:  true,
			expectUpdated: false,
		},
		{
			name:          "update required - cordon",
			currentState:  false,
			desiredState:  true,
			expectUpdated: true,
		},
		{
			name:          "update required - uncordon",
			currentState:  true,
			desiredState:  false,
			expectUpdated: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node := &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-node",
				},
				Spec: corev1.NodeSpec{
					Unschedulable: tt.currentState,
				},
			}

			helper := NewCordonHelper(node)
			updated := helper.UpdateIfRequired(tt.desiredState)
			if updated != tt.expectUpdated {
				t.Errorf("Expected UpdateIfRequired to return %v, got %v", tt.expectUpdated, updated)
			}
			if helper.desired != tt.desiredState {
				t.Errorf("Expected desired state to be %v, got %v", tt.desiredState, helper.desired)
			}
		})
	}
}

func TestCordonHelper_PatchOrReplaceWithContext(t *testing.T) {
	tests := []struct {
		name         string
		node         *corev1.Node
		clientset    *fake.Clientset
		serverDryRun bool
		shouldFail   bool
	}{
		{
			name: "update success: local object should be updated",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "fake-node",
				},
				Spec: corev1.NodeSpec{
					Unschedulable: false,
				},
			},
			clientset: fake.NewClientset(&corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "fake-node",
				},
				Spec: corev1.NodeSpec{
					Unschedulable: false,
				},
			}),
			serverDryRun: false,
			shouldFail:   false,
		},
		{
			name: "update faild: local object should not be updated",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "fake-node",
				},
				Spec: corev1.NodeSpec{
					Unschedulable: false,
				},
			},
			clientset: fake.NewClientset(&corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "fake-node",
				},
				Spec: corev1.NodeSpec{
					Unschedulable: false,
				},
			}),
			serverDryRun: false,
			shouldFail:   true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := NewCordonHelper(tt.node)
			c.desired = true
			if tt.shouldFail {
				tt.clientset.Fake.PrependReactor("update", "nodes", func(action k8stesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, nil, errors.New("fake update error")
				})
				tt.clientset.Fake.PrependReactor("patch", "nodes", func(action k8stesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, nil, errors.New("fake update error")
				})
			}

			err, _ := c.PatchOrReplaceWithContext(context.Background(), tt.clientset, tt.serverDryRun)
			if tt.shouldFail {
				if err == nil {
					t.Errorf("PatchOrReplaceWithContext() failed: update should failed but no error returned")
					return
				}
			} else {
				if err != nil {
					t.Errorf("PatchOrReplaceWithContext() failed: update should succeed but got error: %v", err)
					return
				}
			}

			if err != nil {
				t.Logf("patch/update failed, error: %v, c.node.Spec.Unschedulable shouldn't be updated", err)
				if c.node.Spec.Unschedulable == c.desired {
					t.Errorf("PatchOrReplaceWithContext() failed: update failed but c.node's unschedulable was updated")
				}
			} else {
				t.Log("patch/update successed, c.node.Spec.Unschedulable should be updated")
				if c.node.Spec.Unschedulable != c.desired {
					t.Errorf("PatchOrReplaceWithContext() failed: update succeeded but c.node's unschedulable was not updated")
				}
			}
		})
	}
}

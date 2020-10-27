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

package node

import (
	"errors"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
)

// TestCheckReadyForTests specifically is concerned about the multi-node logic
// since single node checks are in TestReadyForTests.
func TestCheckReadyForTests(t *testing.T) {
	// This is a duplicate definition of the constant in pkg/controller/service/controller.go
	labelNodeRoleMaster := "node-role.kubernetes.io/master"

	fromVanillaNode := func(f func(*v1.Node)) v1.Node {
		vanillaNode := &v1.Node{
			ObjectMeta: metav1.ObjectMeta{Name: "test-node"},
			Status: v1.NodeStatus{
				Conditions: []v1.NodeCondition{
					{Type: v1.NodeReady, Status: v1.ConditionTrue},
				},
			},
		}
		f(vanillaNode)
		return *vanillaNode
	}

	tcs := []struct {
		desc                 string
		nonblockingTaints    string
		allowedNotReadyNodes int
		nodes                []v1.Node
		nodeListErr          error
		expected             bool
		expectedErr          string
	}{
		{
			desc: "Vanilla node should pass",
			nodes: []v1.Node{
				fromVanillaNode(func(n *v1.Node) {}),
			},
			expected: true,
		}, {
			desc:              "Default value for nonblocking taints tolerates master taint",
			nonblockingTaints: `node-role.kubernetes.io/master`,
			nodes: []v1.Node{
				fromVanillaNode(func(n *v1.Node) {
					n.Spec.Taints = []v1.Taint{{Key: labelNodeRoleMaster, Effect: v1.TaintEffectNoSchedule}}
				}),
			},
			expected: true,
		}, {
			desc:              "Tainted node should fail if effect is TaintEffectNoExecute",
			nonblockingTaints: "bar",
			nodes: []v1.Node{
				fromVanillaNode(func(n *v1.Node) {
					n.Spec.Taints = []v1.Taint{{Key: "foo", Effect: v1.TaintEffectNoExecute}}
				})},
			expected: false,
		}, {
			desc:                 "Tainted node can be allowed via allowedNotReadyNodes",
			nonblockingTaints:    "bar",
			allowedNotReadyNodes: 1,
			nodes: []v1.Node{
				fromVanillaNode(func(n *v1.Node) {
					n.Spec.Taints = []v1.Taint{{Key: "foo", Effect: v1.TaintEffectNoExecute}}
				})},
			expected: true,
		}, {
			desc: "Multi-node, all OK",
			nodes: []v1.Node{
				fromVanillaNode(func(n *v1.Node) {}),
				fromVanillaNode(func(n *v1.Node) {}),
			},
			expected: true,
		}, {
			desc: "Multi-node, single blocking node blocks",
			nodes: []v1.Node{
				fromVanillaNode(func(n *v1.Node) {}),
				fromVanillaNode(func(n *v1.Node) {
					n.Spec.Taints = []v1.Taint{{Key: "foo", Effect: v1.TaintEffectNoSchedule}}
				}),
			},
			expected: false,
		}, {
			desc:                 "Multi-node, single blocking node allowed via allowedNotReadyNodes",
			allowedNotReadyNodes: 1,
			nodes: []v1.Node{
				fromVanillaNode(func(n *v1.Node) {}),
				fromVanillaNode(func(n *v1.Node) {
					n.Spec.Taints = []v1.Taint{{Key: "foo", Effect: v1.TaintEffectNoSchedule}}
				}),
			},
			expected: true,
		}, {
			desc:              "Multi-node, single blocking node allowed via nonblocking taint",
			nonblockingTaints: "foo",
			nodes: []v1.Node{
				fromVanillaNode(func(n *v1.Node) {}),
				fromVanillaNode(func(n *v1.Node) {
					n.Spec.Taints = []v1.Taint{{Key: "foo", Effect: v1.TaintEffectNoSchedule}}
				}),
			},
			expected: true,
		}, {
			desc:              "Multi-node, both blocking nodes allowed via separate nonblocking taints",
			nonblockingTaints: "foo,bar",
			nodes: []v1.Node{
				fromVanillaNode(func(n *v1.Node) {}),
				fromVanillaNode(func(n *v1.Node) {
					n.Spec.Taints = []v1.Taint{{Key: "foo", Effect: v1.TaintEffectNoSchedule}}
				}),
				fromVanillaNode(func(n *v1.Node) {
					n.Spec.Taints = []v1.Taint{{Key: "bar", Effect: v1.TaintEffectNoSchedule}}
				}),
			},
			expected: true,
		}, {
			desc:              "Multi-node, one blocking node allowed via nonblocking taints still blocked",
			nonblockingTaints: "foo,notbar",
			nodes: []v1.Node{
				fromVanillaNode(func(n *v1.Node) {}),
				fromVanillaNode(func(n *v1.Node) {
					n.Spec.Taints = []v1.Taint{{Key: "foo", Effect: v1.TaintEffectNoSchedule}}
				}),
				fromVanillaNode(func(n *v1.Node) {
					n.Spec.Taints = []v1.Taint{{Key: "bar", Effect: v1.TaintEffectNoSchedule}}
				}),
			},
			expected: false,
		}, {
			desc:        "Errors from node list are reported",
			nodeListErr: errors.New("Forced error"),
			expected:    false,
			expectedErr: "Forced error",
		},
	}

	// Only determines some logging functionality; not relevant so set to a large value.
	testLargeClusterThreshold := 1000

	for _, tc := range tcs {
		t.Run(tc.desc, func(t *testing.T) {
			c := fake.NewSimpleClientset()
			c.PrependReactor("list", "nodes", func(action k8stesting.Action) (handled bool, ret runtime.Object, err error) {
				nodeList := &v1.NodeList{Items: tc.nodes}
				return true, nodeList, tc.nodeListErr
			})
			checkFunc := CheckReadyForTests(c, tc.nonblockingTaints, tc.allowedNotReadyNodes, testLargeClusterThreshold)
			out, err := checkFunc()
			if out != tc.expected {
				t.Errorf("Expected %v but got %v", tc.expected, out)
			}
			switch {
			case err == nil && len(tc.expectedErr) > 0:
				t.Errorf("Expected error %q nil", tc.expectedErr)
			case err != nil && err.Error() != tc.expectedErr:
				t.Errorf("Expected error %q but got %q", tc.expectedErr, err.Error())
			}
		})
	}
}

func TestReadyForTests(t *testing.T) {
	fromVanillaNode := func(f func(*v1.Node)) *v1.Node {
		vanillaNode := &v1.Node{
			ObjectMeta: metav1.ObjectMeta{Name: "test-node"},
			Status: v1.NodeStatus{
				Conditions: []v1.NodeCondition{
					{Type: v1.NodeReady, Status: v1.ConditionTrue},
				},
			},
		}
		f(vanillaNode)
		return vanillaNode
	}
	_ = fromVanillaNode
	tcs := []struct {
		desc              string
		node              *v1.Node
		nonblockingTaints string
		expected          bool
	}{
		{
			desc: "Vanilla node should pass",
			node: fromVanillaNode(func(n *v1.Node) {
			}),
			expected: true,
		}, {
			desc:              "Vanilla node should pass with non-applicable nonblocking taint",
			nonblockingTaints: "foo",
			node: fromVanillaNode(func(n *v1.Node) {
			}),
			expected: true,
		}, {
			desc: "Tainted node should pass if effect is TaintEffectPreferNoSchedule",
			node: fromVanillaNode(func(n *v1.Node) {
				n.Spec.Taints = []v1.Taint{{Key: "foo", Effect: v1.TaintEffectPreferNoSchedule}}
			}),
			expected: true,
		}, {
			desc: "Tainted node should fail if effect is TaintEffectNoExecute",
			node: fromVanillaNode(func(n *v1.Node) {
				n.Spec.Taints = []v1.Taint{{Key: "foo", Effect: v1.TaintEffectNoExecute}}
			}),
			expected: false,
		}, {
			desc: "Tainted node should fail",
			node: fromVanillaNode(func(n *v1.Node) {
				n.Spec.Taints = []v1.Taint{{Key: "foo", Effect: v1.TaintEffectNoSchedule}}
			}),
			expected: false,
		}, {
			desc:              "Tainted node should pass if nonblocking",
			nonblockingTaints: "foo",
			node: fromVanillaNode(func(n *v1.Node) {
				n.Spec.Taints = []v1.Taint{{Key: "foo", Effect: v1.TaintEffectNoSchedule}}
			}),
			expected: true,
		}, {
			desc: "Node with network not ready fails",
			node: fromVanillaNode(func(n *v1.Node) {
				n.Status.Conditions = append(n.Status.Conditions,
					v1.NodeCondition{Type: v1.NodeNetworkUnavailable, Status: v1.ConditionTrue},
				)
			}),
			expected: false,
		}, {
			desc: "Node fails unless NodeReady status",
			node: fromVanillaNode(func(n *v1.Node) {
				n.Status.Conditions = []v1.NodeCondition{}
			}),
			expected: false,
		},
	}

	for _, tc := range tcs {
		t.Run(tc.desc, func(t *testing.T) {
			out := readyForTests(tc.node, tc.nonblockingTaints)
			if out != tc.expected {
				t.Errorf("Expected %v but got %v", tc.expected, out)
			}
		})
	}
}

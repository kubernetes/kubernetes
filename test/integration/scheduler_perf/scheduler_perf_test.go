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

package benchmark

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
	testutils "k8s.io/kubernetes/test/utils"
	ktesting "k8s.io/kubernetes/test/utils/ktesting"
)

func TestRunOp(t *testing.T) {
	tests := []struct {
		name            string
		op              realOp
		expectedFailure bool
		verifyFunc      func(t *testing.T, tCtx ktesting.TContext, op realOp) error
	}{
		{
			name: "Create Single Node",
			op: &createNodesOp{
				Opcode: createNodesOpcode,
				Count:  1,
			},
			verifyFunc: func(t *testing.T, tCtx ktesting.TContext, op realOp) error {
				createOp, ok := op.(*createNodesOp)
				if !ok {
					return fmt.Errorf("Expected createNodesOp but got %T", op)
				}
				_, err := verifyNodeCount(t, tCtx, createOp.Count)
				return err
			},
		},
		{
			name: "Create Multiple Nodes",
			op: &createNodesOp{
				Opcode: createNodesOpcode,
				Count:  5,
			},
			verifyFunc: func(t *testing.T, tCtx ktesting.TContext, op realOp) error {
				createOp, ok := op.(*createNodesOp)
				if !ok {
					return fmt.Errorf("Expected createNodesOp but got %T", op)
				}
				_, err := verifyNodeCount(t, tCtx, createOp.Count)
				return err
			},
		},
		{
			name: "Create Nodes with Label Strategy",
			op: &createNodesOp{
				Opcode:                   createNodesOpcode,
				Count:                    3,
				LabelNodePrepareStrategy: testutils.NewLabelNodePrepareStrategy("test-label", "value1", "value2", "value3"),
			},
			verifyFunc: func(t *testing.T, tCtx ktesting.TContext, op realOp) error {
				createOp, ok := op.(*createNodesOp)
				if !ok {
					return fmt.Errorf("Expected createNodesOp but got %T", op)
				}

				nodes, err := verifyNodeCount(t, tCtx, createOp.Count)
				if err != nil {
					return err
				}

				strategy := createOp.LabelNodePrepareStrategy
				labelValues := make(map[string]bool)
				for _, v := range strategy.LabelValues {
					labelValues[v] = false
				}

				for _, node := range nodes {
					if labelValue, exists := node.Labels[strategy.LabelKey]; exists {
						if _, ok := labelValues[labelValue]; ok {
							labelValues[labelValue] = true
						} else {
							return fmt.Errorf("Node %s has unexpected label value %s for key %s",
								node.Name, labelValue, strategy.LabelKey)
						}
					} else {
						return fmt.Errorf("Node %s is missing expected label %s", node.Name, strategy.LabelKey)
					}
				}
				return nil
			},
		},
		{
			name: "Create Nodes with Unique Label Strategy",
			op: &createNodesOp{
				Opcode:                  createNodesOpcode,
				Count:                   2,
				UniqueNodeLabelStrategy: testutils.NewUniqueNodeLabelStrategy("unique-test-label"),
			},
			verifyFunc: func(t *testing.T, tCtx ktesting.TContext, op realOp) error {
				createOp, ok := op.(*createNodesOp)
				if !ok {
					return fmt.Errorf("Expected createNodesOp but got %T", op)
				}

				nodes, err := verifyNodeCount(t, tCtx, createOp.Count)
				if err != nil {
					return err
				}

				strategy := createOp.UniqueNodeLabelStrategy
				labelValues := make(map[string]bool)

				for _, node := range nodes {
					if labelValue, exists := node.Labels[strategy.LabelKey]; exists {
						if labelValues[labelValue] {
							return fmt.Errorf("Node %s has duplicate label value %s for key %s",
								node.Name, labelValue, strategy.LabelKey)
						}
						labelValues[labelValue] = true
					} else {
						return fmt.Errorf("Node %s is missing expected label %s", node.Name, strategy.LabelKey)
					}
				}
				return nil
			},
		},
		{
			name: "Create Nodes with Node Allocatable Strategy",
			op: &createNodesOp{
				Opcode: createNodesOpcode,
				Count:  2,
				NodeAllocatableStrategy: testutils.NewNodeAllocatableStrategy(
					map[v1.ResourceName]string{
						v1.ResourceCPU:    "2",
						v1.ResourceMemory: "4Gi",
					},
					nil, // no CSI node allocatable
					nil, // no migrated plugins
				),
			},
			verifyFunc: func(t *testing.T, tCtx ktesting.TContext, op realOp) error {
				createOp, ok := op.(*createNodesOp)
				if !ok {
					return fmt.Errorf("Expected createNodesOp but got %T", op)
				}

				nodes, err := verifyNodeCount(t, tCtx, createOp.Count)
				if err != nil {
					return err
				}

				strategy := createOp.NodeAllocatableStrategy

				for _, node := range nodes {
					for resourceName, expectedValue := range strategy.NodeAllocatable {
						if node.Status.Allocatable == nil {
							return fmt.Errorf("Node %s Status.Allocatable is nil", node.Name)
						}
						if allocatable, exists := node.Status.Allocatable[resourceName]; !exists {
							return fmt.Errorf("Node %s is missing expected allocatable resource %s",
								node.Name, resourceName)
						} else {
							expectedQuantity := resource.MustParse(expectedValue)
							if allocatable.Cmp(expectedQuantity) != 0 {
								return fmt.Errorf("Node %s has incorrect allocatable value for %s: got %s, want %s",
									node.Name, resourceName, allocatable.String(), expectedValue)
							}
						}
					}
				}
				return nil
			},
		},
		{
			name: "Create Nodes with Custom Template",
			op: &createNodesOp{
				Opcode: createNodesOpcode,
				Count:  2,
				NodeTemplatePath: createTempNodeTemplateFile(t, &v1.Node{
					ObjectMeta: metav1.ObjectMeta{
						GenerateName: "custom-node-",
					},
					Status: v1.NodeStatus{
						Capacity: v1.ResourceList{
							v1.ResourcePods:   *resource.NewQuantity(100, resource.DecimalSI),
							v1.ResourceCPU:    resource.MustParse("4"),
							v1.ResourceMemory: resource.MustParse("8Gi"),
						},
						Phase: v1.NodeRunning,
						Conditions: []v1.NodeCondition{
							{Type: v1.NodeReady, Status: v1.ConditionTrue},
						},
					},
				}),
			},
			verifyFunc: func(t *testing.T, tCtx ktesting.TContext, op realOp) error {
				createOp, ok := op.(*createNodesOp)
				if !ok {
					return fmt.Errorf("Expected createNodesOp but got %T", op)
				}

				nodes, err := verifyNodeCount(t, tCtx, createOp.Count)
				if err != nil {
					return err
				}

				for _, node := range nodes {
					podCapacity := node.Status.Capacity[v1.ResourcePods]
					expectedPodCapacity := resource.MustParse("100")
					if podCapacity.Cmp(expectedPodCapacity) != 0 {
						return fmt.Errorf("Node %s has incorrect pod capacity: got %s, want %s",
							node.Name, podCapacity.String(), expectedPodCapacity.String())
					}

					cpuCapacity := node.Status.Capacity[v1.ResourceCPU]
					expectedCPUCapacity := resource.MustParse("4")
					if cpuCapacity.Cmp(expectedCPUCapacity) != 0 {
						return fmt.Errorf("Node %s has incorrect CPU capacity: got %s, want %s",
							node.Name, cpuCapacity.String(), expectedCPUCapacity.String())
					}

					isReady := false
					for _, condition := range node.Status.Conditions {
						if condition.Type == v1.NodeReady && condition.Status == v1.ConditionTrue {
							isReady = true
							break
						}
					}
					if !isReady {
						return fmt.Errorf("Node %s is not in Ready condition as expected", node.Name)
					}
				}
				return nil
			},
		},
		{
			name: "Invalid Node Template Path",
			op: &createNodesOp{
				Opcode:           createNodesOpcode,
				Count:            1,
				NodeTemplatePath: func() *string { path := "non-existent-file.json"; return &path }(),
			},
			expectedFailure: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, tCtx := ktesting.NewTestContext(t)
			client := fake.NewSimpleClientset()
			tCtx = ktesting.WithClients(tCtx, nil, nil, client, nil, nil)

			exec := &WorkloadExecutor{
				tCtx:                         tCtx,
				numPodsScheduledPerNamespace: make(map[string]int),
				nextNodeIndex:                0,
			}

			err := exec.runOp(tt.op, 0)

			if tt.expectedFailure {
				if err == nil {
					t.Fatalf("Expected error for %s but got none", tt.name)
				}
				return
			}

			if err != nil {
				t.Fatalf("Failed to run operation for test %q: %v", tt.name, err)
			}

			if tt.verifyFunc != nil {
				if err := tt.verifyFunc(t, tCtx, tt.op); err != nil {
					t.Fatalf("Verification failed for test %q: %v", tt.name, err)
				}
			}
		})
	}
}

// verifyNodeCount is a helper function to verify the number of nodes.
func verifyNodeCount(t *testing.T, tCtx ktesting.TContext, expectedCount int) ([]v1.Node, error) {
	nodes, err := tCtx.Client().CoreV1().Nodes().List(context.Background(), metav1.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to list nodes: %w", err)
	}
	if got := len(nodes.Items); got != expectedCount {
		return nil, fmt.Errorf("unexpected node count: got %d, want %d", got, expectedCount)
	}
	return nodes.Items, nil
}

// createTempNodeTemplateFile creates a temporary node template file for testing.
// It writes the given node object to the file and registers a cleanup function
// to remove the temporary directory after the test.
func createTempNodeTemplateFile(t *testing.T, node *v1.Node) *string {
	t.Helper()
	dir, err := os.MkdirTemp("", "node-template-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir for node template: %v", err)
	}
	t.Cleanup(func() {
		if err := os.RemoveAll(dir); err != nil {
			t.Errorf("Failed to remove temp dir: %v", err)
		}
	})

	nodeTemplateFile := filepath.Join(dir, "node-template.json")
	f, err := os.Create(nodeTemplateFile)
	if err != nil {
		t.Fatalf("Failed to create node template file %s: %v", nodeTemplateFile, err)
	}
	defer func() {
		if err := f.Close(); err != nil {
			t.Errorf("Failed to close file: %v", err)
		}
	}()

	if err := json.NewEncoder(f).Encode(node); err != nil {
		t.Fatalf("Failed to encode node template to %s: %v", nodeTemplateFile, err)
	}
	return &nodeTemplateFile
}

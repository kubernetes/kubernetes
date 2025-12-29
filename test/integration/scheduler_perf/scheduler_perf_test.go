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

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/component-base/featuregate"
	testutils "k8s.io/kubernetes/test/utils"
	ktesting "k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

type verifyFunc func(t *testing.T, tCtx ktesting.TContext, op realOp) error

func TestRunOp(t *testing.T) {
	tests := []struct {
		name            string
		op              realOp
		expectedFailure bool
		verifyFuncs     []verifyFunc
	}{
		{
			name: "Create Single Node",
			op: &createNodesOp{
				Opcode: createNodesOpcode,
				Count:  1,
			},
			verifyFuncs: []verifyFunc{
				verifyCount(1),
				verifyObj(
					&v1.Node{
						ObjectMeta: metav1.ObjectMeta{
							GenerateName: "node-0-",
						},
						Status: v1.NodeStatus{
							Capacity: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("4"),
								v1.ResourceMemory: resource.MustParse("32Gi"),
								v1.ResourcePods:   resource.MustParse("110"),
							},
							Allocatable: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("4"),
								v1.ResourceMemory: resource.MustParse("32Gi"),
								v1.ResourcePods:   resource.MustParse("110"),
							},
						},
					}),
			},
		},
		{
			name: "Create Multiple Nodes",
			op: &createNodesOp{
				Opcode: createNodesOpcode,
				Count:  5,
			},
			verifyFuncs: []verifyFunc{
				verifyCount(5),
				verifyObj(
					&v1.Node{
						ObjectMeta: metav1.ObjectMeta{
							GenerateName: "node-0-",
						},
						Status: v1.NodeStatus{
							Capacity: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("4"),
								v1.ResourceMemory: resource.MustParse("32Gi"),
								v1.ResourcePods:   resource.MustParse("110"),
							},
							Allocatable: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("4"),
								v1.ResourceMemory: resource.MustParse("32Gi"),
								v1.ResourcePods:   resource.MustParse("110"),
							},
						},
					}),
			},
		},
		{
			name: "Create Nodes with Label Strategy",
			op: &createNodesOp{
				Opcode:                   createNodesOpcode,
				Count:                    3,
				LabelNodePrepareStrategy: testutils.NewLabelNodePrepareStrategy("test-label", "value1", "value2", "value3"),
			},
			verifyFuncs: []verifyFunc{
				verifyCount(3),
				verifyLabelValuesAllowed("test-label", sets.New("value1", "value2", "value3")),
			},
		},
		{
			name: "Create Nodes with Unique Label Strategy",
			op: &createNodesOp{
				Opcode:                  createNodesOpcode,
				Count:                   2,
				UniqueNodeLabelStrategy: testutils.NewUniqueNodeLabelStrategy("unique-test-label"),
			},
			verifyFuncs: []verifyFunc{
				verifyCount(2),
				verifyUniqueLabelValues("unique-test-label"),
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
			verifyFuncs: []verifyFunc{
				verifyCount(2),
				verifyObj(
					&v1.Node{
						ObjectMeta: metav1.ObjectMeta{
							GenerateName: "node-0-",
						},
						Status: v1.NodeStatus{
							Capacity: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("4"),
								v1.ResourceMemory: resource.MustParse("32Gi"),
								v1.ResourcePods:   resource.MustParse("110"),
							},
							Allocatable: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("2"),
								v1.ResourceMemory: resource.MustParse("4Gi"),
								v1.ResourcePods:   resource.MustParse("110"),
							},
						},
					}),
			},
		},
		{
			name: "Create Nodes with Custom Template",
			op: &createNodesOp{
				Opcode: createNodesOpcode,
				Count:  2,
				NodeTemplatePath: createObjTemplateFile(t,
					&v1.Node{
						ObjectMeta: metav1.ObjectMeta{
							GenerateName: "custom-node-",
						},
						Status: v1.NodeStatus{
							Capacity: v1.ResourceList{
								v1.ResourcePods:   resource.MustParse("100"),
								v1.ResourceCPU:    resource.MustParse("4"),
								v1.ResourceMemory: resource.MustParse("8Gi"),
							},
						},
					},
				),
			},
			verifyFuncs: []verifyFunc{
				verifyCount(2),
				verifyObj(
					&v1.Node{
						ObjectMeta: metav1.ObjectMeta{
							GenerateName: "custom-node-",
						},
						Status: v1.NodeStatus{
							Capacity: v1.ResourceList{
								v1.ResourcePods:   resource.MustParse("100"),
								v1.ResourceCPU:    resource.MustParse("4"),
								v1.ResourceMemory: resource.MustParse("8Gi"),
							},
						},
					},
				),
			},
		},
		{
			name: "Invalid Node Template Path",
			op: &createNodesOp{
				Opcode:           createNodesOpcode,
				Count:            1,
				NodeTemplatePath: ptr.To("non-existent-file.json"),
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
					t.Fatalf("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Fatalf("Failed to run operation: %v", err)
			}

			if tt.verifyFuncs != nil {
				for i, vf := range tt.verifyFuncs {
					if err := vf(t, tCtx, tt.op); err != nil {
						t.Fatalf("Verification function %d failed for test %q: %v", i, tt.name, err)
					}
				}
			}
		})
	}
}

// verifyCount returns a verification function that checks if the number of existing objects
// matches the expected count based on the operation type.
func verifyCount(expectedCount int) verifyFunc {
	return func(t *testing.T, tCtx ktesting.TContext, op realOp) error {
		switch op.(type) {
		case *createNodesOp:
			nodes, err := tCtx.Client().CoreV1().Nodes().List(context.Background(), metav1.ListOptions{})
			if err != nil {
				return fmt.Errorf("failed to list nodes: %w", err)
			}
			if got := len(nodes.Items); got != expectedCount {
				return fmt.Errorf("unexpected node count: got %d, want %d", got, expectedCount)
			}
		default:
			return fmt.Errorf("verifyCount doesn't support this operation type: %T", op)
		}
		return nil
	}
}

// verifyLabelValuesAllowed returns a verification function that checks if the label values for a given key.
func verifyLabelValuesAllowed(key string, allowValues sets.Set[string]) verifyFunc {
	return func(t *testing.T, tCtx ktesting.TContext, op realOp) error {
		labelValues, _, err := objLabelValues(t, tCtx, op, key)
		if err != nil {
			return fmt.Errorf("failed to get label values: %w", err)
		}

		for labelValue := range labelValues {
			if !allowValues.Has(labelValue) {
				return fmt.Errorf("Node has unexpected label value %s for key %s", labelValue, key)
			}
		}

		return nil
	}
}

// verifyUniqueLabelValues returns a verification function that checks if the label values for a given key are unique.
func verifyUniqueLabelValues(key string) verifyFunc {
	return func(t *testing.T, tCtx ktesting.TContext, op realOp) error {
		_, duplicatedValues, err := objLabelValues(t, tCtx, op, key)
		if err != nil {
			return fmt.Errorf("failed to get label values: %w", err)
		}

		if duplicatedValues.Len() > 0 {
			return fmt.Errorf("Node has duplicate label values %v for key %s", duplicatedValues.UnsortedList(), key)
		}

		return nil
	}
}

// objLabelValues is a helper function to extract label values from the listed objects.
// The listed objects are dependent on the operation type.
// It returns two sets: one with deduplicated labelValues and second with duplicated labels.
func objLabelValues(t *testing.T, tCtx ktesting.TContext, op realOp, key string) (sets.Set[string], sets.Set[string], error) {
	t.Helper()

	labelValues := sets.New[string]()
	duplicatedValues := sets.New[string]()

	switch op.(type) {
	case *createNodesOp:
		nodes, err := tCtx.Client().CoreV1().Nodes().List(context.Background(), metav1.ListOptions{})
		if err != nil {
			return nil, nil, fmt.Errorf("failed to list nodes for label verification: %w", err)
		}

		for _, node := range nodes.Items {
			if labelValue, exists := node.Labels[key]; exists {
				if labelValues.Has(labelValue) {
					duplicatedValues.Insert(labelValue)
				}
				labelValues.Insert(labelValue)
			} else {
				return nil, nil, fmt.Errorf("Node %s is missing expected label %s", node.Name, key)
			}
		}
	default:
		return nil, nil, fmt.Errorf("verifyLabel doesn't support this operation type: %T", op)
	}

	return labelValues, duplicatedValues, nil
}

// verifyObj checks if listed objects match the expected template object using cmp.Diff.
func verifyObj(expectedObj any) verifyFunc {
	return func(t *testing.T, tCtx ktesting.TContext, op realOp) error {
		var got, want any
		var cmpOpts []cmp.Option
		switch opDetails := op.(type) {
		case *createNodesOp:
			nodesList, listErr := tCtx.Client().CoreV1().Nodes().List(context.Background(), metav1.ListOptions{})
			if listErr != nil {
				return fmt.Errorf("failed to list nodes: %w", listErr)
			}
			gotNodes := nodesList.Items

			expectedNodeTemplate, ok := expectedObj.(*v1.Node)
			if !ok {
				return fmt.Errorf("expectedObj must be *v1.Node when op is *createNodesOp, got %T", expectedObj)
			}

			wantNodes := make([]v1.Node, len(gotNodes))
			// we don't need to verify len(), we just need to check if all of them are the same as the expected one.
			for i := range gotNodes {
				wantNodes[i] = *expectedNodeTemplate
			}

			cmpOpts = []cmp.Option{
				cmpopts.EquateEmpty(),
				cmpopts.IgnoreFields(metav1.ObjectMeta{},
					"UID", "ResourceVersion", "Generation", "CreationTimestamp", "ManagedFields", "SelfLink", "Name",
					"Labels", // verifyObj doesn't care about labels.
				),
				cmpopts.IgnoreFields(v1.NodeStatus{}, // This test isn't interested in these fields.
					"Conditions",
					"Phase",
				),
			}
			got = gotNodes
			want = wantNodes
		default:
			return fmt.Errorf("verifyObj doesn't support this operation type for cmp.Diff: %T", opDetails)
		}

		if diff := cmp.Diff(want, got, cmpOpts...); diff != "" {
			return fmt.Errorf("unexpected difference (-want +got):\\n%s", diff)
		}
		return nil
	}
}

// createObjTemplateFile creates a temporary file with the given object serialized as JSON.
func createObjTemplateFile(t *testing.T, obj any) *string {
	t.Helper()

	dir, err := os.MkdirTemp("", "scheduler-perf-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir for the template: %v", err)
	}
	t.Cleanup(func() {
		if err := os.RemoveAll(dir); err != nil {
			t.Errorf("Failed to remove temp dir: %v", err)
		}
	})

	templateFile := filepath.Join(dir, "template.json")
	f, err := os.Create(templateFile)
	if err != nil {
		t.Fatalf("Failed to create the template file %s: %v", templateFile, err)
	}
	defer func() {
		if err := f.Close(); err != nil {
			t.Errorf("Failed to close file: %v", err)
		}
	}()

	switch obj := obj.(type) {
	case *v1.Node:
		if err := json.NewEncoder(f).Encode(obj); err != nil {
			t.Fatalf("Failed to encode the template to %s: %v", templateFile, err)
		}
	default:
		t.Fatalf("Unsupported object type for template file: %T", obj)
	}
	return &templateFile
}

func TestFeatureGatesMerge(t *testing.T) {
	const (
		FeatureA featuregate.Feature = "FeatureA"
		FeatureB featuregate.Feature = "FeatureB"
		FeatureC featuregate.Feature = "FeatureC"
	)

	tests := []struct {
		name      string
		src       map[featuregate.Feature]bool
		overrides map[featuregate.Feature]bool
		want      map[featuregate.Feature]bool
	}{
		{
			name:      "both nil, return empty map",
			src:       nil,
			overrides: nil,
			want:      map[featuregate.Feature]bool{},
		},
		{
			name:      "both empty, return empty map",
			src:       map[featuregate.Feature]bool{},
			overrides: map[featuregate.Feature]bool{},
			want:      map[featuregate.Feature]bool{},
		},
		{
			name:      "nil src, valid overrides",
			src:       nil,
			overrides: map[featuregate.Feature]bool{FeatureA: true},
			want:      map[featuregate.Feature]bool{FeatureA: true},
		},
		{
			name:      "valid src, nil overrides",
			src:       map[featuregate.Feature]bool{FeatureA: true},
			overrides: nil,
			want:      map[featuregate.Feature]bool{FeatureA: true},
		},
		{
			name:      "distinct features merged",
			src:       map[featuregate.Feature]bool{FeatureA: true},
			overrides: map[featuregate.Feature]bool{FeatureB: false},
			want:      map[featuregate.Feature]bool{FeatureA: true, FeatureB: false},
		},
		{
			name:      "overlap with the same value",
			src:       map[featuregate.Feature]bool{FeatureA: true, FeatureB: true},
			overrides: map[featuregate.Feature]bool{FeatureB: true},
			want:      map[featuregate.Feature]bool{FeatureA: true, FeatureB: true},
		},
		{
			name:      "overlap with override (true to false)",
			src:       map[featuregate.Feature]bool{FeatureA: true},
			overrides: map[featuregate.Feature]bool{FeatureA: false},
			want:      map[featuregate.Feature]bool{FeatureA: false},
		},
		{
			name:      "overlap with override (false to true)",
			src:       map[featuregate.Feature]bool{FeatureA: false},
			overrides: map[featuregate.Feature]bool{FeatureA: true},
			want:      map[featuregate.Feature]bool{FeatureA: true},
		},
		{
			name:      "mixed distinct and overlap",
			src:       map[featuregate.Feature]bool{FeatureA: true, FeatureB: true},
			overrides: map[featuregate.Feature]bool{FeatureB: false, FeatureC: true},
			want:      map[featuregate.Feature]bool{FeatureA: true, FeatureB: false, FeatureC: true},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := featureGatesMerge(tt.src, tt.overrides)
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("Unexpected featureGatesMerge result (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestCompareMetricWithThreshold(t *testing.T) {
	tests := []struct {
		name      string
		items     []DataItem
		threshold float64
		selector  thresholdMetricSelector
		wantErr   bool
	}{
		{
			name:      "no items, should pass",
			items:     []DataItem{},
			threshold: 50,
			selector: thresholdMetricSelector{
				Name:       "TargetMetric",
				DataBucket: "Average",
			},
			wantErr: false,
		},
		{
			name: "zero threshold, should always pass",
			items: []DataItem{
				{
					Labels: map[string]string{"Metric": "TargetMetric"},
					Data:   map[string]float64{"Average": 10},
				},
			},
			threshold: 0,
			selector: thresholdMetricSelector{
				Name:        "TargetMetric",
				DataBucket:  "Average",
				ExpectLower: true,
			},
			wantErr: false,
		},
		{
			name: "metric not found in items, should pass (ignored)",
			items: []DataItem{
				{
					Labels: map[string]string{"Metric": "OtherMetric"},
					Data:   map[string]float64{"Average": 10},
				},
			},
			threshold: 50,
			selector: thresholdMetricSelector{
				Name:       "TargetMetric",
				DataBucket: "Average",
			},
			wantErr: false,
		},
		{
			name: "labels do not match, should pass (ignored)",
			items: []DataItem{
				{
					Labels: map[string]string{"Metric": "TargetMetric", "plugin": "foo"},
					Data:   map[string]float64{"Average": 10},
				},
			},
			threshold: 50,
			selector: thresholdMetricSelector{
				Name:       "TargetMetric",
				Labels:     map[string]string{"plugin": "bar"},
				DataBucket: "Average",
			},
			wantErr: false,
		},
		{
			name: "labels match, value lower than threshold (ExpectLower=false), should fail",
			items: []DataItem{
				{
					Labels: map[string]string{"Metric": "TargetMetric", "plugin": "foo"},
					Data:   map[string]float64{"Average": 10},
				},
			},
			threshold: 50,
			selector: thresholdMetricSelector{
				Name:       "TargetMetric",
				Labels:     map[string]string{"plugin": "foo"},
				DataBucket: "Average",
			},
			wantErr: true,
		},
		{
			name: "missing data bucket in item, should fail",
			items: []DataItem{
				{
					Labels: map[string]string{"Metric": "TargetMetric"},
					Data:   map[string]float64{"Average": 100},
				},
			},
			threshold: 50,
			selector: thresholdMetricSelector{
				Name:       "TargetMetric",
				DataBucket: "99Perc",
			},
			wantErr: true,
		},
		{
			name: "value higher than threshold (ExpectLower=false), should pass",
			items: []DataItem{
				{
					Labels: map[string]string{"Metric": "Throughput"},
					Data:   map[string]float64{"Average": 100},
				},
			},
			threshold: 50,
			selector: thresholdMetricSelector{
				Name:        "Throughput",
				DataBucket:  "Average",
				ExpectLower: false,
			},
			wantErr: false,
		},
		{
			name: "value lower than threshold (ExpectLower=false), should fail",
			items: []DataItem{
				{
					Labels: map[string]string{"Metric": "Throughput"},
					Data:   map[string]float64{"Average": 10},
				},
			},
			threshold: 50,
			selector: thresholdMetricSelector{
				Name:        "Throughput",
				DataBucket:  "Average",
				ExpectLower: false,
			},
			wantErr: true,
		},
		{
			name: "value lower than threshold (ExpectLower=true), should pass",
			items: []DataItem{
				{
					Labels: map[string]string{"Metric": "Latency"},
					Data:   map[string]float64{"Average": 10},
				},
			},
			threshold: 50,
			selector: thresholdMetricSelector{
				Name:        "Latency",
				DataBucket:  "Average",
				ExpectLower: true,
			},
			wantErr: false,
		},
		{
			name: "value higher than threshold (ExpectLower=true), should fail",
			items: []DataItem{
				{
					Labels: map[string]string{"Metric": "Latency"},
					Data:   map[string]float64{"Average": 100},
				},
			},
			threshold: 50,
			selector: thresholdMetricSelector{
				Name:        "Latency",
				DataBucket:  "Average",
				ExpectLower: true,
			},
			wantErr: true,
		},
		{
			name: "value exactly equals threshold, should fail",
			items: []DataItem{
				{
					Labels: map[string]string{"Metric": "Throughput"},
					Data:   map[string]float64{"Average": 50},
				},
			},
			threshold: 50,
			selector: thresholdMetricSelector{
				Name:       "Throughput",
				DataBucket: "Average",
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := compareMetricWithThreshold(tt.items, tt.threshold, tt.selector)
			if err != nil {
				if !tt.wantErr {
					t.Errorf("Expected no error in compareMetricWithThreshold, but got: %v", err)
				}
			} else {
				if tt.wantErr {
					t.Errorf("Expected error %v in compareMetricWithThreshold, but got nil", tt.wantErr)
				}
			}
		})
	}
}

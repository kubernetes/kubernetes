/*
Copyright 2017 The Kubernetes Authors.

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

package scheduler

import (
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

// This file tests the scheduler predicates functionality.

const pollInterval = 100 * time.Millisecond

// TestInterPodAffinity verifies that scheduler's inter pod affinity and
// anti-affinity predicate functions works correctly.
func TestInterPodAffinity(t *testing.T) {
	testCtx := initTest(t, "inter-pod-affinity")
	defer cleanupTest(t, testCtx)
	// Add a few nodes.
	nodes, err := createNodes(testCtx.clientSet, "testnode", nil, 2)
	if err != nil {
		t.Fatalf("Cannot create nodes: %v", err)
	}
	// Add labels to the nodes.
	labels1 := map[string]string{
		"region": "r1",
		"zone":   "z11",
	}
	for _, node := range nodes {
		if err = testutils.AddLabelsToNode(testCtx.clientSet, node.Name, labels1); err != nil {
			t.Fatalf("Cannot add labels to node: %v", err)
		}
		if err = waitForNodeLabels(testCtx.clientSet, node.Name, labels1); err != nil {
			t.Fatalf("Adding labels to node didn't succeed: %v", err)
		}
	}

	cs := testCtx.clientSet
	podLabel := map[string]string{"service": "securityscan"}
	podLabel2 := map[string]string{"security": "S1"}

	tests := []struct {
		pod       *v1.Pod
		pods      []*v1.Pod
		node      *v1.Node
		fits      bool
		errorType string
		test      string
	}{
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "fakename",
					Labels: podLabel2,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "container", Image: imageutils.GetPauseImageName()}},
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "security",
												Operator: metav1.LabelSelectorOpDoesNotExist,
											},
										},
									},
									TopologyKey: "region",
								},
							},
						},
					},
				},
			},
			node:      nodes[0],
			fits:      false,
			errorType: "invalidPod",
			test:      "validates that a pod with an invalid podAffinity is rejected because of the LabelSelectorRequirement is invalid",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "fakename",
					Labels: podLabel2,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "container", Image: imageutils.GetPauseImageName()}},
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "security",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"securityscan"},
											},
										},
									},
									TopologyKey: "region",
								},
							},
						},
					},
				},
			},
			node: nodes[0],
			fits: false,
			test: "validates that Inter-pod-Affinity is respected if not matching",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "fakename",
					Labels: podLabel2,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "container", Image: imageutils.GetPauseImageName()}},
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"securityscan", "value2"},
											},
										},
									},
									TopologyKey: "region",
								},
							},
						},
					},
				},
			},
			pods: []*v1.Pod{{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "fakename2",
					Labels: podLabel,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "container", Image: imageutils.GetPauseImageName()}},
					NodeName:   nodes[0].Name,
				},
			},
			},
			node: nodes[0],
			fits: true,
			test: "validates that InterPodAffinity is respected if matching. requiredDuringSchedulingIgnoredDuringExecution in PodAffinity using In operator that matches the existing pod",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "fakename",
					Labels: podLabel2,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "container", Image: imageutils.GetPauseImageName()}},
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpNotIn,
												Values:   []string{"securityscan3", "value3"},
											},
										},
									},
									TopologyKey: "region",
								},
							},
						},
					},
				},
			},
			pods: []*v1.Pod{{Spec: v1.PodSpec{
				Containers: []v1.Container{{Name: "container", Image: imageutils.GetPauseImageName()}},
				NodeName:   nodes[0].Name},
				ObjectMeta: metav1.ObjectMeta{
					Name:   "fakename2",
					Labels: podLabel}}},
			node: nodes[0],
			fits: true,
			test: "validates that InterPodAffinity is respected if matching. requiredDuringSchedulingIgnoredDuringExecution in PodAffinity using not in operator in labelSelector that matches the existing pod",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "fakename",
					Labels: podLabel2,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "container", Image: imageutils.GetPauseImageName()}},
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"securityscan", "value2"},
											},
										},
									},
									TopologyKey: "region",
									Namespaces:  []string{"diff-namespace"},
								},
							},
						},
					},
				},
			},
			pods: []*v1.Pod{{Spec: v1.PodSpec{
				Containers: []v1.Container{{Name: "container", Image: imageutils.GetPauseImageName()}},
				NodeName:   nodes[0].Name},
				ObjectMeta: metav1.ObjectMeta{
					Name:   "fakename2",
					Labels: podLabel, Namespace: "ns"}}},
			node: nodes[0],
			fits: false,
			test: "validates that inter-pod-affinity is respected when pods have different Namespaces",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "fakename",
					Labels: podLabel,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "container", Image: imageutils.GetPauseImageName()}},
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"antivirusscan", "value2"},
											},
										},
									},
									TopologyKey: "region",
								},
							},
						},
					},
				},
			},
			pods: []*v1.Pod{{Spec: v1.PodSpec{
				Containers: []v1.Container{{Name: "container", Image: imageutils.GetPauseImageName()}},
				NodeName:   nodes[0].Name}, ObjectMeta: metav1.ObjectMeta{
				Name:   "fakename2",
				Labels: podLabel}}},
			node: nodes[0],
			fits: false,
			test: "Doesn't satisfy the PodAffinity because of unmatching labelSelector with the existing pod",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "fakename",
					Labels: podLabel2,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "container", Image: imageutils.GetPauseImageName()}},
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpExists,
											}, {
												Key:      "wrongkey",
												Operator: metav1.LabelSelectorOpDoesNotExist,
											},
										},
									},
									TopologyKey: "region",
								}, {
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"securityscan"},
											}, {
												Key:      "service",
												Operator: metav1.LabelSelectorOpNotIn,
												Values:   []string{"WrongValue"},
											},
										},
									},
									TopologyKey: "region",
								},
							},
						},
					},
				},
			},
			pods: []*v1.Pod{{Spec: v1.PodSpec{
				Containers: []v1.Container{{Name: "container", Image: imageutils.GetPauseImageName()}},
				NodeName:   nodes[0].Name}, ObjectMeta: metav1.ObjectMeta{
				Name:   "fakename2",
				Labels: podLabel}}},
			node: nodes[0],
			fits: true,
			test: "validates that InterPodAffinity is respected if matching with multiple affinities in multiple RequiredDuringSchedulingIgnoredDuringExecution ",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: podLabel2,
					Name:   "fakename",
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "container", Image: imageutils.GetPauseImageName()}},
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpExists,
											}, {
												Key:      "wrongkey",
												Operator: metav1.LabelSelectorOpDoesNotExist,
											},
										},
									},
									TopologyKey: "region",
								}, {
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"securityscan2"},
											}, {
												Key:      "service",
												Operator: metav1.LabelSelectorOpNotIn,
												Values:   []string{"WrongValue"},
											},
										},
									},
									TopologyKey: "region",
								},
							},
						},
					},
				},
			},
			pods: []*v1.Pod{{Spec: v1.PodSpec{
				Containers: []v1.Container{{Name: "container", Image: imageutils.GetPauseImageName()}},
				NodeName:   nodes[0].Name}, ObjectMeta: metav1.ObjectMeta{
				Name:   "fakename2",
				Labels: podLabel}}},
			node: nodes[0],
			fits: false,
			test: "The labelSelector requirements(items of matchExpressions) are ANDed, the pod cannot schedule onto the node because one of the matchExpression items doesn't match.",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "fakename",
					Labels: podLabel2,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "container", Image: imageutils.GetPauseImageName()}},
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"securityscan", "value2"},
											},
										},
									},
									TopologyKey: "region",
								},
							},
						},
						PodAntiAffinity: &v1.PodAntiAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"antivirusscan", "value2"},
											},
										},
									},
									TopologyKey: "node",
								},
							},
						},
					},
				},
			},
			pods: []*v1.Pod{{Spec: v1.PodSpec{
				Containers: []v1.Container{{Name: "container", Image: imageutils.GetPauseImageName()}},
				NodeName:   nodes[0].Name}, ObjectMeta: metav1.ObjectMeta{
				Name:   "fakename2",
				Labels: podLabel}}},
			node: nodes[0],
			fits: true,
			test: "validates that InterPod Affinity and AntiAffinity is respected if matching",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "fakename",
					Labels: podLabel2,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "container", Image: imageutils.GetPauseImageName()}},
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"securityscan", "value2"},
											},
										},
									},
									TopologyKey: "region",
								},
							},
						},
						PodAntiAffinity: &v1.PodAntiAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"antivirusscan", "value2"},
											},
										},
									},
									TopologyKey: "node",
								},
							},
						},
					},
				},
			},
			pods: []*v1.Pod{
				{
					Spec: v1.PodSpec{
						Containers: []v1.Container{{Name: "container", Image: imageutils.GetPauseImageName()}},
						NodeName:   nodes[0].Name,
						Affinity: &v1.Affinity{
							PodAntiAffinity: &v1.PodAntiAffinity{
								RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
									{
										LabelSelector: &metav1.LabelSelector{
											MatchExpressions: []metav1.LabelSelectorRequirement{
												{
													Key:      "service",
													Operator: metav1.LabelSelectorOpIn,
													Values:   []string{"antivirusscan", "value2"},
												},
											},
										},
										TopologyKey: "node",
									},
								},
							},
						},
					},
					ObjectMeta: metav1.ObjectMeta{
						Name:   "fakename2",
						Labels: podLabel},
				},
			},
			node: nodes[0],
			fits: true,
			test: "satisfies the PodAffinity and PodAntiAffinity and PodAntiAffinity symmetry with the existing pod",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "fakename",
					Labels: podLabel2,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "container", Image: imageutils.GetPauseImageName()}},
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"securityscan", "value2"},
											},
										},
									},
									TopologyKey: "region",
								},
							},
						},
						PodAntiAffinity: &v1.PodAntiAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"securityscan", "value2"},
											},
										},
									},
									TopologyKey: "zone",
								},
							},
						},
					},
				},
			},
			pods: []*v1.Pod{{Spec: v1.PodSpec{
				Containers: []v1.Container{{Name: "container", Image: imageutils.GetPauseImageName()}},
				NodeName:   nodes[0].Name}, ObjectMeta: metav1.ObjectMeta{
				Name:   "fakename2",
				Labels: podLabel}}},
			node: nodes[0],
			fits: false,
			test: "satisfies the PodAffinity but doesn't satisfies the PodAntiAffinity with the existing pod",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "fakename",
					Labels: podLabel,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "container", Image: imageutils.GetPauseImageName()}},
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"securityscan", "value2"},
											},
										},
									},
									TopologyKey: "region",
								},
							},
						},
						PodAntiAffinity: &v1.PodAntiAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"antivirusscan", "value2"},
											},
										},
									},
									TopologyKey: "node",
								},
							},
						},
					},
				},
			},
			pods: []*v1.Pod{
				{
					Spec: v1.PodSpec{
						NodeName:   nodes[0].Name,
						Containers: []v1.Container{{Name: "container", Image: imageutils.GetPauseImageName()}},
						Affinity: &v1.Affinity{
							PodAntiAffinity: &v1.PodAntiAffinity{
								RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
									{
										LabelSelector: &metav1.LabelSelector{
											MatchExpressions: []metav1.LabelSelectorRequirement{
												{
													Key:      "service",
													Operator: metav1.LabelSelectorOpIn,
													Values:   []string{"securityscan", "value3"},
												},
											},
										},
										TopologyKey: "zone",
									},
								},
							},
						},
					},
					ObjectMeta: metav1.ObjectMeta{
						Name:   "fakename2",
						Labels: podLabel},
				},
			},
			node: nodes[0],
			fits: false,
			test: "satisfies the PodAffinity and PodAntiAffinity but doesn't satisfies PodAntiAffinity symmetry with the existing pod",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "fakename",
					Labels: podLabel,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "container", Image: imageutils.GetPauseImageName()}},
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpNotIn,
												Values:   []string{"securityscan", "value2"},
											},
										},
									},
									TopologyKey: "region",
								},
							},
						},
					},
				},
			},
			pods: []*v1.Pod{{Spec: v1.PodSpec{
				Containers: []v1.Container{{Name: "container", Image: imageutils.GetPauseImageName()}},
				NodeName:   "machine2"}, ObjectMeta: metav1.ObjectMeta{
				Name:   "fakename2",
				Labels: podLabel}}},
			node: nodes[0],
			fits: false,
			test: "pod matches its own Label in PodAffinity and that matches the existing pod Labels",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "fakename",
					Labels: podLabel,
				},
				Spec: v1.PodSpec{Containers: []v1.Container{{Name: "container", Image: imageutils.GetPauseImageName()}}},
			},
			pods: []*v1.Pod{
				{
					Spec: v1.PodSpec{NodeName: nodes[0].Name,
						Containers: []v1.Container{{Name: "container", Image: imageutils.GetPauseImageName()}},
						Affinity: &v1.Affinity{
							PodAntiAffinity: &v1.PodAntiAffinity{
								RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
									{
										LabelSelector: &metav1.LabelSelector{
											MatchExpressions: []metav1.LabelSelectorRequirement{
												{
													Key:      "service",
													Operator: metav1.LabelSelectorOpIn,
													Values:   []string{"securityscan", "value2"},
												},
											},
										},
										TopologyKey: "zone",
									},
								},
							},
						},
					},
					ObjectMeta: metav1.ObjectMeta{
						Name:   "fakename2",
						Labels: podLabel},
				},
			},
			node: nodes[0],
			fits: false,
			test: "Verify that PodAntiAffinity of an existing pod is respected when PodAntiAffinity symmetry is not satisfied with the existing pod",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "fake-name",
					Labels: podLabel,
				},
				Spec: v1.PodSpec{Containers: []v1.Container{{Name: "container", Image: imageutils.GetPauseImageName()}}},
			},
			pods: []*v1.Pod{
				{
					Spec: v1.PodSpec{NodeName: nodes[0].Name,
						Containers: []v1.Container{{Name: "container", Image: imageutils.GetPauseImageName()}},
						Affinity: &v1.Affinity{
							PodAntiAffinity: &v1.PodAntiAffinity{
								RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
									{
										LabelSelector: &metav1.LabelSelector{
											MatchExpressions: []metav1.LabelSelectorRequirement{
												{
													Key:      "service",
													Operator: metav1.LabelSelectorOpNotIn,
													Values:   []string{"securityscan", "value2"},
												},
											},
										},
										TopologyKey: "zone",
									},
								},
							},
						},
					},
					ObjectMeta: metav1.ObjectMeta{
						Name:   "fake-name2",
						Labels: podLabel},
				},
			},
			node: nodes[0],
			fits: true,
			test: "Verify that PodAntiAffinity from existing pod is respected when pod statisfies PodAntiAffinity symmetry with the existing pod",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "fake-name2"},
				Spec: v1.PodSpec{
					Containers:   []v1.Container{{Name: "container", Image: imageutils.GetPauseImageName()}},
					NodeSelector: map[string]string{"region": "r1"},
					Affinity: &v1.Affinity{
						PodAntiAffinity: &v1.PodAntiAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "foo",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"abc"},
											},
										},
									},
									TopologyKey: "region",
								},
							},
						},
					},
				},
			},
			pods: []*v1.Pod{
				{Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "container", Image: imageutils.GetPauseImageName()}},
					NodeName:   nodes[0].Name}, ObjectMeta: metav1.ObjectMeta{Name: "fakename", Labels: map[string]string{"foo": "abc"}}},
			},
			fits: false,
			test: "nodes[0] and nodes[1] have same topologyKey and label value. nodes[0] has an existing pod that matches the inter pod affinity rule. The new pod can not be scheduled onto either of the two nodes.",
		},
	}

	for _, test := range tests {
		for _, pod := range test.pods {
			var nsName string
			if pod.Namespace != "" {
				nsName = pod.Namespace
			} else {
				nsName = testCtx.ns.Name
			}
			createdPod, err := cs.CoreV1().Pods(nsName).Create(pod)
			if err != nil {
				t.Fatalf("Test Failed: error, %v, while creating pod during test: %v", err, test.test)
			}
			err = wait.Poll(pollInterval, wait.ForeverTestTimeout, podScheduled(cs, createdPod.Namespace, createdPod.Name))
			if err != nil {
				t.Errorf("Test Failed: error, %v, while waiting for pod during test, %v", err, test)
			}
		}
		testPod, err := cs.CoreV1().Pods(testCtx.ns.Name).Create(test.pod)
		if err != nil {
			if !(test.errorType == "invalidPod" && apierrors.IsInvalid(err)) {
				t.Fatalf("Test Failed: error, %v, while creating pod during test: %v", err, test.test)
			}
		}

		if test.fits {
			err = wait.Poll(pollInterval, wait.ForeverTestTimeout, podScheduled(cs, testPod.Namespace, testPod.Name))
		} else {
			err = wait.Poll(pollInterval, wait.ForeverTestTimeout, podUnschedulable(cs, testPod.Namespace, testPod.Name))
		}
		if err != nil {
			t.Errorf("Test Failed: %v, err %v, test.fits %v", test.test, err, test.fits)
		}

		err = cs.CoreV1().Pods(testCtx.ns.Name).Delete(test.pod.Name, metav1.NewDeleteOptions(0))
		if err != nil {
			t.Errorf("Test Failed: error, %v, while deleting pod during test: %v", err, test.test)
		}
		err = wait.Poll(pollInterval, wait.ForeverTestTimeout, podDeleted(cs, testCtx.ns.Name, test.pod.Name))
		if err != nil {
			t.Errorf("Test Failed: error, %v, while waiting for pod to get deleted, %v", err, test.test)
		}
		for _, pod := range test.pods {
			var nsName string
			if pod.Namespace != "" {
				nsName = pod.Namespace
			} else {
				nsName = testCtx.ns.Name
			}
			err = cs.CoreV1().Pods(nsName).Delete(pod.Name, metav1.NewDeleteOptions(0))
			if err != nil {
				t.Errorf("Test Failed: error, %v, while deleting pod during test: %v", err, test.test)
			}
			err = wait.Poll(pollInterval, wait.ForeverTestTimeout, podDeleted(cs, nsName, pod.Name))
			if err != nil {
				t.Errorf("Test Failed: error, %v, while waiting for pod to get deleted, %v", err, test.test)
			}
		}
	}
}

// TestEvenPodsSpreadPredicate verifies that EvenPodsSpread predicate functions well.
func TestEvenPodsSpreadPredicate(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.EvenPodsSpread, true)()

	testCtx := initTest(t, "eps-predicate")
	cs := testCtx.clientSet
	ns := testCtx.ns.Name
	defer cleanupTest(t, testCtx)
	// Add 4 nodes.
	nodes, err := createNodes(cs, "node", nil, 4)
	if err != nil {
		t.Fatalf("Cannot create nodes: %v", err)
	}
	for i, node := range nodes {
		// Apply labels "zone: zone-{0,1}" and "node: <node name>" to each node.
		labels := map[string]string{
			"zone": fmt.Sprintf("zone-%d", i/2),
			"node": node.Name,
		}
		if err = testutils.AddLabelsToNode(cs, node.Name, labels); err != nil {
			t.Fatalf("Cannot add labels to node: %v", err)
		}
		if err = waitForNodeLabels(cs, node.Name, labels); err != nil {
			t.Fatalf("Failed to poll node labels: %v", err)
		}
	}

	pause := imageutils.GetPauseImageName()
	tests := []struct {
		name           string
		incomingPod    *v1.Pod
		existingPods   []*v1.Pod
		fits           bool
		candidateNodes []string // nodes expected to schedule onto
	}{
		// note: naming starts at index 0
		{
			name: "place pod on a 1/1/0/1 cluster with MaxSkew=1, node-2 is the only fit",
			incomingPod: st.MakePod().Namespace(ns).Name("p").Label("foo", "").Container(pause).
				SpreadConstraint(1, "node", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Namespace(ns).Name("p0").Node("node-0").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Namespace(ns).Name("p1").Node("node-1").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Namespace(ns).Name("p3").Node("node-3").Label("foo", "").Container(pause).Obj(),
			},
			fits:           true,
			candidateNodes: []string{"node-2"},
		},
		{
			name: "place pod on a 2/0/0/1 cluster with MaxSkew=2, node-{1,2,3} are good fits",
			incomingPod: st.MakePod().Namespace(ns).Name("p").Label("foo", "").Container(pause).
				SpreadConstraint(2, "node", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Namespace(ns).Name("p0a").Node("node-0").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Namespace(ns).Name("p0b").Node("node-0").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Namespace(ns).Name("p3").Node("node-3").Label("foo", "").Container(pause).Obj(),
			},
			fits:           true,
			candidateNodes: []string{"node-1", "node-2", "node-3"},
		},
		{
			name: "pod is required to be placed on zone0, so only node-1 fits",
			incomingPod: st.MakePod().Namespace(ns).Name("p").Label("foo", "").Container(pause).
				NodeAffinityIn("zone", []string{"zone-0"}).
				SpreadConstraint(1, "node", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Namespace(ns).Name("p0").Node("node-0").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Namespace(ns).Name("p3").Node("node-3").Label("foo", "").Container(pause).Obj(),
			},
			fits:           true,
			candidateNodes: []string{"node-1"},
		},
		{
			name: "two constraints: pod can only be placed to zone-1/node-2",
			incomingPod: st.MakePod().Namespace(ns).Name("p").Label("foo", "").Container(pause).
				SpreadConstraint(1, "zone", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Namespace(ns).Name("p0").Node("node-0").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Namespace(ns).Name("p1").Node("node-1").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Namespace(ns).Name("p3a").Node("node-3").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Namespace(ns).Name("p3b").Node("node-3").Label("foo", "").Container(pause).Obj(),
			},
			fits:           true,
			candidateNodes: []string{"node-2"},
		},
		{
			name: "pod cannot be placed onto any node",
			incomingPod: st.MakePod().Namespace(ns).Name("p").Label("foo", "").Container(pause).
				NodeAffinityNotIn("node", []string{"node-0"}). // mock a 3-node cluster
				SpreadConstraint(1, "zone", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Namespace(ns).Name("p1a").Node("node-1").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Namespace(ns).Name("p1b").Node("node-1").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Namespace(ns).Name("p2a").Node("node-2").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Namespace(ns).Name("p2b").Node("node-2").Label("foo", "").Container(pause).Obj(),
				st.MakePod().Namespace(ns).Name("p3").Node("node-3").Label("foo", "").Container(pause).Obj(),
			},
			fits: false,
		},
		{
			name: "high priority pod can preempt others",
			incomingPod: st.MakePod().Namespace(ns).Name("p").Label("foo", "").Container(pause).Priority(100).
				NodeAffinityNotIn("node", []string{"node-0"}). // mock a 3-node cluster
				SpreadConstraint(1, "zone", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				SpreadConstraint(1, "node", hardSpread, st.MakeLabelSelector().Exists("foo").Obj()).
				Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().ZeroTerminationGracePeriod().Namespace(ns).Name("p1a").Node("node-1").Label("foo", "").Container(pause).Obj(),
				st.MakePod().ZeroTerminationGracePeriod().Namespace(ns).Name("p1b").Node("node-1").Label("foo", "").Container(pause).Obj(),
				st.MakePod().ZeroTerminationGracePeriod().Namespace(ns).Name("p2a").Node("node-2").Label("foo", "").Container(pause).Obj(),
				st.MakePod().ZeroTerminationGracePeriod().Namespace(ns).Name("p2b").Node("node-2").Label("foo", "").Container(pause).Obj(),
				st.MakePod().ZeroTerminationGracePeriod().Namespace(ns).Name("p3").Node("node-3").Label("foo", "").Container(pause).Obj(),
			},
			fits:           true,
			candidateNodes: []string{"node-1", "node-2", "node-3"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			allPods := append(tt.existingPods, tt.incomingPod)
			defer cleanupPods(cs, t, allPods)
			for _, pod := range tt.existingPods {
				createdPod, err := cs.CoreV1().Pods(pod.Namespace).Create(pod)
				if err != nil {
					t.Fatalf("Test Failed: error while creating pod during test: %v", err)
				}
				err = wait.Poll(pollInterval, wait.ForeverTestTimeout, podScheduled(cs, createdPod.Namespace, createdPod.Name))
				if err != nil {
					t.Errorf("Test Failed: error while waiting for pod during test: %v", err)
				}
			}
			testPod, err := cs.CoreV1().Pods(tt.incomingPod.Namespace).Create(tt.incomingPod)
			if err != nil && !apierrors.IsInvalid(err) {
				t.Fatalf("Test Failed: error while creating pod during test: %v", err)
			}

			if tt.fits {
				err = wait.Poll(pollInterval, wait.ForeverTestTimeout, podScheduledIn(cs, testPod.Namespace, testPod.Name, tt.candidateNodes))
			} else {
				err = wait.Poll(pollInterval, wait.ForeverTestTimeout, podUnschedulable(cs, testPod.Namespace, testPod.Name))
			}
			if err != nil {
				t.Errorf("Test Failed: %v", err)
			}
		})
	}
}

var (
	hardSpread = v1.DoNotSchedule
	softSpread = v1.ScheduleAnyway
)

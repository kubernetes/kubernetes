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

package affinities

import (
	"testing"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func TestMergePodAffinities(t *testing.T) {
	tests := []struct {
		name            string
		podAffinity     *api.Affinity
		defaultAffinity *api.Affinity
		expected        *api.Affinity
	}{
		{
			name:            "merge with nil default affinity",
			podAffinity:     &api.Affinity{
				NodeAffinity: &api.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &api.NodeSelector{
						NodeSelectorTerms: []api.NodeSelectorTerm{
							{
								MatchExpressions: []api.NodeSelectorRequirement{
									{Key: "key1", Operator: api.NodeSelectorOpIn, Values: []string{"val1"}},
								},
							},
						},
					},
				},
			},
			defaultAffinity: nil,
			expected:        &api.Affinity{
				NodeAffinity: &api.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &api.NodeSelector{
						NodeSelectorTerms: []api.NodeSelectorTerm{
							{
								MatchExpressions: []api.NodeSelectorRequirement{
									{Key: "key1", Operator: api.NodeSelectorOpIn, Values: []string{"val1"}},
								},
							},
						},
					},
				},
			},
		},
		{
			name:            "merge with nil pod affinity",
			podAffinity:     nil,
			defaultAffinity: &api.Affinity{
				NodeAffinity: &api.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &api.NodeSelector{
						NodeSelectorTerms: []api.NodeSelectorTerm{
							{
								MatchExpressions: []api.NodeSelectorRequirement{
									{Key: "key1", Operator: api.NodeSelectorOpIn, Values: []string{"val1"}},
								},
							},
						},
					},
				},
			},
			expected:        &api.Affinity{
				NodeAffinity: &api.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &api.NodeSelector{
						NodeSelectorTerms: []api.NodeSelectorTerm{
							{
								MatchExpressions: []api.NodeSelectorRequirement{
									{Key: "key1", Operator: api.NodeSelectorOpIn, Values: []string{"val1"}},
								},
							},
						},
					},
				},
			},
		},
		{
			name: "merge two non-nil affinities",
			podAffinity: &api.Affinity{
				NodeAffinity: &api.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &api.NodeSelector{
						NodeSelectorTerms: []api.NodeSelectorTerm{
							{
								MatchExpressions: []api.NodeSelectorRequirement{
									{Key: "key1", Operator: api.NodeSelectorOpIn, Values: []string{"val1"}},
								},
							},
						},
					},
				},
			},
			defaultAffinity: &api.Affinity{
				PodAffinity: &api.PodAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{
						{
							LabelSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"svc": "s1"}},
							TopologyKey:   "zone",
						},
					},
				},
			},
			expected: &api.Affinity{
				NodeAffinity: &api.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &api.NodeSelector{
						NodeSelectorTerms: []api.NodeSelectorTerm{
							{
								MatchExpressions: []api.NodeSelectorRequirement{
									{Key: "key1", Operator: api.NodeSelectorOpIn, Values: []string{"val1"}},
								},
							},
						},
					},
				},
				PodAffinity: &api.PodAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{
						{
							LabelSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"svc": "s1"}},
							TopologyKey:   "zone",
						},
					},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			merged := MergePodAffinities(test.podAffinity, test.defaultAffinity)
			if !apiequality.Semantic.DeepEqual(test.expected, merged) {
				t.Errorf("expected \n%v, \ngot \n%v", test.expected, merged)
			}
		})
	}
}

func TestIsSuperset(t *testing.T) {
	t.Skip("isSuperset is not fully implemented yet")
}

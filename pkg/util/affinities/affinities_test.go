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

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func TestDedupAffinityFields(t *testing.T) {
	t.Run("NodeAffinity", func(t *testing.T) {
		testNodeAffinity(t)
	})
	t.Run("PodAffinity", func(t *testing.T) {
		testPodAffinity(t)
	})
	t.Run("PodAntiAffinity", func(t *testing.T) {
		testPodAntiAffinity(t)
	})
}

func testNodeAffinity(t *testing.T) {
	term1 := api.NodeSelectorTerm{
		MatchExpressions: []api.NodeSelectorRequirement{
			{Key: "key1", Operator: api.NodeSelectorOpIn, Values: []string{"value1"}},
		},
	}
	term2 := api.NodeSelectorTerm{
		MatchExpressions: []api.NodeSelectorRequirement{
			{Key: "key2", Operator: api.NodeSelectorOpIn, Values: []string{"value2"}},
		},
	}
	term3 := api.NodeSelectorTerm{
		MatchExpressions: []api.NodeSelectorRequirement{
			{Key: "key1", Operator: api.NodeSelectorOpIn, Values: []string{"value1"}},
			{Key: "key2", Operator: api.NodeSelectorOpIn, Values: []string{"value2"}},
		},
	}
	term4 := api.NodeSelectorTerm{
		MatchExpressions: []api.NodeSelectorRequirement{
			{Key: "key2", Operator: api.NodeSelectorOpIn, Values: []string{"value2"}},
			{Key: "key3", Operator: api.NodeSelectorOpIn, Values: []string{"value3"}},
		},
	}

	tests := []struct {
		name     string
		affinity *api.Affinity
		expected *api.Affinity
	}{
		{
			name: "RequiredDuringSchedulingIgnoredDuringExecution with duplicates",
			affinity: &api.Affinity{
				NodeAffinity: &api.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &api.NodeSelector{
						NodeSelectorTerms: []api.NodeSelectorTerm{term1, term2, term1},
					},
				},
			},
			expected: &api.Affinity{
				NodeAffinity: &api.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &api.NodeSelector{
						NodeSelectorTerms: []api.NodeSelectorTerm{term1, term2},
					},
				},
			},
		},
		{
			name: "RequiredDuringSchedulingIgnoredDuringExecution with subset",
			affinity: &api.Affinity{
				NodeAffinity: &api.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &api.NodeSelector{
						NodeSelectorTerms: []api.NodeSelectorTerm{term1, term3},
					},
				},
			},
			expected: &api.Affinity{
				NodeAffinity: &api.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &api.NodeSelector{
						NodeSelectorTerms: []api.NodeSelectorTerm{term3},
					},
				},
			},
		},
		{
			name: "PreferredDuringSchedulingIgnoredDuringExecution with duplicates",
			affinity: &api.Affinity{
				NodeAffinity: &api.NodeAffinity{
					PreferredDuringSchedulingIgnoredDuringExecution: []api.PreferredSchedulingTerm{
						{Weight: 1, Preference: term1},
						{Weight: 1, Preference: term2},
						{Weight: 1, Preference: term1},
					},
				},
			},
			expected: &api.Affinity{
				NodeAffinity: &api.NodeAffinity{
					PreferredDuringSchedulingIgnoredDuringExecution: []api.PreferredSchedulingTerm{
						{Weight: 1, Preference: term1},
						{Weight: 1, Preference: term2},
					},
				},
			},
		},
		{
			name: "PreferredDuringSchedulingIgnoredDuringExecution with subset",
			affinity: &api.Affinity{
				NodeAffinity: &api.NodeAffinity{
					PreferredDuringSchedulingIgnoredDuringExecution: []api.PreferredSchedulingTerm{
						{Weight: 1, Preference: term1},
						{Weight: 1, Preference: term3},
					},
				},
			},
			expected: &api.Affinity{
				NodeAffinity: &api.NodeAffinity{
					PreferredDuringSchedulingIgnoredDuringExecution: []api.PreferredSchedulingTerm{
						{Weight: 1, Preference: term3},
					},
				},
			},
		},
		{
			name: "PreferredDuringSchedulingIgnoredDuringExecution with intersection",
			affinity: &api.Affinity{
				NodeAffinity: &api.NodeAffinity{
					PreferredDuringSchedulingIgnoredDuringExecution: []api.PreferredSchedulingTerm{
						{Weight: 1, Preference: term3},
						{Weight: 1, Preference: term4},
					},
				},
			},
			expected: &api.Affinity{
				NodeAffinity: &api.NodeAffinity{
					PreferredDuringSchedulingIgnoredDuringExecution: []api.PreferredSchedulingTerm{
						{Weight: 1, Preference: term3},
						{Weight: 1, Preference: term4},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DedupAffinityFields(tt.affinity)
			if diff := cmp.Diff(tt.expected, got); diff != "" {
				t.Errorf("DedupAffinityFields() returned diff (-want, +got):\n%s", diff)
			}
		})
	}
}

func testPodAffinity(t *testing.T) {
	term1 := api.PodAffinityTerm{
		LabelSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"key1": "value1"}},
		TopologyKey:   "node",
	}
	term2 := api.PodAffinityTerm{
		LabelSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"key2": "value2"}},
		TopologyKey:   "node",
	}
	term3 := api.PodAffinityTerm{
		LabelSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"key1": "value1", "key2": "value2"}},
		TopologyKey:   "node",
	}

	tests := []struct {
		name     string
		affinity *api.Affinity
		expected *api.Affinity
	}{
		{
			name: "RequiredDuringSchedulingIgnoredDuringExecution with duplicates",
			affinity: &api.Affinity{
				PodAffinity: &api.PodAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{term1, term2, term1},
				},
			},
			expected: &api.Affinity{
				PodAffinity: &api.PodAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{term1, term2},
				},
			},
		},
		{
			name: "RequiredDuringSchedulingIgnoredDuringExecution with subset",
			affinity: &api.Affinity{
				PodAffinity: &api.PodAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{term1, term3},
				},
			},
			expected: &api.Affinity{
				PodAffinity: &api.PodAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{term3},
				},
			},
		},
		{
			name: "PreferredDuringSchedulingIgnoredDuringExecution with duplicates",
			affinity: &api.Affinity{
				PodAffinity: &api.PodAffinity{
					PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
						{Weight: 1, PodAffinityTerm: term1},
						{Weight: 1, PodAffinityTerm: term2},
						{Weight: 1, PodAffinityTerm: term1},
					},
				},
			},
			expected: &api.Affinity{
				PodAffinity: &api.PodAffinity{
					PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
						{Weight: 1, PodAffinityTerm: term1},
						{Weight: 1, PodAffinityTerm: term2},
					},
				}},
		},
		{
			name: "PreferredDuringSchedulingIgnoredDuringExecution with subset",
			affinity: &api.Affinity{
				PodAffinity: &api.PodAffinity{
					PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
						{Weight: 1, PodAffinityTerm: term1},
						{Weight: 1, PodAffinityTerm: term3},
					},
				},
			},
			expected: &api.Affinity{
				PodAffinity: &api.PodAffinity{
					PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
						{Weight: 1, PodAffinityTerm: term3},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DedupAffinityFields(tt.affinity)
			if diff := cmp.Diff(tt.expected, got); diff != "" {
				t.Errorf("DedupAffinityFields() returned diff (-want, +got):\n%s", diff)
			}
		})
	}
}

func testPodAntiAffinity(t *testing.T) {
	term1 := api.PodAffinityTerm{
		LabelSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"key1": "value1"}},
		TopologyKey:   "node",
	}
	term2 := api.PodAffinityTerm{
		LabelSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"key2": "value2"}},
		TopologyKey:   "node",
	}
	term3 := api.PodAffinityTerm{
		LabelSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"key1": "value1", "key2": "value2"}},
		TopologyKey:   "node",
	}

	tests := []struct {
		name     string
		affinity *api.Affinity
		expected *api.Affinity
	}{
		{
			name: "RequiredDuringSchedulingIgnoredDuringExecution with duplicates",
			affinity: &api.Affinity{
				PodAntiAffinity: &api.PodAntiAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{term1, term2, term1},
				},
			},
			expected: &api.Affinity{
				PodAntiAffinity: &api.PodAntiAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{term1, term2},
				},
			},
		},
		{
			name: "RequiredDuringSchedulingIgnoredDuringExecution with subset",
			affinity: &api.Affinity{
				PodAntiAffinity: &api.PodAntiAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{term1, term3},
				},
			},
			expected: &api.Affinity{
				PodAntiAffinity: &api.PodAntiAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{term3},
				},
			},
		},
		{
			name: "PreferredDuringSchedulingIgnoredDuringExecution with duplicates",
			affinity: &api.Affinity{
				PodAntiAffinity: &api.PodAntiAffinity{
					PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
						{Weight: 1, PodAffinityTerm: term1},
						{Weight: 1, PodAffinityTerm: term2},
						{Weight: 1, PodAffinityTerm: term1},
					},
				},
			},
			expected: &api.Affinity{
				PodAntiAffinity: &api.PodAntiAffinity{
					PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
						{Weight: 1, PodAffinityTerm: term1},
						{Weight: 1, PodAffinityTerm: term2},
					},
				},
			},
		},
		{
			name: "PreferredDuringSchedulingIgnoredDuringExecution with subset",
			affinity: &api.Affinity{
				PodAntiAffinity: &api.PodAntiAffinity{
					PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
						{Weight: 1, PodAffinityTerm: term1},
						{Weight: 1, PodAffinityTerm: term3},
					},
				},
			},
			expected: &api.Affinity{
				PodAntiAffinity: &api.PodAntiAffinity{
					PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
						{Weight: 1, PodAffinityTerm: term3},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DedupAffinityFields(tt.affinity)
			if diff := cmp.Diff(tt.expected, got); diff != "" {
				t.Errorf("DedupAffinityFields() returned diff (-want, +got):\n%s", diff)
			}
		})
	}
}

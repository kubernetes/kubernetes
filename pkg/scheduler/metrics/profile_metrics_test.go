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

package metrics

import (
	"strings"
	"testing"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
)

func TestRecordGeneratedPlacements(t *testing.T) {
	InitMetrics()
	registry := metrics.NewKubeRegistry()
	registry.MustRegister(GeneratedPlacementsTotal)

	RecordGeneratedPlacements("test-profile", PodGroup, 3)
	RecordGeneratedPlacements("test-profile", PodGroup, 2)
	RecordGeneratedPlacements("test-profile-2", PodGroup, 4)
	RecordGeneratedPlacements("test-profile", CompositePodGroup, 1)

	want := `
		# HELP scheduler_generated_placements_total [ALPHA] Number of candidate placements generated when scheduling pod groups, by scheduler profile and entity type.
		# TYPE scheduler_generated_placements_total counter
		scheduler_generated_placements_total{profile="test-profile",type="compositepodgroup"} 1
		scheduler_generated_placements_total{profile="test-profile",type="podgroup"} 5
		scheduler_generated_placements_total{profile="test-profile-2",type="podgroup"} 4
	`
	if err := testutil.GatherAndCompare(registry, strings.NewReader(want), "scheduler_generated_placements_total"); err != nil {
		t.Errorf("unexpected generated_placements_total metric output:\n%v", err)
	}
}

func TestObservePlacementEvaluation(t *testing.T) {
	InitMetrics()
	registry := metrics.NewKubeRegistry()
	registry.MustRegister(PlacementEvaluations)
	registry.MustRegister(PlacementEvaluationDuration)

	ObservePlacementEvaluation(FeasibleResult, "test-profile", PodGroup, 0.5)
	ObservePlacementEvaluation(FeasibleResult, "test-profile", PodGroup, 0.5)
	ObservePlacementEvaluation(InfeasibleResult, "test-profile", PodGroup, 0.1)
	ObservePlacementEvaluation(FeasibleResult, "test-profile-2", PodGroup, 0.2)
	ObservePlacementEvaluation(InfeasibleResult, "test-profile-2", PodGroup, 0.3)
	ObservePlacementEvaluation(FeasibleResult, "test-profile", CompositePodGroup, 0.4)
	ObservePlacementEvaluation(InfeasibleResult, "test-profile", CompositePodGroup, 0.6)

	wantCounter := `
		# HELP scheduler_placement_evaluations_total [ALPHA] Number of candidate placements evaluated when scheduling pod groups, by result, scheduler profile, and entity type. 'feasible' means the pod group fit into the placement, while 'infeasible' means it did not.
		# TYPE scheduler_placement_evaluations_total counter
		scheduler_placement_evaluations_total{profile="test-profile",result="feasible",type="compositepodgroup"} 1
		scheduler_placement_evaluations_total{profile="test-profile",result="feasible",type="podgroup"} 2
		scheduler_placement_evaluations_total{profile="test-profile",result="infeasible",type="compositepodgroup"} 1
		scheduler_placement_evaluations_total{profile="test-profile",result="infeasible",type="podgroup"} 1
		scheduler_placement_evaluations_total{profile="test-profile-2",result="feasible",type="podgroup"} 1
		scheduler_placement_evaluations_total{profile="test-profile-2",result="infeasible",type="podgroup"} 1
	`
	if err := testutil.GatherAndCompare(registry, strings.NewReader(wantCounter), "scheduler_placement_evaluations_total"); err != nil {
		t.Errorf("unexpected placement_evaluations_total metric output:\n%v", err)
	}

	// The duration histogram is recorded alongside the counter, so its sample
	// count per profile/result/type must match the number of evaluations recorded above.
	for _, tc := range []struct {
		profile    string
		result     string
		entityType string
		wantCount  uint64
	}{
		{"test-profile", FeasibleResult, PodGroup, 2},
		{"test-profile", InfeasibleResult, PodGroup, 1},
		{"test-profile-2", FeasibleResult, PodGroup, 1},
		{"test-profile-2", InfeasibleResult, PodGroup, 1},
		{"test-profile", FeasibleResult, CompositePodGroup, 1},
		{"test-profile", InfeasibleResult, CompositePodGroup, 1},
	} {
		gotCount, err := testutil.GetHistogramMetricCount(PlacementEvaluationDuration.WithLabelValues(tc.result, tc.profile, tc.entityType))
		if err != nil {
			t.Errorf("Failed to get sample count for placement_evaluation_duration_seconds{profile=%q,result=%q,type=%q}: %v", tc.profile, tc.result, tc.entityType, err)
			continue
		}
		if gotCount != tc.wantCount {
			t.Errorf("placement_evaluation_duration_seconds{profile=%q,result=%q,type=%q}: got %d samples, want %d", tc.profile, tc.result, tc.entityType, gotCount, tc.wantCount)
		}
	}
}

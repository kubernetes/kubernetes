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

	RecordGeneratedPlacements("test-profile", 3)
	RecordGeneratedPlacements("test-profile", 2)
	RecordGeneratedPlacements("test-profile-2", 4)

	want := `
		# HELP scheduler_generated_placements_total [ALPHA] Number of candidate placements generated when scheduling pod groups.
		# TYPE scheduler_generated_placements_total counter
		scheduler_generated_placements_total{profile="test-profile"} 5
		scheduler_generated_placements_total{profile="test-profile-2"} 4
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

	ObservePlacementEvaluation(FeasibleResult, "test-profile", 0.5)
	ObservePlacementEvaluation(FeasibleResult, "test-profile", 0.5)
	ObservePlacementEvaluation(InfeasibleResult, "test-profile", 0.1)
	ObservePlacementEvaluation(FeasibleResult, "test-profile-2", 0.2)
	ObservePlacementEvaluation(InfeasibleResult, "test-profile-2", 0.3)

	wantCounter := `
		# HELP scheduler_placement_evaluations_total [ALPHA] Number of candidate placements evaluated when scheduling pod groups, by result. 'feasible' means the pod group fit into the placement, while 'infeasible' means it did not.
		# TYPE scheduler_placement_evaluations_total counter
		scheduler_placement_evaluations_total{profile="test-profile",result="feasible"} 2
		scheduler_placement_evaluations_total{profile="test-profile",result="infeasible"} 1
		scheduler_placement_evaluations_total{profile="test-profile-2",result="feasible"} 1
		scheduler_placement_evaluations_total{profile="test-profile-2",result="infeasible"} 1
	`
	if err := testutil.GatherAndCompare(registry, strings.NewReader(wantCounter), "scheduler_placement_evaluations_total"); err != nil {
		t.Errorf("unexpected placement_evaluations_total metric output:\n%v", err)
	}

	// The duration histogram is recorded alongside the counter, so its sample
	// count per profile/result must match the number of evaluations recorded above.
	for _, tc := range []struct {
		profile   string
		result    string
		wantCount uint64
	}{
		{"test-profile", FeasibleResult, 2},
		{"test-profile", InfeasibleResult, 1},
		{"test-profile-2", FeasibleResult, 1},
		{"test-profile-2", InfeasibleResult, 1},
	} {
		gotCount, err := testutil.GetHistogramMetricCount(PlacementEvaluationDuration.WithLabelValues(tc.result, tc.profile))
		if err != nil {
			t.Errorf("Failed to get sample count for placement_evaluation_duration_seconds{profile=%q,result=%q}: %v", tc.profile, tc.result, err)
			continue
		}
		if gotCount != tc.wantCount {
			t.Errorf("placement_evaluation_duration_seconds{profile=%q,result=%q}: got %d samples, want %d", tc.profile, tc.result, gotCount, tc.wantCount)
		}
	}
}

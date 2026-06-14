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

func TestPlacementsGenerated(t *testing.T) {
	InitMetrics()
	registry := metrics.NewKubeRegistry()
	registry.MustRegister(placementTotal)

	PlacementsGenerated("test-profile", 3)
	PlacementsGenerated("test-profile", 2)

	want := `
		# HELP scheduler_placement_total [ALPHA] Number of candidate placements generated when scheduling pod groups.
		# TYPE scheduler_placement_total counter
		scheduler_placement_total{profile="test-profile"} 5
	`
	if err := testutil.GatherAndCompare(registry, strings.NewReader(want), "scheduler_placement_total"); err != nil {
		t.Error(err)
	}
}

func TestPlacementEvaluated(t *testing.T) {
	InitMetrics()
	registry := metrics.NewKubeRegistry()
	registry.MustRegister(placementEvaluations)
	registry.MustRegister(placementEvaluationDuration)

	PlacementEvaluated(FeasibleResult, "test-profile", 0.5)
	PlacementEvaluated(FeasibleResult, "test-profile", 0.5)
	PlacementEvaluated(InfeasibleResult, "test-profile", 0.1)

	wantCounter := `
		# HELP scheduler_placement_evaluations_total [ALPHA] Number of candidate placements evaluated when scheduling pod groups, by result. 'feasible' means the pod group fit into the placement, while 'infeasible' means it did not.
		# TYPE scheduler_placement_evaluations_total counter
		scheduler_placement_evaluations_total{profile="test-profile",result="feasible"} 2
		scheduler_placement_evaluations_total{profile="test-profile",result="infeasible"} 1
	`
	if err := testutil.GatherAndCompare(registry, strings.NewReader(wantCounter), "scheduler_placement_evaluations_total"); err != nil {
		t.Error(err)
	}

	// The duration histogram is recorded alongside the counter, so its sample
	// count per result must match the number of evaluations recorded above.
	for result, wantCount := range map[string]uint64{
		FeasibleResult:   2,
		InfeasibleResult: 1,
	} {
		gotCount, err := testutil.GetHistogramMetricCount(placementEvaluationDuration.WithLabelValues(result, "test-profile"))
		if err != nil {
			t.Errorf("Failed to get sample count for result %q: %v", result, err)
			continue
		}
		if gotCount != wantCount {
			t.Errorf("placement_evaluation_duration_seconds{result=%q}: got %d samples, want %d", result, gotCount, wantCount)
		}
	}
}
